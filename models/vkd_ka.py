import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss


class VKDBert(BertForSequenceClassification):
    """Vanilla KD, directly learns from the concat of teacher distribution"""

    def __init__(self, config,
                 teachers=None,
                 kd_alpha=1,
                 temperature=5.0, eval_strategy='student', overlap_class_num=0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.teachers = teachers
        self.temperature = temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.kd_loss = None
        self.eval_strategy = eval_strategy
        self.overlap_class_num = overlap_class_num

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):
        kd_loss = None
        if self.training:
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)  # bsz, num_class + 2 * over_lap

            t_logits = []
            with torch.no_grad():
                for t_model in self.teachers:
                    t_output = t_model(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict, )
                    t_logits.append(t_output[0])
            # logit kl loss
            concat_teacher_logits = torch.cat(t_logits, dim=1)  # may be need normalize

            kd_loss = self.kl_kd_loss(F.log_softmax(student_logits / self.temperature, dim=-1),
                                      F.softmax(concat_teacher_logits / self.temperature,
                                                dim=-1)) * self.temperature ** 2
            # merge overlap class if necessary
            if self.overlap_class_num > 0:
                assert len(self.teachers) == 2, 'This code only used for 2 teachers now'
                student_logits = torch.cat([student_logits[:, self.teachers[0].config.num_labels],
                                            student_logits[:,
                                            self.teachers[0].config.num_labels + self.overlap_class_num]],
                                           dim=1)  # bsz, num_class
        else:  # use student model for inference
            # -1: teacher_concat 0: student
            # 1 - N : N teacher
            if self.eval_strategy == -1:  #
                t_logits = []
                for t_model in self.teachers:
                    t_output = t_model(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict, )
                    t_logits.append(t_output[0])
                # logit kl loss
                concat_teacher_logits = torch.cat(t_logits, dim=1)  # may be need normalize
                student_logits = concat_teacher_logits
                # merge overlap class if necessary
                if self.overlap_class_num > 0:
                    assert len(self.teachers) == 2, 'This code only used for 2 teachers now'
                    student_logits = torch.cat([student_logits[:, self.teachers[0].config.num_labels],
                                                student_logits[:,
                                                self.teachers[0].config.num_labels + self.overlap_class_num]],
                                               dim=1)  # bsz, num_class
            elif self.eval_strategy == 0:
                student_outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                pooled_output = student_outputs[1]
                pooled_output = self.dropout(pooled_output)
                student_logits = self.classifier(pooled_output)
                # merge overlap class if necessary
                if self.overlap_class_num > 0:
                    assert len(self.teachers) == 2, 'This code only used for 2 teachers now'
                    student_logits = torch.cat([student_logits[:, self.teachers[0].config.num_labels],
                                                student_logits[:,
                                                self.teachers[0].config.num_labels + self.overlap_class_num]],
                                               dim=1)  # bsz, num_class
            elif self.eval_strategy in [k for k in range(1, len(self.teachers) + 1)]:
                t_model = self.teachers[self.eval_strategy - 1]
                t_output = t_model(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, )
                t_logit = t_output[0]
                prefix_len, suffix_len = 0, self.num_labels
                for i in range(len(self.teachers)):  # get prefix
                    if i == self.eval_strategy - 1:
                        suffix_len = self.num_labels - prefix_len - self.teachers[i].num_labels
                        break
                    prefix_len += self.teachers[i].num_labels
                t_prob = F.softmax(t_logit, dim=-1)
                bsz = t_prob.size(0)
                prefix_pad = torch.zeros((bsz, prefix_len)).to(t_prob.device)
                suffix_pad = torch.zeros((bsz, suffix_len)).to(t_prob.device)
                student_logits = torch.cat([prefix_pad, t_prob, suffix_pad], dim=-1)
                # merge overlap class if necessary
                if self.overlap_class_num > 0:
                    assert len(self.teachers) == 2, 'This code only used for 2 teachers now'
                    student_logits = torch.cat([student_logits[:, self.teachers[0].config.num_labels],
                                                student_logits[:,
                                                self.teachers[0].config.num_labels + self.overlap_class_num]],
                                               dim=1)  # bsz, num_class
            else:
                raise ValueError("Unsupported evaluation mode")

        loss = 0.0
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))
        if kd_loss is not None:
            self.kd_loss = kd_loss
            loss = self.kd_alpha * kd_loss

        output = (student_logits,)  # + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
