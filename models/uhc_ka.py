import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss


class UHCBert(BertForSequenceClassification):
    def __init__(self, config,
                 teachers=None,
                 kd_alpha=1,
                 temperature=5.0, eval_strategy='student'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.teachers = teachers
        self.temperature = temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.kd_loss = None
        self.eval_strategy = eval_strategy

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
            student_logits = self.classifier(pooled_output)

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

            kd_loss = 0.
            original_prob = F.softmax(student_logits / self.temperature, dim=-1)  # q(Y = l )
            label_cnt = 0
            for t_model, t_logit in zip(self.teachers, t_logits):
                t_num = t_model.num_labels
                split_prob = original_prob[:, label_cnt: label_cnt + t_num] / torch.sum(
                    original_prob[:, label_cnt: label_cnt + t_num], dim=-1, keepdim=True)
                kd_loss += self.kl_kd_loss(torch.log(split_prob), F.softmax(t_logit / self.temperature, dim=-1))
                label_cnt += t_num

            kd_loss = kd_loss * self.temperature ** 2

        else:  # use student model for inference

            if self.eval_strategy == 'teacher_concat':
                t1_outputs = self.teacher1(
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

                t2_outputs = self.teacher2(
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

                # logit kl loss
                t1_logits = t1_outputs[0]
                t2_logits = t2_outputs[0]
                student_logits = torch.cat([t1_logits, t2_logits], dim=-1)
                student_outputs = (None, None, None)
            else:
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
            # print(student_logits.size())
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
