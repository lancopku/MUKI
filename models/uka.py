import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np


class UKABert(BertForSequenceClassification):
    """adaptively learns from the teacher model"""

    def __init__(self, config,
                 teacher1=None, teacher2=None,
                 kd_temperature=1.0,
                 teacher_score_temperature=0.2,
                 eval_strategy='student',
                 vkd_weight=1.0,
                 consistency_schedule=False,
                 hard_teacher=1,
                 no_mc=0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.kd_temperature = kd_temperature
        self.teacher_score_temperature = teacher_score_temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.kd_loss = None
        self.eval_strategy = eval_strategy
        self.cnt = 0
        self.vkd_weight = vkd_weight
        self.vkd_loss = 0.0
        self.consistency_schedule = (consistency_schedule == 1)
        self.hard_teacher = (hard_teacher == 1)
        self.no_mc = (no_mc == 1)  # do not use monte-carlo results

    def _get_entropy_upper_bound(self, num_class):
        avg_prob = 1 / num_class * torch.ones((1, num_class))
        ent = avg_prob * torch.log(avg_prob)
        return - ent.sum()

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
                return_dict=None, teacher_weight=None):
        kd_loss = None
        if self.training:  # kd training
            assert self.teacher1 is not None and self.teacher2 is not None, " hold a None teacher reference"
            student_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            # teacher forward
            with torch.no_grad():
                t1_outputs = self.teacher1(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
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
                    output_hidden_states=False,
                    return_dict=return_dict,
                )
            # logit kl loss
            t1_logits = t1_outputs[0]  # shape of first half label
            t2_logits = t2_outputs[0]  #
            t1_prob = F.softmax(t1_logits / self.kd_temperature, dim=-1)
            t2_prob = F.softmax(t2_logits / self.kd_temperature, dim=-1)

            zero_pad_t1 = torch.zeros_like(t1_prob)
            zero_pad_t2 = torch.zeros_like(t2_prob)
            student_prob = F.softmax(student_logits / self.kd_temperature, dim=-1)  # q(Y = l )

            t1_pad = torch.cat([t1_prob, zero_pad_t2], dim=-1)
            t2_pad = torch.cat([zero_pad_t1, t2_prob], dim=-1)
            kd_loss = 0.
            if self.no_mc:
                # print('use run-time teacher estimation')
                # use run-time estimation
                t1_ent = -torch.sum(torch.log(t1_prob) * t1_prob, dim=-1) / self._get_entropy_upper_bound(t1_logits.size(1))  # bsz
                t2_ent = -torch.sum(torch.log(t2_prob) * t2_prob, dim=-1) / self._get_entropy_upper_bound(t2_logits.size(1))  # bsz
                teacher_weight = torch.cat([1 - t1_ent.unsqueeze(1), 1 - t2_ent.unsqueeze(1)], dim=1)

            assert teacher_weight is not None and len(teacher_weight.size()) == 2, "Required teacher weight"
            if self.hard_teacher:
                partial_prob = torch.where(torch.argmax(teacher_weight, dim=-1).unsqueeze(1) == 0, t1_pad, t2_pad)
            else:
                soft_teacher_weights = F.softmax(teacher_weight / self.teacher_score_temperature, dim=-1)  # bsz ,2
                partial_prob = soft_teacher_weights[:, 0].unsqueeze(1) * t1_pad \
                               + soft_teacher_weights[:, 1].unsqueeze(1) * t2_pad

            if self.consistency_schedule:
                kl_fct = nn.KLDivLoss(reduction='none')
                consistency_score = (teacher_weight[:, 0] - teacher_weight[:, 1]).abs() + 1e-6
                # weighted by two teacher uncertainty difference
                weighted_kd_loss = torch.mean(kl_fct(torch.log(student_prob), partial_prob).sum(-1) * consistency_score)
                kd_loss += self.vkd_weight * weighted_kd_loss
            else:
                kd_loss += self.vkd_weight * self.kl_kd_loss(torch.log(student_prob), partial_prob)
            self.vkd_loss = kd_loss.detach().cpu().item()  # record

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

                t1_prob = F.softmax(t1_logits / self.kd_temperature, dim=-1)
                t2_prob = F.softmax(t2_logits / self.kd_temperature, dim=-1)
                t1_entropy = - torch.sum(t1_prob * torch.log(t1_prob + 1e-6), dim=-1)  # bsz
                t2_entropy = - torch.sum(t2_prob * torch.log(t2_prob + 1e-6), dim=-1)  # bsz
                # print(torch.mean(t1_entropy), torch.mean(t2_entropy))
                t1_score = 1 - t1_entropy / self._get_entropy_upper_bound(t1_prob.size(1))  # bsz
                t2_score = 1 - t2_entropy / self._get_entropy_upper_bound(t2_prob.size(1))  # bsz

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
                student_prob = F.softmax(student_logits, dim=-1)

                t1_score = torch.sum(student_prob[:, :self.num_labels // 2], dim=-1).unsqueeze(1)
                t2_score = torch.sum(student_prob[:, self.num_labels // 2:], dim=-1).unsqueeze(1)

                for t1s, t2s in zip(t1_score, t2_score):
                    print(t1s, t2s)

                ensemble = torch.cat([t1_prob * t1_score, t2_prob * t2_score], dim=-1)  # bsz, num_class

                student_logits = ensemble  # torch.cat([t1_logits, t2_logits], dim=-1)
                student_outputs = (None, None, None)
            elif self.eval_strategy == 't1':
                t1_outputs = self.teacher1(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
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
                    output_hidden_states=True,
                    return_dict=return_dict,
                )

                hiddens = t1_outputs['hidden_states'][-1]  # last layer
                pooled = self.teacher1.bert.pooler(hiddens)  # pooled rep
                #np.save("pooled_t1_%d.npy" % self.cnt, pooled.detach().cpu().numpy())
                #np.save("label_t1_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                self.cnt += 1
                student_logits = torch.cat([t1_outputs[0], torch.zeros_like(t2_outputs[0])],
                                           dim=-1)  # torch.cat([t1_logits, t2_logits], dim=-1)
                student_outputs = (None, None, None)
            elif self.eval_strategy == 't2':
                t1_outputs = self.teacher1(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
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
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
                hiddens = t2_outputs['hidden_states'][-1]  # last layer
                pooled = self.teacher2.bert.pooler(hiddens)  # pooled rep

                #np.save("label_t2_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                #np.save("pooled_t2_%d.npy" % self.cnt, pooled.detach().cpu().numpy())
                self.cnt += 1
                student_logits = torch.cat([torch.zeros_like(t1_outputs[0]), t2_outputs[0]],
                                           dim=-1)  # torch.cat([t1_logits, t2_logits], dim=-1)
                student_outputs = (None, None, None)
            elif self.eval_strategy == 'embedding_student':
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
                student_logits = self.classifier(pooled_output)
                student_prob = F.softmax(student_logits, dim=-1)

                np.save("pooled_output/label_stu_margin_hard_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                np.save("pooled_output/pooled_stu_margin_hard_%d.npy" % self.cnt, pooled_output.detach().cpu().numpy())
                self.cnt += 1
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
        loss = 0.0
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))

        if kd_loss is not None and self.training:  # record eval loss during eval
            loss = kd_loss

        output = (student_logits,)  # + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
