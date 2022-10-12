import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss


class UKABert(BertForSequenceClassification):
    """adaptively learns from the teacher model"""

    def __init__(self, config,
                 teachers=None,
                 kd_temperature=1.0,
                 teacher_score_temperature=0.2,
                 eval_strategy='student',
                 vkd_weight=1.0,
                 consistency_schedule=False,
                 hard_teacher=1,
                 no_mc=0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.teachers = teachers  # list of teacher model
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

            t_logits = []
            # teacher forward
            with torch.no_grad():
                for t_model in self.teachers:
                    teacher_output = t_model(input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids,
                                             position_ids=position_ids,
                                             head_mask=head_mask,
                                             inputs_embeds=inputs_embeds,
                                             output_attentions=output_attentions,
                                             output_hidden_states=False,
                                             return_dict=return_dict, )
                    t_logits.append(teacher_output[0])

            teacher_probs = [F.softmax(t_logit / self.kd_temperature, dim=-1) for t_logit in t_logits]

            # pad teacher model
            prefix_len = 0
            padded_probs = []
            for t_prob in teacher_probs:
                bsz = t_prob.size(0)
                prefix_pad = torch.zeros((bsz, prefix_len)).to(t_prob.device)
                pad_prob = torch.cat([prefix_pad, t_prob], dim=-1)
                suffix_pad = torch.zeros((bsz, self.num_labels - prefix_len - t_prob.size(1))).to(t_prob.device)
                pad_prob = torch.cat([pad_prob, suffix_pad], dim=-1)
                padded_probs.append(pad_prob)
                prefix_len += t_prob.size(1)

            student_prob = F.softmax(student_logits / self.kd_temperature, dim=-1)  # q(Y = l )

            kd_loss = 0.
            if self.no_mc:
                teacher_ents = [
                    -torch.sum(torch.log(t_prob) * t_prob, dim=-1) / self._get_entropy_upper_bound(t_prob.size(1)) for
                    t_prob in teacher_probs]
                teacher_weight = torch.cat([1 - t_ent.unsqueeze(1) for t_ent in teacher_ents], dim=1)  # bsz, n_teacher
            teacher_prob_stacked = torch.stack(padded_probs, dim=0).transpose(1, 0)  # bsz, num_teacher, num_labels
            assert teacher_weight is not None and len(teacher_weight.size()) == 2, "Required teacher weight"
            if self.hard_teacher:
                teacher_index = torch.argmax(teacher_weight, dim=-1).unsqueeze(1)  # bsz, 1
                indices = teacher_index.repeat(1, student_prob.size(1))  # bsz, num_labels
                indices = indices.unsqueeze(1)  # bsz, 1, num_labels
                partial_prob = torch.gather(teacher_prob_stacked, dim=1, index=indices).squeeze()
                assert partial_prob.size() == (teacher_weight.size(0), self.num_labels)
            else:
                soft_teacher_weights = F.softmax(teacher_weight / self.teacher_score_temperature, dim=-1)  # bsz , t_num
                soft_teacher_weights = soft_teacher_weights.unsqueeze(-1)  # bsz, t_num, 1
                partial_prob = torch.sum(soft_teacher_weights * teacher_prob_stacked, dim=1)  # bsz, num_labels

            if self.consistency_schedule:
                kl_fct = nn.KLDivLoss(reduction='none')
                scores, _ = torch.topk(teacher_weight, k=2, dim=-1)
                consistency_score = (scores[:, 0] - scores[:, 1]).abs() + 1e-6
                # weighted by two teacher uncertainty difference
                weighted_kd_loss = torch.mean(kl_fct(torch.log(student_prob), partial_prob).sum(-1) * consistency_score)
                kd_loss += self.vkd_weight * weighted_kd_loss
            else:
                kd_loss += self.vkd_weight * self.kl_kd_loss(torch.log(student_prob), partial_prob)
            self.vkd_loss = kd_loss.detach().cpu().item()  # record

        elif self.eval_strategy == 'analysis':  # compare KL & gold teacher
            bsz = input_ids.size(0)
            student_logits = torch.zeros((bsz, self.num_labels)).to(input_ids.device)
            t_logits = []
            with torch.no_grad():
                for t_model in self.teachers:
                    teacher_output = t_model(input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids,
                                             position_ids=position_ids,
                                             head_mask=head_mask,
                                             inputs_embeds=inputs_embeds,
                                             output_attentions=output_attentions,
                                             output_hidden_states=False,
                                             return_dict=return_dict, )
                    t_logits.append(teacher_output[0])

            teacher_probs = [F.softmax(t_logit, dim=-1) for t_logit in t_logits]
            gold_teacher = teacher_probs[-1]  # gold
            cat_logit = torch.cat(t_logits[:-1], dim=-1)  # bsz, num_labels
            cat_prob = F.softmax(cat_logit, dim=-1)

            cat2gold = self.kl_kd_loss(torch.log(cat_prob), gold_teacher)
            prefix_len = 0
            padded_probs = []
            for t_prob in teacher_probs[:-1]:
                bsz = t_prob.size(0)
                prefix_pad = torch.zeros((bsz, prefix_len)).to(t_prob.device)
                pad_prob = torch.cat([prefix_pad, t_prob], dim=-1)
                suffix_pad = torch.zeros((bsz, self.num_labels - prefix_len - t_prob)).to(t_prob.device)
                pad_prob = torch.cat([pad_prob, suffix_pad], dim=-1)
                padded_probs.append(pad_prob)
                prefix_len += t_prob.size(1)

            teacher_prob_stacked = torch.stack(padded_probs, dim=0).transpose(1, 0)  # bsz, num_teacher, num_labels

            soft_teacher_weights = F.softmax(teacher_weight / self.teacher_score_temperature, dim=-1)  # bsz , t_num
            soft_teacher_weights = soft_teacher_weights.unsqueeze(-1)  # bsz, t_num, 1
            soft_prob = torch.sum(soft_teacher_weights * teacher_prob_stacked, dim=1)  # bsz, num_labels
            teacher_index = torch.argmax(teacher_weight, dim=-1).unsqueeze(1)  # bsz, 1
            indices = teacher_index.repeat(1, student_logits.size(1))  # bsz, num_labels
            indices = indices.unsqueeze(1)  # bsz, 1, num_labels
            hard_prob = torch.gather(teacher_prob_stacked, dim=1, index=indices).squeeze()

            hard2gold = self.kl_kd_loss(torch.log(hard_prob), gold_teacher)
            soft2gold = self.kl_kd_loss(torch.log(soft_prob), gold_teacher)
            print('concat: ', cat2gold, 'hard: ', hard2gold, 'soft: ', soft2gold)
        else:  # use student model for inference

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
