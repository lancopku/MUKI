import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification
from .cfl import CFLFCBlock, CFLLoss
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

class OracleRelationMarginLoss(nn.Module):
    def __init__(self, max_class, mode='auto', instance_wise=False):
        super(OracleRelationMarginLoss, self).__init__()
        # margin contrastive learning
        assert  mode in ['auto', 'linear-dec', 'linear-inc', 'inv-dec', 'inv-inc']
        self.mode = mode
        self.reduction = 'mean' if not instance_wise else 'none'
        if mode == 'auto':
            self.weights = nn.Parameter(torch.ones((max_class)).uniform_(0, 1))
        elif mode == 'linear-dec':
            k = (1-0.1) / (max_class-2) if max_class > 2 else 0
            self.weights = [0] + [k * i + 0.1 for i in range(max_class -1 )][::-1]  if k > 0 else [0., 1.] # linear decay
        elif mode == 'linear-inc':
            k = (1-0.1) / (max_class-2) if max_class > 2 else 0
            self.weights =  [0.] + [k * i + 0.1 for i in range(max_class -1  )] if k > 0 else [0., 1.]
        elif mode == 'inv-dec':
            self.weights = [0.]  + [ 1 / i  for i in range(1, max_class  )]
        elif mode == 'inv-inc':
            self.weights = [0.]  + [ 1 / i  for i in range(1, max_class  )][::-1]
        else:
            raise ValueError("Unsupported weight type %s" % mode)

    def forward(self,  stu_emb, t1_prob, t2_prob, classifier_weight,
                s_threshold=0.0, t_threshold=0.0, margin=1, cosine=False, labels=None):
        # regularize the student embedding by teacher relation for each instance
        if not cosine:
            loss_fct = torch.nn.TripletMarginLoss(margin=margin, reduction=self.reduction)
        else:
            loss_fct = torch.nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=margin, reduction=self.reduction)

        offset = t1_prob.size(1)
        pos_emb = torch.index_select(classifier_weight, dim=0, index=labels) # ground-truth label embedding as pos

        # construct a label relation vector according to two teacher output
        t1_num_label = t1_prob.size(1)
        t2_num_label = t2_prob.size(1)

        _, t1_ind = torch.topk(t1_prob, k=t1_num_label, dim=-1, largest=True)
        _, t2_ind = torch.topk(t2_prob, k=t2_num_label, dim=-1, largest=True)

        anchor_emb = stu_emb
        classifier_weight = classifier_weight.detach() # detach weight
        loss = 0.
        for i in range(1, t1_num_label): # from t2_view
            t1_min = t1_ind[:, i]
            neg_emb = torch.index_select(classifier_weight, dim=0, index=t1_min)
            weights = self.weights if self.mode != 'auto' else torch.sigmoid(self.weights)
            loss +=  weights[i] * loss_fct(anchor_emb, pos_emb, neg_emb)

        for i in range(1, t2_num_label): # from t1_view
            t2_min = t2_ind[:, i]
            neg_emb = torch.index_select(classifier_weight, dim=0, index=t2_min + offset)
            weights = self.weights if self.mode != 'auto' else torch.sigmoid(self.weights)
            loss += weights[i] * loss_fct(anchor_emb, pos_emb, neg_emb)

        return loss

class RelationMarginLoss(nn.Module):
    def __init__(self, max_class, mode='auto', instance_wise=False):
        super(RelationMarginLoss, self).__init__()
        # margin contrastive learning
        assert  mode in ['auto', 'linear-dec', 'linear-inc', 'inv-dec', 'inv-inc']
        self.mode = mode
        self.reduction = 'mean' if not instance_wise else 'none'
        if mode == 'auto':
            self.weights = nn.Parameter(torch.ones((max_class)).uniform_(0, 1))
        elif mode == 'linear-dec':
            k = (1-0.1) / (max_class-2) if max_class > 2 else 0 
            self.weights = [0] + [k * i + 0.1 for i in range(max_class -1 )][::-1]  if k > 0 else [0., 1.] # linear decay
        elif mode == 'linear-inc':
            k = (1-0.1) / (max_class-2) if max_class > 2 else 0 
            self.weights =  [0.] + [k * i + 0.1 for i in range(max_class -1  )] if k > 0 else [0., 1.]
        elif mode == 'inv-dec':
            self.weights = [0.]  + [ 1 / i  for i in range(1, max_class  )]
        elif mode == 'inv-inc':
            self.weights = [0.]  + [ 1 / i  for i in range(1, max_class  )][::-1]
        else:
            raise ValueError("Unsupported weight type %s" % mode)

    def forward(self,  stu_emb, t1_prob, t2_prob, classifier_weight,
                s_threshold=0.0, t_threshold=0.0, margin=1, cosine=False, labels=None):
        # regularize the student embedding by teacher relation for each instance
        if not cosine:
            loss_fct = torch.nn.TripletMarginLoss(margin=margin, reduction=self.reduction)
        else:
            loss_fct = torch.nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=margin, reduction=self.reduction)

        offset = t1_prob.size(1)

        # construct a label relation vector according to two teacher output
        t1_max = torch.argmax(t1_prob, dim=-1)
        t2_max = torch.argmax(t2_prob, dim=-1)
        t1_num_label = t1_prob.size(1)
        t2_num_label = t2_prob.size(1)

        _, t1_ind = torch.topk(t1_prob, k=t1_num_label, dim=-1, largest=True)
        _, t2_ind = torch.topk(t2_prob, k=t2_num_label, dim=-1, largest=True)

        anchor_emb = stu_emb
        classifier_weight = classifier_weight.detach() # detach weight
        loss = 0.
        for i in range(1, t1_num_label): # from t2_view
            t1_min = t1_ind[:, i ]
            pos_emb = torch.index_select(classifier_weight, dim=0, index=t2_max + offset)
            neg_emb = torch.index_select(classifier_weight, dim=0, index=t1_min)
            weights = self.weights if self.mode != 'auto' else torch.sigmoid(self.weights)
            loss +=  weights[i] * loss_fct(anchor_emb, pos_emb, neg_emb)

        for i in range(1, t2_num_label): # from t1_view
            t2_min = t2_ind[:, i]
            pos_emb = torch.index_select(classifier_weight, dim=0, index=t1_max)
            neg_emb = torch.index_select(classifier_weight, dim=0, index=t2_min + offset)
            weights = self.weights if self.mode != 'auto' else torch.sigmoid(self.weights)
            loss += weights[i] * loss_fct(anchor_emb, pos_emb, neg_emb)

        return loss


class AKABert(BertForSequenceClassification):
    """adaptively learns from the teacher model"""

    def __init__(self, config,
                 teacher1=None, teacher2=None,
                 kd_alpha=1,
                 temperature=0.2, eval_strategy='student',
                 sts_weight=0.0,vkd_weight=1.0,relation_weight=0.0, margin=1.0, mode='auto', max_class=-1,
                 dynamic_margin=False, oracle_margin_loss=True, consistency_schedule=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.temperature = temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.kd_loss = None
        self.eval_strategy = eval_strategy
        self.cnt = 0
        self.sts_weight = sts_weight  # self-boosted superivsion
        self.vkd_weight = vkd_weight
        self.vkd_loss = 0.0
        self.relation_loss = 0.0
        self.relation_weight = relation_weight
        self.margin_loss = RelationMarginLoss(max_class=max_class, mode=mode, instance_wise=dynamic_margin) \
            if not oracle_margin_loss else OracleRelationMarginLoss(max_class=max_class, mode=mode, instance_wise=dynamic_margin)
        self.margin = margin
        self.dynamic_margin = dynamic_margin
        self.consistency_schedule = consistency_schedule
        

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
        if self.training:
            assert self.teacher1 is not None and self.teacher2 is not None, " hold a None teacher reference"
            student_outputs = self.bert(
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
            # logit kl loss
            t1_logits = t1_outputs[0]  # shape of first half label
            t2_logits = t2_outputs[0]  #
            t1_prob = F.softmax(t1_logits, dim=-1)
            t2_prob = F.softmax(t2_logits, dim=-1)

            zero_pad_t1 = torch.zeros_like(t1_prob)
            zero_pad_t2 = torch.zeros_like(t2_prob)
            student_prob = F.softmax(student_logits, dim=-1)  # q(Y = l )
            label_per_teacher = self.num_labels // 2

            t1_pad = torch.cat([t1_prob, zero_pad_t2], dim=-1)
            t2_pad = torch.cat([zero_pad_t1, t2_prob], dim=-1)
            kd_loss = 0. 
            if teacher_weight is not None:
                if len(teacher_weight.size()) == 1:  # hard:
                    partial_prob = torch.where(teacher_weight.unsqueeze(1) == 0, t1_pad, t2_pad)
                elif len(teacher_weight.size()) == 2:  # soft
                    soft_teacher_weights = F.softmax(teacher_weight / self.temperature, dim=-1) # bsz ,2
                    partial_prob = soft_teacher_weights[:, 0].unsqueeze(1) * t1_pad + soft_teacher_weights[:, 1].unsqueeze(1) * t2_pad 
                else:
                    raise ValueError("Unsupported teacher weight shape")
            else:
                partial_prob = torch.where(torch.argmax(student_logits, dim=-1).unsqueeze(1) < label_per_teacher, t1_pad, t2_pad)


            if self.consistency_schedule:
                assert teacher_weight.size() == 2
                kl_fct = nn.KLDivLoss(reduction='none')
                consistency_score = (teacher_weight[:, 0] - teacher_weight[:, 1]).abs() + 1e-6
                # weighted by two teacher uncertainty difference
                weighted_kd_loss =  torch.mean(kl_fct(torch.log(student_prob), partial_prob).sum(-1) * consistency_score)
                kd_loss += self.vkd_weight * weighted_kd_loss
            else:
                kd_loss += self.vkd_weight * self.kl_kd_loss(torch.log(student_prob), partial_prob)
            self.vkd_loss = kd_loss.detach().cpu().item() # record
            if self.relation_weight > 0:
                relation_loss = self.margin_loss(pooled_output, t1_prob, t2_prob, self.classifier.weight,
                                                 margin=self.margin, cosine=False, labels=labels)
                # deal with dynamic loss
                if self.dynamic_margin:
                    stu_entropy = - torch.sum(student_prob * torch.log(student_prob + 1e-6), dim=-1)
                    # higher confidence need less relation_loss
                    confidence =  stu_entropy / self._get_entropy_upper_bound(student_prob.size(1))  # bsz
                    relation_loss = torch.mean(relation_loss * confidence) # mean over
                    # print(confidence, relation_loss)
                self.relation_loss = relation_loss.detach().cpu().item()
                kd_loss += self.relation_weight * relation_loss

            if self.sts_weight > 0 : # add student-aided knowledge fusion
                #t1_weight = torch.sum(student_prob[:, :label_per_teacher], dim=-1, keepdim=True) # bsz,
                #t2_weight = torch.sum(student_prob[:, label_per_teacher:], dim=-1, keepdim=True) # bsz,
                #fused_prob = t1_weight * t1_pad + t2_weight * t2_pad
                kd_loss += self.sts_weight * self.kl_kd_loss(torch.log(student_prob), partial_prob)

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

                t1_prob = F.softmax(t1_logits / self.temperature, dim=-1)
                t2_prob = F.softmax(t2_logits / self.temperature, dim=-1)
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
            elif self.eval_strategy == 'embedding_t1':
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
                hiddens = t1_outputs['hidden_states'][-1] # last layer
                pooled = self.teacher1.bert.pooler(hiddens) # pooled rep
                np.save("pooled_t1_%d.npy"%  self.cnt, pooled.detach().cpu().numpy())
                np.save("label_t1_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                self.cnt += 1
                student_logits = torch.cat( [t1_outputs[0], torch.zeros_like(t1_outputs[0])], dim=-1)  # torch.cat([t1_logits, t2_logits], dim=-1)
                student_outputs = (None, None, None)
            elif self.eval_strategy == 'embedding_t2':
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
                hiddens = t2_outputs['hidden_states'][-1] # last layer
                pooled = self.teacher2.bert.pooler(hiddens) # pooled rep

                np.save("label_t2_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                np.save("pooled_t2_%d.npy"%  self.cnt, pooled.detach().cpu().numpy())
                self.cnt += 1
                student_logits = torch.cat( [t2_outputs[0], torch.zeros_like(t2_outputs[0])], dim=-1)  # torch.cat([t1_logits, t2_logits], dim=-1)
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
                np.save("pooled_output/pooled_stu_margin_hard_%d.npy"%  self.cnt, pooled_output.detach().cpu().numpy())
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

        if kd_loss is not None and self.training: # record eval loss during eval
            loss = kd_loss

        output = (student_logits,)  # + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
