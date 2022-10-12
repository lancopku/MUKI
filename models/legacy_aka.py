import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification
from .cfl import CFLFCBlock, CFLLoss
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)  #
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


def consistency(t1_d, t2_d):
    relative_t1 = t1_d.unsqueeze(2) - t1_d.unsqueeze(1)
    relative_t2 = t2_d.unsqueeze(2) - t2_d.unsqueeze(1)

    ind1 = torch.where(relative_t1 >= 0, 1, 0)  # N, N, N
    ind2 = torch.where(relative_t2 >= 0, 1, 0)  # N, N, N
    consistent = (ind1 == ind2)
    # consistent= consistent.int().fill_diagonal_(0)
    print('teacher consistentcy score:', consistent.float().mean())


class MutualCalibratedLoss(nn.Module):
    def forward(self, t1_mapped, t1_origin, t2_mapped, t2_origin, labels=None, num_labels=-1):
        bsz = t1_mapped.size(0)
        # N x C
        # N x N x C
        with torch.no_grad():
            t1_d = pdist(t1_origin, squared=False)
            mean_td = t1_d[t1_d > 0].mean()
            t1_d = (t1_d / mean_td).view(-1)

            t2_d = pdist(t2_origin, squared=False)
            mean_td = t2_d[t2_d > 0].mean()
            t2_d = (t2_d / mean_td).view(-1)

        consistency(t1_d.view(bsz, bsz), t2_d.view(bsz, bsz))

        # higher distance ->
        t1_idx = torch.argsort(t1_d, descending=True)  # ascending order of distance
        t2_idx = torch.argsort(t2_d, descending=True)  # ascending order of distance

        if labels is not None:
            t1_label_indicator = torch.lt(labels, num_labels // 2)
            t2_label_indicator = ~t1_label_indicator
            # 1 denotes that two instances belone to one teacher
            same_teacher_indicator = (
                        t1_label_indicator.unsqueeze(1).float() * t2_label_indicator.unsqueeze(0).float()).view(
                -1)  # N x N
            # print('t1 same teacher accuracy:', same_teacher_indicator[t1_idx[: bsz // 2 ]].mean())
            # print('t2 same teacher accuracy:', same_teacher_indicator[t2_idx[: bsz // 2 ]].mean())

        # T1 self-alignment loss

        t1m_d = pdist(t1_mapped, squared=False)
        mean_td = t1m_d[t1m_d > 0].mean()
        t1m_d = (t1m_d / mean_td).view(-1)

        t2m_d = pdist(t2_mapped, squared=False)
        mean_td = t2m_d[t2m_d > 0].mean()
        t2m_d = (t2m_d / mean_td).view(-1)

        t1_sa = F.smooth_l1_loss(t1m_d, t1_d, reduction='mean')
        # t1_sa = F.smooth_l1_loss(t1m_d[t1_idx[: bsz // 2 ]], t1_d[t1_idx[: bsz // 2 ]], reduction='mean')
        # T1, calibrated by T2 for pair of larger distances

        t1_ca = F.smooth_l1_loss(t1m_d, t2_d, reduction='mean')
        # t1_ca = F.smooth_l1_loss(t1m_d[t1_idx[bsz // 2 :]], t2_d[t1_idx[bsz // 2 :]], reduction='mean')
        # T2 self-alignment loss
        t2_sa = F.smooth_l1_loss(t2m_d, t2_d, reduction='mean')
        # t2_sa = F.smooth_l1_loss(t2m_d[t2_idx[: bsz // 2]], t2_d[t2_idx[: bsz // 2]], reduction='mean')
        # T2, calibrated by T1 for pair of larger distances

        t2_ca = F.smooth_l1_loss(t2m_d, t1_d, reduction='mean')
        # t2_ca = F.smooth_l1_loss(t2m_d[t2_idx[bsz // 2:]], t1_d[t2_idx[bsz // 2:]], reduction='mean')

        loss = t1_sa + t1_ca + t2_sa + t2_ca
        # print('fusion loss: ', t1_sa, t1_ca, t2_sa, t2_ca)
        return loss


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)  # can be a scalar, when Py


class OracleLoss(nn.Module):
    def forward(self, t1_prob, t2_prob, classifier_weight, threshold=0.5):
        oracle_relation_gs = torch.tensor([[0.9895, 1.0048, 0.9879, 1.0280],
                                           [0.9651, 0.9748, 1.0000, 1.0264],
                                           [1.0067, 0.9939, 1.0166, 1.0122],
                                           [0.9793, 1.0044, 0.9858, 1.0247]]).to(classifier_weight.device)
        weight_1 = classifier_weight[:t1_prob.size(1), :]  # n_class1, hidden_dim
        weight_2 = classifier_weight[t1_prob.size(1):, :]  # n_class2, hidden_dim

        dist = torch.cdist(weight_1.unsqueeze(0),
                           weight_2.unsqueeze(0), p=2).squeeze()
        dist = dist / dist[dist > 0].mean()
        loss = F.smooth_l1_loss(dist, oracle_relation_gs, reduction='mean')
        return loss


class AlignLoss(nn.Module):
    def forward(self, t1_prob, t2_prob, classifier_weight, threshold=0.0):
        if threshold > 0:  # confidence threshold
            conf1, _ = torch.max(t1_prob, dim=-1)
            conf2, _ = torch.max(t2_prob, dim=-1)
            t1_prob = t1_prob[(conf1 > threshold) & (conf2 > threshold)]
            t2_prob = t2_prob[(conf1 > threshold) & (conf2 > threshold)]

        # cls_weight, n_dim x n_class, can be viewed as a prototypical vector
        # construct a label relation vector according to two teacher output
        t1_max = torch.argmax(t1_prob, dim=-1)
        t2_max = torch.argmax(t2_prob, dim=-1)
        t1_min = torch.argmin(t1_prob, dim=-1)
        t2_min = torch.argmin(t2_prob, dim=-1)

        # classifier_weight: [n_dim, n_class]  n_class = n_class_t1 + n_class_t2
        t1_max_oh = F.one_hot(t1_max, num_classes=t1_prob.size(1)).float()  # N, cls_a
        t1_min_oh = F.one_hot(t1_min, num_classes=t1_prob.size(1)).float()  #
        t2_max_oh = F.one_hot(t2_max, num_classes=t2_prob.size(1)).float()
        t2_min_oh = F.one_hot(t2_min, num_classes=t2_prob.size(1)).float()

        t1_similar = (torch.matmul(t1_max_oh.T, t2_max_oh))
        t1_dissimilar = (torch.matmul(t1_max_oh.T, t2_min_oh))
        co_matrix1 = t1_similar - t1_dissimilar

        t2_similar = (torch.matmul(t2_max_oh.T, t1_max_oh))
        t2_dissimilar = (torch.matmul(t2_max_oh.T, t1_min_oh))
        co_matrix2 = t2_similar - t2_dissimilar
        # merge to a unified relation view
        co_matrix1 = F.softmax(co_matrix1, dim=-1)
        co_matrix2 = F.softmax(co_matrix2, dim=-1)
        co_matrix = (co_matrix1 + co_matrix2.T) / 2  # n_class1, n_class 2
        weight_1 = classifier_weight[:t1_prob.size(1), :]  # n_class1, hidden_dim
        weight_2 = classifier_weight[t1_prob.size(1):, :]  # n_class2, hidden_dim

        dist = torch.cdist(weight_1.unsqueeze(0),
                           weight_2.unsqueeze(0))

        loss = F.kl_div(F.log_softmax(dist.squeeze(0), dim=-1), co_matrix, reduction='batchmean')
        return loss


class RelationMarginLoss(nn.Module):

    def forward(self, stu_emb, t1_prob, t2_prob, classifier_weight, s_threshold=0.0, t_threshold=0.0, margin=1,
                cosine=False, hard_neg=False):
        # regularize the student embedding by teacher relation for each instance
        if not cosine:
            loss_fct = torch.nn.TripletMarginLoss(margin=margin)
        else:
            loss_fct = torch.nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=margin)

        offset = t1_prob.size(1)

        # construct a label relation vector according to two teacher output
        t1_max = torch.argmax(t1_prob, dim=-1)
        t2_max = torch.argmax(t2_prob, dim=-1)
        t1_num_label = t1_prob.size(1)
        t2_num_label = t2_prob.size(1)
        # linear decay negative pairs
        _, t1_ind = torch.topk(t1_prob, k=t1_num_label, dim=-1, largest=True)
        _, t2_ind = torch.topk(t2_prob, k=t2_num_label, dim=-1, largest=True)

        anchor_emb = stu_emb
        classifier_weight = classifier_weight.detach()  # detach weight
        weights = [0.1 * i + 0.2 for i in range(10)][::-1]  # linear decay
        loss = 0.
        for i in range(1, t1_num_label):  # from t2_view
            t1_min = t1_ind[:, i]
            pos_emb = torch.index_select(classifier_weight, dim=0, index=t2_max + offset)
            neg_emb = torch.index_select(classifier_weight, dim=0, index=t1_min)
            loss += weights[i] * loss_fct(anchor_emb, pos_emb, neg_emb)

        for i in range(1, t2_num_label):  # from t1_view
            t2_min = t2_ind[:, i]
            pos_emb = torch.index_select(classifier_weight, dim=0, index=t1_max)
            neg_emb = torch.index_select(classifier_weight, dim=0, index=t2_min + offset)
            loss += weights[i] * loss_fct(anchor_emb, pos_emb, neg_emb)

        return loss


class AKABert(BertForSequenceClassification):
    """adaptively learns from the teacher model"""

    def __init__(self, config,
                 teacher1=None, teacher2=None,
                 kd_alpha=1,
                 temperature=5.0, eval_strategy='student',
                 kd_hard_alpha=0.0,
                 kd_soft_alpha=1.0,
                 sts_weight=0.0,
                 rkd_angle_weight=0.0,
                 rkd_dist_weight=0.0, vkd_weight=1.0, align_weight=0.0, relation_weight=0.0, margin=1.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.temperature = temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.kd_loss = None
        self.kd_hard_alpha = kd_hard_alpha
        self.kd_soft_alpha = kd_soft_alpha
        self.eval_strategy = eval_strategy
        self.pseudo_label_loss = CrossEntropyLoss()
        self.mse_loss = MSELoss()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.cnt = 0
        self.sts_weight = sts_weight  # self-boosted superivsion
        self.rkd_angle_weight = rkd_angle_weight  # relational kd weight
        self.rkd_dist_weight = rkd_dist_weight  # relational kd weight
        self.vkd_weight = vkd_weight
        self.angle_criterion = RKdAngle()
        self.dist_criterion = RkdDistance()
        self.fusion_loss = OracleLoss()  # AlignLoss()
        self.align_weight = align_weight
        self.vkd_loss = 0.0
        self.relation_loss = 0.0
        self.relation_weight = relation_weight
        self.margin_loss = RelationMarginLoss()
        self.margin = margin

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

            if teacher_weight is not None:
                if len(teacher_weight.size()) == 1:  # hard:
                    partial_prob = torch.where(teacher_weight.unsqueeze(1) == 0, t1_pad, t2_pad)
                elif len(teacher_weight.size()) == 2:  # soft
                    pass
                else:
                    raise ValueError("Unsupported teacher weight shape")
            else:
                partial_prob = torch.where(labels.unsqueeze(1) < label_per_teacher, t1_pad, t2_pad)

            kd_loss = self.vkd_weight * self.kl_kd_loss(torch.log(student_prob), partial_prob)
            self.vkd_loss = kd_loss.detach().cpu().item()
            if self.relation_weight > 0:
                relation_loss = self.margin_loss(pooled_output, t1_prob, t2_prob, self.classifier.weight,
                                                 margin=self.margin, cosine=False, hard_neg=True)
                self.relation_loss = relation_loss.detach().cpu().item()
                kd_loss += self.relation_weight * relation_loss

            if self.align_weight > 0:
                align_loss = self.fusion_loss(t1_prob, t2_prob, self.classifier.weight, 0.9)
                self.relation_loss = (self.align_weight * align_loss).detach().cpu().item()
                kd_loss += self.align_weight * align_loss

            if self.sts_weight > 0:  # add student-aided knowledge fusion
                t1_weight = torch.sum(student_prob[:, :label_per_teacher], dim=-1, keepdim=True)  # bsz,
                t2_weight = torch.sum(student_prob[:, label_per_teacher:], dim=-1, keepdim=True)  # bsz,
                fused_prob = t1_weight * t1_pad + t2_weight * t2_pad
                kd_loss += self.sts_weight * self.kl_kd_loss(torch.log(student_prob), fused_prob)

            if self.rkd_dist_weight > 0 or self.rkd_angle_weight > 0:  # add relational kd
                t1_hidden = t1_outputs["hidden_states"][-1]
                with torch.no_grad():
                    t1_hidden = self.teacher1.bert.pooler(t1_hidden)

                t2_hidden = t2_outputs["hidden_states"][-1]
                with torch.no_grad():
                    t2_hidden = self.teacher2.bert.pooler(t2_hidden)

                t1_hidden = torch.mean(t1_outputs["hidden_states"][-2], dim=1)
                t2_hidden = torch.mean(t2_outputs["hidden_states"][-2], dim=1)
                # fused_teacher feature
                t1_mapped = self.dense1(t1_hidden)
                t2_mapped = self.dense2(t2_hidden)

                # align t1 & t2
                fusion_loss = self.fusion_loss(t1_mapped, t1_hidden, t2_mapped, t2_hidden, labels,
                                               num_labels=student_prob.size(-1))

                # Align fused teacher & student
                fused_teacher_hidden = (t1_mapped + t2_mapped) / 2  # F.normalize(
                student_norm = pooled_output  #
                angle_loss = self.angle_criterion(student_norm, fused_teacher_hidden)
                dist_loss = self.dist_criterion(student_norm, fused_teacher_hidden)
                kd_loss += self.rkd_angle_weight * angle_loss + self.rkd_dist_weight * dist_loss + fusion_loss * 10

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
                hiddens = t1_outputs['hidden_states'][-1]  # last layer
                pooled = self.teacher1.bert.pooler(hiddens)  # pooled rep
                np.save("pooled_t1_%d.npy" % self.cnt, pooled.detach().cpu().numpy())

                np.save("label_t1_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                self.cnt += 1
                student_logits = torch.cat([t1_outputs[0], torch.zeros_like(t1_outputs[0])],
                                           dim=-1)  # torch.cat([t1_logits, t2_logits], dim=-1)
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
                hiddens = t2_outputs['hidden_states'][-1]  # last layer
                pooled = self.teacher2.bert.pooler(hiddens)  # pooled rep

                np.save("label_t2_%d.npy" % self.cnt, labels.detach().cpu().numpy())
                np.save("pooled_t2_%d.npy" % self.cnt, pooled.detach().cpu().numpy())
                self.cnt += 1
                student_logits = torch.cat([t2_outputs[0], torch.zeros_like(t2_outputs[0])],
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
