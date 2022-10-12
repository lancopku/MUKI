import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

class MCBert(BertForSequenceClassification):
    """adaptively learns from the teacher model"""

    def __init__(self, config,
                 teachers=None,
                 kd_alpha=1,
                 temperature=5.0, eval_strategy='student',
                 sts_weight=0.0, vkd_weight=1.0, relation_weight=0.0, margin=1.0, mode='auto'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.teachers = None  
        self.temperature = temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.kd_loss = None
        self.eval_strategy = eval_strategy
        self.cnt = 1
        self.acc = 0.0
        self.sts_weight = sts_weight  # self-boosted superivsion
        self.vkd_weight = vkd_weight
        self.vkd_loss = 0.0
        self.relation_loss = 0.0
        self.relation_weight = relation_weight
        self.margin = margin
        self.sacc = 0.0

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
        monte_carlo_K = 16 
        if self.training or not self.training:
            # assert self.teacher1 is not None and self.teacher2 is not None, " hold a None teacher reference"
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
                # print(self.teacher1.config.hidden_dropout_prob, self.teacher2.config.hidden_dropout_prob)
                probs = []
                for m in range(monte_carlo_K):
                    for i, t_model in enumerate(self.teachers):
                        # print(i, t_model.device, input_ids.device)
                        teacher_output = t_model(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 position_ids=position_ids,
                                                 head_mask=head_mask,
                                                 inputs_embeds=inputs_embeds,
                                                 output_attentions=output_attentions,
                                                 output_hidden_states=False,
                                                 return_dict=return_dict, )

                        teacher_logit = teacher_output[0]
                        teacher_prob = F.softmax(teacher_logit, dim=-1)
                        # print(teacher_prob)
                        if m == 0:
                            probs.append(teacher_prob)  #
                        else:
                            probs[i] += teacher_prob
                #print(len(probs))
                probs = [prob / monte_carlo_K for prob in probs]
                #print(probs)
                uncertainty = [-torch.sum(torch.log(prob) * prob, dim=-1) / self._get_entropy_upper_bound(
                    prob.size(1)) for prob in probs]
                # print(uncertainty)
                uncertainty_score = [1 - u.unsqueeze(1) for u in uncertainty]
                #print(uncertainty_score)
                teacher_score = torch.cat(uncertainty_score, dim=1)  # bsz, num_teacher
                # print(teacher_score.shape)
                np.save("teacher_md/gs_4teacher_%d.npy" % self.cnt , teacher_score.detach().cpu().numpy()) 
                print('original label', labels) 
                #teacher_label = torch.where(labels < 4, 0, 1 )
                teacher_label = torch.where( labels < 2 , 11, labels) # # 3, 12
                # print(teacher_label) #
                teacher_label = torch.where( teacher_label < 4,  10, teacher_label)

                teacher_label = torch.where(teacher_label < 6, 9, teacher_label) 
                teacher_label = torch.where(teacher_label < 8, 8, teacher_label) 
                teacher_label =  11 - teacher_label 
                print('gold teacher', teacher_label)

                               
                #
                predicted_teacher = torch.argmax(teacher_score, dim=-1)
                print(predicted_teacher)
                ground_truth_teacher = teacher_label.long()  #torch.where(labels < self.num_labels // 2, 0, 1)
 
                # for t1e, t2e, gtt in zip(t1_ent, t2_ent, ground_truth_teacher):
                # print(t1e, t2e, gtt)
                self.acc += torch.mean((predicted_teacher == ground_truth_teacher).float())

                print(self.acc / self.cnt)
                self.cnt += 1
                # print('accuracy: ', )

        loss = 0.0
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))

        output = (student_logits,)  # + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
