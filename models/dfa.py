import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from .cfl import CFLFCBlock, CFLLoss
from torch.nn import CrossEntropyLoss, MSELoss


class AmalBlock(nn.Module):
    def __init__(self, student_config, teacher_configs):
        super(AmalBlock, self).__init__()
        self.student_hidden_num = student_config.hidden_size #num_hidden_layers
        self.teacher_hidden_nums = [t_config.hidden_size for t_config in teacher_configs]
        self.teacher_total_hidden = sum(self.teacher_hidden_nums)
        self.t_encoder = nn.Linear(self.teacher_total_hidden, self.student_hidden_num)
        self.t_decoder = nn.Linear(self.student_hidden_num, self.teacher_total_hidden)
        self.s_encoder = nn.Linear(self.student_hidden_num, self.student_hidden_num)

    def forward(self, fs, fts):
        cat_fts = torch.cat(fts, dim=-1)
        rep = self.t_encoder(cat_fts)
        _fts = self.t_decoder(rep)
        _fts = torch.split(_fts, self.teacher_hidden_nums, dim=-1)
        _fs = self.s_encoder(fs)
        return rep, _fs, _fts


class DFABert(BertForSequenceClassification):
    def __init__(self, config,
                 teachers, teacher_configs=None,
                 teacher_number=2,
                 kd_alpha=1, almal_alpha=1, rec_alpha=1.0,
                 temperature=5.0, eval_strategy='student', align_number=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.kd_alpha = kd_alpha
        self.almal_alpha = almal_alpha
        self.rec_alpha = rec_alpha
        self.teachers = teachers
        self.temperature = temperature
        self.kl_kd_loss = nn.KLDivLoss(reduction='batchmean')
        if align_number == -1:
            align_number = config.num_hidden_layers
        self.align_number = align_number

        self.almal_blocks = nn.ModuleList(
            AmalBlock(config, teacher_configs) for _ in range(align_number))
        self.kd_loss = None
        self.almal_loss = None
        self.rec_loss = None
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
                output_hidden_states=True,
                return_dict=return_dict,
            )

            pooled_output = student_outputs[1]
            pooled_output = self.dropout(pooled_output)
            student_logits = self.classifier(pooled_output)

            t_logits, t_hiddens = [], []
            with torch.no_grad():
                for t_model in self.teachers:
                    t_output = t_model(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds,
                                       output_attentions=output_attentions,
                                       output_hidden_states=True,
                                       return_dict=return_dict, )
                    t_logits.append(t_output[0])
                    # deal with the mismatch layer number
                    t_hiddens.append(t_output['hidden_states'][-1:-(1 + t_model.config.num_hidden_layers):-(
                            t_model.config.num_hidden_layers // self.config.num_hidden_layers)])
            concat_teacher_logits = torch.cat(t_logits, dim=1)  # may be need normalize

            kd_loss = self.kl_kd_loss(F.log_softmax(student_logits / self.temperature, dim=-1),
                                      F.softmax(concat_teacher_logits / self.temperature,
                                                dim=-1)) * self.temperature ** 2
            student_hiddens = student_outputs['hidden_states'][-1:-(1 + self.align_number): -1]
            assert len(student_hiddens) == len(t_hiddens[0]) == len(t_hiddens[1]), \
                "feature numbeer mismatch Stu: %d T1: %d T2: %d" % (
                    len(student_hiddens), len(t_hiddens[0]), len(t_hiddens[1]))
            # DFA learning

            almal_loss, rec_loss = 0, 0
            for i, block in enumerate(self.almal_blocks):
                fts = [t_hidden[i] for t_hidden in t_hiddens]
                rep, _fs, _fts = block(student_hiddens[i], fts)
                almal_loss += F.mse_loss(_fs, rep.detach())
                rec_loss += sum([F.mse_loss(_ft, ft) for (_ft, ft) in zip(_fts, fts)])
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
            # print(student_logits.size())
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(student_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))
        if kd_loss is not None:
            self.almal_loss = almal_loss
            self.rec_loss = rec_loss
            self.kd_loss = kd_loss
            loss = self.kd_alpha * kd_loss + self.rec_alpha * rec_loss + self.almal_alpha * almal_loss

        output = (student_logits,)  # + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
