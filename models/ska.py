import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss


class SKABert(BertForSequenceClassification):
    """Vanilla KD, directly learns from the concat of teacher distribution"""

    def __init__(self, config,
                 teacher1=None, teacher2=None,
                 kd_alpha=1,
                 temperature=5.0, eval_strategy='student'):
        super().__init__(config)
        self.num_labels = 2  # config.num_labels teacher selection
        self.kd_alpha = kd_alpha
        self.teacher1 = teacher1
        self.teacher2 = teacher2
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
        student_logits = self.classifier(pooled_output)  # teacher selection

        loss = 0.0
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(student_logits.view(-1, self.num_labels), labels.view(-1))

        output = (student_logits,)  # + student_outputs[2:]
        return ((loss,) + output) if loss is not None else output
