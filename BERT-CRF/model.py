from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 恢复单层 BiLSTM
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size // 2,
                              num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        
        # 仅使用 BERT 最后一层负载
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 序列对齐
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        
        padded_sequence_output, _ = self.bilstm(padded_sequence_output)
        padded_sequence_output = self.dropout(padded_sequence_output)
        logits = self.classifier(padded_sequence_output)
        
        # 调试信息：检查 logits 分布
        # if not self.training:
        #     print(f"Logits mean: {logits.mean().item()}, max: {logits.max().item()}, min: {logits.min().item()}")
        
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_mask[:, 0] = True
            # CRF Loss
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
        return outputs
