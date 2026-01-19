from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels

        # 显式在 config 中开启 output_hidden_states
        config.output_hidden_states = True
        self.bert = BertModel(config)
        # 还原回传统的单 dropout，以便 R-Drop 正常工作
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 新增投影层，将 4 层拼接后的维度映射回 hidden_size
        self.fusion_projection = nn.Linear(config.hidden_size * 4, config.hidden_size)
        
        self.bilstm = nn.LSTM(config.hidden_size, config.hidden_size // 2,
                              num_layers=1, bidirectional=True, batch_first=True)
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        # hidden_states 会因为 config 中的设置而返回
        # transformers 返回格式通常是 (last_hidden_state, pooler_output, hidden_states)
        hidden_states = outputs[2]
        
        # 融合最后 4 层的特征 (Feature Fusion)
        fusion_layers = torch.cat([hidden_states[-1], hidden_states[-2], 
                                 hidden_states[-3], hidden_states[-4]], dim=-1)
        
        sequence_output = self.fusion_projection(fusion_layers)

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        
        # 经过 BiLSTM 层
        padded_sequence_output, _ = self.bilstm(padded_sequence_output)
        
        # 还原为单次 dropout
        padded_sequence_output = self.dropout(padded_sequence_output)
        logits = self.classifier(padded_sequence_output)
        
        outputs = (logits,)
        if labels is not None:
            # 确保 mask 的第一位是 True，解决 torchcrf 的校验报错
            loss_mask = labels.gt(-1)
            loss_mask[:, 0] = True
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs
