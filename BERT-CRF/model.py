from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
import torch


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        # ensure BERT will return hidden states (for layer fusion) on older transformers versions
        config.output_hidden_states = True
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 移除 Scalar Mix 参数，确保模型纯净
        # self.num_layers = config.num_hidden_layers + 1 # includes embedding layer
        # self.weights = nn.Parameter(torch.zeros(self.num_layers))
        # self.gamma = nn.Parameter(torch.ones(1))

        # 恢复单层 BiLSTM (已注释，彻底不初始化)
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
        
        # Scalar Mix: combine all hidden states
        # support both object and tuple outputs
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 2:
            hidden_states = outputs[2]
        else:
            raise RuntimeError("BERT did not return hidden states. Ensure config.output_hidden_states=True when initializing the model.")
        
        # Calculate Scalar Mix (Disabled)
        # weights = torch.softmax(self.weights, dim=0)
        # sequence_output = torch.stack(hidden_states, dim=0) # (num_layers, batch, seq_len, hidden)
        # sequence_output = (weights.view(-1, 1, 1, 1) * sequence_output).sum(dim=0)
        # sequence_output = sequence_output * self.gamma

        # 简化模型：直接使用最后一层 Hidden State
        sequence_output = hidden_states[-1]

        # 序列对齐
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  if starts.nonzero().size(0) > 0 else layer[:0]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        
        # 移除 BiLSTM，回归纯 BERT-CRF
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
