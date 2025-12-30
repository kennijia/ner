import torch
from transformers import BertTokenizer

from model import BertNER
import config


def load_model_and_tokenizer():
    """
    加载分词器和模型
    """
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model,
        do_lower_case=True
    )

    model = BertNER.from_pretrained(config.model_dir)
    model.to(config.device)
    model.eval()

    return model, tokenizer


def predict(text, model, tokenizer):
    """
    对单条中文文本进行实体识别
    """
    # ===== 1. 按字符切分 =====
    tokens = list(text)

    # ===== 2. 编码为 BERT 输入 =====
    encoding = tokenizer.encode_plus(
        tokens,
        is_split_into_words=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )

    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)

    token_type_ids = encoding.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(config.device)

    # ===== 3. 构造 token_starts（当前写法：全部递增）=====
    input_token_starts = torch.arange(
        input_ids.size(1)
    ).unsqueeze(0).to(config.device)

    # ===== 4. 前向推理 + CRF 解码 =====
    with torch.no_grad():
        logits = model(
            (input_ids, input_token_starts),
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]

        # 保证 mask 与 logits 长度一致
        if logits.size(1) != attention_mask.size(1):
            attention_mask = attention_mask[:, :logits.size(1)]

        attention_mask = attention_mask.bool()
        pred_ids = model.crf.decode(
            logits,
            mask=attention_mask
        )[0]

    # ===== 5. id → label =====
    id2label = {v: k for k, v in config.label2id.items()}

    # 尝试直接取对应长度，不再跳过第一位
    # 注意：前提是你的模型输出确实不包含 CLS
    curr_pred_ids = pred_ids[:len(tokens)] 

    pred_labels = [
        id2label[i]
        for i in curr_pred_ids
    ]

    # ===== 6. 实体抽取 =====
    entities = []
    entity = None

    for idx, (char, label) in enumerate(zip(tokens, pred_labels)):
        if label.startswith('B-'):
            if entity:
                entities.append(entity)
            entity = {
                'type': label[2:],
                'start': idx,
                'text': char
            }

        elif (
            label.startswith('I-')
            and entity
            and label[2:] == entity['type']
        ):
            entity['text'] += char

        else:
            if entity:
                entities.append(entity)
                entity = None

    if entity:
        entities.append(entity)

    return entities


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()

    text = input("请输入一句中文：")
    entities = predict(text, model, tokenizer)

    print("识别到的实体：")
    for ent in entities:
        print(f"{ent['type']}: {ent['text']} (起始位置: {ent['start']})")
