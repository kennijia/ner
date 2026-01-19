import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import config
from model import BertNER
from metrics import f1_score, bad_case
from transformers import BertTokenizer
from utils import FGM, EMA


def train_epoch_with_ema(train_loader, model, optimizer, scheduler, epoch, ema):
    # set model to training mode
    model.train()
    fgm = FGM(model)
    
    train_losses = 0
    # 构造辅助损失的类别权重 Tensor
    weights = torch.ones(len(config.label2id)).to(config.device)
    for label, w in config.class_weights.items():
        if f'B-{label}' in config.label2id:
            weights[config.label2id[f'B-{label}']] = w
        if f'I-{label}' in config.label2id:
            weights[config.label2id[f'I-{label}']] = w
        if f'S-{label}' in config.label2id:
            weights[config.label2id[f'S-{label}']] = w

    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0) 
        batch_masks[:, 0] = True
        
        label_masks = batch_labels.gt(-1)
        label_masks[:, 0] = True

        # R-Drop: 跑两次
        loss1, logits1 = model((batch_data, batch_token_starts),
                              token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        loss2, logits2 = model((batch_data, batch_token_starts),
                              token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        
        # 1. CRF Loss (NLL)
        crf_loss = (loss1 + loss2) / 2
        
        # 2. 辅助加权 CE Loss (增强难分类别)
        active_loss = label_masks.view(-1)
        active_logits = logits1.view(-1, model.num_labels)
        active_labels = torch.where(
            active_loss, batch_labels.view(-1), torch.tensor(0).type_as(batch_labels)
        ).long()
        aux_loss = F.cross_entropy(active_logits, active_labels, weight=weights, reduction='mean')
        
        # 3. KL 一致性损失
        p_loss = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='none')
        mask_expand = label_masks.unsqueeze(-1).expand_as(p_loss)
        kl_loss = ((p_loss * mask_expand).sum() + (q_loss * mask_expand).sum()) / 2
        
        # 总 Loss
        loss = (crf_loss + config.aux_loss_alpha * aux_loss + config.rdrop_alpha * kl_loss) / config.gradient_accumulation_steps
        
        train_losses += loss.item() * config.gradient_accumulation_steps
        loss.backward()

        # 对抗训练
        fgm.attack(epsilon=0.5) 
        loss_adv, _ = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
        (loss_adv / config.gradient_accumulation_steps).backward() 
        fgm.restore() 

        # 达到累积步数后更新参数
        if (idx + 1) % config.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            # 更新 EMA
            ema.update()
        
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    return train_loss


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir, writer=None):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertNER.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    
    # 统一管理 EMA 实例
    ema = EMA(model, config.ema_decay)
    ema.register()

    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        # 传递 ema 实例到 train_epoch 内部进行 update
        train_loss = train_epoch_with_ema(train_loader, model, optimizer, scheduler, epoch, ema)
        
        # 评估前应用 EMA 影子权重
        ema.apply_shadow()
        val_metrics = evaluate(dev_loader, model, mode='dev')
        # 评估后恢复原始权重，以便继续训练
        ema.restore()

        val_f1 = val_metrics['f1']
        val_loss = val_metrics['loss']
        logging.info("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_loss, val_f1))
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/dev', val_loss, epoch)
            writer.add_scalar('F1/dev', val_f1, epoch)
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {}".format(best_val_f1))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    
    # 应用 EMA (如果有的话，通常在测试阶段应用)
    # 注意：为了逻辑严密，我们需要在 train.py 全局创建一个 ema 实例
    # 这里我们临时通过手动方式模拟，更严谨的做法是在 train 里持有它。
    
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    if mode == 'test':
        assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics


if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)
