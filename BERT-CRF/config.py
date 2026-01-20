import os
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data', 'my') + '/'
train_dir = data_dir + 'admin_train.npz'
test_dir = data_dir + 'admin_test.npz'
files = ['admin_train', 'admin_test']
bert_model = 'pretrained_bert_models/bert-base-chinese/'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.path.join(base_dir, 'experiments', 'admin_split') + '/'
log_dir = os.path.join(model_dir, 'train.log')
case_dir = os.path.join(base_dir, 'case', 'bad_case.txt')

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False  # 改为 False，确保从头微调

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# 更换更强的预训练模型路径
bert_model = roberta_model  # 切换到 chinese_roberta_wwm_large_ext

# hyper-parameter
learning_rate = 5e-5 # 提高整体学习率
weight_decay = 0.01
clip_grad = 1.0 

batch_size = 8 
gradient_accumulation_steps = 1 # 减少累积步数，增加更新频率

epoch_num = 100 
min_epoch_num = 10
patience = 0.00001
patience_num = 50 

# R-Drop 超参数
rdrop_alpha = 1.0 

# EMA 平滑系数
ema_decay = 0.99 

# 辅助损失权重
aux_loss_alpha = 1.0 

# 为不同层设置不同的学习率
bert_lr = 3e-5 # 提高 BERT 学习率
head_lr = 1e-3 # 大幅提高 head 学习率

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

# 这里的 labels 需要替换为你最新的 5 个标签
labels = ['ORG', 'ACTION', 'OBJ', 'LEVEL_KEY', 'VALUE']

# 类别权重（根据你的训练表现微调）
# ACTION 和 LEVEL_KEY 表现较差，这里赋予更高的权重（用于加权 Loss 或者作为论文分析点）
class_weights = {
    'ORG': 1.0,
    'ACTION': 3.0, # 维持高权重
    'OBJ': 2.0,
    'LEVEL_KEY': 2.5, # 维持高权重
    'VALUE': 1.0
}

# 自动构建 BIOS 标签映射，确保每个标签都有唯一 ID，避免 predict 出 None
label2id = {'O': 0}
for i, label in enumerate(labels):
    label2id[f'B-{label}'] = i * 3 + 1
    label2id[f'I-{label}'] = i * 3 + 2
    label2id[f'S-{label}'] = i * 3 + 3

id2label = {i: label for label, i in label2id.items()}
num_labels = len(id2label)

# 确保训练输出路径正确
exp_dir = model_dir
