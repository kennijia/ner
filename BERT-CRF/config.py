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
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 8
epoch_num = 50
# epoch_num = 1
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

# 这里的 labels 需要替换为你最新的 5 个标签
labels = ['ORG', 'ACTION', 'OBJ', 'LEVEL_KEY', 'VALUE']

# 自动构建 BIO 标签映射
label2id = {'O': 0}
for i, label in enumerate(labels):
    label2id[f'B-{label}'] = i * 2 + 1
    label2id[f'I-{label}'] = i * 2 + 2
    label2id[f'S-{label}'] = i * 2 + 1 # 简单起见，S 映射到 B（也可以单独加）

id2label = {i: label for label, i in label2id.items()}
num_labels = len(label2id)

# 确保训练输出路径正确
exp_dir = model_dir
