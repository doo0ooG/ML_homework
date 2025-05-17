# 定义 数据/模型/日志 路径
log_path = './experiment/train.log' # 训练信息的日志

chn_tokenizer = './tokenizer/chn.model'  # 中文分词器
eng_tokenizer = './tokenizer/eng.model'  # 英文分词器

train_data_path = './data/json/train.json' # 训练集
dev_data_path = './data/json/dev.json'     # 验证集
test_data_path = './data/json/test.json'   # 测试集

# 设备
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 计算设备

# 超参数
batch_size = 32
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
epoch_num = 20
early_stop = 5
lr = 3e-4
weight_decay = 1e-4
