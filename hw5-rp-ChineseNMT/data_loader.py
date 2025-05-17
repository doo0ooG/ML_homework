from torch.utils.data import Dataset
from utils import chinese_tokenizer_load
from utils import english_tokenizer_load
from pathlib import Path
import json
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

import config
DEVICE = config.device

class Batch:
    @staticmethod
    def subsequent_mask(size):
        """
        生成用于遮蔽未来词的矩阵 (1, size, size)。
        - 主对角线以下为True（当前及之前位置可见）
        - 主对角线上为False（未来位置不可见）
        
        Args:
            size (int): 序列长度
        
        Returns:
            torch.Tensor: subsequent mask
        """
        attn_shape = (1, size, size)
        mask = torch.tril(torch.ones(attn_shape, dtype=torch.bool))
        return mask

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        构建Decoder输入的标准mask，综合遮掉padding位置和未来词。
        
        Args:
            tgt (torch.Tensor): 目标输入序列 (batch_size, seq_len)
            pad (int): Padding token的id
        
        Returns:
            torch.Tensor: 目标mask (batch_size, 1, seq_len) & (1, seq_len, seq_len)
        """
        # 标记出非pad的位置
        tgt_mask = (tgt != pad).unsqueeze(1)
        # 叠加遮蔽未来词
        tgt_mask = tgt_mask & Batch.subsequent_mask(tgt.size(1)).to(tgt_mask.device)
        return tgt_mask

    def __init__(self, src_text, tgt_text, src, tgt=None, pad=0):
        """
        初始化Batch，封装源序列、目标序列及相关mask。

        Args:
            src_text (List[str]): 源语言文本（原始句子）
            tgt_text (List[str]): 目标语言文本（原始句子）
            src (torch.Tensor): 源语言token id序列
            tgt (torch.Tensor, optional): 目标语言token id序列
            pad (int): Padding token id
        """
        self.src_text = src_text
        self.tgt_text = tgt_text

        self.src = src.to(DEVICE)
        self.src_mask = (src != pad).unsqueeze(-2)  # (batch_size, 1, src_len)

        if tgt is not None:
            tgt = tgt.to(DEVICE)
            self.tgt = tgt[:, :-1]    # Decoder输入部分 (BOS到tokenN-1)
            self.tgt_y = tgt[:, 1:]   # Decoder监督目标 (token1到EOS)
            self.tgt_mask = Batch.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()  # 有效token数，用于loss归一化

class MTDataset(Dataset):
    """
    机器翻译任务的数据集类（Machine Translation Dataset）。
    支持自动分词器加载，英文/中文句子对加载与排序，供DataLoader使用。
    """
    def __init__(self, data_path):
        """
        初始化数据集。

        Args:
            data_path (str or Path): 存储数据的json文件路径。
        """
        self.data_path = Path(data_path)
        
        # 加载中英文分词器
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()

        # 特殊token的id
        self.PAD = self.sp_eng.pad_id()  # Padding id (通常为0)
        self.BOS = self.sp_eng.bos_id()  # Begin of sentence id (通常为2)
        self.EOS = self.sp_eng.eos_id()  # End of sentence id (通常为3)

        # 加载数据集，并按英文句子长度升序排序
        self.out_en_sent, self.out_cn_sent = self.get_dataset(sort=True)

    @staticmethod
    def len_argsort(seq):
        """
        根据句子长度升序排列，返回原索引的排序列表。

        Args:
            seq (List[str]): 句子列表。
        
        Returns:
            List[int]: 排序后的索引列表。
        """
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, sort):
        """
        加载json数据，并根据英文句子长度排序（可选）。

        Args:
            sort (bool): 是否按英文句子长度排序。

        Returns:
            Tuple[List[str], List[str]]: 分别返回英文句子列表和中文句子列表。
        """
        with self.data_path.open('r', encoding='utf-8') as f:
            dataset = json.load(f)

        out_en_sent = []
        out_ch_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])
            out_ch_sent.append(dataset[idx][1])

        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_ch_sent = [out_ch_sent[i] for i in sorted_index]

        return out_en_sent, out_ch_sent

    def __len__(self):
        """
        返回数据集大小。

        Returns:
            int: 样本数量
        """
        return len(self.out_en_sent)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本对（英文句子，中文句子）。

        Args:
            idx (int): 样本索引。
        
        Returns:
            Tuple[str, str]: 英文句子, 中文句子
        """
        return self.out_en_sent[idx], self.out_cn_sent[idx]
    
    def collate_fn(self, batch):
        """
        自定义collate_fn函数，将原始样本batch处理成可供Transformer训练的格式。

        Args:
            batch (List[Tuple[str, str]]): 一个batch的样本对 (英文句子, 中文句子)
        
        Returns:
            Batch: 处理好的Batch对象，包含src/trg张量和mask
        """
        # 拆分源语言和目标语言句子
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        # 分词，并在两端加上BOS/EOS
        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # 把不同长度的句子padding成统一长度
        batch_input = pad_sequence([torch.LongTensor(src_token) for src_token in src_tokens],
                                    batch_first=True, 
                                    padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(tgt_token) for tgt_token in tgt_tokens],
                                    batch_first=True, 
                                    padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)