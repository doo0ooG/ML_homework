import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len = 512, d_model = 512, dropout = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(0, seq_len, device = x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(position)
        x = x + pos_emb
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model = 512, nhead = 8, 
                 num_encoder_layers = 6, num_decoder_layers = 6, dim_feedforward = 2048, dropout = 0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len = 512, d_model = d_model, dropout = dropout)

        self.transformer = nn.Transformer(d_model = d_model, 
                                          nhead = nhead,
                                          num_encoder_layers = num_encoder_layers,
                                          num_decoder_layers = num_decoder_layers,
                                          dim_feedforward = dim_feedforward,
                                          dropout = dropout,
                                          batch_first= True)
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None,
                src_key_padding_mask = None, tgt_key_padding_mask = None):
        src = self.src_embed(src) * math.sqrt(self.transformer.d_model)
        src = self.pos_embed(src)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.transformer.d_model)
        tgt = self.pos_embed(tgt)

        output = self.transformer(
            src,
            tgt,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            src_key_padding_mask = src_key_padding_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = src_key_padding_mask
        )

        return self.generator(output)
    
