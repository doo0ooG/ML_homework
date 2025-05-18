import config
from tqdm import tqdm
import logging
import torch
from data_loader import Batch
import sentencepiece as spm
import sacrebleu
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import math

scaler = GradScaler()

sp_chn = spm.SentencePieceProcessor()
sp_chn.Load(config.chn_tokenizer)

def run_epoch(dataloader, model, criterion, optimizer = None):
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader):
        with autocast(enabled = True):
            output = model(src = batch.src,
                        tgt = batch.tgt,
                        tgt_mask = batch.tgt_mask,
                        src_key_padding_mask = batch.src_key_padding_mask,
                        tgt_key_padding_mask = batch.tgt_key_padding_mask)

            loss = criterion(output.reshape(-1, output.size(-1)), batch.tgt_y.reshape(-1))
        

        if optimizer is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_tokens += batch.ntokens.item() 

    return total_loss / total_tokens

def train(train_dataloader, val_dataloader, model, criterion, optimizer):
    best_bleu = 0.0
    early_stop_counter = 0

    for epoch in range(1, config.epoch + 1):
        logging.info(f"Epoch {epoch} start")

        model.train()
        train_loss = run_epoch(train_dataloader, model, criterion, optimizer)
        logging.info(f"Epoch {epoch}, train loss: {train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(val_dataloader, model, criterion)
        logging.info(f"Epoch {epoch}, val loss: {val_loss:.4f}")

        bleu = compute_bleu_score(model, val_dataloader)
        logging.info(f"Epoch {epoch}, bleu score: {bleu:.4f}")

        if bleu > best_bleu:
            best_bleu = bleu
            early_stop_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            logging.info(f"Epoch {epoch}, new Best BLEU: {bleu:.4f}, save best model -------------")
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.early_stop:
            logging.info("Early stopping triggered!")
            break

def test(test_dataloader, model, criterion):
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    with torch.no_grad():
        test_loss = run_epoch(test_dataloader, model, criterion)
        logging.info(f"Test loss: {test_loss:.4f}")

        bleu = compute_bleu_score(model, test_dataloader)
        logging.info(f"Test BLEU: {bleu:.4f}")

def compute_bleu_score(model, dataloader, sp_chn, bos_idx=2, eos_idx=3):
    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    all_references = []
    all_hypotheses = []

    for batch in dataloader:
        src = batch.src
        src_key_padding_mask = batch.src_key_padding_mask

        preds = greedy_decode(model, src, src_key_padding_mask, bos_idx, eos_idx)

        for pred_tokens, ref_text in zip(preds, batch.tgt_text):
            # 处理预测
            pred_tokens = pred_tokens.tolist()
            if eos_idx in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index(eos_idx)]
            pred_sentence = sp_chn.decode_ids(pred_tokens)

            # 处理参考答案（batch.tgt_text 是 token id 列表）
            ref_tokens = [tok for tok in ref_text if tok != 0 and tok != bos_idx and tok != eos_idx]
            ref_sentence = sp_chn.decode_ids(ref_tokens)

            all_hypotheses.append(pred_sentence)
            all_references.append([ref_sentence])  # 注意嵌套一层列表是 sacrebleu 格式

    return bleu.corpus_score(all_hypotheses, all_references).score



def greedy_decode(model, src, src_key_padding_mask, bos_idx, eos_idx, max_len=100):
    model.eval()
    with torch.no_grad():
        src_embed = model.src_embed(src) * math.sqrt(model.transformer.d_model)
        src_embed = model.pos_embed(src_embed)
        memory = model.transformer.encoder(src_embed, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            tgt_embed = model.tgt_embed(ys) * math.sqrt(model.transformer.d_model)
            tgt_embed = model.pos_embed(tgt_embed)

            tgt_len = ys.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).masked_fill(0 == 0, float('-inf'))
            tgt_mask = tgt_mask.masked_fill(torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)) == 1, float(0.0))

            out = model.transformer.decoder(
                tgt_embed, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = model.generator(out[:, -1])  # [B, vocab_size]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)  # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token == eos_idx).all():
                break

        return ys[:, 1:]  # 去掉 <bos>