import config
from tqdm import tqdm
import logging
import torch
from data_loader import Batch
import sentencepiece as spm
import sacrebleu

sp_chn = spm.SentencePieceProcessor()
sp_chn.Load(config.chn_tokenizer)

def run_epoch(dataloader, model, criterion, optimizer = None):
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader):
        output = model(src = batch.src,
                       tgt = batch.tgt,
                       tgt_mask = batch.tgt_mask,
                       src_key_padding_mask = batch.src_key_padding_mask,
                       tgt_key_padding_mask = batch.tgt_key_padding_mask)
        
        # logging.info(output.shape)
        # logging.info(output.reshape(-1, output.size(-1)).shape)
        # logging.info(batch.tgt_y.reshape(-1).shape)
        # assert 1 == 2
        loss = criterion(output.reshape(-1, output.size(-1)), batch.tgt_y.reshape(-1))
        total_loss += loss.item()
        total_tokens += batch.ntokens.item() 

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

def compute_bleu_score(model, dataloader):
    model.eval()
    total_refs = []
    total_hyps = []

    # 只取前 5% 的 batch
    dataloader_list = list(dataloader)
    num_batches = max(1, int(len(dataloader_list) * config.compute_bleu_data_ratio))
    limited_batches = dataloader_list[:num_batches]

    for batch in tqdm(limited_batches, desc=f"Computing BLEU ({config.compute_bleu_data_ratio * 100}%)"):
        output = greedy_decode(model, 
                               batch.src, 
                               batch.src_key_padding_mask, 
                               max_len=config.max_len, 
                               start_symbol=config.bos_idx)

        for pred_tokens, tgt_tokens in zip(output.tolist(), batch.tgt_y.tolist()):
            pred_ids = [id for id in pred_tokens if id != config.padding_idx and id != config.eos_idx]
            tgt_ids = [id for id in tgt_tokens if id != config.padding_idx and id != config.eos_idx]
            if config.eos_idx in pred_ids:
                pred_ids = pred_ids[:pred_ids.index(config.eos_idx)]

            pred_text = sp_chn.decode_ids(pred_ids)
            ref_text = sp_chn.decode_ids(tgt_ids)

            # logging.info(pred_text)
            # logging.info(ref_text)
            # assert 1 == 2

            total_hyps.append(pred_text)
            total_refs.append([ref_text])

    bleu = sacrebleu.corpus_bleu(total_hyps, total_refs)
    return bleu.score


def greedy_decode(model, src, src_key_padding_mask, max_len = 60, start_symbol = 2):
    memory = model.transformer.encoder(model.pos_embed(model.src_embed(src)), 
                                       src_key_padding_mask = src_key_padding_mask)
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).long().to(src.device)
    # logging.info(ys.shape)
    # logging.info(ys)
    # assert 1 == 2

    for i in range(max_len - 1):
        tgt_mask = Batch.subsequent_mask(ys.size(1)).to(src.device)
        out = model.transformer.decoder(
            model.pos_embed(model.tgt_embed(ys)),
            memory,
            tgt_mask = tgt_mask,
            memory_key_padding_mask = src_key_padding_mask
        )

        out = model.generator(out[:, -1])
        next_word = torch.argmax(out, dim = -1).unsqueeze(1)
        ys = torch.cat([ys, next_word], dim = 1)
        # logging.info(ys.shape)
    return ys