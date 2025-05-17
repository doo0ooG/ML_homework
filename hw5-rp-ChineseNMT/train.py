import config
from tqdm import tqdm
import logging
import torch

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

    for epoch in range(config.epoch):
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

    for batch in tqdm(dataloader):
        src_tensor = batch.src
        src_key_padding_mask = batch.src_key_padding_mask

        output = greedy_decode(model, src_tensor, src_key_padding_mask, 
                               max_len = config.max_len, 
                               start_symbol = config.bos_idx)
        
    return 0