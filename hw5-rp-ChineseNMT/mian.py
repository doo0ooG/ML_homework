import utils
import config
import logging
from data_loader import MTDataset
from torch.utils.data import DataLoader
import torch

def run():
    utils.set_logger(config.log_path)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)
    # logging.info(dev_dataset[295][0])
    # logging.info(dev_dataset[295][1])
    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    logging.info("-------- Get Dataloader! --------")

    model = make_model(config.src_vocab_size,
                       config.tgt_vocab_size,
                       config.n_layers,
                       config.d_model,
                       config.d_ff,
                       config.n_heads,
                       config.dropout)
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0, reduction = 'sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    train(train_dataloader, dev_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)

if __name__ == "__main__":
    run()