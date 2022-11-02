import torch
from torch.utils.data import DataLoader

from datasets.newsGroupDataset import NewsGroupDataset
from utils.preprocessing import tokenizer
from utils.device import get_device


# vocab = get_vocab(data_iter=NewsGroupDataset(subset='train'))


def collate_batch(batch, vocab):
    device = get_device()
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def get_news_group_data_loader(dataset, vocab, batch_size=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_batch(batch, vocab=vocab))
