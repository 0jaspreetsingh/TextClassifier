import torch
from tqdm import tqdm
import time

from utils.logging import get_logger

logger = get_logger(__name__)


def train(model, criterion, optimizer, dataloader, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(tqdm(dataloader, desc="-" * 50)):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches '
                        '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                    total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
    return total_acc, total_count, log_interval, start_time


def validate(model, criterion, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(tqdm(dataloader, desc="-" * 50)):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count
