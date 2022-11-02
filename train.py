import torch
from torch.utils.data import random_split
import time
from datasets.newsGroupDataLoader import get_news_group_data_loader
from datasets.newsGroupDataset import NewsGroupDataset
from model.modelHelper import train, validate
from model.textClassificationModel import TextClassificationModel
from utils.device import get_device
from utils.earlyStopping import EarlyStopping
from utils.logging import get_logger
from utils.preprocessing import get_vocab
from os import mkdir, path

device = get_device()
logger = get_logger(__name__)

train_dataset = NewsGroupDataset(subset='train')

# build vocabulary from training data
vocab = get_vocab(train_dataset)
logger.info("saving vocabulary")
if not path.exists("./output"): mkdir("output")
torch.save(vocab, "./output/vocab.pt")
logger.info("vocabulary saved successfully")
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Hyper-parameters
EPOCHS = 20  # epoch
LR = 5  # learning rate
BATCH_SIZE = 32  # batch size for training

train_dataloader = get_news_group_data_loader(dataset=split_train_, vocab=vocab, batch_size=BATCH_SIZE)
valid_dataloader = get_news_group_data_loader(dataset=split_valid_, vocab=vocab, batch_size=BATCH_SIZE)

num_class = len(list(train_dataset.target_names))
vocab_size = len(vocab)
emsize = 300
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

# Hyper-parameters for model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
early_stopping = EarlyStopping(verbose=True)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    total_acc, total_count, log_interval, start_time = train(model=model, criterion=criterion, optimizer=optimizer,
                                                             dataloader=train_dataloader, epoch=epoch)
    accu_val = validate(model=model, criterion=criterion, dataloader=valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    logger.info('-' * 59)
    logger.info('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch, time.time() - epoch_start_time, accu_val))
    logger.info('-' * 59)
    early_stopping(total_acc, model)
    if (early_stopping.early_stop):
        break
