import torch
from datasets.newsGroupDataLoader import get_news_group_data_loader
from datasets.newsGroupDataset import NewsGroupDataset
from model.modelHelper import validate
from model.textClassificationModel import TextClassificationModel
from utils.device import get_device
from utils.logging import get_logger

logger = get_logger(__name__)
device = get_device()
logger.info("loading vocabulary")

logger.info("Loading vocabulary...")
vocab = torch.load("./output/vocab.pt")

test_dataset = NewsGroupDataset(subset='test')
test_dataloader = get_news_group_data_loader(dataset=test_dataset, vocab=vocab, batch_size=32)

num_class = len(list(test_dataset.target_names))
vocab_size = len(vocab)
emsize = 300
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
model.load_state_dict(torch.load("./output/checkpoint.pt"))

criterion = torch.nn.CrossEntropyLoss()
logger.info('Checking the results of test dataset.')
accu_test = validate(model=model, criterion=criterion, dataloader=test_dataloader)
logger.info('test accuracy {:8.3f}'.format(accu_test))
