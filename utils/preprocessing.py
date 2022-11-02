import re
import string
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from utils import en, stopwords
from utils.logging import get_logger

logger = get_logger(__name__)


# tokenization
def tokenizer(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    tokens = [token.text for token in en.tokenizer(nopunct)]
    return [x for x in tokens if x not in stopwords]


def yield_tokens(data_iter):
    for _, text in tqdm(data_iter, desc="-" * 50):
        yield tokenizer(text)


def get_vocab(data_iter):
    logger.info("Building Vocabulary from dataloader")
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    logger.info("Vocabulary built from dataloader with a total of %d words", len(vocab))
    return vocab
