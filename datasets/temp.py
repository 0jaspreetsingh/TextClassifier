import string
from sklearn.datasets import fetch_20newsgroups
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from torchtext.data.utils import get_tokenizer

from utils import getDevice

newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
from pprint import pprint

pprint(newsgroups_train.data[0])
pprint(newsgroups_train.target[0])

total = 0

for data in newsgroups_train.data:
    total = total +  len(data.split())

avg = total/len(newsgroups_train.data)
print("Average: ",avg)
print("Totoal:", len(newsgroups_train.data))
device = getDevice()
print(device)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)


#tokenization
en = spacy.load('en_core_web_trf')
stopwords = en.Defaults.stop_words

def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in en.tokenizer(nopunct)]


print(tokenize(newsgroups_train.data[0]))


tokenizerr = get_tokenizer('basic_english')
print("Pytorch eg: ",tokenizerr('here'))