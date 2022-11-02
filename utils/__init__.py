import spacy

en = spacy.load('en_core_web_trf')
stopwords = en.Defaults.stop_words
