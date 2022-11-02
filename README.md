# TextClassifier

## Classification model to specify group of a given news

https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

### Libraries used:
1. `pytorch`
2. `torchtext`
3. `numpy`
4. `spacy`
5. `sklearn`

Please refer `environment.yml` to get a complete list of the libraries used

To train the model run `train.py` to generate vocabulary and model data (generated in the `output` folder)

`validate.py` test the accuracy of the trained model using test data


### Approach used for the project

1. Every article contains words such as `"the", "a", "and", "nobody"`  which are referred as stopwords in the document which provide no useful information for the model classification. So in the first step these words are removed from each document.
2. Similarly punctuation in the document provide no useful information so it is also eleminated from the document.
3. Finally each news article in train dataset is tokenized (split in words) and represented in an array like structure using vocab builder. Here each word in a document is referred by array index. eg if vocab is: `["apple" , "cat", "tree"]`. Numeric representation of `"apple"` will be `0`
4. Each news category deals with words unique for a given category, for instance `rec.autos` news category mention words such as `"car", "drive", "km"` so a sequential neural network is used for the classification.
5. Since every document will be of different length, embeed layer is used as a first layer in the neural network.
