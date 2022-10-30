from datasets.newsGroupDataset import NewsGroupDataset

data_iter = NewsGroupDataset(subset='train')
print(data_iter[0])