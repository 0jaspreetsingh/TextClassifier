from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset

class NewsGroupDataset(Dataset):

    # subset = 'train' , 'test' , all 
    def __init__(self, subset="all"):
        self.newsgroups = fetch_20newsgroups(subset=subset)
        self.target_names = self.newsgroups.target_names

    def __len__(self):
        return len(self.newsgroups.target)

    def __getitem__(self, idx):
        return self.newsgroups.target[idx], self.newsgroups.data[idx]

    def getTargetName(self, idx):
        return self.newsgroups.target_names[self.newsgroups.target[idx]]

    