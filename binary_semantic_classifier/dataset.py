import torch
import pandas as pd

class IMDB_Dataset:
    def __init__(self, data_path, max_length, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reviews = []
        self.labels = []
        self.load_data()

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def load_data(self):
        df = pd.read_csv(self.data_path)
        self.reviews = df['review'].tolist()
        self.labels = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).tolist()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        ids = self.tokenizer.get_ids(self.reviews[idx], pad_length=self.max_length)
        labels = self.labels[idx]

        return torch.tensor(ids), torch.tensor(labels)
    
