import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TextClassificationDataset(Dataset):
  """
  Dataset class for text classification task.
  created from CSV file
  Args:
      tokenizer (Tokenizer): The tokenizer used to encode the text.
      file_path (str): The path to the file containing the speech classification data.
  """
  def __init__(self, tokenizer, file_path, target, feature):
    self.tokenizer = tokenizer

    if not os.path.exists(file_path):
      raise FileNotFoundError(f"The file {file_path} does not exist.")

    meta_data = pd.read_csv(file_path)

    self.texts = meta_data[feature].tolist()
    self.labels = meta_data[target].tolist()


  def __len__(self):
    return len(self.texts)

  def __getitem__(self, index):
    label, text = self.labels[index], self.texts[index]
    input_ids = self.tokenizer(text, padding='max_length',max_length=512, truncation=True,return_tensors='pt', return_token_type_ids=False, return_attention_mask=False)
    input_ids = input_ids['input_ids'].squeeze()
    label_tensor = torch.tensor(label, dtype=torch.long)

    return input_ids, label_tensor

