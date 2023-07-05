from config import *
from dependencies import *
from dataset import *

train_data = Qasper_Dataset(
                data_path = qasper_classification_path,
                data_type = "train_data",
                tokenizer = tokenizer,
                attributes = attributes,
                max_token_length = 512,
                sample = None
                )

val_data = Qasper_Dataset(
                    data_path = qasper_classification_path,
                    data_type = "validation_data",
                    tokenizer = tokenizer,
                    attributes = attributes,
                    max_token_length = 512,
                    sample = None
                    )

train_dataloader = DataLoader(
                  train_data,
                  batch_size = batch_size,
                  num_workers = 2,
                  shuffle = True
              )

val_dataloader = DataLoader(
                  val_data,
                  batch_size = batch_size,
                  num_workers = 2,
                  shuffle = True
              )