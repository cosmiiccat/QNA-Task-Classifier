from dataloader import *
from modelling import *

model = Qasper_Classifier(bert_model, len(attributes))
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
  {
      "params": [
          p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
      ],
      "weight_decay": 0.001,
  },
  {
      "params": [
          p for n, p in param_optimizer if any(nd in n for nd in no_decay)
      ],
      "weight_decay": 0.0,
  },
]

num_train_steps = int(len(train_data) / batch_size * n_epochs)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)