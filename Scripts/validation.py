from config import *
from dependencies import *

def eval_fn(data_loader, model):

  model.eval()

  tot_logits = []
  tot_labels = []

  with torch.no_grad():
    for idx, data in tqdm(enumerate(data_loader), total = len(data_loader)):
      print(f"Validation Loop {idx}")
      input_ids = data['input_ids']
      attention_mask = data['attention_mask']
      labels = data['labels']

      input_ids = input_ids.to(device, dtype=torch.long)
      attention_mask = attention_mask.to(device, dtype=torch.long)
      labels = labels.to(device, dtype=torch.float)

      logits = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                )
      logits = logits.cpu().detach().numpy().tolist()
      labels = labels.cpu().detach().numpy().tolist()

      tot_logits.extend(logits)
      tot_labels.extend(labels)

    return (
      tot_logits, 
      tot_labels
   )