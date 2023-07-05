from dependencies import *
from config import *

def train_fn(data_loader, model, optimizer, device, scheduler):

  model.train()

  for idx, data in tqdm(enumerate(data_loader), total = len(data_loader)):
    # print(data.keys())
    print(f"Training loop {idx}")
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']

    input_ids = input_ids.to(device, dtype=torch.long)
    attention_mask = attention_mask.to(device, dtype=torch.long)
    labels = labels.to(device, dtype=torch.float)

    optimizer.zero_grad()
    logits = model(
                  input_ids = input_ids,
                  attention_mask = attention_mask,
              )
    
    loss = torch.zeros(1, requires_grad=True)
    # if labels is not None:
    print(f"type - {type(logits)} type - {type(labels)}")
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    loss = loss_fn(logits.view(-1, len(attributes)), labels.view(-1, len(attributes)))
    print(f"loss - {loss}, logits - {logits}")
    print(f"loss__ - {loss.requires_grad}, logits__ - {logits.requires_grad}")

    loss.backward()
    optimizer.step()
    scheduler.step()