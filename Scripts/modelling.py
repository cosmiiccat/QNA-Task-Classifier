from config import *
from dataloader import *

class Qasper_Classifier(nn.Module):

    def __init__(self, model, n_labels):

        super(Qasper_Classifier, self).__init__()
        self.pretrained_model = model # bert model   
        self.n_labels = n_labels

        self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        print(f"hidden - {self.hidden.weight.requires_grad}")
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.n_labels)
        print(f"classifier - {self.classifier.weight.requires_grad}")
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        # print(f"loss - {self.loss_func.weight.requires_grad}")
        self.dropout = nn.Dropout(0.3)
        # print(f"dropout - {self.dropout.weight.requires_grad}")


    def forward(self, input_ids, attention_mask, labels = None):
        # print("Forward Propagation started")
        # input_ids.requires_grad = True; attention_mask,requires_grad = True
        print(f"input - {input_ids.requires_grad}, attention_mask - {attention_mask.requires_grad}")
        output = self.pretrained_model(
                                    input_ids = input_ids,
                                    attention_mask = attention_mask
                                       )
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # loss = 0
        # loss.requires_grad = True
        # logits.requires_grad = True
        # labels.required_grad = True
        # loss = Variable(loss, requires_grad = True)
        return logits