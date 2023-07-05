from dependencies import *

class Qasper_Dataset(Dataset):

    def __init__(self, data_path, data_type, tokenizer, attributes, max_token_length = 128, sample = None):
        self.data_path = data_path
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.max_token_length = max_token_length
        self.data = self.__load_data__()

    def __load_data__(self):
        data_fd = open(self.data_path)
        self.data = json.load(data_fd)
        return (
            self.data['data'][self.data_type]
        )

    def __len__(self):
        return (
            len(self.data)
        )

    def __getitem__(self, index):
        context = ""
        for i in range(len(self.data[index][0])):
            if i != 0:
                context = context + " "
            context = context + self.data[index][0][i]
        # context = self.data[index][0]
        question = self.data[index][1]
        labels = [self.data[index][2][self.attributes[0]], self.data[index][2][self.attributes[1]], self.data[index][2][self.attributes[2]], self.data[index][2][self.attributes[3]]]
        labels = torch.FloatTensor(labels)
        # labels = Variable(labels, requires_grad = True)
        tokens = self.tokenizer.encode_plus(
                                            question,
                                            context,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_token_length,
                                            return_attention_mask = True
                                            )
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}