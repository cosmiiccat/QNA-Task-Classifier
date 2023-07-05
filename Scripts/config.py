# All config parameters will be declared here
from dependencies import *

train_data_path = r"C:\Users\DELL\Desktop\Qasper_qna\Data\qasper-train-v0.3.json"
test_data_path = r"C:\Users\DELL\Desktop\Qasper_qna\Data\qasper-test-v0.3.json"
model_name = r"bert-base-uncased"

new_qasper_data_path = r"C:\Users\DELL\Desktop\Qasper_qna\Data\new_qasper.json"
qasper_classification_path = r"C:\Users\DELL\Desktop\Qasper_qna\Data\qasper_classification.json"

device = r"cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
attributes = [
                "unanswerable",
                "extractive_spans",
                "yes_no",
                "abstractive"
            ]
max_token_length = 512
bert_model = AutoModel.from_pretrained(model_name, return_dict = True)
batch_size = 8

n_labels = 4
lr = 1.5e-6
warmup = 0.2
weight_decay = 0.001
n_epochs = 100
MODEL_PATH = r"C:\Users\DELL\Desktop\Qasper_qna\models\model.bin"



