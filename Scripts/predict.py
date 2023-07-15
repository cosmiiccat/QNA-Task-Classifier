# from dependencies import *
# from config import *

# from config import MODEL_PATH

import torch

context = "Hello my name name is Preetam."
question = "What is my name?"

print("Loading Model...")

classification_model = torch.load(r"C:\Users\DELL\Desktop\Qasper_qna\models\classifier.bin")

# result = classification_model(context, question)

# print(result)

