from modelling import *
from training import *
from validation import *
from optimizer import *
from metric import *
from dependencies import *

def __run__():
    best_accuracy = 0
    model.to(device)
    print(model)

    for epoch in range(n_epochs):
        train_fn(train_dataloader, model, optimizer, device, scheduler)
        logits, labels = eval_fn(val_dataloader, model)
        logits = clipping_fn(logits, 1.0, 0.0)
        accuracy = metric_accuracy(logits, labels)
        overall_accuracy = (accuracy[0] + accuracy[1] + accuracy[2] + accuracy[3])/len(accuracy)
        
        print(f"Detailed accuracy after {epoch} epoch:")
        print(f"unanswerable accuarcy: {accuracy[0]}")
        print(f"extractive accuarcy: {accuracy[1]}")
        print(f"yes_no accuarcy: {accuracy[2]}")
        print(f"abstractive accuarcy: {accuracy[3]}")
        print(f"Overall accuarcy: {overall_accuracy}")
        print(f"Best accuarcy: {best_accuracy}")

        if overall_accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = overall_accuracy
            print(best_accuracy)
            print("Model Updated")

if __name__ == "__main__":
    __run__()


