import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.DataLoader import LoadItem
from dataloader.Model import SimpleClassification

if __name__ == '__main__':

    root_train_dir = "/Users/rakshitbhatt/Documents/GalaxEye /Disease Classification/Potato/Train/"
    root_test_dir = "/Users/rakshitbhatt/Documents/GalaxEye /Disease Classification/Potato/Test/"

    device = "cuda" if torch.cuda.is_available() else "cpu"


    def generate_labels(unstructured_labels):
        labels = torch.tensor([[0, 0, 0]])
        for label in unstructured_labels:
            if label == 0:
                labels = torch.cat((labels, torch.tensor([[1.0, 0.0, 0.0]])), dim=0)
            elif label == 1:
                labels = torch.cat((labels, torch.tensor([[0.0, 1.0, 0.0]])), dim=0)
            else:
                labels = torch.cat((labels, torch.tensor([[0.0, 0.0, 1.0]])), dim=0)
        return labels[1:]


    def ValidateModel(model_state_dict):
        test_loader = LoadItem(root_test_dir)
        model = torch.load(model_state_dict)
        model = model.to(device)
        test_iterator = DataLoader(test_loader, batch_size=8, shuffle=False)

        for idx, val_examples in test_iterator:
            pred_labels = model(val_examples[1])
            actual_labels = generate_labels(val_examples[0])

            # define your metric here

        return 0


    def TrainModel(batch_size=8, epochs=1):
        data_loader = LoadItem(root_train_dir)

        train_model = SimpleClassification().to(device)
        loss_fxn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(train_model.parameters(), lr=0.0001)
        loss_history = {}

        for epoch in range(epochs):
            data_interator = DataLoader(data_loader,
                                        batch_size=batch_size,
                                        shuffle=True)
            print("Starting Epoch number {}".format(epoch + 1))
            loss_history[str(epoch)] = []

            for idx, batch in enumerate(data_interator):

                optimizer.zero_grad()
                images = batch[1].to(device)
                labels = generate_labels(batch[0]).to(device)
                out = train_model(images)
                loss = loss_fxn(out, labels)

                if idx // 10:
                    # print("the loss value calculated is {}".format(loss))
                    loss_history[str(epoch)].append(loss)

                loss.backward()
                optimizer.step()


    # Put the model on training
    TrainModel(batch_size=4, epochs=1)
