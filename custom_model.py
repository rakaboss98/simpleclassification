import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.DataLoader import LoadItem
from models.Model import SimpleClassification

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


    def ValidateModel(path_to_checkpoint):
        test_loader = LoadItem(root_test_dir)
        test_model = SimpleClassification()
        test_model.load_state_dict(torch.load(path_to_checkpoint))
        test_model = test_model.to(device)
        test_model.eval()
        test_iterator = DataLoader(test_loader, batch_size=8, shuffle=False)

        accuracy = 0
        category = 0.0
        total_positives = 0
        actual_positives = 0
        recall = 0

        for idx, val_examples in enumerate(test_iterator):
            print("batch Number is: {}".format(idx))
            pred_labels = test_model(val_examples[1])
            pred_labels = torch.max(pred_labels, 1)
            actual_labels = generate_labels(val_examples[0])
            actual_labels = torch.max(actual_labels, 1)
            # print("The predicted labels are: {}".format(pred_labels))
            # print("The actual labels are: {}".format(actual_labels))

            # define your metrics here
            correct_predictions = (pred_labels.indices == actual_labels.indices)
            accuracy += torch.sum(correct_predictions == True)

            temp_total_positives = pred_labels.indices == category
            temp_actual_positives = temp_total_positives == (actual_labels == category)
            actual_positives += torch.sum(temp_actual_positives == True)
            total_positives += torch.sum(temp_total_positives == True)

            # Recall and confusion metric to this model

        print("the accuracy of the model is {}".format(accuracy))
        print("the precision of category {} is {}".format(category, actual_positives / total_positives))
        print("the recall of category {} is {}".format(category, recall))

        return None


    ValidateModel('checkpoint/saved_model.pt')


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

            torch.save(train_model.state_dict(), 'checkpoint/saved_model.pt')

            # Test the trained model
            ValidateModel('checkpoint/saved_model.pt')

    # Put the model on training
    # TrainModel(batch_size=4, epochs=1)