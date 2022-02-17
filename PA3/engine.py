# Modified by Colin Wang, Weitang Liu
import torch
import numpy as np

from data import *
from model import baseline


def prepare_model_baseline(device, args=None):
    # load model, criterion, optimizer, and learning rate scheduler
    model = baseline().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer


def train_model_baseline(model, criterion, optimizer, device, dataloaders, args=None):
    # get data loaders
    loaders = get_dataloaders("./food-101/train.csv", "./food-101/test.csv", transform=transform_test)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_loss = float('inf')

    for epoch in range(25):
        train_l, val_l = [], []
        correct_train, total_train = 0, 0
        correct_val, total_val = 0, 0
        mb = 0

        print("Epoch", epoch, "starts")
        for X, y in loaders[0]:  # training set
            print("minibatch", mb)
            # move to GPU
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)  # forward pass
            loss_train = criterion(y_pred, y)  # compute loss
            train_l.append(loss_train.item())

            # accuracy
            preds = torch.argmax(y_pred.data, 1)
            total_train += y.shape[0]
            correct_train += torch.sum(preds == y)

            # zero gradients to backward and update weights
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            for X_val, y_val in loaders[1]:  # validation set
                # move to GPU
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                val_pred = model(X_val)
                loss_val = criterion(val_pred, y_val)
                val_l.append(loss_val.item())

                preds = torch.argmax(val_pred.data, 1)
                total_val += y_val.shape[0]
                correct_val += torch.sum(preds == y_val)

            mb += 1


        train_loss.append(np.mean(train_l))
        train_acc.append(float(correct_train) / float(total_train))

        val_loss_avg = np.mean(val_l)
        val_loss.append(val_loss_avg)
        val_acc.append(float(correct_val) / float(total_val))

        if val_loss_avg < best_val_loss:  # if current validation loss is less
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "checkpoint.pt")

        print("Epoch", epoch, "ends")

    model = baseline()
    model.load_state_dict(torch.load("checkpoint.pt"))

    # return the model with weight selected by best performance
    return model, train_loss, val_loss, train_acc, val_acc


# add your own functions if necessary
def evaluate_test(test_loader, net, device):
    total = 0
    correct = 0
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        outputs = net(X)
        predictions = torch.argmax(outputs.data, 1)

        total += y.shape[0]

        correct += torch.sum(predictions == y)

    return float(correct) / float(total)
