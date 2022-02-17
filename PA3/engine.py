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


def train_model_baseline(model, criterion, optimizer, device, dataloaders, max_epoch=25, args=None):
    model.train()

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_loss = float('inf')

    for epoch in range(25):
        train_l = []
        correct_train, total_train = 0, 0

        print("Epoch", epoch, "starts")
        print("Going through training set")
        for batch_idx, (X, y) in enumerate(dataloaders[0]):  # training set
            # move to GPU
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # zero gradients
            output = model(X)  # forward pass
            loss = criterion(output, y)  # compute loss
            loss.backward()
            optimizer.step()

            train_l.append(loss.item())

            # accuracy
            preds = torch.argmax(output.data, 1)
            total_train += y.shape[0]
            correct_train += torch.sum(preds == y)

        train_loss.append(np.mean(train_l))
        train_acc.append(float(correct_train) / float(total_train))

        print("Going through validation set")
        val_los, val_ac = evaluate(dataloaders[1], model, criterion, device)

        val_loss.append(val_los)
        val_acc.append(val_ac)

        if val_los < best_val_loss:  # if current validation loss is less
            best_val_loss = val_los
            torch.save(model.state_dict(), "checkpoint.pt")

        print("Epoch", epoch, "ends")

    model = baseline().to(device)
    model.load_state_dict(torch.load("checkpoint.pt"))

    # return the model with weight selected by best performance
    return model, train_loss, val_loss, train_acc, val_acc


# add your own functions if necessary
def evaluate(dataloader, model, criterion, device, is_test=False):
    test = "Validation"
    if is_test:
        test = "Test"

    model.eval()
    loss = []
    acc = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            los = criterion(output, target)
            loss.append(los.item())

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(dataloader.dataset)
    loss = np.mean(loss)

    print('\n{} set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        test, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

    return loss, acc
