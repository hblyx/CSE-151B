# Modified by Colin Wang, Weitang Liu
import torch
import numpy as np

from data import *
from model import baseline, custom1


def prepare_model(device, model_type="baseline"):
    # load model, criterion, optimizer, and learning rate scheduler
    model = create_model(device, model_type=model_type)

    criterion = torch.nn.CrossEntropyLoss()
    if model_type == "custom3" or model_type == "custom":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        return model, criterion, optimizer, scheduler
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    return model, criterion, optimizer

def create_model(device, model_type="baseline"):
    if model_type == "custom1":
        model = custom1().to(device)
    elif model_type == "custom2":
        model = baseline().to(device)
    elif model_type == "custom3":
        model = baseline().to(device)
    elif model_type == "custom":
        model = custom1().to(device)
    else:
        model = baseline().to(device)

    return model


def train_model(model, criterion, optimizer, device, dataloaders, max_epoch=25, scheduler=None, model_type="baseline"):
    model.train()

    cp_name = model_type + ".pt"

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    best_val_loss = float('inf')

    for epoch in range(max_epoch):
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
            torch.save(model.state_dict(), cp_name)

        if scheduler is not None:
            scheduler.step()

        print("Epoch", epoch, "ends")

    model = create_model(device, model_type=model_type)
    model.load_state_dict(torch.load(cp_name))

    # return the model with weight selected by best performance
    return model, train_loss, val_loss, train_acc, val_acc


# add your own functions if necessary
def evaluate(dataloader, model, criterion, device, is_test=False):
    test = "Validation"
    if is_test:
        test = "Test"

    model.eval()
    loss = []
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
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
