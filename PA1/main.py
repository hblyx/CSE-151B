import argparse
import data
import network
import numpy as np

from matplotlib import pyplot as plt


def main(hyperparameters):
    pass # the practical implement and pass dataset in the notebook

hyperparameters = {"epochs": 100,
                   "batch_size": 64,
                   "learning_rate": 0.001,
                   "k_folds": 10}

# training and validation
def cross_validation(dataset, hyperparameters, activation, loss, out_dim): # same method in the notebook
    regressor = network.Network(hyperparameters, activation, loss, out_dim)

    losses_train = []
    losses_val = []

    accs_val = []

    for train, val in data.generate_k_fold_set(dataset, k=hyperparameters["k_folds"]):
        for epoch in range(hyperparameters["epochs"]):
            train = data.shuffle(train)
            loss_train = [] # training loss per epoch including all mini-batchs
            for minibatch in data.generate_minibatches(train, batch_size=hyperparameters["batch_size"]): # SGD
                loss_batch, _ = regressor.train(minibatch)
                loss_train.append(loss_batch)

            loss_val, acc_val = regressor.test(val) # use the current weight to get a validation loss

            if len(losses_val) != 0 and loss_val > losses_val[-1]: # if the validation loss rise up
                break # early stopping

            loss_train = np.mean(loss_train) # average all mini-batchs
            losses_train.append(loss_train)
            losses_val.append(loss_val)
            accs_val.append(acc_val)

    plt.plot(losses_train)
    plt.plot(losses_val)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss plot")
    plt.legend(["Training Loss", "Validation Loss"])

    return regressor, np.mean(accs_val)

main(hyperparameters)
