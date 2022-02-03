################################################################################
# CSE 151b: Programming Assignment 2
# Code snippet by Eric Yang Yu, Ajit Kumar, Savyasachi
# Winter 2022
################################################################################
import copy

from data import write_to_file, generate_minibatches
from neuralnet import *


def train(x_train, y_train, x_val, y_val, config, experiment=None):
    """
    Train your model here using batch stochastic gradient descent and early stopping. Use config to set parameters
    for training like learning rate, momentum, etc.

    Args:
        x_train: The train patterns
        y_train: The train labels
        x_val: The validation set patterns
        y_val: The validation set labels
        config: The configs as specified in config.yaml
        experiment: An optional dict parameter for you to specify which experiment you want to run in train.

    Returns:
        5 things:
            training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
            best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None

    model = NeuralNetwork(config=config)

    # mini-batched SGD
    for epoch in range(config["epochs"]):
        loss_train = []  # training loss per epoch including all mini-batchs
        acc_train = []
        for batch in generate_minibatches((x_train, y_train), config["batch_size"]):
            x, t = batch
            best_model = copy.deepcopy(model)

            model.forward(x, targets=t)
            model.backward()  # batch SDG within backward()

            # get training loss and accuracy of this batch
            batch_loss, batch_acc = test(model, x, t)
            loss_train.append(batch_loss)
            acc_train.append(batch_acc)

        # finish a epoch
        # check validation
        loss_val, acc_val = test(model, x_val, y_val)

        if config["early_stop"]: # if early stop is enabled
            if len(val_loss) != 0 and loss_val > val_loss[-1]:  # if the validation loss rise up
                break  # early stopping

        loss_train = np.mean(loss_train)  # average all batchs in this epoch
        acc_train = np.mean(acc_train)  # average all batchs in this epoch

        # store the loss and accuracy
        train_loss.append(loss_train)
        train_acc.append(acc_train)
        val_loss.append(loss_val)
        val_acc.append(acc_val)
        # update the best model
        best_model = copy.deepcopy(model)

    return train_acc, val_acc, train_loss, val_loss, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and returns loss and accuracy on the test set.

    Args:
        model: The trained model to run a forward pass on.
        x_test: The test patterns.
        y_test: The test labels.

    Returns:
        Loss, Test accuracy
    """
    # return loss, accuracy
    _, loss = model.forward(x_test, targets=y_test)
    acc = model.accuracy()

    return loss, acc


def train_mlp(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function trains a single multi-layer perceptron and plots its performances.

    NOTE: For this function and any of the experiments, feel free to come up with your own ways of saving data
            (i.e. plots, performances, etc.). A recommendation is to save this function's data and each experiment's
            data into separate folders, but this part is up to you.
    """
    # train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
        train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    write_to_file('./results.pkl', data)


def activation_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests all the different activation functions available and then plots their performances.
    """
    raise NotImplementedError('Activation Experiment not implemented')


def topology_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests performance of various network topologies, i.e. making
    the graph narrower and wider by halving and doubling the number of hidden units.

    Then, we change number of hidden layers to 2 of equal size instead of 1, and keep
    number of parameters roughly equal to the number of parameters of the best performing
    model previously.
    """
    raise NotImplementedError('Topology Experiment not implemented')


def regularization_experiment(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """
    This function tests the neural network with regularization.
    """
    raise NotImplementedError('Regularization Experiment not implemented')


def check_gradients(train_data, config):
    """
    Check the network gradients computed by back propagation by comparing with the gradients computed using numerical
    approximation.
    """

    # functions to do the experiment
    def get_weight_grad(name, layer, idx, nn, is_bias=False):
        # we arbitrary pick some weights to do this experiment
        if is_bias:  # bias is stored different, so use different method
            weight = nn.layers[layer].b[0][idx]
            grad = nn.layers[layer].d_b[idx]
        else:
            weight = nn.layers[layer].w[0][idx]
            grad = nn.layers[layer].d_w[0][idx]

        return weight, grad

    def get_loss(weight, layer, idx, nn, is_bias=False):
        # get E(w + ep) and E(w - ep)
        higher = weight + epsilon
        lower = weight - epsilon

        if is_bias:
            nn.layers[layer].b[0][idx] = higher
            _, higher_loss = nn.forward(small_x, targets=small_y)
            nn.layers[layer].b[0][idx] = lower
            _, lower_loss = nn.forward(small_x, targets=small_y)

            # reset nn
            nn.layers[layer].b[0][idx] = weight
        else:
            nn.layers[layer].w[0][idx] = higher
            _, higher_loss = nn.forward(small_x, targets=small_y)
            nn.layers[layer].w[0][idx] = lower
            _, lower_loss = nn.forward(small_x, targets=small_y)

            # reset nn
            nn.layers[layer].w[0][idx] = weight

        return higher_loss, lower_loss

    def get_estimate(higher, lower):
        # use the (E(w + ep) - E(w - ep)) / 2*ep to estimate the gradient
        est = (higher - lower) / (2 * epsilon)
        return est

    def diff_grad(grad, est):
        # check the difference between gradience and estimated gradient
        diff = np.abs((grad - est))
        return diff

    def check_grad(name, layer, idx, nn, is_bias=False):
        # overall check the gradient implement
        weight, grad = get_weight_grad(name, layer, idx, nn, is_bias=is_bias)
        higher, lower = get_loss(weight, layer, idx, nn, is_bias=is_bias)
        est = get_estimate(higher, lower)
        diff = diff_grad(grad, est)

        return diff, grad, est

    # shuffle dataset
    imgs, labs = train_data

    shuffled_idx = np.random.permutation(len(train_data[1]))

    imgs = imgs[shuffled_idx]
    labs = labs[shuffled_idx]

    # get an example of data to do this experiment
    small_set = imgs[: 1], labs[:1]
    small_x, small_y = small_set[0], small_set[1]

    # get a network first
    nn = NeuralNetwork(config)
    output = nn(small_x, targets=small_y)

    # get weights by backpropagation
    nn.backward()

    epsilon = 0.01

    # bias of output weight
    b_o_diff, b_o_grad, b_o_est = check_grad("output bias", 2, 7, nn, is_bias=True)
    # bias of hidden weight
    b_h_diff, b_h_grad, b_h_est = check_grad("hidden bias", 0, 7, nn, is_bias=True)
    # weights of hidden to output
    w_ho_1_diff, w_ho_1_grad, w_ho_1_est = check_grad("hidden-output weight_1", 2, 7, nn, is_bias=False)
    w_ho_2_diff, w_ho_2_grad, w_ho_2_est = check_grad("hidden-output weight_2", 2, 8, nn, is_bias=False)
    # weights of input to hidden
    w_ih_1_diff, w_ih_1_grad, w_ih_1_est = check_grad("input-hidden weight_1", 0, 7, nn, is_bias=False)
    w_ih_2_diff, w_ih_2_grad, w_ih_2_est = check_grad("input-hidden weight_2", 0, 8, nn, is_bias=False)

    # get an array to output the result (ready to import to a table)
    return np.array([["output bias", b_o_diff, b_o_grad, b_o_est],
                     ["hidden bias", b_h_diff, b_h_grad, b_h_est],
                     ["hidden-output weight_1", w_ho_1_diff, w_ho_1_grad, w_ho_1_est],
                     ["hidden-output weight_2", w_ho_2_diff, w_ho_2_grad, w_ho_2_est],
                     ["input-hidden weight_1", w_ih_1_diff, w_ih_1_grad, w_ih_1_est],
                     ["input-hidden weight_2", w_ih_2_diff, w_ih_2_grad, w_ih_2_est]
                     ])
