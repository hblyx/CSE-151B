import torch
import torch.nn as nn
import torchvision.models as models


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s

    return num_features


class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()

        # for the baseline model, we use the transform_test to pre-process the data, which means we normalize and
        # corp center(224) pixels. Thus, the first input shape is (3, 224,224), (C, W, H).
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        # after convolution, we get fully connected layer
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=20)  # 20 classes

    def forward(self, x):



class custom(nn.Module):
    pass


class resnet(nn.Module):
    pass


class vgg(nn.Module):
    pass


def get_model(args):
    model = None
    return model
