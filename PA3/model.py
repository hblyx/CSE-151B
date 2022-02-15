import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.conv1_bn = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv2_bn = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv3_bn = nn.BatchNorm2d(num_features=128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.conv4_bn = nn.BatchNorm2d(num_features=128)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        # after convolution, we get fully connected layer
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=128)
        self.dropout = nn.Dropout()

        self.fc2 = nn.Linear(in_features=128, out_features=20)  # 20 classes

    def forward(self, x):
        # conv1(out_channels=64, kernel_size=3)+BN+ReLU
        x = F.relu(self.conv1_bn(self.conv1(x)))
        # conv2(out_channels=128, kernel_size=3)+BN+ReLU
        x = F.relu(self.conv2_bn(self.conv2(x)))
        # conv3(out_channels=128, kernel_size=3)+BN+ReLU -> maxpool1(kernel_size=3)
        x = F.max_pool1d(F.relu(self.conv3_bn(self.conv3(x))), kernel_size=3)
        # conv4(out_channels=128, kernel_size=3, stride=2)+BN+ReLU
        x = F.relu(self.conv4_bn(self.conv4(x)))
        # adaptive_avg_pool(output_size=1x1)
        x = self.aap(x)
        # fc1(out_features=128)+dropout+ReLU
        x = F.relu(self.dropout(self.fc1(x)))
        # fc2(out_features=num_classes)
        x = self.fc2(x)

        return x


class custom(nn.Module):
    pass


class resnet(nn.Module):
    pass


class vgg(nn.Module):
    pass


def get_model(args):
    model = None
    return model
