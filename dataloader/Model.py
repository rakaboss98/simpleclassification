import torch.cuda
import torch.nn as nn
import torch.nn.functional as functional

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleClassification(nn.Module):
    def __init__(self):
        super(SimpleClassification, self).__init__()
        self.simple_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),  # 256x256x3
            nn.BatchNorm2d(num_features=32),  # 254x254x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 147x147x32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),  # 73x73x64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 35x35x64

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),  # 16x16x64
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7x64

        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.simple_model(x)
        out = torch.flatten(out, 1)
        print("the shape of flattened out is {}".format(out.shape))
        out = self.linear(out)
        return out
