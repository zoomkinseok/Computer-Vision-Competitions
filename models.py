import torch.nn as nn
import torch.nn.functional as F

'''
    Result report
    1. Simple CNN
        Epoch [300/300]
        Train Loss: 0.4453	Train Accuracy: 86.39%
        Validation Loss: 0.4931	Validation Accuracy: 81.04%

    2. ResNet
        Epoch [300/300]
        Train Loss: 0.3549	Train Accuracy: 95.83%
        Validation Loss: 0.3910	Validation Accuracy: 92.18%
'''

'''
    1. CNN model
'''
class Covnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)

        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)
'''
    2. ResNet
'''

class Block_1(nn.Module):
    def __init__(self, channels, stride=1):
        super(Block_1, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += x
        return out

class Block_2(nn.Module):
    def __init__(self, channels, stride=1):
        super(Block_2, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 2 * channels, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(2 * channels, 2 * channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1,
                      stride=2)
        )
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class ResNet(nn.Module):


    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.stage1 = self.build_stage(64, nblk_stage1, 1)
        self.stage2 = self.build_stage(64, nblk_stage2, 2)
        self.stage3 = self.build_stage(128, nblk_stage3, 3)
        self.stage4 = self.build_stage(256, nblk_stage4, 4)
        self.avg_pool = nn.AvgPool2d(4, stride=4)
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 2 * 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 2),
        )


    def build_stage(self, planes, num_block, num_stage):
        layers = []
        i = 0
        if num_stage == 1:
            layers.append(Block_1(planes))
            while i < (num_block - 1):
                layers.append(Block_1(planes))
                i = i + 1
        else:
            layers.append(Block_2(planes))
            while i < (num_block - 1):
                layers.append(Block_1(2 * planes))
                i = i + 1
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return F.softmax(out, dim=1)
