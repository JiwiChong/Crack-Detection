import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM

class LeNet5(nn.Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.n_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Output layer
        return x


class CNN_0FC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rgb = args.rgb
        self.input_channel = 3 if self.rgb else 1
        self.use_cbam = args.use_cbam
        factor = 2 if args.half else 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.first_cbam = CBAM(args.first_dim, args.first_channel)
        self.dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(self.input_channel, int(64/factor), 3)  # originally 3, 64, 3
        self.conv2 = nn.Conv2d(int(64/factor), int(64/factor), 1)  # originally 64, 64, 1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(int(64/factor))  # originally 64

        self.second_cbam = CBAM(args.second_dim, args.second_channel)
        self.conv3 = nn.Conv2d(int(64/factor), int(128/factor), 3)   # originally 64, 128, 3
        self.conv4 = nn.Conv2d(int(128/factor), int(128/factor), 1)  # originally 128, 128, 1
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(int(128/factor))  # originally 128
 
        self.conv5 = nn.Conv2d(int(128/factor), int(256/factor), 3)  # originally 128, 256, 3
        self.conv6 = nn.Conv2d(int(256/factor), int(256/factor), 1)  # originally 256, 256, 1
        self.pool3 = nn.MaxPool2d(2, 2)
        self.batchnorm3 = nn.BatchNorm2d(int(256/factor))  # originally 256

        self.conv7 = nn.Conv2d(int(256/factor), int(512/factor), 3) # originally 256, 512, 3
        self.conv8 = nn.Conv2d(int(512/factor), int(512/factor), 1)  # originally 512, 512, 1
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batchnorm4 = nn.BatchNorm2d(int(512/factor))  # originally 512

        self.out_fc = nn.Linear(int(2048/factor), args.n_classes) # originally 2048


    def forward(self, x):
        x = F.relu(self.conv1(x)) if not self.use_cbam else self.first_cbam(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.batchnorm1(self.pool1(x))

        x = F.relu(self.conv3(x)) if not self.use_cbam else self.second_cbam(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.batchnorm2(self.pool2(x))

        x = self.dropout(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.batchnorm3(self.pool3(x))

        x = self.dropout(F.relu(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = self.batchnorm4(self.pool4(x))
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.out_fc(x))

        return x


class Ensemble(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nets = nn.ModuleList()
        for _ in range(3):
            self.nets.append(CNN_0FC(args))

    def forward(self, inputs):
        outputs = []
        for net in self.nets:
            outputs.append(net(inputs))
        with torch.no_grad():
            ens_outs = torch.stack(outputs).mean(0)


        # ensemble output and the models output
        return ens_outs, outputs
    
class MultiScaleNet_Branch1(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights.view(8,1,3,3).repeat(1,3,1,1)
        self.weights.requires_grad = True
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(self.weights)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        seq_one_out = F.relu(self.batchnorm1(self.conv1(x)))
        maxp1_out = self.maxpool1(seq_one_out)

        seq_two_out = F.relu(self.batchnorm2(self.conv2(maxp1_out)))
        maxp2_out = self.maxpool2(seq_two_out)

        seq_thr_out = F.relu(self.batchnorm3(self.conv3(maxp2_out)))
        maxp3_out = self.maxpool3(seq_thr_out)

        return maxp3_out

class MultiScaleNet_Branch2(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.weights = weights.view(8,1,3,3).repeat(1,3,1,1)
        self.weights.requires_grad = True
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(self.weights)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 64, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        maxp1_out = self.maxpool1(x)

        seq_one_out = F.relu(self.batchnorm1(self.conv1(maxp1_out)))
        maxp2_out = self.maxpool2(seq_one_out)

        seq_two_out = F.relu(self.batchnorm2(self.conv2(maxp2_out)))
        maxp3_out = self.maxpool3(seq_two_out)

        return maxp3_out

class MultiScaleNet_Branch3(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights.view(8,1,3,3).repeat(1,3,1,1)
        self.weights.requires_grad = True

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.weights)
        self.batchnorm = nn.BatchNorm2d(8)
        self.maxpool3 = nn.MaxPool2d(2, 2)

    
    def forward(self, x):
        maxp1_out = self.maxpool1(x)
        maxp2_out = self.maxpool2(maxp1_out)
        seq_out = F.relu(self.batchnorm(self.conv(maxp2_out)))
        maxp3_out = self.maxpool3(seq_out)

        return maxp3_out
    

class MultiScaleNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.weights = torch.Tensor([[[0.11, 0.11, 0.11],
                                    [0.11, 0.11, 0.11],
                                    [0.11, 0.11, 0.11]],

                                    [[0., -1., 0.],
                                    [-1., 4., -1.],
                                    [0., -1., 0.]],

                                    [[-1., 0., -1.],
                                    [0., 4., 0.],
                                    [-1., 0., -1.]],
                                    
                                    [[1., 2., 1.],
                                    [0., 0., 0.],
                                    [-1., -2., -1.]],
                                    
                                    [[-2., -1, 0.],
                                    [-1., 0., 1.],
                                    [0., 1., 2.]],

                                    [[1., 0., -1.],
                                    [2., 0., -2.],
                                    [1., 0., -1.]],

                                    [[-1., -1., -1.],
                                    [-1., 8., -1.],
                                    [-1., -1., -1.]],
                                    
                                    [[1., -2., 1.],
                                    [-2., 4., -2.],
                                    [1., -2., 1.]]])
        self.branch_one = MultiScaleNet_Branch1(self.weights)
        self.branch_two = MultiScaleNet_Branch2(self.weights)
        self.branch_three = MultiScaleNet_Branch3(self.weights)

        self.dropout = nn.Dropout(p=0.2)

        self.conv_two = nn.Conv2d(136, 128, 3, padding=1)  # originally 3, 64, 3
        self.batchnorm_two = nn.BatchNorm2d(128)  # originally 64
        self.pool_two = nn.MaxPool2d(2, 2)
        
        self.conv_three = nn.Conv2d(128, 256, 3, padding=1)  # originally 64, 64, 1
        self.batchnorm_three = nn.BatchNorm2d(256)  # originally 64
        self.pool_three = nn.MaxPool2d(2, 2)

        self.conv_four = nn.Conv2d(256, 256, 3, padding=1)  # originally 64, 64, 1
        self.batchnorm_four = nn.BatchNorm2d(256)  # originally 64
        self.pool_four = nn.MaxPool2d(2, 2)

        self.out_fc = nn.Linear(1024, 2) # originally 2048


    def forward(self, x):
        branch_one_out = self.branch_one(x)
        branch_two_out = self.branch_two(x)
        branch_three_out = self.branch_three(x)
        concat_out = torch.cat((branch_one_out, branch_two_out, branch_three_out),dim=1)

        stretch1_out = self.pool_two(F.relu(self.batchnorm_two(self.conv_two(concat_out))))
        stretch2_out = self.pool_three(F.relu(self.batchnorm_three(self.conv_three(stretch1_out))))
        stretch3_out = self.pool_four(F.relu(self.batchnorm_four(self.conv_four(stretch2_out))))

        flat_out = torch.flatten(stretch3_out, 1)
        out = F.softmax(self.out_fc(flat_out), dim=1)

        return out