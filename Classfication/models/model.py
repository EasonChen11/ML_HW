import torch.nn as nn
import torch

# In project 5, you need to adjust the model architecture.
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        ################################ you need to modify the cnn model here ################################

        # after convolutoin, the feature map size = ((origin + padding*2 - kernel_size) / stride) + 1
        # input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)   # ((224+2*1-3)/1)+1=224  # output_shape=(64,224,224)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # output_shape=(64,112,112) # (224)/2

        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)   # output_shape=(128,112,112)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    # output_shape=(64,56,56)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)   # output_shape=(64,56,56)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)    # output_shape=(64,28,28)

        self.fc1 = nn.Linear(64*28*28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(512, 12)
        # =================================================================================================== #

    def forward(self, x):
        
        ################################ you need to modify the cnn model here ################################
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.fc3(out)
        # =================================================================================================== #

        return out