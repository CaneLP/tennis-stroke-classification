import torch
import torch.nn as nn
import torch.nn.functional as F


class TennisStrokeCNNClassifier1(nn.Module):
    def __init__(self, num_classes=4):
        
        super(TennisStrokeCNNClassifier1, self).__init__()
        self.num_classes = num_classes
        
        self.pad1 = nn.ReplicationPad2d(1)
        
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        
        self.fc1 = nn.Linear(32 * 7 * 3, 128)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
                
        
    def _num_flatten_features(self, x):
        num_features = 1
        for s in x.size()[1:]:  # Don't use batch size dim
            num_features *= s
        return num_features    
    
    
    def forward(self, x):
        x = self.pad1(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        
        x = x.view(-1, self._num_flatten_features(x))
        x = self.relu4(self.fc1(x))
        x = self.dropout2(x)
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)
        
        x = F.log_softmax(x, dim=1)
        
        return x

    
    
class TennisStrokeCNNClassifier2(nn.Module):
    def __init__(self, num_classes=4):
        
        super(TennisStrokeCNNClassifier2, self).__init__()
        
        self.num_classes = num_classes
        
        self.pad1 = nn.ReplicationPad2d(1)
        
        negative_slope = 0.01
        
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.LeakyReLU(negative_slope)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.LeakyReLU(negative_slope)
        
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu3 = nn.LeakyReLU(negative_slope)
        self.dropout2 = nn.Dropout(p=0.6)
        
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU(negative_slope)

        self.conv5 = nn.Conv2d(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.relu5 = nn.LeakyReLU(negative_slope)

        self.fc1 = nn.Linear(256, 256)
        self.relu6 = nn.LeakyReLU(negative_slope)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)
        self.relu7 = nn.LeakyReLU(negative_slope)
        self.dropout4 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(128, 64)
        self.relu8 = nn.LeakyReLU(negative_slope)
        self.fc4 = nn.Linear(64, num_classes)
            
        
    def _num_flatten_features(self, x):
        num_features = 1
        for s in x.size()[1:]:  # Dont' use batch size dim
            num_features *= s
        return num_features    
    
        
    def forward(self, x):
        x = self.pad1(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        
        x = self.bn3(self.conv3(x))
        x = self.pool1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        
        x = self.relu4(self.bn4(self.conv4(x)))
        
        x = self.pool2(self.bn5(self.conv5(x)))
        x = self.relu5(x)
                
        x = x.view(-1, self._num_flatten_features(x))
        x = self.relu6(self.fc1(x))
        x = self.dropout3(x)
        
        x = self.relu7(self.fc2(x))
        x = self.dropout4(x)
        
        x = self.relu8(self.fc3(x))
        x = self.fc4(x)
        
        x = F.log_softmax(x, dim=1)

        return x
    