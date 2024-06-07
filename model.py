import torch
import torch.nn as nn


class BiLSTM_Conv1d(nn.Module):
    def __init__(self):
        super(BiLSTM_Conv1d, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(80, 40)
        self.bn3 = nn.BatchNorm1d(40)
        self.fc4 = nn.Linear(40, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10, 1)

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc_conv_1 = nn.Linear(128, 64)
        self.fc_conv_2 = nn.Linear(64, 16)  # 假设最终分类为10类

    def forward(self, x_1, x_2):
        h0 = torch.zeros(4, x_1.size(0), 128).to(x_1.device)
        c0 = torch.zeros(4, x_1.size(0), 128).to(x_1.device)

        x_1, _ = self.lstm(x_1, (h0, c0))
        x_1 = x_1[:, -1, :]
        x_1 = self.bn(x_1)

        x_1 = self.relu(self.bn1(self.fc1(x_1)))
        x_1 = self.relu(self.bn2(self.fc2(x_1)))

        x_2 = x_2.unsqueeze(1)
        x_2 = self.pool(torch.relu(self.conv1(x_2)))
        x_2 = self.pool(torch.relu(self.conv2(x_2)))
        x_2 = x_2.view(-1, 32 * 4)  # 展平特征图
        x_2 = torch.relu(self.fc_conv_1(x_2))
        x_2 = self.fc_conv_2(x_2)

        output = torch.cat((x_1, x_2), dim=1)
        output = self.relu(self.bn3(self.fc3(output)))
        output = self.relu(self.bn4(self.fc4(output)))
        output = torch.sigmoid(self.fc5(output))

        return output.squeeze(1).to(x_1.device)


class BiLSTM_Conv1d_2(nn.Module):
    def __init__(self):
        super(BiLSTM_Conv1d_2, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(80, 40)
        self.bn3 = nn.BatchNorm1d(40)
        self.fc4 = nn.Linear(40, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.fc5 = nn.Linear(10, 1)

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc_conv_1 = nn.Linear(128, 64)
        self.fc_conv_2 = nn.Linear(64, 16)  # 假设最终分类为10类

    def forward(self, x_1, x_2):
        h0 = torch.zeros(6, x_1.size(0), 256).to(x_1.device)
        c0 = torch.zeros(6, x_1.size(0), 256).to(x_1.device)

        x_1, _ = self.lstm(x_1, (h0, c0))
        x_1 = x_1[:, -1, :]
        x_1 = self.bn(x_1)

        x_1 = self.relu(self.bn1(self.fc1(x_1)))
        x_1 = self.relu(self.bn2(self.fc2(x_1)))

        x_2 = x_2.unsqueeze(1)
        x_2 = self.pool(torch.relu(self.conv1(x_2)))
        x_2 = self.pool(torch.relu(self.conv2(x_2)))
        x_2 = x_2.view(-1, 32 * 4)  # 展平特征图
        x_2 = torch.relu(self.fc_conv_1(x_2))
        x_2 = self.fc_conv_2(x_2)

        output = torch.cat((x_1, x_2), dim=1)
        output = self.relu(self.bn3(self.fc3(output)))
        output = self.relu(self.bn4(self.fc4(output)))
        output = torch.sigmoid(self.fc5(output))

        return output.squeeze(1).to(x_1.device)
