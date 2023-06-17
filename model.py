import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CNN(nn.Module):
    def __init__(self, num_channel, dropout):
        super(CNN, self).__init__()
        self.num_channel = num_channel
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 2, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(in_features=self.num_channel * 160, out_features=109)
        # [1,2,3,....,109]

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = x.reshape(-1, self.num_channel * 160)
        x = self.dropout(x)
        x = self.linear1(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class CNN_(nn.Module):
    def __init__(self, num_channel):
        super(CNN_, self).__init__()
        self.num_channel = num_channel
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 2, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=2)
        return x


class VCNN(nn.Module):
    def __init__(self, num_channel, dropout):
        super(VCNN, self).__init__()
        self.num_channel = num_channel

        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel * 8)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=self.num_channel * 4)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=self.num_channel * 2)

        self.linear1 = nn.Linear(in_features=self.num_channel * 2 * 160, out_features=109)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        # print(x.shape)
        x = x.view(-1, self.num_channel * 2 * 160)
        x = self.linear1(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class N_VCNN(nn.Module):
    def __init__(self, num_channel, dropout):
        super(N_VCNN, self).__init__()
        self.num_channel = num_channel

        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel * 8)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=self.num_channel * 4)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=self.num_channel * 2)
        self.conv5 = nn.Conv1d(in_channels=self.num_channel * 2, out_channels=self.num_channel, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features=self.num_channel * 160, out_features=109)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu((self.conv5(x)))
        # print(x.shape)
        x = x.view(-1, self.num_channel * 160)
        x = self.linear1(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class VCNN_(nn.Module):
    def __init__(self, num_channel, dropout):
        super(VCNN_, self).__init__()
        self.num_channel = num_channel

        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.num_channel * 8)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=self.num_channel * 4)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=self.num_channel * 2)
        self.conv5 = nn.Conv1d(in_channels=self.num_channel * 2, out_channels=self.num_channel, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn4(self.conv4(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu((self.conv5(x)))
        # print(x.shape)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(CNN_LSTM, self).__init__()
        self.CNN_ = CNN_(num_channel=num_channel)

        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = self.CNN_.forward(x)

        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.view(-1, self.lstm_hidden_size * 160)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class CNN_BLSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(CNN_BLSTM, self).__init__()
        self.CNN_ = CNN_(num_channel=num_channel)
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num,
                            bidirectional=True)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 2 * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = self.CNN_.forward(x)
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.view(-1, self.lstm_hidden_size * 2 * 160)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1


class VCNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(VCNN_LSTM, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=self.num_channel * 2, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=109)

    def forward(self, x, hidden=None):
        # x = self.bn1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        # x = F.max_pool1d(x, kernel_size=2)
        # print(x.shape)
        x = x.view(160, -1, self.num_channel * 2)
        x, hidden = self.lstm(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.lstm_hidden_size * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class N_VCNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_LSTM, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.VCNN = VCNN_(num_channel, dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = self.VCNN.forward(x)
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.view(-1, self.lstm_hidden_size * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class VCNN_BLSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(VCNN_BLSTM, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)

        # self.linear1 = nn.Linear(in_features=64*160, out_features=lstm_input_size)
        self.lstm = nn.LSTM(input_size=self.num_channel * 2, hidden_size=self.lstm_hidden_size, num_layers=lstm_num,
                            bidirectional=True)
        # self.lstm1 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.lstm2 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.linear2 = nn.Linear(in_features=lstm_input_size, out_features=200)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 2 * 160, out_features=109)

    def forward(self, x, hidden=None):
        # x = self.bn1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        # x = F.max_pool1d(x, kernel_size=2)
        # print(x.shape)
        x = x.view(160, -1, self.num_channel * 2)
        x, hidden = self.lstm(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.lstm_hidden_size * 2 * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class N_VCNN_BLSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_BLSTM, self).__init__()
        self.num_channel = num_channel
        self.hidden_size = num_channel * 2

        self.NVCNN = VCNN_(num_channel, dropout)
        self.blstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.hidden_size, dropout=dropout,
                             num_layers=lstm_num, bidirectional=True)
        self.linear = nn.Linear(in_features=self.hidden_size * 2 * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = self.NVCNN.forward(x)
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.blstm(x, hidden)
        x = x.view(-1, self.hidden_size * 2 * 160)
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)

class VCNN_GRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(VCNN_GRU, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)

        # self.linear1 = nn.Linear(in_features=64*160, out_features=lstm_input_size)
        self.gru = nn.GRU(input_size=self.num_channel * 2, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        # self.lstm1 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.lstm2 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.linear2 = nn.Linear(in_features=lstm_input_size, out_features=200)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=109)

    def forward(self, x, hidden=None):
        # x = self.bn1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        # x = F.max_pool1d(x, kernel_size=2)
        # print(x.shape)
        x = x.view(160, -1, self.num_channel * 2)
        x, hidden = self.gru(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.lstm_hidden_size * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class N_VCNN_GRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_GRU, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2
        self.NVCNN = VCNN_(num_channel=num_channel, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        # self.linear1 = nn.Linear(in_features=64*160, out_features=lstm_input_size)
        self.gru = nn.GRU(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        # self.lstm1 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.lstm2 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.linear2 = nn.Linear(in_features=lstm_input_size, out_features=200)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = self.NVCNN.forward(x)
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.gru(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.lstm_hidden_size * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class VCNN_BGRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(VCNN_BGRU, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.bn1 = nn.BatchNorm1d(num_features=self.num_channel * 16)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 16, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)

        # self.linear1 = nn.Linear(in_features=64*160, out_features=lstm_input_size)
        self.gru = nn.GRU(input_size=self.num_channel * 2, hidden_size=self.lstm_hidden_size, num_layers=lstm_num,
                          bidirectional=True)
        # self.linear2 = nn.Linear(in_features=lstm_input_size, out_features=200)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 2 * 160, out_features=109)

    def forward(self, x, hidden=None):
        # x = self.bn1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        # x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        # x = F.max_pool1d(x, kernel_size=2)
        # print(x.shape)
        x = x.view(160, -1, self.num_channel * 2)
        x, hidden = self.gru(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.lstm_hidden_size * 2 * 160)
        x = self.dropout(x)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class CNN_GRU(nn.Module):
    def __init__(self, num_channel, dropout, gru_num):
        super(CNN_GRU, self).__init__()
        self.num_channel = num_channel
        self.gru_hidden_size = num_channel * 2
        self.bn1 = nn.BatchNorm1d(num_features=num_channel)
        self.conv1 = nn.Conv1d(in_channels=self.num_channel, out_channels=self.num_channel * 2, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.num_channel * 2, out_channels=self.num_channel * 4, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv3 = nn.Conv1d(in_channels=self.num_channel * 4, out_channels=self.num_channel * 8, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.conv4 = nn.Conv1d(in_channels=self.num_channel * 8, out_channels=self.num_channel * 16, kernel_size=(3,),
                               stride=(1,),
                               padding_mode='replicate',
                               padding=1)
        self.dropout = nn.Dropout(p=dropout)

        # self.linear1 = nn.Linear(in_features=64*160, out_features=lstm_input_size)
        self.gru = nn.GRU(input_size=self.num_channel, hidden_size=self.gru_hidden_size, num_layers=gru_num,
                          dropout=dropout)
        # self.lstm1 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.lstm2 = nn.LSTMCell(input_size=64*3, hidden_size=320)
        # self.linear2 = nn.Linear(in_features=lstm_input_size, out_features=200)
        self.linear3 = nn.Linear(in_features=self.gru_hidden_size * 160, out_features=109)

    def forward(self, x, hidden=None):
        # print(x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=2)
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.gru(x, hidden)
        # print(x.shape)
        x = x.view(-1, self.gru_hidden_size * 160)
        x = self.linear3(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class LSTM_CNN(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(LSTM_CNN, self).__init__()
        self.dropout = dropout
        self.num_channel = num_channel
        self.hidden_size = self.num_channel * 2

        self.CNN = CNN(num_channel=self.hidden_size, dropout=self.dropout)
        self.lstm1 = nn.LSTM(input_size=self.num_channel, hidden_size=self.hidden_size, num_layers=lstm_num)

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm1(x, hidden)
        x = x.view(-1, self.hidden_size, 160)
        x = self.CNN.forward(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class BLSTM_CNN(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(BLSTM_CNN, self).__init__()
        self.dropout = dropout
        self.num_channel = num_channel
        self.hidden_size = self.num_channel * 2

        self.CNN = CNN(num_channel=self.hidden_size * 2, dropout=self.dropout)
        self.lstm1 = nn.LSTM(input_size=self.num_channel, hidden_size=self.hidden_size, num_layers=lstm_num,
                             bidirectional=True)

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm1(x, hidden)
        x = x.view(-1, self.hidden_size * 2, 160)
        x = self.CNN.forward(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class LSTM_VCNN(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(LSTM_VCNN, self).__init__()
        self.dropout = dropout
        self.num_channel = num_channel
        self.hidden_size = self.num_channel * 2

        self.VCNN = VCNN(num_channel=self.hidden_size, dropout=self.dropout)
        self.lstm1 = nn.LSTM(input_size=self.num_channel, hidden_size=self.hidden_size, num_layers=lstm_num)

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm1(x, hidden)
        x = x.view(-1, self.hidden_size, 160)
        x = self.VCNN.forward(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class LSTM_(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(LSTM_, self).__init__()
        self.num_channel = num_channel

        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.num_channel * 2, dropout=dropout,
                            num_layers=lstm_num)
        self.linear = nn.Linear(in_features=self.num_channel * 2 * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.view(-1, self.num_channel * 2 * 160)
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class GRU_(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(GRU_, self).__init__()
        self.num_channel = num_channel

        self.gru = nn.GRU(input_size=self.num_channel, hidden_size=self.num_channel * 2, dropout=dropout,
                          num_layers=lstm_num)
        self.linear = nn.Linear(in_features=self.num_channel * 2 * 160, out_features=109)

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.gru(x, hidden)
        x = x.view(-1, self.num_channel * 2 * 160)
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


if __name__ == "__main__":
    net = CNN(dropout=0.3, num_channel=9)
    print(net)
    # lstm_h = (torch.randn(2, 2, 64*3), torch.randn(2, 2, 64*3))
    inputs = torch.randn(2, 9, 160)
    # print(inputs.shape)
    results, h = net(inputs)
    print(results[0])
