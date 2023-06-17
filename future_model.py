# encoding=utf-8
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
        self.linear1 = nn.Linear(in_features=self.num_channel * 160, out_features=512)

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


class CNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(CNN_LSTM, self).__init__()
        self.CNN_ = CNN_(num_channel=num_channel)
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=512)

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
        self.linear1 = nn.Linear(in_features=self.num_channel * 160, out_features=512)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu((self.conv5(x)))
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


class N_VCNN_(nn.Module):
    def __init__(self, num_channel, dropout):
        super(N_VCNN_, self).__init__()
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


class N_VCNN_LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_LSTM, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2

        self.N_VCNN = N_VCNN_(num_channel, dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=512)

    def forward(self, x, hidden=None):
        x = self.N_VCNN.forward(x)
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


class N_VCNN_GRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(N_VCNN_GRU, self).__init__()
        self.num_channel = num_channel
        self.lstm_hidden_size = num_channel * 2
        self.N_VCNN = N_VCNN_(num_channel=num_channel, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        self.gru = nn.GRU(input_size=self.num_channel, hidden_size=self.lstm_hidden_size, num_layers=lstm_num)
        self.linear3 = nn.Linear(in_features=self.lstm_hidden_size * 160, out_features=512)

    def forward(self, x, hidden=None):
        x = self.N_VCNN.forward(x)
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


class GRU(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(GRU, self).__init__()
        self.num_channel = num_channel

        self.gru = nn.GRU(input_size=self.num_channel, hidden_size=self.num_channel * 2, dropout=dropout,
                          num_layers=lstm_num)
        self.linear = nn.Linear(in_features=self.num_channel * 2 * 160, out_features=512)

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


class LSTM(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(LSTM, self).__init__()
        self.num_channel = num_channel

        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.num_channel * 2, dropout=dropout,
                            num_layers=lstm_num)
        self.linear = nn.Linear(in_features=self.num_channel * 2 * 160, out_features=512)

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        print(x.shape)
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

class LSTM_with_Attention(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num=2):
        super(LSTM_with_Attention, self).__init__()
        self.num_channel = num_channel
        self.lstm = nn.LSTM(input_size=self.num_channel, hidden_size=self.num_channel, dropout=dropout,
                            num_layers=lstm_num)
        self.linear = nn.Linear(in_features=self.num_channel * 160, out_features=512)
        self.attention_weight = nn.Parameter(torch.randn(self.num_channel * 160, self.num_channel * 160))

    def forward(self, x, hidden=None):
        x = x.view(160, -1, self.num_channel)
        x, hidden = self.lstm(x, hidden)
        x = x.view(-1, self.num_channel * 160)  # N,20480
        attention = F.softmax(torch.mm(x,self.attention_weight), dim=1)
        # print(attention.shape)  # N，self.num_channel * 2 * 160  ## N, 20480
        x = torch.mul(x , attention)   # N,20480
        x = self.linear(x)
        return x, hidden

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)

class F_LSTM_CNN(nn.Module):
    def __init__(self, num_channel, dropout, lstm_num):
        super(F_LSTM_CNN, self).__init__()
        self.dropout = dropout
        self.num_channel = num_channel
        self.hidden_size = self.num_channel * 2
        self.CNN = CNN(num_channel=self.hidden_size, dropout=dropout)
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


class EEGNet(nn.Module):
    def __init__(self, num_channel=64, dropout=0.25):
        super(EEGNet, self).__init__()
        self.num_channel = num_channel
        self.dropout = dropout

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, self.num_channel), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # 全连接层
        # 此维度将取决于数据中每个样本的时间戳数。
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 10, 512)

    def forward(self, x):
        x = x.view(-1, 1, 160, self.num_channel)

        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dropout)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling3(x)
        # 全连接层
        x = x.reshape(-1, 4 * 2 * 10)
        x = self.fc1(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data)  # normal: mean=0, std=1
            # if isinstance(m, nn.Conv1d):
            #     init.normal_(m.weight.data)


class EEGLstmNet3fc_82_200(nn.Module):
    def __init__(self, num_channel=64, dropout=0.25):
        super(EEGNet, self).__init__()
        self.num_channel = num_channel
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (101, 1), padding=(50, 0)),
            nn.BatchNorm2d(8, False),
            nn.Sigmoid()
        )

        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (1, 64), groups=8),
            nn.BatchNorm2d(16, False),
            nn.Sigmoid(),
            nn.AvgPool2d((4, 1))
        )

        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8),
            nn.BatchNorm2d(16, False),
            nn.Sigmoid(),
            nn.Conv2d(16, 16, (1, 1), padding=0),
            nn.BatchNorm2d(16, False),
            nn.Sigmoid(),
            nn.AvgPool2d((2, 1))
        )

        # LSTM Layer
        self.rnn = nn.LSTM(
            input_size=16,
            hidden_size=16 * 25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （batch,time_step,input）时是Ture
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(16 * 25 * 2 * 2, False),
            # nn.Dropout(0.15),
            nn.Linear(16 * 25 * 2 * 2, 800),
            nn.BatchNorm1d(800, False),
            nn.Sigmoid(),
            # nn.Linear(800, 400),
            # nn.Dropout(0.15),
            # nn.BatchNorm1d(800, False),
            # nn.Sigmoid(),
            nn.Linear(800, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.dropout(x, 0.15)

        # Layer 2
        x = self.conv2(x)
        x = F.dropout(x, 0.15)

        # Layer 3
        x = self.conv3(x)
        x = F.dropout(x, 0.15)
        #
        # # LSTM Layer
        x = x.view(-1, 16, 25)
        x = x.permute(0, 2, 1)
        x, (h_n, h_c) = self.rnn(x, None)
        x = self.fc(x[:, -1, :])
        return x


def main():
    # x = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    # print(x)
    # z = F.softmax(x,dim=0)
    # print(z)
    # y = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    # print(y)
    # z = torch.mm(x,y)
    # print(z)

    net = LSTM_with_Attention(64, 0.2, 2)
    # net = N_VCNN(64, 0.2)
    #  print(net)
    # lstm_h = (torch.randn(2, 2, 64*3), torch.randn(2, 2, 64*3))
    inputs = torch.randn(3, 64, 160)
    # print(inputs.shape)
    # results = net(inputs)
    results, _ = net(inputs)
    print(results.shape)


if __name__ == "__main__":
    main()
