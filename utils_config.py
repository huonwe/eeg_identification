import os
import sys
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from future_model import *
from datasets import eegDataset

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

lstm_seri = ['lstm', 'blstm']
gru_seri = ['gru', 'bgru']


def get_config(network):
    cfg = edict()
    # COMMON
    cfg.lr = 0.0001
    cfg.dropout = 0.3
    cfg.batch_size = 64  # 64
    cfg.network = network
    cfg.num_lstm = 2

    return cfg


def get_model(cfg, num_channel, **kwargs):
    # cnn / n_vcnn / lstm / cnn_lstm / cnn_gru /
    # n_vcnn_lstm / vcnn_lstm / vcnn_gru /
    # lstm_cnn / lstm_vcnn / cnn_blstm / vcnn_blstm / blstm_cnn

    # vcnn
    # n_vcnn_gru
    # gru 2

    # n_vcnn_blstm
    if cfg.network == "cnn":
        return CNN(num_channel, cfg.dropout)
    elif cfg.network == "vcnn":
        return VCNN(num_channel, cfg.dropout)
    elif cfg.network == "n_vcnn":
        return N_VCNN(num_channel, cfg.dropout)
    elif cfg.network == "lstm":
        return LSTM(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "gru":
        return GRU(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "cnn_lstm":
        return CNN_LSTM(num_channel, cfg.dropout, cfg.num_lstm)
    elif cfg.network == "cnn_gru":
        return CNN_GRU(num_channel, cfg.dropout, cfg.num_lstm)
    elif cfg.network == "cnn_blstm":
        return CNN_BLSTM(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "vcnn_lstm":
        return VCNN_LSTM(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "n_vcnn_lstm":
        return N_VCNN_LSTM(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "vcnn_gru":
        return VCNN_GRU(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "n_vcnn_gru":
        return N_VCNN_GRU(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "n_vcnn_blstm":
        return N_VCNN_BLSTM(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "vcnn_blstm":
        return VCNN_BLSTM(num_channel, cfg.dropout, cfg.num_lstm)
    elif cfg.network == "vcnn_bgru":
        return VCNN_BGRU(num_channel, cfg.dropout, cfg.num_lstm)
    elif cfg.network == "lstm_cnn":
        return LSTM_CNN(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "lstm_vcnn":
        return LSTM_VCNN(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "blstm_cnn":
        return BLSTM_CNN(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "f_lstm_cnn":
        return F_LSTM_CNN(num_channel=num_channel, dropout=cfg.dropout, lstm_num=cfg.num_lstm)
    elif cfg.network == "eegnet":
        return EEGNet(num_channel=num_channel, dropout=cfg.dropout)

    else:
        raise ValueError()


def prepare_data(batch_size, channel, path):
    if not path:
        path = "../datasets/train.hdf5"
    print("preparing data: " + path)
    dataset = eegDataset(path, channel)
    length = len(dataset)
    train_size, validate_size = int(0.8 * length), int(0.1 * length)

    train_set, validate_set, test_set = torch.utils.data.random_split(dataset, [train_size, validate_size,
                                                                                length - train_size - validate_size])
    if channel == "P":
        num_channel = 55 - 47 + 1
    elif channel == "C":
        num_channel = 14 - 8 + 1
    elif channel == "CP":
        num_channel = 21 - 15 + 1
    elif channel == "P-C":
        num_channel = 14 - 8 + 1 + 55 - 47 + 1
    elif channel == "C-CP":
        num_channel = 21 - 8 + 1
    elif channel == "P-CP":
        num_channel = 21 - 15 + 1 + 55 - 47 + 1
    elif channel == "P-C-CP":
        num_channel = 21 - 8 + 1 + 55 - 47 + 1
    elif channel == "128":
        num_channel = 128
    elif channel == "64":
        num_channel = 64
    elif channel == "4":
        num_channel = 4
    elif channel == "8":
        num_channel = 8
    elif channel == "16":
        num_channel = 16
    elif channel == "32":
        num_channel = 32
    else:
        assert "illegal channel"

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=validate_set,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)
    return train_loader, val_loader, test_loader, num_channel
