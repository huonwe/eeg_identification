# encoding=utf8
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader


class eegDataset(Dataset):
    def __init__(self, hdf5Path, channel):
        self.data = h5py.File(hdf5Path, 'r')
        self.eeg_data = self.data['data']
        self.eeg_labels = self.data['labels']
        self.channel = channel

    def __getitem__(self, item):
        eeg_label = self.eeg_labels[item]
        label = np.argmax(eeg_label)
        label = np.array(label)
        label = torch.from_numpy(label)

        self.prepared_data = np.array(self.eeg_data[item]).transpose((1, 0))

        if self.channel == "P":
            data = torch.from_numpy(self.prepared_data[46:55])
        elif self.channel == "C":
            data = torch.from_numpy(self.prepared_data[7:14])
        elif self.channel == "CP":
            data = torch.from_numpy(self.prepared_data[14:21])
        elif self.channel == "P-C":
            data = torch.from_numpy(
                np.concatenate((self.prepared_data[7:14], self.prepared_data[46:55]), axis=0))
        elif self.channel == "C-CP":
            data = torch.from_numpy(self.prepared_data[7:21])
        elif self.channel == "P-CP":
            data = torch.from_numpy(
                np.concatenate((self.prepared_data[14:21], self.prepared_data[46:55]), axis=0))
        elif self.channel == "P-C-CP":
            data = torch.from_numpy(
                np.concatenate((self.prepared_data[7:21], self.prepared_data[46:55]), axis=0))
        elif self.channel == "64":
            data = torch.from_numpy(self.prepared_data)
        elif self.channel == "128":
            data = torch.from_numpy(self.prepared_data)
        elif self.channel == "4":
            data = torch.from_numpy(
                np.concatenate((self.prepared_data[31:32], self.prepared_data[35:36], self.prepared_data[48:49],
                                self.prepared_data[52:53]), axis=0)
            )
        elif self.channel == "8":
            data = np.concatenate((self.prepared_data[31:32], self.prepared_data[35:36], self.prepared_data[48:49],
                                   self.prepared_data[52:53]), axis=0)
            data = np.concatenate(
                (data[:], self.prepared_data[8:9], self.prepared_data[10:11], self.prepared_data[12:13],
                 self.prepared_data[50:51]), axis=0)
            data = torch.from_numpy(data)
        elif self.channel == "16":
            data = np.concatenate((self.prepared_data[31:32], self.prepared_data[35:36], self.prepared_data[48:49],
                                   self.prepared_data[52:53]), axis=0)
            data = np.concatenate(
                (data[:], self.prepared_data[8:9], self.prepared_data[10:11], self.prepared_data[12:13],
                 self.prepared_data[50:51]), axis=0)
            data = np.concatenate(
                (data[:], self.prepared_data[21:22], self.prepared_data[23:24], self.prepared_data[29:30],
                 self.prepared_data[37:38], self.prepared_data[40:41], self.prepared_data[41:42],
                 self.prepared_data[46:47], self.prepared_data[54:55]), axis=0)
            data = torch.from_numpy(data)
        elif self.channel == "32":
            data = np.concatenate((self.prepared_data[31:32], self.prepared_data[35:36], self.prepared_data[48:49],
                                   self.prepared_data[52:53]), axis=0)
            data = np.concatenate(
                (data[:], self.prepared_data[8:9], self.prepared_data[10:11], self.prepared_data[12:13],
                 self.prepared_data[50:51]), axis=0)
            data = np.concatenate(
                (data[:], self.prepared_data[21:22], self.prepared_data[23:24], self.prepared_data[29:30],
                 self.prepared_data[37:38], self.prepared_data[40:41], self.prepared_data[41:42],
                 self.prepared_data[46:47], self.prepared_data[54:55]), axis=0)

            data = np.concatenate(
                (data[:], self.prepared_data[22:23], self.prepared_data[26:27], self.prepared_data[33:34],
                 self.prepared_data[57:58], self.prepared_data[0:1], self.prepared_data[2:3],
                 self.prepared_data[3:4], self.prepared_data[4:5], self.prepared_data[6:7],
                 self.prepared_data[14:15], self.prepared_data[16:17], self.prepared_data[17:18],
                 self.prepared_data[18:19], self.prepared_data[20:21], self.prepared_data[60:61],
                 self.prepared_data[62:63]), axis=0)
        else:
            assert "illegal channel"

        return data, label

    def __len__(self):
        return len(self.eeg_labels)

    def __del__(self):
        self.data.close()


if __name__ == '__main__':
    d = eegDataset('./train.hdf5', channel="32")
    train_loader = DataLoader(dataset=d,
                              batch_size=1,
                              shuffle=False,
                              drop_last=True)
    for i, datat in enumerate(train_loader):
        print(datat[0].shape)
        break
