import os

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle
import argparse
from utils_config import get_config, get_model, prepare_data
from TripletLoss import TripletLoss
from navigator import Navigator

lstm_seri = ['lstm', 'blstm']
gru_seri = ['gru', 'bgru']
navigator = Navigator()


def train(args):
    device = torch.device("cuda:" + args.ctx if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.network)
    print(device)
    print(cfg)
    print("maxEpoch: ", args.maxEpoch)
    train_loader, val_loader, test_loader, num_channel = prepare_data(batch_size=cfg.batch_size, channel=args.channel,
                                                                      path=args.path)
    model_path = "./models/" + cfg.network + "/" + str(args.maxEpoch) + "/model" + "-" + args.channel + ".pth"
    if not os.path.exists("./models/" + cfg.network):
        os.mkdir("./models/" + cfg.network)
    if not os.path.exists("./models/" + cfg.network + "/" + str(args.maxEpoch)):
        # print("./models/" + cfg.network+"/"+str(args.maxEpoch))
        os.mkdir("./models/" + cfg.network + "/" + str(args.maxEpoch))

    train_loss = []
    val_loss = []
    val_acc = []

    network = get_model(cfg, num_channel=num_channel).to(device)

    if args.resume:
        t1 = time.time()
        network.load_state_dict(torch.load(model_path))
        t2 = time.time()

        train_acc = pickle.load(
            "./results/" + cfg.network + "/" + args.maxEpoch + "/" + args.channel + "_" + "train_acc.pkl")
        train_loss = pickle.load(
            "./results/" + cfg.network + "/" + args.maxEpoch + "/" + args.channel + "_" + "train_loss.pkl")
        val_acc = pickle.load(
            "./results/" + cfg.network + "/" + args.maxEpoch + "/" + args.channel + "_" + "val_acc.pkl")
        val_loss = pickle.load(
            "./results/" + cfg.network + "/" + args.maxEpoch + "/" + args.channel + "_" + "val_loss.pkl")
    else:
        network.initialize()
    print(cfg.network + ":" + args.channel)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.TripletMarginLoss()
    criterion = TripletLoss()
    optimizer = optim.Adam(network.parameters(), lr=cfg.lr)
    epoch = 0
    num_print = 0
    stop = False
    t3 = time.time()
    while not stop:
        if epoch > args.maxEpoch:
            stop = True
            recording = [train_loss, val_acc, val_loss]
            name = ["train_loss", "val_acc", "val_loss"]
            if not os.path.exists("./results/" + cfg.network):
                os.mkdir("./results/" + cfg.network)
            if not os.path.exists("./results/" + cfg.network + "/" + str(args.maxEpoch)):
                os.mkdir("./results/" + cfg.network + "/" + str(args.maxEpoch))
            for index, r in enumerate(recording):
                # print(index)
                file = open("./results/" + cfg.network + "/" + str(args.maxEpoch) + "/" + args.channel + "_" + name[
                    index] + ".pkl", "wb")
                pickle.dump(r, file)
                file.close()
            break
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            net_combine = cfg.network.split("_")
            if "lstm" in net_combine or "blstm" in net_combine:
                outputs, hidden = network(inputs)
                for h in hidden:
                    h.detach_()
            elif "gru" in net_combine or "bgru" in net_combine:
                outputs, hidden = network(inputs)
                hidden.detach_()
            else:
                outputs = network(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # _, predicted = torch.max(outputs.data, 1)
            # correct = (predicted == labels).sum().item()
            # total = labels.size(0)
            # acc = correct / total
            # running_acc += acc
            if i == 0:
                continue
            if i % 200 == 0:  # print every 200 mini-batches
                num_print += 1
                loss_avg = running_loss / 200
                print('[%d, %5d] loss: %.6f' % (epoch, i, loss_avg))
                running_loss = 0
                train_loss.append(loss_avg)

                if loss_avg < 0.001:
                    stop = True
                    recording = [train_loss, val_acc, val_loss]
                    name = ["train_loss", "val_acc", "val_loss"]
                    if not os.path.exists("./results/" + cfg.network):
                        os.mkdir("./results/" + cfg.network)
                    if not os.path.exists("./results/" + cfg.network + "/" + str(args.maxEpoch)):
                        os.mkdir("./results/" + cfg.network + "/" + str(args.maxEpoch))
                    for index, r in enumerate(recording):
                        # print(index)
                        file = open(
                            "./results/" + cfg.network + "/" + str(args.maxEpoch) + "/" + args.channel + "_" + name[
                                index] + ".pkl", "wb")
                        pickle.dump(r, file)
                        file.close()
                    break

                if num_print % 10 == 0:
                    # torch.save(network.state_dict(), model_path)
                    # test(cfg, num_channel=num_channel, channel=args.channel, test_loader=test_loader, ctx=args.ctx)
                    total_loss = []
                    total_acc = []
                    navigator.clear()

                    print("validating...")
                    dist_class = [[], []]
                    v_count = 0
                    for _, v_data in enumerate(val_loader):
                        v_input, v_label = v_data
                        v_input, v_label = v_input.to(device), v_label.to(device)

                        if "lstm" in net_combine or "blstm" in net_combine or "gru" in net_combine or "bgru" in net_combine:
                            v_outputs, v_hidden = network(v_input)
                        else:
                            v_outputs = network(v_input)

                        v_loss = criterion(v_outputs, v_label).item()
                        total_loss.append(v_loss)
                        total = v_label.size(0)

                        for x in range(total):
                            for y in range(x + 1, total - x):
                                v_count += 1
                                dist = navigator.distance(v_outputs[x], v_outputs[y])
                                truth_same = (v_label[x] == v_label[y]).item()
                                # print(dist, truth_same)
                                if truth_same:
                                    dist_class[0].append(dist)
                                else:
                                    dist_class[1].append(dist)

                    try:
                        print("Same Max Dist: %s, Min Dist: %s" % (np.max(dist_class[0]), np.min(dist_class[0])))
                        print("Same Avg Dist: %s, std: %s" % (np.mean(dist_class[0]), np.std(dist_class[0], ddof=1)))
                    except ValueError:
                        print("The Same is Empty")
                    try:
                        print("Diff Max Dist: %s, Min Dist: %s" % (np.max(dist_class[1]), np.min(dist_class[1])))
                        print("Diff Avg Dist: %s, std: %s" % (np.mean(dist_class[1]), np.std(dist_class[1], ddof=1)))
                    except ValueError:
                        print("The Diff is Empty")

                    try:
                        print("Adjusting Threshold: %s->%s" % (
                        navigator.threshold, np.mean(dist_class[0]) + 3 * np.std(dist_class[0], ddof=1)))
                        navigator.threshold = np.mean(dist_class[0]) + 3 * np.std(dist_class[0], ddof=1)
                    except ValueError:
                        print("Adjust Failed")

                    this_ = 0
                    for t in dist_class[0]:
                        if t < navigator.threshold:
                            this_ += 1
                    for f in dist_class[1]:
                        if f > navigator.threshold:
                            this_ += 1
                    this_acc = this_ / v_count

                    this_loss = np.mean(total_loss)
                    # print(len(total_loss))
                    print("validation loss: ", this_loss)
                    print("validation acc: ", this_acc)
                    val_loss.append(this_loss)
                    val_acc.append(this_acc)

        print("saving model...")
        epoch += 1
        torch.save(network.state_dict(), model_path)
    t4 = time.time()
    test(cfg, num_channel=num_channel, channel=args.channel, test_loader=test_loader, ctx=args.ctx,
         maxEpoch=args.maxEpoch)
    print("fin")
    if args.resume:
        print("Time Model Resume Loading : ", t2 - t1)
    # print("Time per Epoch : ",t4-t3)


def test(cfg, channel, num_channel, ctx, maxEpoch, test_loader=None):
    assert test_loader is not None
    device = torch.device("cuda:" + ctx if torch.cuda.is_available() else "cpu")
    print("testing ", cfg.network)
    maxEpoch = str(maxEpoch)
    model_path = "./models/" + cfg.network + "/" + maxEpoch + "/model" + "-" + channel + ".pth"
    network = get_model(cfg, num_channel)

    t5 = time.time()
    network.load_state_dict(torch.load(model_path))
    t6 = time.time()

    print("Time Model Test Loading : ", t6 - t5)

    network.eval().to(device)

    net_combine = cfg.network.split("_")

    criterion = TripletLoss()
    index = 0

    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0
        store_tl = []
        store_tp = []

        t_correct = 0
        t_count = 0

        t7 = time.time()
        dist_class = [[], []]
        for _, t_data in enumerate(test_loader):
            t_input, t_label = t_data
            t_input, t_label = t_input.to(device), t_label.to(device)
            # print(t_label.shape)

            if "lstm" in net_combine or "blstm" in net_combine or "gru" in net_combine or "bgru" in net_combine:
                t_outputs, t_hidden = network(t_input)
            else:
                t_outputs = network(t_input)

            # if not t_outputs:
            #     raise KeyError()
            # if cfg.network == "cnn_pure":
            #     t_outputs = network(t_input)
            # else:
            #     t_outputs, _ = network(t_input)
            t_loss = criterion(t_outputs, t_label).item()
            test_loss += t_loss
            t_total = t_label.size(0)


            for x in range(t_total):
                for y in range(x + 1, t_total - x):
                    t_count += 1
                    dist = navigator.distance(t_outputs[x], t_outputs[y])
                    truth_same = (t_label[x] == t_label[y]).item()
                    # print(dist, truth_same)
                    if truth_same:
                        dist_class[0].append(dist)
                    else:
                        dist_class[1].append(dist)
            index += 1

            store_tl.extend(label.item() for label in t_label)
            store_tp.extend(t_outputs.cpu().numpy())
        t8 = time.time()

        try:
            print("Same Max Dist: %s, Min Dist: %s" % (np.max(dist_class[0]), np.min(dist_class[0])))
            print("Same Avg Dist: %s, std: %s" % (np.mean(dist_class[0]), np.std(dist_class[0], ddof=1)))
        except ValueError:
            print("The Same is Empty")
        try:
            print("Diff Max Dist: %s, Min Dist: %s" % (np.max(dist_class[1]), np.min(dist_class[1])))
            print("Diff Avg Dist: %s, std: %s" % (np.mean(dist_class[1]), np.std(dist_class[1], ddof=1)))
        except ValueError:
            print("The Diff is Empty")

        try:
            print("Adjusting Threshold: %s->%s" % (
                navigator.threshold, np.mean(dist_class[0]) + 3 * np.std(dist_class[0], ddof=1)))
            navigator.threshold = np.mean(dist_class[0]) + 3 * np.std(dist_class[0], ddof=1)
        except ValueError:
            print("Adjust Failed")

        this_ = 0
        for t in dist_class[0]:
            if t < navigator.threshold:
                this_ += 1
        for f in dist_class[1]:
            if f > navigator.threshold:
                this_ += 1
        this_acc = this_ / t_count

        # t_acc = t_correct / count
        # test_acc += t_acc

        print("Time per Enumerate : ", (t8 - t7) / index)

        print("test_loss: ", test_loss / len(test_loader))
        print("test_acc: ", this_acc)

        print("threshold: ", navigator.threshold)

        if not os.path.exists("./results/" + cfg.network):
            os.mkdir("./results/" + cfg.network)
        if not os.path.exists("./results/" + cfg.network + "/" + maxEpoch):
            os.mkdir("./results/" + cfg.network + "/" + maxEpoch)
        test_label = pd.DataFrame(store_tl)
        test_label.to_csv("./results/" + cfg.network + "/" + maxEpoch + "/" + channel + '_test_label.csv')
        test_predict = pd.DataFrame(store_tp)
        test_predict.to_csv("./results/" + cfg.network + "/" + maxEpoch + "/" + channel + '_test_predict.csv')
        test_loss_s = pd.DataFrame([test_loss / len(test_loader)])
        test_loss_s.to_csv("./results/" + cfg.network + "/" + maxEpoch + "/" + channel + '_test_loss.csv')
        test_acc_s = pd.DataFrame([test_acc / len(test_loader)])
        test_acc_s.to_csv("./results/" + cfg.network + "/" + maxEpoch + "/" + channel + '_test_acc.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train option")
    parser.add_argument('network', type=str, help='Pytorch EEG Traing')
    parser.add_argument('--channel', default="64", type=str, help='P/C/CP/P-C/C-CP/P-CP/P-C-CP/64')
    parser.add_argument('--path', default="../datasets/train.hdf5", type=str, help='dataset path')
    parser.add_argument('--ctx', default="0", type=str, help='cuda')
    parser.add_argument('--maxEpoch', default=200, type=int, help='max epoch num')
    parser.add_argument('--resume', default=False, type=bool, help='resume model')
    train(parser.parse_args())
