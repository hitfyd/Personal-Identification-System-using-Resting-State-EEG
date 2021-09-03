import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()


# 设置全局随机数种子，同时用于记录实验数据
def setup_seed(seed):
    global global_seed
    global_seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Z-score标准化，执行完成后删除基线数据，即事件开始前的数据
def z_score_standardization(data):
    baseline = data.reshape([-1])
    mean = baseline.mean(-1)
    std = baseline.std(-1)
    data -= mean
    data /= std
    return data


def split_samples(data, window=80, sliding=0.5):
    cnt = 0
    data_split = []
    while int((cnt + 1) * window) <= data.shape[1]:
        # print(len(data_i))
        left = int(cnt * window)
        right = int((cnt + 1) * window)
        data_split.append(data[:, left:right])
        cnt += sliding
    return data_split


# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# 计算等错误率（EER）
def compute_EER(intra_class, inter_class, threshold=-1.):
    intra_class.sort()
    inter_class.sort(reverse=True)
    print(intra_class[:100])
    print(inter_class[:100])
    print(min(intra_class), max(intra_class), min(inter_class), max(inter_class))
    frr, far, eer = 0., 0., 0.
    threshold_list, undulate = [], False
    while True:
        threshold_list.append(threshold)
        frr_num = sum(np.array(intra_class) <= threshold)
        far_num = sum(np.array(inter_class) >= threshold)
        frr = 100. * frr_num / len(intra_class)
        far = 100. * far_num / len(inter_class)
        eer = (frr + far) / 2
        print('threshold:{:.6f}\teer:{:.6f}%\tfrr:{}/{} ({:.6f}%)\tfar:{}/{} ({:.6f}%)'.format(
            threshold, eer, frr_num, len(intra_class), frr, far_num, len(inter_class), far))
        if len(threshold_list) >= 2 and abs(threshold_list[-2] - threshold_list[-1]) < 1e-6:
            break
        if abs(frr - far) < 1e-3:
            break
        elif frr > far:
            if undulate:
                old_threshold = threshold_list[-2]
                for i in range(1, len(threshold_list) + 1):
                    if threshold_list[-i] < threshold:
                        old_threshold = threshold_list[-i]
                        break
                threshold = (old_threshold + threshold) / 2.
            else:
                threshold *= 2.
        else:
            undulate = True
            old_threshold = threshold_list[-2]
            for i in range(1, len(threshold_list) + 1):
                if threshold_list[-i] > threshold:
                    old_threshold = threshold_list[-i]
                    break
            threshold = (old_threshold + threshold) / 2.
    return threshold, frr, far, eer


def train(model, device, train_loader, optimizer, epoch, classes=109):
    # 创建一个空矩阵存储混淆矩阵
    conf_matrix = torch.zeros(classes, classes)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx + 1) % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        conf_matrix = confusion_matrix(pred, labels=target, conf_matrix=conf_matrix)
    accuracy = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Training Dataset\tEpoch：{}\tAccuracy: [{}/{} ({:.6f})]\tAverage Loss: {:.6f}'.format(
        epoch, correct, len(train_loader.dataset), 1. * correct / len(train_loader.dataset), train_loss))
    # print(conf_matrix)
    return accuracy, train_loss, conf_matrix


def test(model, device, test_loader, classes=109):
    # 创建一个空矩阵存储混淆矩阵
    conf_matrix = torch.zeros(classes, classes)
    model.eval()
    test_loss = 0
    correct = 0
    intra_class, inter_class = [], []
    with torch.no_grad():
        for data, target in test_loader:
            # test_start_time = time.time()
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
            # test_end_time = time.time()
            # print('model test time: {}'.format(test_end_time - test_start_time))
            conf_matrix = confusion_matrix(pred, labels=target, conf_matrix=conf_matrix)
            output_numpy = output.cpu().numpy()
            target_numpy = target.cpu().numpy()
            for i in range(len(target_numpy)):
                target_index = target_numpy[i]
                intra_class.append(output_numpy[i][target_index])
                inter_class.extend(output_numpy[i][:target_index])
                inter_class.extend(output_numpy[i][target_index + 1:])

    intra_class.sort()
    inter_class.sort(reverse=True)

    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # print(conf_matrix)
    return test_accuracy, test_loss, conf_matrix, intra_class, inter_class


# 重置权重，用于多次训练模型时
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def set_data_loader(data, labels, batch_size=256):
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run(record, model, optimizer, device, train_data, train_labels, test_data, test_labels,
        batch_size=256, epochs=200, classes=109, fold=0):
    train_loader = set_data_loader(train_data, train_labels, batch_size)
    test_loader = set_data_loader(test_data, test_labels, batch_size)

    model.zero_grad()
    model.apply(weight_reset)

    test_accuracy_list, test_loss_list, train_accuracy_list, train_loss_list = [], [], [], []
    best_test_accuracy, best_state_dict, best_test_conf_matrix, best_intra_class, best_inter_class = 0, {}, [], [], []
    for epoch in range(epochs):
        train_accuracy, train_loss, train_conf_matrix = train(model, device, train_loader, optimizer, epoch, classes)
        test_accuracy, test_loss, test_conf_matrix, intra_class, inter_class = test(model, device, test_loader, classes)

        # 更新最佳测试结果，记录测试精度、当前模型参数、混淆矩阵、真集结果、假集结果
        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_state_dict = model.state_dict()
            best_test_conf_matrix = test_conf_matrix
            best_intra_class, best_inter_class = intra_class, inter_class

        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)

    threshold, frr, far, eer = compute_EER(best_intra_class, best_inter_class)
    print(best_test_accuracy)

    # 记录实验数据
    torch.save(best_state_dict, '{}/{}_params.pkl'.format(record.record_dir, fold))

    frr_num = sum(np.array(best_intra_class) <= threshold)
    far_num = sum(np.array(best_inter_class) >= threshold)
    record.append('fold:{}\nbest_accuracy: {:.6f}%\n'
                  'threshold:{:.6f}\teer:{:.6f}%\tfrr:{}/{} ({:.6f}%)\tfar:{}/{} ({:.6f}%)'.format(
        fold, best_test_accuracy,
        threshold, eer, frr_num, len(best_intra_class), frr, far_num, len(best_inter_class), far))
    record.append('train_accuracy_list: {}'.format(train_accuracy_list))
    record.append('train_loss_list: {}'.format(train_loss_list))
    record.append('test_accuracy_list: {}'.format(test_accuracy_list))
    record.append('test_loss_list: {}'.format(test_loss_list))
    np.savetxt('{}/{}_best_intra_class.txt'.format(record.record_dir, fold), best_intra_class)
    np.savetxt('{}/{}_best_inter_class.txt'.format(record.record_dir, fold), best_inter_class)
    np.savetxt('{}/{}_best_test_conf_matrix.txt'.format(record.record_dir, fold), best_test_conf_matrix.cpu().numpy())
    return best_test_accuracy, threshold, frr, far, eer
