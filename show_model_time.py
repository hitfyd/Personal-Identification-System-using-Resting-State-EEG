import time
from glob import glob

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import cnn
import util

parameters = {'data_path': '/media/hit/1/EEG_Personal_Identification/mnenpz/',  # resting state EEG数据路径
              'data_type': 'EO&EC',  # 'EO', 'EC', 'EO&EC'
              'BATCH_SIZE': 64, 'EPOCHS': 10,
              'DEVICE': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
              'channels': range(64),
              'ica_enable': True, 'conv_layers': 3, 'ica_output': 64, 'conv_filters': 32, 'fc1_output': 512,
              'dropout': 0.5,
              'learn_rate': 3e-4, 'classes': 109,
              'window_length': 80, 'sliding_ratio': 0.5  # 'window_length' === 'points'
              }

train_data, train_labels, test_data, test_labels = [], [], [], []
# 读取npz
label = 0
npzFiles = glob(parameters['data_path'] + '*.npz')
for file in npzFiles:
    npz = np.load(file)
    if len(npz['eyes_open_data']) > 0:
        eyes_open_data = npz['eyes_open_data'][parameters['channels'], :]
        eyes_closed_data = npz['eyes_closed_data'][parameters['channels'], :]
        eyes_open_data = util.z_score_standardization(eyes_open_data)
        eyes_closed_data = util.z_score_standardization(eyes_closed_data)

        train_data_seg = util.split_samples(eyes_open_data[:, :-1920], parameters['window_length'],
                                            parameters['sliding_ratio'])
        test_data_seg = util.split_samples(eyes_open_data[:, -1920:], parameters['window_length'],
                                           parameters['sliding_ratio'])

        if parameters['data_type'] == 'EO':
            pass
        elif parameters['data_type'] == 'EC':
            train_data_seg = util.split_samples(eyes_closed_data[:, :-1920], parameters['window_length'],
                                                parameters['sliding_ratio'])
            test_data_seg = util.split_samples(eyes_closed_data[:, -1920:], parameters['window_length'], 1)
        elif parameters['data_type'] == 'EO&EC':
            train_data_seg.extend(util.split_samples(eyes_closed_data[:, :-1920], parameters['window_length'],
                                                     parameters['sliding_ratio']))
            test_data_seg.extend(
                util.split_samples(eyes_closed_data[:, -1920:], parameters['window_length'], 1))

        train_labels_seg = [label] * len(train_data_seg)
        test_labels_seg = [label] * len(test_data_seg)

        train_data.extend(train_data_seg)
        train_labels.extend(train_labels_seg)
        test_data.extend(test_data_seg)
        test_labels.extend(test_labels_seg)

        label += 1
# 转换数据类型和维度，适应模型输入
train_data = np.array(train_data, dtype=np.float32)
train_data = np.expand_dims(train_data, 1)
train_data = train_data.transpose([0, 1, 3, 2])  # 将时间维度和信道维度交换，信道维度作为最后一维
train_labels = np.array(train_labels, dtype=np.longlong)
test_data = np.array(test_data, dtype=np.float32)
test_data = np.expand_dims(test_data, 1)
test_data = test_data.transpose([0, 1, 3, 2])  # 将时间维度和信道维度交换，信道维度作为最后一维
test_labels = np.array(test_labels, dtype=np.longlong)
# 构建模型和优化器
load_start_time = time.time()
model = cnn.CNN(channels=len(parameters['channels']), points=parameters['window_length'],
                ica_enable=parameters['ica_enable'], conv_layers=parameters['conv_layers'],
                classes=parameters['classes'], ica_output=parameters['ica_output'],
                conv_filters=parameters['conv_filters'], fc1_output=parameters['fc1_output'],
                dropout=parameters['dropout']).to(parameters['DEVICE'])

for state_dict_path in ['D:/cross_test/14_params.pkl', 'D:/cross_test/32_params.pkl', 'D:/cross_test/64_params.pkl']:
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    load_end_time = time.time()
    print('model load time: {}'.format(load_end_time - load_start_time))
    optimizer = optim.Adam(model.parameters(), lr=parameters['learn_rate'])

    train_loader = util.set_data_loader(train_data, train_labels, parameters['BATCH_SIZE'])
    test_loader = util.set_data_loader(test_data, test_labels, parameters['BATCH_SIZE'])
    util.test(model, parameters['DEVICE'], test_loader, parameters['classes'])
