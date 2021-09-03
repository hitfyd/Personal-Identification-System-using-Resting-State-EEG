from glob import glob

import numpy as np
import torch
import torch.optim as optim

import cnn
import util
from ExperimentRecord import ExperimentRecord

window_length_list = [80, 160, 320]
sliding_ratio_list = [0.5, 1]

for window_length in window_length_list:
    for sliding_ratio in sliding_ratio_list:
            parameters = {'data_path': '/media/hit/1/EEG_Personal_Identification/mnenpz/',     # resting state EEG数据路径
                          'data_type': 'EO',    # 'EO', 'EC', 'EO&EC'
                          'BATCH_SIZE': 64, 'EPOCHS': 2000,
                          'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          'channels': range(64),
                          'ica_enable': True, 'conv_layers': 3, 'ica_output': 64, 'conv_filters': 32, 'fc1_output': 512, 'dropout': 0.5,
                          'learn_rate': 3e-4, 'classes': 109,
                          'window_length': window_length, 'sliding_ratio': sliding_ratio,    # 'window_length' === 'points'
                          'setup_seed': 1000
                          }

            record = ExperimentRecord(extra='window_length{}_sliding_ratio{}'.format(window_length, sliding_ratio))
            record.append('parameters: {}'.format(parameters))
            util.setup_seed(parameters['setup_seed'])

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
            model = cnn.CNN(channels=len(parameters['channels']), points=parameters['window_length'],
                            ica_enable=parameters['ica_enable'], conv_layers=parameters['conv_layers'],
                            classes=parameters['classes'], ica_output=parameters['ica_output'],
                            conv_filters=parameters['conv_filters'], fc1_output=parameters['fc1_output'],
                            dropout=parameters['dropout']).to(parameters['DEVICE'])
            optimizer = optim.Adam(model.parameters(), lr=parameters['learn_rate'])

            train_loader = util.set_data_loader(train_data, train_labels, parameters['BATCH_SIZE'])
            test_loader = util.set_data_loader(test_data, test_labels, parameters['BATCH_SIZE'])
            best_test_accuracy, threshold, frr, far, eer = util.run(record, model, optimizer, parameters['DEVICE'],
                                                                    train_data, train_labels, test_data, test_labels,
                                                                    parameters['BATCH_SIZE'], parameters['EPOCHS'],
                                                                    parameters['classes'])
