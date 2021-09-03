from glob import glob

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold

import cnn
import util
from ExperimentRecord import ExperimentRecord

# 选择的14、32通道的编号（从1开始）
# [1, 7, 26, 28, 30, 32, 36, 38, 41, 42, 47, 55, 61, 63]
# [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 24, 30, 32, 34, 36, 38, 39, 40, 41, 42, 45, 46, 47, 49, 51, 53, 55, 61, 62, 63]
# 从0开始
channels_list = [[0, 6, 25, 27, 29, 31, 35, 37, 40, 41, 46, 54, 60, 62],
                 [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 23, 29, 31, 33, 35, 37, 38, 39, 40, 41, 44, 45, 46, 48, 50,
                  52, 54, 60, 61, 62],
                 range(64)]
data_type_list = ['EO', 'EC', 'EO&EC']

for data_type in data_type_list:
    for channels in channels_list:
        parameters = {'data_path': '/media/hit/1/EEG_Personal_Identification/mnenpz/',  # resting state EEG数据路径
                      'data_type': data_type,  # 'EO', 'EC', 'EO&EC'
                      'BATCH_SIZE': 64, 'EPOCHS': 2000,
                      'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                      'channels': channels,
                      'ica_enable': True, 'conv_layers': 3, 'ica_output': 64, 'conv_filters': 32, 'fc1_output': 512, 'dropout': 0.5,
                      'learn_rate': 3e-4, 'classes': 109,
                      'window_length': 80, 'sliding_ratio': 0.5,  # 'window_length' === 'points'
                      'setup_seed': 1000
                      }

        record = ExperimentRecord(
            extra='{}{}'.format(parameters['data_type'], len(parameters['channels'])))
        record.append('parameters: {}'.format(parameters))
        util.setup_seed(parameters['setup_seed'])

        # 交叉验证
        best_test_accuracy_list, threshold_list, frr_list, far_list, eer_list = [], [], [], [], []
        for fold in range(5):
            test_start = fold * 1920
            test_end = test_start + 1920
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

                    train_data_seg = util.split_samples(eyes_open_data[:, :test_start], parameters['window_length'],
                                                        parameters['sliding_ratio'])
                    train_data_seg.extend(util.split_samples(eyes_open_data[:, test_end:], parameters['window_length'],
                                                        parameters['sliding_ratio']))
                    test_data_seg = util.split_samples(eyes_open_data[:, test_start:test_end], parameters['window_length'],
                                                       parameters['sliding_ratio'])

                    if parameters['data_type'] == 'EO':
                        pass
                    elif parameters['data_type'] == 'EC':
                        train_data_seg = util.split_samples(eyes_closed_data[:, :test_start], parameters['window_length'],
                                                        parameters['sliding_ratio'])
                        train_data_seg.extend(util.split_samples(eyes_closed_data[:, test_end:], parameters['window_length'],
                                                        parameters['sliding_ratio']))
                        test_data_seg = util.split_samples(eyes_closed_data[:, test_start:test_end], parameters['window_length'], 1)
                    elif parameters['data_type'] == 'EO&EC':
                        train_data_seg.extend(util.split_samples(eyes_closed_data[:, :test_start], parameters['window_length'],
                                               parameters['sliding_ratio']))
                        train_data_seg.extend(
                            util.split_samples(eyes_closed_data[:, test_end:], parameters['window_length'],
                                               parameters['sliding_ratio']))
                        test_data_seg.extend(
                            util.split_samples(eyes_closed_data[:, test_start:test_end], parameters['window_length'], 1))

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
                                                                    parameters['classes'], fold)
            best_test_accuracy_list.append(best_test_accuracy)
            threshold_list.append(threshold)
            frr_list.append(frr)
            far_list.append(far)
            eer_list.append(eer)

        record.append('best_test_accuracy average: {}\tstd:{}'.format(np.mean(best_test_accuracy_list),
                                                                      np.std(best_test_accuracy_list)))
        record.append('threshold_list average: {}\tstd:{}'.format(np.mean(threshold_list), np.std(threshold_list)))
        record.append('frr_list average: {}\tstd:{}'.format(np.mean(frr_list), np.std(frr_list)))
        record.append('far_list average: {}\tstd:{}'.format(np.mean(far_list), np.std(far_list)))
        record.append('eer_list average: {}\tstd:{}'.format(np.mean(eer_list), np.std(eer_list)))
