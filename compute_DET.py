import csv
from glob import glob

import numpy as np

import util

for path in glob('record_new/hyper_parameters_test_window*'):
    intra_class = np.loadtxt(path + '/0_best_intra_class.txt')
    inter_class = np.loadtxt(path + '/0_best_inter_class.txt')

    # util.compute_EER(intra_class.tolist(), inter_class.tolist())
    threshold_list = np.unique(intra_class)
    print(len(threshold_list), threshold_list)
    frr_list, far_list = [], []
    for threshold in threshold_list:
        frr_num = sum(intra_class <= threshold)
        far_num = sum(inter_class >= threshold)
        frr = 100. * frr_num / len(intra_class)
        far = 100. * far_num / len(inter_class)
        print(threshold, frr, far)
        frr_list.append(frr)
        far_list.append(far)
    print(frr_list)
    print(far_list)
    with open(path + 'det.csv', 'a', newline='', encoding='utf-8') as record:
        writer = csv.writer(record)
        writer.writerow(frr_list)
        writer.writerow(far_list)
