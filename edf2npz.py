import mne
import os
import numpy as np
import pyedflib as pl


def read_edf(root_dir, subject):
    eyes_open_file = '{}{}/{}R01.edf'.format(root_dir, subject, subject)
    f = pl.EdfReader(eyes_open_file)
    n = f.signals_in_file
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in range(n):
        sigbufs[i, :] = f.readSignal(i)
    eyes_open_data = sigbufs[:, :9600]  # 截取前60秒数据

    eyes_closed_file = '{}{}/{}R02.edf'.format(root_dir, subject, subject)
    f = pl.EdfReader(eyes_closed_file)
    n = f.signals_in_file
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in range(n):
        sigbufs[i, :] = f.readSignal(i)
    eyes_closed_data = sigbufs[:, :9600]  # 截取前60秒数据

    return eyes_open_data, eyes_closed_data


def mne_read_edf(root_dir, subject):
    eyes_open_file = '{}{}/{}R01.edf'.format(root_dir, subject, subject)
    eyes_open_raw = mne.io.read_raw_edf(eyes_open_file, preload=True)
    eyes_open_data = eyes_open_raw.get_data()[:, :9600]   # 截取前60秒数据

    eyes_closed_file = '{}{}/{}R02.edf'.format(root_dir, subject, subject)
    eyes_closed_raw = mne.io.read_raw_edf(eyes_closed_file, preload=True)
    eyes_closed_data = eyes_closed_raw.get_data()[:, :9600]

    return eyes_open_data, eyes_closed_data


if __name__ == '__main__':
    root_dir = "/media/hit/1/EEG_Personal_Identification/files/"
    save_dir = '/media/hit/1/EEG_Personal_Identification/npz/'
    # save_dir = '/media/hit/1/EEG_Personal_Identification/mnenpz/'
    for subject in os.listdir(root_dir):
        print(subject)
        if not os.path.isdir(root_dir + subject):
            continue
        eyes_open_data, eyes_closed_data = read_edf(root_dir, subject)
        # eyes_open_data, eyes_closed_data = mne_read_edf(root_dir, subject)
        np.savez(save_dir + subject + '.npz',
                 eyes_open_data=eyes_open_data,
                 eyes_closed_data=eyes_closed_data,
                 subject=subject)

