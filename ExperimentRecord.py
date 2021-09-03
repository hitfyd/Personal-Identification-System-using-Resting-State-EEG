import csv
import os
import sys
import time


class ExperimentRecord(object):
    def __init__(self, record_root='record_new/', extra=''):
        self.record_root = record_root
        self.extra = extra
        # 获取当前执行的py文件绝对路径
        self.run_py_path = os.path.abspath(sys.argv[0])
        # 提取当前执行的py文件名称
        self.run_py_name = os.path.split(self.run_py_path)[-1].split(".")[0]
        self.time = time.strftime("%Y%m%d%H%M%S")

        if self.extra == '':
            self.record_dir = '{}{}_{}/'.format(self.record_root, self.run_py_name, self.time)
        else:
            self.record_dir = '{}{}_{}_{}/'.format(self.record_root, self.run_py_name, self.extra, self.time)
        # 确保记录目录存在
        os.makedirs(self.record_dir, exist_ok=True)
        self.basic_record = self.record_dir + 'basic.txt'
        print(self.run_py_path, self.record_dir)

    def append(self, row_record):
        assert isinstance(row_record, str) or isinstance(row_record, list)
        # 打开记录文件并追加记录
        with open(self.basic_record, 'a+') as f:
            f.write(row_record)
            f.write('\n')
