import os
import argparse
from yolo import yolo
from test import test


if __name__ == '__main__':
    training = True
    if training: yolo()
    else:
        arg = argparse.ArgumentParser(description='YOLO-v3 on VOC2007')
        arg.add_argument('-d', '--data_dir', default='./voc_dataset/voc2007/JPEGImages/', help='Input data directory')
        arg.add_argument('-a', '--annotation_dir', default='./voc_dataset/voc2007/Annotations/', help='Input annotation directory')
        arg.add_argument('-b', '--batch_size', default=8, help='Input batch size')
        args = arg.parse_args()
        test(args)
