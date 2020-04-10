import argparse
import os
import numpy as np
import json
import cv2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from dataloader import Dataloader
from voc import parse_voc
from model import create_model, dummy_loss
from utils.utils import evaluate
from predict import predict


# def test(args):
def test():
    config_path = 'config.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # data_dir = args.data_dir
    # batch_size = args.batch_size
    images_path = config['train']['train_image_folder']
    annotation_path = config['train']['train_annot_folder']
    labels = config['model']['labels']

    anchors = config['model']['anchors']

    test_list, label_list = parse_voc(images_path, annotation_path, labels)
    label_list = labels
    test_list = test_list[:int(len(test_list) * 0.1)]

    test_generator = Dataloader(
        train_list=test_list,
        label_list=label_list,
        anchors=config['model']['anchors'],
        max_box_per_image=42,
        batch_size=config['train']['batch_size']
    )

    json_file = open('model_architecture.json', 'r')
    infer_model_json = json_file.read()
    json_file.close()
    test_model = model_from_json(infer_model_json)
    test_model.load_weights('best_weights.h5')

    test_d = test_generator[0][0][0]
    # print(test_d.shape)
    # average_precisions = evaluate(test_model, test_data)

    test_data = []
    test_data.append(test_d[0])

    predict(test_model, test_data, anchors)

    average_precisions = evaluate(test_model, test_generator)
    print('\n')
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
