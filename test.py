import os
import argparse
import json
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from dataloader import Dataloader
from voc_parser import parse_voc
from model import create_model, dummy_loss
from utils.utils import evaluate


def test(args):
    config_path = 'config.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    data_dir = args.data_dir
    annotation_dir = args.annotation_dir
    batch_size = args.batch_size

    images_path = data_dir
    annotation_path = annotation_dir
    labels = config['model']['labels']

    # images_path = config['train']['train_image_folder']
    # annotation_path = config['train']['train_annot_folder']
    # batch_size=config['train']['batch_size']

    test_list, label_list = parse_voc(images_path, annotation_path, labels)
    label_list = labels

    max_box_per_image = max([len(images['object']) for images in test_list])

    test_generator = Dataloader(
        train_list=test_list,
        label_list=label_list,
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        batch_size=batch_size
    )

    json_file = open('logs/models/model_architecture.json', 'r')
    infer_model_json = json_file.read()
    json_file.close()
    test_model = model_from_json(infer_model_json)
    test_model.load_weights('./logs/weights/voc.h5')

    average_precisions = evaluate(test_model, test_generator)
    print('\n')
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
