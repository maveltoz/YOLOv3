import argparse
import os
import numpy as np
import json
import tensorflow as tf
import tensorflow.keras
import cv2
from voc import parse_voc
from dataloader import Dataloader
from model import create_model, dummy_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from utils.utils import evaluate
from tensorflow.keras.models import model_from_json


tf.config.experimental_run_functions_eagerly(True)


class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback()


def yolo():
    config_path = 'config.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    images_path = config['train']['train_image_folder']
    annotation_path = config['train']['train_annot_folder']
    labels = config['model']['labels']

    train_list, label_list = parse_voc(images_path, annotation_path, labels)
    label_list = labels

    train_valid_split = int(0.9 * len(train_list))
    np.random.seed(0)
    np.random.shuffle(train_list)
    np.random.seed()
    valid_list = train_list[train_valid_split:]
    train_list = train_list[:train_valid_split]

    max_box_per_image = max([len(images['object']) for images in train_list])

    train_generator = Dataloader(
        train_list=train_list,
        label_list=label_list,
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size']
    )

    valid_generator = Dataloader(
        train_list=valid_list,
        # train_list=train_list,
        label_list=label_list,
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size']
    )

    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    train_model, infer_model = create_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
    )

    optimizer = Adam(lr=config['train']['learning_rate'], clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    train_model.load_weights('voc.h5')
    callbacks = [OnEpochEnd([train_generator.on_epoch_end]),
        ModelCheckpoint('voc.h5', monitor='loss', save_best_only=True, save_weights_only=True)]

    train_model.fit(
        train_generator,
        #validation_data=valid_generator,
        #steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        callbacks=callbacks
    )

    model_json = infer_model.to_json()
    with open('model_architecture.json', 'w') as json_file:
        json_file.write(model_json)

    json_file = open('model_architecture.json', 'r')
    infer_model_json = json_file.read()
    json_file.close()
    test_model = model_from_json(infer_model_json)
    test_model.load_weights('voc.h5')

    average_precisions = evaluate(test_model, valid_generator)
    print('\n')
    for label, average_precision in average_precisions.items():
       print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
