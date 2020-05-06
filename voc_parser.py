import os
import numpy as np
import xml.etree.ElementTree as ET
import pickle


def parse_voc(train_image_folder, train_annot_folder, labels=[]):
    train_list = []
    label_list = {}

    for annot in sorted(os.listdir(train_annot_folder)):
        img = {'object': []}

        try:
            tree = ET.parse(train_annot_folder + annot)
        except Exception as e:
            print(e)
            print('Ignore this bad annotation: ' + train_annot_folder + annot)
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = train_image_folder + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] == 'hand':
                            continue
                        if obj['name'] == 'head':
                            continue
                        if obj['name'] == 'foot':
                            continue

                        if obj['name'] in label_list:
                            label_list[obj['name']] += 1
                        else:
                            label_list[obj['name']] = 1

                        # if len(labels) > 0 and obj['name'] not in labels:
                        if len(labels) > 0 and obj['name'] not in label_list:
                            print(img['filename'], ' and ', obj['name'],
                                  '  have no annotations! Please revise the list of labels in the config.json.')

                            # return None, None
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            train_list += [img]

    label_list = sorted(label_list.keys())

    return train_list, label_list
