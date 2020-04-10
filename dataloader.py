from tensorflow.keras import utils
import numpy as np
import math
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from utils.bbox import BoundBox, bbox_iou
from utils.utils import normalize


class Dataloader(utils.Sequence):
    def __init__(self,
                 train_list,
                 label_list,
                 anchors,
                 max_box_per_image=42,
                 batch_size=1,
                 ):
        # self.is_epoch_0 = 0
        self.train_list = train_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.max_box_per_image = max_box_per_image
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.net_h = 416
        self.net_w = 416
        self.downsample = 32
        self.on_epoch_end()
        np.random.shuffle(self.train_list)

    def __len__(self):
        return math.ceil(len(self.train_list) / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.train_list))
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        base_grid_h, base_grid_w = self.net_h // self.downsample, self.net_w // self.downsample

        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        x_batch = np.zeros((self.batch_size, self.net_h, self.net_w, 3))

        t_batch = np.zeros((self.batch_size, 1, 1, 1, self.max_box_per_image, 4))

        yolo_1 = np.zeros((self.batch_size, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.label_list)))
        yolo_2 = np.zeros((self.batch_size, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.label_list)))
        yolo_3 = np.zeros((self.batch_size, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3,
                           4 + 1 + len(self.label_list)))

        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((self.batch_size, 1))
        dummy_yolo_2 = np.zeros((self.batch_size, 1))
        dummy_yolo_3 = np.zeros((self.batch_size, 1))

        for idx, data_idx in enumerate(indexes):
            img = cv2.imread(self.train_list[data_idx]['filename'])
            objs = self.train_list[data_idx]['object']

            true_box_index = 0
            aug_img, aug_objs = self.augmentation(img, objs)

            for obj in aug_objs:
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0, 0, obj.x2 - obj.x1, obj.y2 - obj.y1)

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                center_x = .5 * (obj.x1 + obj.x2)
                center_y = .5 * (obj.y1 + obj.y2)

                center_x = center_x / float(self.net_w) * grid_w
                center_y = center_y / float(self.net_h) * grid_h

                w = np.log((obj.x2 - obj.x1) / float(max_anchor.xmax))
                h = np.log((obj.y2 - obj.y1) / float(max_anchor.ymax))
                
                box = [center_x, center_y, w, h]

                obj_idx = self.label_list.index(obj.label)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                yolo[idx, grid_y, grid_x, max_index % 3] = 0
                yolo[idx, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[idx, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[idx, grid_y, grid_x, max_index % 3, 5 + obj_idx] = 1

                true_box = [center_x, center_y, obj.x2 - obj.x1, obj.y2 - obj.y1]
                t_batch[idx, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

            x_batch[idx] = normalize(aug_img)

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def augmentation(self, img, objs):
        images = np.array([img])
        _bbs = []

        for obj in objs:
            _bbs.append(BoundingBox(x1=obj['xmin'], y1=obj['ymin'], x2=obj['xmax'], y2=obj['ymax'], label=obj['name']))

        bbs = BoundingBoxesOnImage(_bbs, shape=img.shape)

        seq = iaa.Sequential([
            iaa.Resize(size={"height": self.net_h, "width": self.net_w}),
        ])

        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_images(images)[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        return image_aug, bbs_aug

    def num_classes(self):
        return len(self.label_list)

    def size(self):
        return len(self.train_list)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.train_list[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.label_list.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.train_list[i]['filename'])
