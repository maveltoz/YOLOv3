from tensorflow.keras import utils
import numpy as np
import math
import cv2
from utils.bbox import BoundBox, bbox_iou
from utils.images import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes
from utils.utils import normalize
# import imgaug as ia
# import imgaug.augmenters as iaa
# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Dataloader(utils.Sequence):
    def __init__(self,
                 train_list,
                 label_list,
                 anchors,
                 max_box_per_image=42,
                 batch_size=1,
                 ):
        self.train_list = train_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.max_box_per_image = max_box_per_image
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.net_h = 416
        self.net_w = 416
        self.downsample = 32
        self.min_input_size = 224
        self.max_input_size = 480
        self.min_net_size = (self.min_input_size // self.downsample) * self.downsample
        self.max_net_size = (self.max_input_size // self.downsample) * self.downsample
        self.jitter = 0.3
        self.on_epoch_end()
        np.random.shuffle(self.train_list)

    def __len__(self):
        return math.ceil(len(self.train_list) / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.train_list))
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.train_list):
            r_bound = len(self.train_list)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((self.batch_size, net_h, net_w, 3))
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

        true_box_index = 0

        for instance_count, train_instace in enumerate(self.train_list[l_bound:r_bound]):
            aug_img, aug_objs = self.augmentation(train_instace, net_h, net_w)

            for obj in aug_objs:
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0,
                                       0,
                                       obj['xmax'] - obj['xmin'],
                                       obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h

                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))

                box = [center_x, center_y, w, h]

                obj_indx = self.label_list.index(obj['name'])

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

            x_batch[instance_count] = normalize(aug_img)

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(self.min_net_size / self.downsample,
                                                           self.max_net_size / self.downsample + 1)
            # print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def augmentation(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)
        image = image[:, :, ::-1]

        image_h, image_w, _ = image.shape

        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)

        if new_ar < 1:
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)

        im_sized = random_distort_image(im_sized)

        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w,
                                          image_h)

        return im_sized, all_objs

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

