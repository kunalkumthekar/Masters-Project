import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from yolo_utils import KeyPointPair, draw_kpp
import math


# [] used if we want to index out the elements, {} used as order doesnt matter


def read_annotations(img_dir, edge_threshold, side_threshold):

    all_imgs = []

    # creating empty box of list of arrays where images would be later stored in
    seen_labels = {}
    # labels would be soon appended in this empty dict list

    for ann_file_name in sorted(os.listdir(img_dir)):

        # creates a list of names in the given image directory (img_dir)
        # sorted() function
        ext = os.path.splitext(ann_file_name)[1]

        # splits the pathname into the its name and its extension (basically ext is index 1 of the total file name eg:filenam.ext))

        if ext == ".txt":
            img = {"object": []}
            # instantisation

            img_file_name = img_dir + ann_file_name

            # technically contains the absolute(abs) path to the file ann_file_name

            # print("img_file_name= ", img_file_name)

            # print( "ann_file_name= ", ann_file_name)

            # read in predefined order
            file = open(img_file_name, "r")

            for line in file:

                # each line is a keypoint pair to this picture
                # file is a list of strings read from the txt file where each line is treated as a string (return or enter at the end of each line is used to determine the end of each string)
                vWords = line.split()

                # splitting the line string into a list of elements, where each element is a sub string of the main string called 'line', using whitespace as the spacer element
                obj = {}

                # an empty dict is created

                img["filename"] = os.path.splitext(img_file_name)[0] + ".bmp"
                # absolute path of the image file for the current annotation file being read, is stored as a string in img['filename']
                # where img is a dictionary

                if (edge_threshold < float(vWords[5])) and (
                    side_threshold < float(vWords[6])
                ):

                    obj["name"] = vWords[0]
                    # in the empty obj dict a class name is created.
                    # this name is appended by index 0 of vWords which is nothing but class

                    # the following if block is for incrementing the number of classes in the empty list of seen_lables={}
                    if obj["name"] in seen_labels:
                        seen_labels[obj["name"]] += 1
                    # now as class (0) is now visible in seen_labels for every iteration 1 is appended
                    else:
                        seen_labels[obj["name"]] = 1
                    # initally if obj['name'] is not in seen_labels then its appended by 1

                    # two keypoints which give pose (x0, y0) and direction (x1, y1) in pixel coordinates, all following coordinates are from object polygon and thus ignored
                    obj["x0"] = float(vWords[1])
                    # x0
                    obj["y0"] = float(vWords[2])
                    # y0

                    obj["alpha"] = float(vWords[3])

                    # obj['x1'] = float( vWords[3] )
                    # x1

                    # obj['y1'] = float( vWords[4] )
                    # y1

                    # zu diesem img das keypoint-pair
                    # The dict called obj is stored in the list called object, which is appended with a new dict instance of obj for every iteration of the loop
                    img["object"] += [obj]

                    # the list called object is in itself stored as a key-value pair within a dictionary called img. the dict img contains two elements {'object', 'filename(abs path to img file)'}
                else:
                    print("Shaft was not considered for training \n")
            file.close()

            if len(img["object"]) > 0:
                # if the object dict is no longer empty then store all the string of data stored in img, in list of all_imgs
                all_imgs += [img]
    return all_imgs, seen_labels


class YoloBatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True, jitter=True, norm=None):

        print("***YoloBatchGenerator Called***")
        self.generator = None

        # images and the returned variable all_imgs from the above method are similar
        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        # self.image_counter = 0

        # for debugging
        print("\n Length of images before augmentation: \t")
        lengthBeforeAug = len(self.images)
        print(lengthBeforeAug)

        # end of debugging

        ia.seed(1)

        # augmentors by https://github.com/aleju/imgaug
        def sometimes(aug):
            return iaa.Sometimes(0.8, aug)

        # hier wird die Augmentation definiert, aber nicht ausgeführt.

        # Define our sequence of augmentation steps that will be applied to every image

        # All augmenters with per_channel=0.5 will sample one value _per image_

        # in 50% of all cases. In all other cases they will sample new values

        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                sometimes(
                    iaa.Affine(
                        # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1),},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-10, 10),
                        # rotate by -45 to +45 degrees
                        # shear=(-5, 5),
                        # shear by -16 to +16 degrees
                        # order=[0, 1],
                        # use nearest neighbour or bilinear interpolation (fast)
                        # cval=(0, 255),
                        # if mode is constant, use a cval between 0 and 255
                        # mode=ia.ALL
                        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                        mode="edge",
                    )
                ),
                iaa.SomeOf(
                    (0, 5),
                    [
                        # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                        # convert images into their superpixel representation
                        iaa.OneOf(
                            [
                                iaa.GaussianBlur((0, 3.0)),
                                # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)),
                                # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)),
                                # blur image using local medians with kernel sizes between 2 and 7
                            ]
                        ),
                        #    iaa.CoarseSalt(0.01, size_percent=(0.002, 0.01)),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        # sharpen images
                        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                        # emboss images
                        # search either for all edges or for directed edges
                        # sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        # ])),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.1 * 255), per_channel=0
                        ),
                        # add gaussian noise to images
                        iaa.OneOf(
                            [
                                iaa.Dropout((0.01, 0.1), per_channel=0),
                                # randomly remove up to 10% of the pixels
                                # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                            ]
                        ),
                        # iaa.Invert(0.05, per_channel=True),
                        # invert color channels
                        iaa.Add((-15, 15), per_channel=0),
                        # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0),
                        # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0),
                        # improve or worsen the contrast
                        # iaa.Grayscale(alpha=(0.0, 1.0)),
                        # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                        # move pixels locally around (with random strengths)
                        # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                        # sometimes move parts of the image around
                    ],
                ),
            ],
            random_order=True,
        )

        if shuffle:
            np.random.shuffle(self.images)

        # for debugging
        print("\n Length of images after augmentation: \t")
        lengthAfterAug = len(self.images)
        print(lengthAfterAug)

    #
    #

    # end of debugging

    def __len__(self):
        """[Divides the length of total number of images available by the 
            the batch size and returns the value as an integer.
            Any float value obtained by division is rounded down.
            The default len() method calls object.__len__
            eg: called upon in generating warmup_batches in yolo_frontend]
            [For further explanation, refer :https://www.programiz.com/python-programming/methods/built-in/len]

        Returns:
            [int] -- [nb_imgs/batch_size]
        """
        return int(np.ceil(float(len(self.images)) / self.config["BATCH_SIZE"]))

    def num_classes(self):
        """return the length of element labels defined in the config.json file
        which is nothing but list of 1 element i.e 0

        Returns:
            [int] -- [length of the list called LABELS]
        """
        return len(self.config["LABELS"])

    def size(self):
        return len(self.images)

    #
    #
    #
    #
    #
    # Marked for cleaning
    #
    #
    #
    #
    #
    #
    #
    #
    #

    # def load_annotation(self, i):

    #     """[summary]

    #     Arguments:

    #         i {[type]} -- [description]

    #     Returns:

    #         [type] -- [description]

    #     """

    #     annots = []

    #     for obj in self.images[i]['object']:

    #         annot = [obj['x0'], obj['y0'], obj['x1'], obj['y1'], self.config['LABELS'].index(obj['name'])]

    #         annots += [annot]

    #     if len(annots) == 0: annots = [[]]

    #     return np.array(annots)

    # def load_image(self, i):

    #     return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        # liefert einen kompletten batch(Get a complete batch)
        """https://www.geeksforgeeks.org/__getitem__-and-__setitem__-in-python/
        https://stackoverflow.com/questions/43627405/understanding-getitem-method/43627975

        Arguments:
            idx {[int]} -- [describes the position of the batch from the batches of images which are
            passed into training per epoch]

        Returns:
            [type] -- [batches of image data and labels]
        """

        # list[lower_boundary:upper_boundary]
        l_bound = idx * self.config["BATCH_SIZE"]
        # lower boundary of the index range  eg: if idx(position of batch=0) then lower boundary=0*16=0

        # upper boundary of index range
        r_bound = (idx + 1) * self.config["BATCH_SIZE"]

        best_anchor = 0

        # TRUE_KPP_BUFFER == max_kpp_per_image (hier z.B. 10 in config), list of keypoints x0, y0, x1, y1 ein Keypointpaar per grid cell, same as in y_batch

        # but in ascending order, box-by-box

        # <batchsize> <BOX == nb_box == len( anchors )//2>, <for each box 4*keypoint pairs +  1*confidency + one-hot labels> == desired network output tensor
        y_batch = np.zeros(
            (
                r_bound - l_bound,
                self.config["GRID_H"],
                self.config["GRID_W"],
                self.config["KPP"],
                3 + 1 + len(self.config["LABELS"]),
            )
        )

        # i_img = 0
        keypoints_on_images = []
        # contains all keypoints in the image, ready for augmentation
        images_batch = []

        # keypoints-list aufbauen
        num_images = len(self.images)
        instance_src_index = l_bound % num_images

        for instance_count in range(r_bound - l_bound):
            train_instance = self.images[instance_src_index]

            # augment input image and fix object's position and size
            image_name = train_instance["filename"]
            img = cv2.imread(image_name)
            img = img[:, :, 1]
            # green channel only
            img = np.expand_dims(img, -1)
            # reattach a dimension

            images_batch.append(img)

            # construct output from object's x, y, w, h
            true_kpp_index = 0

            keypoints_on_one_image = []
            all_objs = train_instance["object"]
            for obj in all_objs:
                keypoints_on_one_image.append(
                    ia.Keypoint(x=float(obj["x0"]), y=float(obj["y0"]))
                )
                keypoints_on_one_image.append(
                    ia.Keypoint(x=float(obj["x1"]), y=float(obj["y1"]))
                )

            keypoints_on_images.append(
                ia.KeypointsOnImage(keypoints_on_one_image, shape=img.shape)
            )

            # instance_src_index = (instance_src_index + 1) % num_images
            instance_src_index += 1

        if self.jitter:
            ia.seed(134)
            aug_pipe_det = self.aug_pipe.to_deterministic()
            # so that the augmentation of the images and the keypoints effect the same transformations
            x_batch = aug_pipe_det.augment_images(images_batch)
            # augmented images
            keypoints_batch_aug = aug_pipe_det.augment_keypoints(keypoints_on_images)
        # augmented keypoints
        else:
            x_batch = images_batch
            keypoints_batch_aug = keypoints_on_images

        x_batch = np.reshape(
            x_batch,
            (r_bound - l_bound, self.config["IMAGE_H"], self.config["IMAGE_W"], 1),
        )
        x_batch = self.norm(x_batch)

        # enter augmented keypoints in y_batch
        num_images = len(self.images)
        instance_src_index = l_bound % num_images

        # each loop is for one image
        for instance_count in range(r_bound - l_bound):
            train_instance = self.images[instance_src_index]
            all_objs = train_instance["object"]
            obj_count = 0
            for obj in all_objs:
                # each loop is for one part/shaft in the image
                if obj["name"] in self.config["LABELS"]:

                    kp0_x = (
                        keypoints_batch_aug[instance_count].keypoints[obj_count * 2].x
                    )
                    # object_KeyPointsOnImage.list_
                    kp0_x = kp0_x / (
                        float(self.config["IMAGE_W"]) / self.config["GRID_W"]
                    )
                    # convert pixel coordinate values to grid coordinate values eg. x = 128 is converted to 16 in grid coordinates
                    kp0_y = (
                        keypoints_batch_aug[instance_count].keypoints[obj_count * 2].y
                    )
                    kp0_y = kp0_y / (
                        float(self.config["IMAGE_H"]) / self.config["GRID_H"]
                    )

                    kp1_x = (
                        keypoints_batch_aug[instance_count]
                        .keypoints[obj_count * 2 + 1]
                        .x
                    )
                    kp1_x = kp1_x / (
                        float(self.config["IMAGE_W"]) / self.config["GRID_W"]
                    )
                    kp1_y = (
                        keypoints_batch_aug[instance_count]
                        .keypoints[obj_count * 2 + 1]
                        .y
                    )
                    kp1_y = kp1_y / (
                        float(self.config["IMAGE_H"]) / self.config["GRID_H"]
                    )

                    dx = kp1_x - kp0_x
                    dy = kp1_y - kp0_y

                    alpha = self.alphaAngle(dy, dx)
                    # match to 0.1...0.9

                    # Determine the grid cell to which the keypoint belongs.
                    grid_x = int(np.floor(kp0_x))
                    # these are the grid coordinates, e.g. in the 4x4 grid into which the image is divided
                    grid_y = int(np.floor(kp0_y))

                    if (
                        grid_x >= 0
                        and grid_y >= 0
                        and grid_x < self.config["GRID_W"]
                        and grid_y < self.config["GRID_H"]
                    ):
                        obj_indx = self.config["LABELS"].index(obj["name"])
                        # label-class-number

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        best_anchor = 0
                        # vorerst nur ein anchor-keypoint je grid_cell
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0] = kp0_x
                        # keypoint0 in grid-Koordinaten LUC
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 1] = kp0_y
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 2] = alpha
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 3] = 1.0
                        # confidence
                        y_batch[
                            instance_count, grid_y, grid_x, best_anchor, 4 + obj_indx
                        ] = 1.0
                        # one-hot class

                        true_kpp_index += 1
                        true_kpp_index = true_kpp_index % self.config["TRUE_KPP_BUFFER"]
                    # avoid overflow

                    # self.image_counter += 1

                    obj_count += 1
            instance_src_index = (instance_src_index + 1) % num_images

        return x_batch, y_batch

    # image normalized and y_batch in grid coordinates

    def alphaAngle(self, dy, dx):
        alpha = math.atan2(dy, dx)
        if alpha < 0.0:
            alpha = alpha + math.pi

        alpha = alpha / math.pi * 0.8 + 0.1
        # match to 0.1...0.9
        return alpha

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)


# shuffle along the first axis only
