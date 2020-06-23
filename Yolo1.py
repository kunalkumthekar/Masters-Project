import argparse
import os
import numpy as np

os.chdir("../")

print(os.getcwd())
from yolo_preprocessing import read_annotations
from yolo_frontend import SpecialYOLO
import json
import sys


# run with command line -c yolo_config.json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0" for gpu usage

# sys.path.append('C:/Users/thond/AppData/Local/Programs/Python/Python36/Lib/site-packages')

# argparser called for declaring different options passed as arguments(eg: help as -h, config as -c)
argparser = argparse.ArgumentParser(
    description="Train and validate YOLO_v2 model on any dataset"
)

argparser.add_argument("-c", "--conf", help="path to configuration file")


def _main_(args):
    config_path = args.conf
    print(config_path)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    #   config = json.loads(open(config_path).read())

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    # in read_annotaions config(path to train/valid images) is passed as argument for img_dir.
    # the two outputs obtained from the read_annotation function(all_img, seen_labels) is then asigned to train_images/valid_images and train_labels/valid_labels respectively
    train_imgs, train_labels = read_annotations(
        config["train"]["train_image_folder"],
        config["train"]["obj_edge_vis_thresh"],
        config["train"]["obj_side_vis_thresh"],
    )
    valid_imgs, valid_labels = read_annotations(
        config["valid"]["valid_image_folder"],
        config["train"]["obj_edge_vis_thresh"],
        config["train"]["obj_edge_side_thresh"],
    )

    # print( "train_imgs= ", train_imgs ,"\n")

    # parse annotations of the validation set, if any, otherwise split the training set

    np.random.shuffle(train_imgs)
    np.random.shuffle(valid_imgs)

    #  print( "train_imgs= ", train_imgs ,"\n") # debug print shuffled dataset

    # check if all labels are contained in train annotations
    if len(config["model"]["labels"]) > 0:
        overlap_labels = set(config["model"]["labels"]).intersection(
            set(train_labels.keys())
        )

        print("Seen labels:\t", train_labels)
        print("Given labels:\t", config["model"]["labels"])
        print("Overlap labels:\t", overlap_labels)

        if len(overlap_labels) < len(config["model"]["labels"]):
            print(
                "Some labels have no annotations! Please revise the list of labels in the config.json file!"
            )
    else:
        print("No labels are provided. Train on all seen labels.")
        config["model"]["labels"] = train_labels.keys()

    ###############################
    #   Construct the model
    ###############################

    yolo = SpecialYOLO(
        input_width=config["model"]["input_width"],
        input_height=config["model"]["input_height"],
        labels=config["model"]["labels"],
        max_kpp_per_image=config["model"]["max_kpp_per_image"],
    )

    ###############################
    #   Start the training process
    ###############################

    yolo.train(
        train_imgs=train_imgs,
        valid_imgs=valid_imgs,
        train_times=config["train"]["train_times"],
        valid_times=config["valid"]["valid_times"],
        nb_epochs=config["train"]["nb_epochs"],
        learning_rate=config["train"]["learning_rate"],
        batch_size=config["train"]["batch_size"],
        warmup_epochs=config["train"]["warmup_epochs"],
        object_scale=config["train"]["object_scale"],
        no_object_scale=config["train"]["no_object_scale"],
        coord_scale=config["train"]["coord_scale"],
        class_scale=config["train"]["class_scale"],
        direction_scale=config["train"]["direction_scale"],
        saved_weights_name=config["train"]["saved_weights_name"],
        debug=config["train"]["debug"],
    )


if __name__ == "__main__":
    args = argparser.parse_args()
    # testein
    args.conf = "yolo_config.json"

    _main_(args)
