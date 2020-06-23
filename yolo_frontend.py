from keras.models import Model
from keras.layers import (
    Reshape,
    Activation,
    Conv2D,
    Input,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Lambda,
)
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import numpy as np
import os
import cv2
from yolo_utils import decode_netout
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from yolo_preprocessing import YoloBatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# to make sure our monitoring parameter gets most improved results
# from yolo_backend import TinyYoloFeature
import keras
import sys
import matplotlib.pyplot as plt


class SpecialYOLO(object):
    def __init__(
        self, input_width, input_height, labels, max_kpp_per_image
    ):  # max_kpp_per_image is not required acc to professor

        self.input_width = input_width
        self.input_height = input_height

        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.nb_kpp = 1  # predefined number of keypoint pairs per grid cell
        self.class_wt = np.ones(self.nb_class, dtype="float32")
        self.anchors = [
            0.5,
            0.5,
        ]  # the model classifies and regresses bounding boxes with reference to anchor boxes of multiple scales and aspect ratios
        self.max_kpp_per_image = max_kpp_per_image  # kpp = key point pairs

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image = Input(
            shape=(self.input_height, self.input_width, 1)
        )  # declaring the input image shape by input function, required for building conv layer, 1:=B&W

        num_layer = 0
        # intial layers assigned as 0

        # stack 1

        x = Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv_" + str(num_layer),
            use_bias=False,
        )(input_image)
        # 16 kernels of 3x3 size
        x = BatchNormalization(name="norm_" + str(num_layer))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        num_layer += 1

        # stack 2
        for i in range(0, 2):
            x = Conv2D(
                32 * (2 ** i),
                (3, 3),
                strides=(1, 1),
                padding="same",
                name="conv_" + str(num_layer),
                use_bias=False,
            )(x)
            x = BatchNormalization(name="norm_" + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            num_layer += 1

        # stack 3
        for i in range(0, 10):
            x = Conv2D(
                64,
                (3, 3),
                strides=(1, 1),
                padding="same",
                name="conv_" + str(num_layer),
                use_bias=False,
            )(x)
            x = BatchNormalization(name="norm_" + str(num_layer))(x)
            x = LeakyReLU(alpha=0.1)(x)
            num_layer += 1

        x = Conv2D(
            3 + 1 + self.nb_class,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv_" + str(num_layer),
            use_bias=False,
        )(x)
        # 3:= x,y coord, angle between 2 points of kp
        x = BatchNormalization(name="norm_" + str(num_layer))(x)
        # 1:=1confidence of object in grid cell
        x = LeakyReLU(alpha=0.1)(x)
        num_layer += 1

        # make the object detection layer
        output = Conv2D(
            self.nb_kpp * (3 + 1 + self.nb_class),
            # (x,y,alpha+conf+no.of classes) #nb_classes are one hot encoded
            (1, 1),
            strides=(1, 1,),
            # in this case (1,1) kernel is used which tightens up the possibility of probaibility extraction)
            padding="same",
            name="DetectionLayer",
            kernel_initializer="lecun_normal",
        )(x)

        print("x.shape=", x.shape.as_list())
        self.grid_h = x.shape.as_list()[1]
        self.grid_w = x.shape.as_list()[2]

        print("self.grid_h, self.grid_w=", self.grid_h, self.grid_w)

        output = Reshape(
            (self.grid_h, self.grid_w, self.nb_kpp, 3 + 1 + self.nb_class)
        )(x)
        # this output detects the kpp with respect to the grid

        # For debugging
        print(x)

        print("model_1 input shape=", input_image.shape)
        print("model_2 output shape=", output.shape)

        self.model = Model(inputs=input_image, outputs=output)

        # print a summary of the whole model
        tf.compat.v1.keras.utils.plot_model(
            self.model,
            to_file="model.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
        )
        self.model.summary(positions=[0.25, 0.60, 0.80, 1.0])
        # just printing the model with a diff position parameter than default
        tf.logging.set_verbosity(tf.logging.INFO)
        ## v1 Logging and summary of just the info part and not the errors or warnings

    def custom_loss(self, y_true, y_pred):
        # shape of y_pred (y_true should have same shape): <batch_size> <gridsize_x> <gridsize_y> <nb_anchors> <x0 ,y0, x1 ,y1 ,conf ,classes one-hot>
        # y_true, y_pred are the data for a whole batch
        # y_true are Grid-coordinate (here 0...4) so 3 elements i.e batch_size, gridsize_x, gridsize_y, nb_anchors
        # y_pred are cell-coordinate (0...1 if within the Cell)

        # netout must be in image_width and image_height units, i.e. in the interval [0...1]

        mask_shape = tf.shape(y_true)[
            :4
        ]  # for mask_shape [:4] represents:= (batch_size(:0), nb_grid_x(:1), nb_grid_y(:2), nb_anchors(:3))

        # cell_x and cell_y then contain their respective x and y coordinates for each grid_cell as lookup coordinates
        cell_x = tf.to_float(
            tf.reshape(
                tf.tile(tf.range(self.grid_w), [self.grid_h]),
                (1, self.grid_h, self.grid_w, 1, 1),
            )
        )  # v1 .to.float depracated use tf.cast for v2
        # creating transpose of cell_x to create cell_y
        cell_y = tf.to_float(
            tf.reshape(
                tf.transpose(
                    tf.reshape(
                        tf.tile(tf.range(self.grid_h), [self.grid_w]),
                        (self.grid_w, self.grid_h),
                    )
                ),
                (1, self.grid_h, self.grid_w, 1, 1),
            )
        )
        print(cell_x.shape)
        print(cell_y.shape)
        # tf.tile= constructs tensor by replicating the input(here "tf.range(self.grid_h), [self.grid_w])") multiple times        %%tf.tile(inputs,multiples,name)%%
        # tf.range function("""input""") represesnts creating a seq of numbers from grid width to grid height
        ####basically we are tiling tf.range function by multiples of (1, self.grid_h, self.grid_w, 1, 1) thats grid shape of y_pred or y_true

        # cell_grid contains this coordinate LUT for each batch and for each keypoint
        cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.nb_kpp, 1]
        )
        print(cell_grid.shape)
        # simply creating a grid of concatenated (cellx,celly) and y_pred.shape
        # Intialize masks with 0
        # main motive of adding masks is to add 2 additional arrays to represent id an i/p or o/p is actually present
        coord_mask = tf.zeros(
            mask_shape, dtype="float32"
        )  # used to locate the coordinates of the keypoints
        conf_mask = tf.zeros(
            mask_shape, dtype="float32"
        )  # Used to tell the confidence of the located keypoints
        class_mask = tf.zeros(
            mask_shape, dtype="float32"
        )  # Used to identify the class of the object

        # These two variables used for Visualisation process later on
        seen = tf.Variable(0.0, dtype="float32")
        total_recall = tf.Variable(0.0, dtype="float32")

        """
        Adjust prediction
        """
        ### adjust predicted keypoint pair coordinates
        ##True Position is identified by addition of grid_coordinate+cell coordinate (Truue_position=GCS+CCS)
        pred_kp0_xy = (
            tf.sigmoid(y_pred[..., :2]) + cell_grid
        )  # transform to grid coordinates, predicted keypoint0 grid coordinates are the first two elements in last dimension (x0,y0)

        pred_alpha = y_pred[
            ..., 2:3
        ]  # predicted keypoint1 grid coordinates are the elements 2 to 3 in last dimension thats angle between (x1,y1)

        ### adjust (=limit to [0...1]) predicted confidence           #could use softmax???
        pred_kpp_conf = tf.sigmoid(y_pred[..., 3])  # predicted keypoint pair confidence

        ### adjust predicted class probabilities
        pred_kpp_class = y_pred[
            ..., 4:
        ]  # one or more classes starting with element 5 in last dimension

        """
        Adjust ground truth
        """
        ### keypoint0 x and y
        true_kp0_xy = y_true[..., 0:2]  # unit grid cells, LUC

        true_alpha = y_true[..., 2:3]  # alpha
        ### keypoint-pair confidence
        true_kpp_conf = y_true[..., 3]
        ### The argmax function determines the index tensor of the maximum arguments over the last dimension. But here the last axis is specified
        ### i.e. the shape of the result vector determines the argmax of the lowest dimension. The lowest dimension is removed from the shape.
        ### Here Shape of the result vector: (nb_batches, grid_x, grid_y, nb_kpp ), where the last dimension contains the respective argmax (here always 1)
        true_kpp_class = y_true[..., 4:]
        true_kpp_class_argmax = tf.argmax(
            y_true[..., 4:], -1
        )  # axis=-1 represents the last axis (last element of the array)

        """
        Determine the masks

        In this section the masks are generated by appending their initial value (zero) to their respective values.
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        # this is the confidence for each keypoint pair, multiplied by the coord_scale. A dimension is appended to this.
        coord_mask = tf.expand_dims(y_true[..., 3], axis=-1) * self.coord_scale
        # tf.expand_dims simply changes the dimension shape at the end(axis=-1)-> (x0,y0,x1,y1,confidence,nb_anchors)
        # At the end conf_mask set all elements to no_object_scale or to object_scale. #### both of them represented in the .json file
        # penalize the confidence difference of all keypoints which are farer away from true keypoints
        print(coord_mask.shape)
        conf_mask = conf_mask + 1.0
        # all set to 1 which were previously just an array of zeros(tf.zeros) set in the masking sections
        # conf_mask.shape==(nb_batches, nb_grid_x, nb_grid_y, nb_anchors)

        # penalize the confidence difference of all keypoints which are reponsible for corresponding ground truth keypoint0
        conf_mask = conf_mask + y_true[..., 3] * self.object_scale
        # set the cells containing keypoints to 6 [prev conf.mask was set to 1...now confmask is added to nb_anchor and multiplied by 5(i.e object_sclae), this sets it to 6]

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = (
            y_true[..., 3]
            * tf.gather(self.class_wt, true_kpp_class_argmax)
            * self.class_scale
        )

        """
        Warm-up training  

        All datapoints in the image (even ones not considered as a keypoint) are shown interest during the training part with warmup batches, by setting all elements in the coord_mask to one.
        This is done to tune weights for parameters that may or may not be a part of the hypothesis in prediction.
        Once the training through warmup batches are done, the  elements in the coord_mask are once again set to their original values.
        This is done so that interest for loss computation is only shown at datapoints where we expect the keypoint coordinates to be.
        """
        no_kpp_mask = tf.to_float(coord_mask < self.coord_scale / 2.0)  # ??
        print(no_kpp_mask.shape)
        seen = tf.assign_add(seen, 1.0)
        # tf.cond is a conditional function that passes first lambda function if tf.less() statement is true, else passes the second lambda function if tf.less() is  false
        # "tf.less(seen, self.warmup_batches+1)":== passes the truth table for seen<self.warmup.batches+1 (i.e if arg_x less than arg_y then true and so on...)
        # warmup_batches is defined below
        true_kp0_xy, coord_mask = tf.cond(
            tf.less(seen, self.warmup_batches + 1),
            lambda: [
                true_kp0_xy + (0.5 + cell_grid) * no_kpp_mask,
                tf.ones_like(coord_mask),
            ],
            lambda: [true_kp0_xy, coord_mask],
        )

        """
        Finalize the loss
        """
        # tf.reduce_sum adds the elements of arrays passed. As in our case no axis is passed as an argument, the function will return a single int as output
        # syntax below simply gives the length of the respective masks by adding all of their elements and redicing the, to single integer as output
        nb_coord_kpp = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_kpp = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_kpp = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        # summing up the elements of the array which are squared element wise with the help of tf.square 1e-6) / 2.
        # (true_kp0_xy-pred_kp0_xy) * coord_mask):== basically computes loss in first part and then confirms the presence by multiplying the mask on the diff
        # tf.square makes the diff in the true and predicted values directionless as direction is of no concern for us
        # loss computed is divided by the respective number of attributes to compute a loss/attribute
        loss_kp0_xy = (
            tf.reduce_sum(tf.square(true_kp0_xy - pred_kp0_xy) * coord_mask)
            / (nb_coord_kpp + 1e-6)
            / 2.0
        )
        loss_alpha = (
            tf.reduce_sum(
                tf.square(true_alpha - pred_alpha) * coord_mask * self.direction_scale
            )
            / (nb_coord_kpp + 1e-6)
            / 2.0
        )
        # direction.scale from .json
        loss_conf = (
            tf.reduce_sum(tf.square(true_kpp_conf - pred_kpp_conf) * conf_mask)
            / (nb_conf_kpp + 1e-6)
            / 2.0
        )
        # 1e-6 is just a small number which is used so that if nb_conf_kpp is 0, the fraction should not get invalid output

        # test
        # class_mask_expanded = tf.expand_dims( class_mask, -1)
        # logit basically is the raw prediction that comes out of last layer of the neural network        #logits are mini batch entries
        # input from this logit is given to softmax activation to create the probabilities
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_kpp_class_argmax, logits=pred_kpp_class
        )

        # loss_class  = tf.reduce_sum(tf.square(true_kpp_class-pred_kpp_class)*class_mask_expanded) / (nb_class_kpp + 1e-6)/2. # * tf.expand_dims( class_mask, axis=-1 ))  / (nb_class_kpp  + 1e-6) / 2.

        # testaus
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_kpp + 1e-6)
        loss_class = tf.sigmoid(loss_class)

        # tf.less() is the pred; lambda1:true statement passed; lambda2:false statement passed
        loss = tf.cond(
            tf.less(seen, self.warmup_batches + 1),
            lambda: loss_kp0_xy
            + loss_alpha
            + loss_conf
            + loss_class
            + 10,  # Adding bias as 10     #bias of 10 is added so that the activation function(leaky relu which limits till 10)is always activated
            lambda: loss_kp0_xy + loss_alpha + loss_conf + loss_class,
        )

        if self.debug:  # debug puts out an boolean output true/false
            nb_true_kpp = tf.reduce_sum(y_true[..., 3])
            nb_pred_kpp = tf.reduce_sum(
                tf.to_float(true_kpp_conf > 0.5) * tf.to_float(pred_kpp_conf > 0.3)
            )

            current_recall = nb_pred_kpp / (nb_true_kpp + 1e-6)
            # tf.assign_add(ref,add) which is explained in the next line
            total_recall = tf.assign_add(
                total_recall, current_recall
            )  # it simply updates ref=total_recall and adds value=current_recall to it
            # now we print all the losses
            loss = tf.Print(
                loss, [loss_kp0_xy], message="Loss Keyp0 \t", summarize=1000
            )
            loss = tf.Print(loss, [loss_alpha], message="Loss alpha \t", summarize=1000)
            loss = tf.Print(loss, [loss_conf], message="Loss Conf \t", summarize=1000)
            loss = tf.Print(loss, [loss_class], message="Loss Class \t", summarize=1000)
            loss = tf.Print(loss, [loss], message="Total Loss \t", summarize=1000)
            loss = tf.Print(
                loss, [current_recall], message="Current Recall \t", summarize=1000
            )
            loss = tf.Print(
                loss, [total_recall / seen], message="Average Recall \t", summarize=1000
            )

        return loss

    def load_weights(
        self, weight_path
    ):  # this function is used to call the weights path which later called upon by yolo_predict file
        self.model.load_weights(weight_path)
        self.model.save(weight_path + "full")

        print("input layer name=")
        print([node.op.name for node in self.model.inputs])
        print("output layer name=")
        print([node.op.name for node in self.model.outputs])

        return self.model.output.shape[1:3]

    def normalize(self, image):  # CONVERTING THE IMAGE FORMAT FROM 0-255 to 0-1.
        return image / 255.0  # returning the normalised image

    def train(
        self,
        train_imgs,  # the list of images to train the model
        valid_imgs,  # the list of images used to validate the model
        train_times,  # the number of time to repeat the training set and thus later create number of training_images= training_times*train_gen
        valid_times,  # the number of times to repeat the validation set, and thus later create number of val_images= val_times*val_gen
        nb_epochs,  # number of epoches
        learning_rate,  # the learning rate
        batch_size,  # the size of the batch
        warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
        object_scale,  # .json output
        no_object_scale,  # .json output
        coord_scale,  # .json output
        class_scale,  # .json output
        direction_scale,  # .json output
        saved_weights_name="shaft.h5",
        debug=False,
    ):

        self.batch_size = batch_size
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale
        self.direction_scale = direction_scale
        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        ##Creating a variable having the following arguments which is used for generating the augmented training and validaion datasets
        generator_config = {
            "IMAGE_H": self.input_height,
            "IMAGE_W": self.input_width,
            "GRID_H": self.grid_h,
            "GRID_W": self.grid_w,
            "KPP": self.nb_kpp,
            "LABELS": self.labels,
            "CLASS": len(self.labels),
            "BATCH_SIZE": self.batch_size,
            "TRUE_KPP_BUFFER": self.max_kpp_per_image,
        }
        # passing the arguments in YoloBatchGenerator to obtain the train_gen,valid_gen datasets
        train_generator = YoloBatchGenerator(
            train_imgs, generator_config, norm=self.normalize
        )
        valid_generator = YoloBatchGenerator(
            valid_imgs, generator_config, norm=self.normalize, jitter=False
        )

        # Debug starts
        # train_generator[0]   = train_generator.__getitem__(0)
        print("Length of train generator: \t", len(train_generator))
        print("\nLength of valid generator: \t", len(valid_generator))

        # Debug ends

        self.warmup_batches = warmup_epochs * (
            train_times * len(train_generator) + valid_times * len(valid_generator)
        )  # declaring the warmup_batches as a function of warmup_epochs

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
        )  # parameters beta1 and beta2 cntrol the decay of the learning rate
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################
        # These callbacks are used to make changes to the training process to save time/save model
        early_stop = EarlyStopping(
            monitor="val_loss",  # used to monitor validation_loss and helps in stopping the training val_loss doesnt reduce for more than 2/3 epochs
            min_delta=0.001,
            patience=3,  # 2or3
            mode="min",
            verbose=1,
        )
        checkpoint = ModelCheckpoint(
            saved_weights_name,  # this callback saves the best model having the lowest monitoring factor(i.e val_loss)
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            period=1,
        )
        tensorboard = TensorBoard(
            log_dir=os.path.expanduser("~/logs/"),
            histogram_freq=0,
            # write_batch_performance=True,
            write_graph=True,
            write_images=True,
        )

        ############################################
        # Start the training process
        ############################################

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) * train_times,
            epochs=warmup_epochs + nb_epochs,  # 3+30 by default
            verbose=2 if debug else 1,
            validation_data=valid_generator,
            validation_steps=len(valid_generator) * valid_times,
            callbacks=[early_stop, checkpoint, tensorboard],
            workers=3,  # formerly 3
            max_queue_size=8,
        )  # 8
        # use_multiprocessing = False)

        ############################################
        # Compute mAP on the validation set
        ############################################

        ##### test prediction ###########################

        print("test prediction start\n")
        image = cv2.imread(
            "D:\\Master Project\\github\\kk\\Masters-Project\\Img_000000.bmp"
        )
        # just a sample test image
        image = image[:, :, 0]
        # blue channel only # [dim_1 == img_width (length 128), dim_2 == img_height(length 128), dim_3 == colour channel(length 3)]  [128, 128, 3]
        image = np.expand_dims(image, -1)

        self.predict(image)
        print("test prediction end\n")
        ##### test prediction ende ######################

    def predict(self, image):

        print("image.shape=", image.shape)
        image_h, image_w, image_color_depth = image.shape
        # image = cv2.resize(image, (self.input_width, self.input_height))
        image = self.normalize(image)
        # dividing the entire color depth(0-255) by 255. to get it into form (0-1)

        input_image = image[:, :, ::-1]
        # flip rgb to bgr or vice versa  # refer temp.py # seems to mirror/flip the image rather than flipping the values in 3rd dimension
        # the above expression can be deemed unneccessary since only one colour channel is passed through

        input_image = np.expand_dims(input_image, 0)
        # dummy_array = np.zeros((1,1,1,1,self.max_kpp_per_image,4))

        # netout = self.model.predict([input_image])# add dummy_array
        netout = self.model.predict([input_image])[0]  # add dummy_array # Why?

        print("netout=", [netout])  # print the netout

        netout_decoded = decode_netout(
            netout, self.anchors, image_w, image_h, self.nb_class
        )

        for kpp in netout_decoded:
            print(kpp.x0, kpp.y0, kpp.alpha_norm, kpp.c, kpp.classes)

        return netout_decoded
