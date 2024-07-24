# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 19:40:29 2023

@author: kaueu
"""

#import matplotlib.pylab as plt



import torch
import monai
import numpy as np
import glob
import tensorflow.keras as k
import torch as pt
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as k
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from monai.data import DataLoader
from monai.transforms import (Transform, AddChanneld, Compose, LoadImaged, Transposed, ScaleIntensityd,
                              RandAxisFlipd, RandRotated, RandAxisFlipd, Resized,Resize,
                              RandBiasFieldd, ScaleIntensityRangePercentilesd, RandAdjustContrastd,
                              RandHistogramShiftd, RandGibbsNoised, RandRicianNoised, AsChannelLastd, ToNumpyd,
                              ToTensor)

#
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb
import sys


class DataGenerator(k.utils.Sequence):
    """
        class to be fed into model.fit_generator method of tf.keras model

        uses a pytorch dataloader object to create a new generator object that can be used by tf.keras
        dataloader in pytorch must be used to load image data
        transforms on the input image data can be done with pytorch, model fitting still with tf.keras

        ...

        Attributes
        ----------
        gen : torch.utils.data.dataloader.DataLoader
            pytorch dataloader object; should be able to load image data for pytorch model
        ncl : int
            number of classes of input data; equal to number of outputs of model
    """
    def __init__(self, gen, ncl):
        """
            Parameters
            ----------
            gen : torch.utils.data.dataloader.DataLoader
                pytorch dataloader object; should be able to load image data for pytorch model
            ncl : int
                number of classes of input data; equal to number of outputs of model
        """
        self.gen = gen
        self.iter = iter(gen)
        self.ncl = ncl

    def __getitem__(self, _):
        """
            function used by model.fit_generator to get next input image batch

            Variables
            ---------
            ims : np.ndarray
                image inputs; tensor of (batch_size, height, width, channels); input of model
            lbs : np.ndarray
                labels; tensor of (batch_size, number_of_classes); correct outputs for model
        """
        # catch when no items left in iterator
        try:
            x = next(self.iter)  # generation of data handled by pytorch dataloader
            ims, lbs = x['img'], x['label']
        # catch when no items left in iterator
        except StopIteration:
            self.iter = iter(self.gen)  # reinstanciate iteator of data
            x = next(self.iter)  # generation of data handled by pytorch dataloader
            ims, lbs = x['img'], x['label']
        # swap dimensions of image data to match tf.keras dimension ordering
        ims = ims.numpy()#np.swapaxes(np.swapaxes(ims.numpy(), 1, 3), 1, 2)
        ims = tf.convert_to_tensor(ims)
        #print(ims.shape)
        # convert labels to one hot representation
        lbs = lbs#self.ncl#np.eye(self.ncl)[lbs]
        #print(lbs)
        return ims, tf.convert_to_tensor(lbs)

    def __len__(self):
        """
            function that returns the number of batches in one epoch
        """
        return len(self.gen)
    

#Start running sweep

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    

#
def generate_model(base_model,ishape=(128, 128, 128, 1)):
     input_image = tf.keras.layers.Input(shape=ishape)                                  
     x = base_model(input_image)
     flat = tf.keras.layers.Flatten()(x)
     out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)
     model = tf.keras.models.Model(inputs=input_image, outputs=out)    
     return model
 
def vgg_like_3d(ishape=(128, 128, 128)):
     """
     VGG like 3D model for image classification.
     :param ishape: Input shape
     :return: Tensorflow model
     """
     input_image = tf.keras.layers.Input(shape=(ishape[0], ishape[1], ishape[2], 1))
     conv1 = tf.keras.layers.Conv3D(30, kernel_size=(3, 3, 3), activation="relu")(
         input_image
     )
     conv3 = tf.keras.layers.Conv3D(
         30, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
     )(conv1)
     conv3_drop = tf.keras.layers.Dropout(0.1)(conv3)
     conv4 = tf.keras.layers.Conv3D(60, kernel_size=(3, 3, 3), activation="relu")(
         conv3_drop
     )
     conv5 = tf.keras.layers.Conv3D(
         60, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
     )(conv4)
     conv5_drop = tf.keras.layers.Dropout(0.1)(conv5)
 
     conv6 = tf.keras.layers.Conv3D(120, kernel_size=(3, 3, 3), activation="relu")(
         conv5_drop
     )
     conv7 = tf.keras.layers.Conv3D(
         120, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
     )(conv6)
     conv7_drop = tf.keras.layers.Dropout(0.1)(conv7)
     conv8 = tf.keras.layers.Conv3D(240, kernel_size=(3, 3, 3), activation="relu")(
         conv7_drop
     )
     conv9 = tf.keras.layers.Conv3D(
         240, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"
     )(conv8)
     conv9_drop = tf.keras.layers.Dropout(0.1)(conv9)
 
     flat = tf.keras.layers.Flatten()(conv9_drop)
 
     out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)
 
     model = tf.keras.models.Model(inputs=input_image, outputs=out)
 
     return model
def train(config=None):
    wandb.init(
       project="my-awesome-project",
       tags=['debug'],
       config = wandb.config
       )
   
    fn_keys = ("img", "label")
    img_path = r"YOUR_PATH"
    
    f = open(img_path+"patient_id.txt", "r")
    
    if wandb.config.image_type == 'Whole_Brain':
        files = [img_path+file.replace('\n','')+'_orig.nii.gz' for file in f]#[20:48]
    elif wandb.config.image_type == 'Slant':
            files = [img_path+file.replace('\n','')+'_slant.nii.gz' for file in f]#[20:48]
    elif wandb.config.image_type == 'DKT':
            files = [img_path+file.replace('\n','')+'_dkt.nii.gz' for file in f]#[20:48]
    
    print('\n'.join(files))
    labels = [1 if f.split('/')[-1][0] == "N" else 0 for f in files]
    print(labels)
    # Creates dictionary to store nifti files' paths and their sex labels (1-> male; 0-> female)
    filenames = [{"img": x, "label": y} for (x,y) in zip(files,labels)]
    
    #print(filenames[0])
    
    ####
    #split the training and validation set
    new_filenames = np.array(filenames)
    len_filenames = len(filenames)
    half = len(filenames)//2
    new_filenames[0::2] = filenames[0:half]
    new_filenames[1::2] = filenames[half:]
    print(new_filenames)
    
    
    percentage_split = 0.8
    train_filenames = new_filenames[:int(len_filenames*percentage_split)]
    val_filenames = new_filenames[int(len_filenames*percentage_split):]
     
    #######
    train_transforms = Compose(
        [            
            LoadImaged("img"),
            #Resize('trilinear',(64, 64, 64)),            
            AddChanneld("img"),
            ScaleIntensityd("img",minv = 0.0, maxv = 1.0),
            RandRotated("img",range_x=np.pi / 4, prob=0.5, keep_size=True),
            #NO#RandBiasFieldd("img", prob=0.5,degree=2),
            RandAdjustContrastd("img", prob = 0.5, gamma=(0.5,1.5)),
            #NO#RandHistogramShiftd("img", num_control_points=3, prob = 0.5),            
            Resized(keys=["img"], spatial_size=( 128, 128, 128), mode=('nearest')),    
            Transposed("img",(1,2,3,0)),
            RandGibbsNoised("img",prob=0.4,alpha=(0.1,0.4))
       ]
    )
    num_augmented_samples = 5  # Number of augmented samples for each original sample
    
    #ds = monai.data.Dataset(filenames, train_transforms)
    #loader = DataLoader(ds, batch_size=4, shuffle = True, num_workers=5)
    
    
    ######
    
    augmented_data     = []
    augmented_data_val = []
    
    for data in train_filenames:
        print(data)
        augmented_data.extend([train_transforms(data) for _ in range(num_augmented_samples)])
    
    loader = DataLoader(augmented_data, batch_size=12, shuffle=True, num_workers=0)
    
    for data in val_filenames:
        print(data)
        augmented_data_val.extend([train_transforms(data) for _ in range((num_augmented_samples+3))])
    
    loader_val = DataLoader(augmented_data_val, batch_size=12, shuffle=True, num_workers=0)
    
    
    ############
    
    #x = next(iter(loader))
    #print(x['img'].shape)
    if wandb.config.transferlearning == 'None':
        weight = None
    elif wandb.config.transferlearning == 'ImageNet':
        weight = 'imagenet'
    elif wandb.config.transferlearning == 'RadImageNet':
        weight = 'radimagenet'

    if wandb.config.network_name == 'VGG16':
        model = vgg_like_3d()
    elif wandb.config.network_name == 'VGG19':
        from tensorflow.keras.applications.vgg19 import VGG19
        base_model = VGG19(include_top=False, weights=weight, input_shape=(128,128,128), classes=1,classifier_activation='relu')
        model = generate_model(base_model)
    elif wandb.config.network_name == 'ResNet50':
        from tensorflow.keras.applications.resnet import ResNet50
        base_model = ResNet50(include_top=False, weights=weight, input_shape=(128,128,128), classes=1)
        model = generate_model(base_model)
    elif wandb.config.network_name == 'VGG16u':
        from tensorflow.keras.applications.vgg16 import VGG16
        base_model = VGG16(include_top=False, weights=weight, input_shape=(128,128,128), classes=1,classifier_activation='relu')
        model = generate_model(base_model)
   
    model.summary()
    #early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    
    
    model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
                loss=wandb.config.loss,#"binary_crossentropy",
                metrics=[wandb.config.metric]#["accuracy"],
            )
    
    dataloader     = DataGenerator(loader,    ncl=2) # ncl represents the number of classes for the model
    dataloader_val = DataGenerator(loader_val,ncl=2)
    
    img,label = dataloader.__getitem__(0)
    print(img[0].shape)
    
    
    
    history = model.fit_generator(generator       = dataloader, 
                                  epochs          = 400,
                                  validation_data = dataloader_val,
                                  callbacks=[
                                      WandbMetricsLogger(log_freq=5),
                                      WandbModelCheckpoint('models',save_model=False)]
                                  )
    #                    callbacks=[early_stopping])# fitting the model using the datagenerator (custom parameter choices)
      
    #model.save('path/to/model/name.h5') # save the model (optional but useful)

#create the sweep
#sweep_id = sys.argv[1]
image_type = sys.argv[1]
network_name = sys.argv[2]
loss = sys.argv[3]
metric = sys.argv[4]
transferlearning = sys.argv[5]


sweep_config = {
    #"name": "models_TS",
    "method": "random",
    "parameters": {
        "image_type":{'value':image_type},#values "orig","dkt","slant"
        "network_name":{'value':network_name},# values "vgg16", resnet40", "vgg19"
        "loss":{'value':loss}, #"sparse_categorical_crossentropy", "focal_loss", "focal+dice"
        "metric":{'value':metric},#value "accuracy"
        "transferlearning":{'value':transferlearning}# values "none", "imagenet", "radimagenet"
        }
    }

wandb.login()    
    
sweep_id = wandb.sweep(sweep_config, project="Tourette_models")
wandb.agent(sweep_id, train, count=1)
