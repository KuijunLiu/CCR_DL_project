import itertools

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
import tensorflow_hub as hub
    

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")




import os
import shutil
import sys

import numpy as np
import time
import datetime
import matplotlib as mpl
mpl.use('Agg')  # to avoid use of DISPLAY
import matplotlib.pyplot as plt
import cv2
import subprocess

data_root = "data"
model_root = "models"

def train_classifier(data_name = "", tfhub_module = "efficientnet_b0", batch_size = 32, num_epoch = 20, l_r = 0.01, model_name = "", resume=None, limit = 50000, augmentation=False, verbose=1, remove_existing_model=False, kTop = 2):
    print("BEGINNING OF CLASSIFICATION")
    start_time = time.time()
    print ("model name :",model_name)
    print ("data ame :",data_name)
    
    if os.path.exists(os.path.join(model_root, model_name)) and not resume:
        print("WARNING: MODEL NAME EXISTS... ..  . .  .  .")
        if not remove_existing_model:
            PLEASE_RENAME_OR_REMOVE_THE_MODEL
            
        shutil.rmtree(os.path.join(catch_dir_model, model_name))
        os.mkdir(os.path.join(model_root, model_name))
    elif not resume or resume != model_name:
        os.mkdir(os.path.join(model_root, model_name))
    
    catch_logs = os.path.join(os.path.abspath(model_root) , model_name)
    if not os.path.exists(os.path.join(catch_logs, "logs")):
        print("MAKING LOG DIRECTORY: ", os.path.join(catch_logs, "logs"))
        if not os.path.exists(catch_logs):
            os.mkdir(catch_logs)
        os.mkdir(os.path.join(catch_logs, "logs"))
    
    
    
    train_data_dir = os.path.join(data_root, data_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = os.path.join(catch_logs, "logs"))
    
    module_handle, IMAGE_SIZE = select_tfhub_model(tfhub_module)
    
    txt_result = "input image shape: " + str(IMAGE_SIZE) + "\n"
    txt_result += "tfhub_module: " + module_handle + "\n"
    txt_result += "\n"
    with open(os.path.join(os.path.join(model_root , model_name), "Result_Summary.txt"), "a") as text_file:
        text_file.write(txt_result)
        txt_result = ""
        
    seed=666
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=batch_size,
                       interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        train_data_dir, subset="validation", shuffle=True, seed=seed, **dataflow_kwargs)
    
    
    if augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=10,
          horizontal_flip=False,
          width_shift_range=0.2, height_shift_range=0.2,
          shear_range=0.2, zoom_range=0.2,
          **datagen_kwargs)
    else:
        train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, subset="training", shuffle=True, seed=seed, **dataflow_kwargs)
    do_fine_tuning = True
    print("Building model with", module_handle)
    
    model, lrschedule = create_classification_model(module_handle = module_handle, num_classes=train_generator.num_classes, IMAGE_SIZE=IMAGE_SIZE, do_fine_tuning=do_fine_tuning, l_r=l_r, num_epochs=num_epoch+0.001, kTop=kTop)
    
    train_steps = train_generator.samples // train_generator.batch_size
    steps_per_epoch = np.min([1000, train_steps])
    validation_steps = valid_generator.samples // valid_generator.batch_size
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.path.join(model_root, model_name), "model_weights"),save_weights_only=True,save_freq='epoch',verbose=1)
    print("Checkpoint Will Be Saved Here: ", os.path.join(model_root, model_name))
    
    if resume:
        print("Loading Weights...")
        model.load_weights(os.path.join(os.path.join(model_root, resume), "model_weights"))
        
    if num_epoch > 0:  
        print("Start training for "+str(num_epoch)+" epochs")
        hist=model.fit(train_generator,epochs=num_epoch,steps_per_epoch=steps_per_epoch,validation_data=valid_generator,validation_steps=np.min([300,validation_steps]),callbacks=[tensorboard_callback, lrschedule, model_checkpoint_callback])
        print("Saving Model")
        model.save(os.path.join(model_root, model_name))  
    
    return model, train_generator, valid_generator
 





def select_tfhub_model(model_name = "efficientnet_b0"):
    model_handle_map = {
      "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
      "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
      "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
      "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
      "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
      "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
      "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
      "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
      "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
      "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature-vector/4",
      "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
      "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
      "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
      "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
      "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
      "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
      "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature-vector/4",
      "resnet_50": "https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
      "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
      "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
      "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
      "mobilenet_v1_025_192": "https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/5",
      "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
      "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
      "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
      "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
      "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
      "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
      "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
    }

    model_image_size_map = {
      "efficientnet_b0": 224,
      "efficientnet_b1": 240,
      "efficientnet_b2": 260,
      "efficientnet_b3": 300,
      "efficientnet_b4": 380,
      "efficientnet_b5": 456,
      "efficientnet_b6": 528,
      "efficientnet_b7": 600,
      "inception_v3": 299,
      "inception_resnet_v2": 299,
      "nasnet_large": 331,
      "pnasnet_large": 331,
    }

    model_handle = model_handle_map.get(model_name)
    pixels = model_image_size_map.get(model_name, 224)

    print(f"Selected model: {model_name} : {model_handle}")

    IMAGE_SIZE = (pixels, pixels)
    print(f"Input size {IMAGE_SIZE}")

    return model_handle, IMAGE_SIZE

def create_classification_model(module_handle, num_classes, IMAGE_SIZE, do_fine_tuning = True, l_r = 0.005, num_epochs = 100, kTop = 2, inference = False):
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
        hub.KerasLayer(module_handle, trainable=do_fine_tuning, name="module"),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(num_classes,
                              kernel_regularizer=tf.keras.regularizers.l2(0.0001), name="dense")
                    
    ])
    
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary()
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,decay_steps=5000, decay_rate=0.9)
    
    #lrdecay = l_r / num_epochs
    lrdecay = 1 / num_epochs
    def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + lrdecay * epoch)
    lrschedule = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=l_r,beta_1=0.9,beta_2=0.999,epsilon=1e-07,name="Adam")
    optimizer = tf.keras.optimizers.SGD(lr=l_r, momentum=0.9)
    metric_top2 = tf.keras.metrics.TopKCategoricalAccuracy(k=kTop)
    metric_loss = tf.keras.metrics.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
#     metric_auc = tf.keras.metrics.AUC(curve="PR")
    if inference:
        new_model = tf.keras.models.Model(inputs=model.inputs,outputs=[model.get_layer(name="dense").output, model.get_layer(name="module").output])
    #     model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),metrics=['accuracy', metric_top2, metric_loss])
        new_model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),metrics=['accuracy', metric_top2, metric_loss])
        return new_model, lrschedule
    else:
        model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),metrics=['accuracy', metric_top2, metric_loss])
        return model, lrschedule
    

def one_hot(y_):
    y_ = np.asarray(y_).reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)] 

