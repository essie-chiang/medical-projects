import os
import numpy as np
import math
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from absl import app
from absl import flags
import keras.backend as K
import TrainValTensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model import UnetModel
from data_process import DataReader


from PIL import Image
import matplotlib.pyplot as plt
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 2,
                     'Batch Size')
flags.DEFINE_integer('num_epochs', 50,
                     'number of epochs to run')
flags.DEFINE_integer('layer', 3,
                     'number of layers to run')



def run_vae(self):
    shape = (240, 240)
    no_train, no_test, lower_train, lower_test, full_train, full_test = get_bbox_set(shape)
    print("@@@ run model {} layers @@@".format(FLAGS.layer))

    model = self.get_vae_model()
    print("vae model: ", model)
    tb = keras.callbacks.TensorBoard(
        log_dir='./logs/{}_{}'.format(FLAGS.batch_size, FLAGS.num_epochs),
        histogram_freq=2,
        batch_size=FLAGS.batch_size,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None)

    if not os.path.exists('vae'):
        os.mkdir('vae')
    filepath = "vae/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    lossckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    psnrckpt = ModelCheckpoint(filepath, monitor='val_psnr', verbose=1, save_best_only=True, mode='max')
    cblist = [tb, lossckpt, psnrckpt]

    model.fit(no_train, full_train,
              epochs=1,
              batch_size=2,
              validation_data=(no_test, full_test),
              callbacks=cblist)

    res = model.predict(no_test)
    return res



def vae(argv):
    unet = UnetModel()
    print("run_vae:", unet)
    res = unet.run_vae()
    print("vae shape", res.shape)

def run_main(argv):
    shape=(240, 240)
    no_train, no_test,lower_train,lower_test,full_train,full_test, label_train, label_test = get_bbox_set(shape)
    unet = UnetModel(shape=shape)
    print("@@@ run model {} layers with input @@@".format(FLAGS.layer))
    model = unet.get_model()
    tb = keras.callbacks.TensorBoard(
            log_dir='./logs/{}_{}'.format(FLAGS.batch_size,FLAGS.num_epochs),
            histogram_freq=2, 
            batch_size=FLAGS.batch_size, 
            write_graph=True, 
            write_grads=False, 
            write_images=True, 
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None, 
            embeddings_data=None)

    if not os.path.exists('main'):
        os.mkdir('main')
    filepath = "main/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    lossckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    psnrckpt = ModelCheckpoint(filepath, monitor='val_psnr', verbose=1, save_best_only=True, mode='max')
    cblist = [tb, lossckpt, psnrckpt]

    model.fit([no_train, lower_train], [full_train, label_train],
              epochs=1,
              batch_size=2,
              validation_data=([no_test, lower_test], [full_test, label_test]),
              callbacks=cblist)

    res_full, res_label = model.predict([no_test, lower_test])
    print(res_full.shape)
    print(res_label.shape)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("save model to disk")

def run_small_main(argv):
    shape=(240, 240)
    no_train, no_test,lower_train,lower_test,full_train,full_test, label_train, label_test = get_bbox_set(shape)
    unet = UnetModel(shape=shape, task='aug')
    print("@@@ run model {} layers with input @@@".format(FLAGS.layer))
    model = unet.get_model()

    if not os.path.exists('smallmain'):
        os.mkdir('smallmain')
    filepath = "smallmain/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    lossckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    psnrckpt = ModelCheckpoint(filepath, monitor='val_psnr', verbose=1, save_best_only=True, mode='max')
    cblist = [lossckpt, psnrckpt]

    model.fit([no_train, lower_train, label_train], [full_train],
              epochs=1,
              batch_size=2,
              validation_data=([no_test, lower_test, label_test], [full_test]),
              callbacks=cblist)

    res_full = model.predict([no_test, lower_test, label_test])
    print(res_full.shape)


def run_seg(argv):
    datareader = DataReader()
    datareader.init()
    no_train, no_test, label_train, label_test = datareader.get_seg_data()
    unet = UnetModel(task='seg')
    print("@@@ run model {} layers with input @@@".format(FLAGS.layer))
    model = unet.get_model()

    if not os.path.exists('seg'):
        os.mkdir('seg')
    filepath = "seg/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    lossckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    psnrckpt = ModelCheckpoint(filepath, monitor='val_psnr', verbose=1, save_best_only=True, mode='max')
    cblist = [lossckpt, psnrckpt]

    model.fit([no_train], [label_train],
              epochs=1,
              batch_size=2,
              validation_data=([no_test], [label_test]),
              callbacks=cblist)

    res_full = model.predict([no_test])
    print(res_full.shape)


def run_attention(argv):
    shape=(240, 240)
    no_train, no_test, lower_train, lower_test, full_train, full_test, label_train, label_test = get_label_set()
    unet = UnetModel(shape=shape, task='attention')
    print("@@@ run model {} layers with input @@@".format(FLAGS.layer))
    model = unet.get_model()



    if not os.path.exists('attention'):
        os.mkdir('attention')
    filepath = "attention/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    lossckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    psnrckpt = ModelCheckpoint(filepath, monitor='val_psnr', verbose=1, save_best_only=True, mode='max')
    cblist = [lossckpt, psnrckpt]

    model.fit([no_train], [label_train],
              epochs=1,
              batch_size=2,
              validation_data=([no_test], [label_test]),
              callbacks=cblist)

    res_full = model.predict([no_test])
    print(res_full.shape)

def run_orig(self):
    shape = (240, 240)
    no_train, no_test, merge_train, merge_test, full_train, full_test = get_dicom()
    print("@@@ run model {} layers @@@".format(FLAGS.layer))

    unet = UnetModel(shape=shape, task='orig')
    model = unet.get_model()
    print("vae model: ", model)
    tb = keras.callbacks.TensorBoard(
        log_dir='./logs/{}_{}'.format(FLAGS.batch_size, FLAGS.num_epochs),
        histogram_freq=2,
        batch_size=FLAGS.batch_size,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None)

    if not os.path.exists('orig'):
        os.mkdir('orig')
    filepath = "orig/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    lossckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    psnrckpt = ModelCheckpoint(filepath, monitor='val_psnr', verbose=1, save_best_only=True, mode='max')
    cblist = [tb, lossckpt, psnrckpt]

    model.fit([no_train, merge_train], [full_train],
              epochs=1,
              batch_size=2,
              validation_data=([no_test, merge_test], [full_test]),
              callbacks=cblist)

#    res = model.predict([no_test, lower_test])
#    return res

if __name__ == '__main__':
    app.run(run_seg)
