
import random, numpy as np, cv2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback

from trainers.trainer_base import TrainerBase


class SegmentionTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(SegmentionTrainer, self).__init__(model, data, config)
        self.model = model
        self.data = data
        self.config = config
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.config.hdf5_path + self.config.exp_name + '_best_weights.h5',
                verbose=1,
                monitor='val_loss',
                mode='auto',
                save_best_only=True
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.checkpoint,
                write_images=True,
                write_graph=True,
            )
        )

    def train(self):
        no_train, no_test, label_train, label_test = self.data
        hist = self.model.fit([no_train], [label_train],
                              batch_size=self.config.batch_size,
                              epochs=self.config.epochs,
                              verbose=1,
                              callbacks=self.callbacks,
                              validation_data=([no_test], [label_test]),
                              )
        self.model.save_weights(self.config.hdf5_path + self.config.exp_name + '_last_weights.h5', overwrite=True)


