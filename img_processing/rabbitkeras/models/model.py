from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from mylayer import MaskLayer
from absl import app
from absl import flags
import keras.backend as K
import TrainValTensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from loss import psnr, ssim, dice_coef_loss
from tensorflow.keras.optimizers import Adam
from mylayer.MaskLayer import MaskLayer



class UnetModel():
    def __init__(self, config=None):
        if config is None:
            from tool.config_utils import  process_config
            try:
                config = process_config('segmention_config.json')
                print(config)
            except Exception as e:
                print('[Exception] Config Error, %s' % e)
                exit(0)
        self.layers = config.layers
        self.shape = (config.height, config.width)
        self.task = config.task

    def get_model(self):
        if self.layers == 1:
            return self.get_u1_model()
        elif self.layers == 2:
            return self.get_u2_model()

        if self.task == 'brain':
            return self.get_seg_model()
        elif self.task == 'aug' or self.task == 'aug_patch':
            return self.get_aug_model()
        elif self.task == 'aug_label':
            return self.get_aug_label_model()
        elif self.task == 'attention':
            return self.get_attiont_unet()
        elif self.task == 'orig':
            return self.get_u3_model()


    def get_vae_model(self):
        print("get vae model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input = input_no

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.concatenate(inputs=[concat1, x])

        x = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(x)  ##?

        pred = x

        autoencoder = keras.Model(inputs=input_no, outputs=pred)
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer=sgd, loss=psnr, metrics=[psnr])
        autoencoder.summary()

        return autoencoder

    def get_u1_model(self):
        print("get u1 model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input_lower = layers.Input(shape=(w, h, 1))

        input = layers.subtract([input_lower, input_no])

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.concatenate(inputs=[concat1, x])

        x = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(x)  ##?

        pred = layers.add(inputs=[input_no, x])

        autoencoder = keras.Model(inputs=[input_no, input_lower], outputs=pred)
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer='adam', loss=dice_coef_loss, metrics=[psnr, ssim])
        autoencoder.summary()

        return autoencoder

    def get_u2_model(self):
        print("get u2 model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input_lower = layers.Input(shape=(w, h, 1))

        input = layers.subtract([input_lower, input_no])

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        print("concat 1:", concat1)

        print(x)
        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat2 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        print("concat 2:", concat2)

        print(x)
        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)

        print(x)
        x = layers.concatenate(inputs=[concat2, x])
        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        print(x)

        x = layers.concatenate(inputs=[concat1, x])
        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        print(x)

        x = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(x)  ##?

        pred = layers.add(inputs=[input_no, x])

        autoencoder = keras.Model(inputs=[input_no, input_lower], outputs=pred)
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[psnr, ssim])
        autoencoder.summary()

        return autoencoder

    def get_u3_model(self):
        print("get u3 model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input_lower = layers.Input(shape=(w, h, 1))

        input = layers.subtract([input_lower, input_no])

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat2 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat3 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        #
        res = layers.Conv2D(192, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat3, x])

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat2, x])

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat1, x])

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])

        x = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(x)  ##?

        pred = layers.add(inputs=[input_no, x])

        autoencoder = keras.Model(inputs=[input_no, input_lower], outputs=pred)
        #    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder.compile(optimizer='sgd', loss='mae', metrics=[psnr])
        autoencoder.summary()

        return autoencoder

    def get_aug_label_model(self):
        print("get aug model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input_lower = layers.Input(shape=(w, h, 1))
        input_mask = layers.Input(shape=(w, h, 1))

#        input_sub = layers.subtract([input_lower, input_no])
        input = layers.concatenate([input_lower, input_no])

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input_lower, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat2 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat3 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        #
        res = layers.Conv2D(192, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat3, x])

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat2, x])

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat1, x])

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])

        x = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(x)  ##?

        pred_aug = layers.add(inputs=[input_no, x], name='pred_aug')
        pred_mask = layers.multiply(inputs=[x, input_mask], name='pred_mask')

        #        pred_aug = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(pred_aug)  ##?

        #        autoencoder = keras.Model(inputs=[input_no, input_lower, input_mask], outputs=pred_aug)
        autoencoder = keras.Model(inputs=[input_no, input_lower, input_mask], outputs=[pred_aug, pred_mask])

#        autoencoder.compile(optimizer='adam', loss='mae', metrics=[psnr])
        autoencoder.compile(optimizer='adam',
                            loss={'pred_aug': 'mae',
                                  'pred_mask': 'mae'},
                            loss_weights={'pred_aug': 0.5,
                                          'pred_mask': 0.5})
        autoencoder.summary()

        return autoencoder

    def get_aug_model(self):
        print("get aug model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input_lower = layers.Input(shape=(w, h, 1))
#        input_mask = layers.Input(shape=(w, h, 1))

        input_sub = layers.subtract([input_lower, input_no])
        input = layers.concatenate([input_lower, input_no])

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input_sub)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input_lower, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat2 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat3 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        #
        res = layers.Conv2D(192, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat3, x])

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat2, x])

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat1, x])

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])

        x = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(x)  ##?

        pred_aug = layers.add(inputs=[input_no, x])

#        pred_aug = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same')(pred_aug)  ##?

#        autoencoder = keras.Model(inputs=[input_no, input_lower, input_mask], outputs=pred_aug)
        autoencoder = keras.Model(inputs=[input_no, input_lower], outputs=pred_aug)

        autoencoder.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[psnr, ssim])
        #        autoencoder.compile(optimizer='sgd',
        #                            loss={'pred_aug': 'mae',
        #                                  'pred_seg': 'binary_crossentropy'},
        #                            loss_weights={'pred_aug': 0.5,
        #                                          'pred_seg': 0.5})
        autoencoder.summary()

        return autoencoder

    def get_seg_model(self):
        print("get seg model")
        w, h = self.shape
        input = layers.Input(shape=(w, h, 1))

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat2 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat3 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        #
        res = layers.Conv2D(192, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat3, x])

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat2, x])

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[concat1, x])

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])

        pred_seg = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same', name='pred_seg')(x)  ##?

        autoencoder = keras.Model(inputs=[input], outputs=pred_seg)
        autoencoder.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return autoencoder

    def unet_gate_signal(self, input):
        x = layers.Conv2D(K.int_shape(input)[3]*2, kernel_size=(1, 1), strides=(1, 1), padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def attn_gate_block(self, x, g, inter_shape):
        shape_x = 32
        shape_g = 16

        theta_x = layers.Conv2D(16, (2, 2), strides=(2, 2), padding='same')(x)
        shape_theta_x = K.int_shape(theta_x)
        phi_g = layers.Conv2D(16, (1, 1), padding='same')(g)
        upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), strides = (shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), padding='name')(phi_g)

        act_xg = layers.add([upsample_g, theta_x], activation='relu')
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)

        upsample_psi = self.expend_as(upsample_psi, shape_x[3])

        y = layers.multiply([upsample_psi, x])

        result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization()(result)

        return result_bn

    def expend_as(self, tensor, rep):
        my_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat


    def get_attiont_unet(self):
        print("get attention model")
        w, h = self.shape
        input_no = layers.Input(shape=(w, h, 1))
        input_lower = layers.Input(shape=(w, h, 1))
        mask = layers.Input(shape=(w, h, 1))

        input = layers.subtract([input_lower, input_no])

        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[input, x])
        concat1 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat2 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        concat3 = x
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        #
        res = layers.Conv2D(192, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        gate = self.unet_gate_signal(x)
        attn_1 = self.attn_gate_block(concat3, gate, inter_shape=32)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[attn_1, x])

        res = layers.Conv2D(96, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        gate = self.unet_gate_signal(x)
        attn_2 = self.attn_gate_block(concat2, gate, inter_shape=32)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[attn_2, x])

        res = layers.Conv2D(48, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])
        gate = self.unet_gate_signal(x)
        attn_3 = self.attn_gate_block(concat1, gate, inter_shape=32)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.concatenate(inputs=[attn_3, x])

        res = layers.Conv2D(24, kernel_size=(1, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(24, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.add(inputs=[res, x])

        pred_add = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same', name='pred_add')(x)  ##?
        pred_seg = layers.Conv2D(1, kernel_size=(1, 1), activation='relu', padding='same', name='pred_seg')(x)  ##?

        pred_mask_add = layers.multiply([pred_add, mask])
        input_decay = MaskLayer()(input_no)

        pred_aug = layers.add(inputs=[input_decay, pred_mask_add], name='pred_aug')

        autoencoder = keras.Model(inputs=[input_no, input_lower, mask], outputs=pred_aug)

        autoencoder.compile(optimizer='sgd', loss='mae', metrics=[psnr])
        #        autoencoder.compile(optimizer='sgd',
        #                            loss={'pred_aug': 'mae',
        #                                  'pred_seg': 'binary_crossentropy'},
        #                            loss_weights={'pred_aug': 0.5,
        #                                          'pred_seg': 0.5})
        autoencoder.summary()

        return autoencoder

