
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

class MaskLayer(Layer):

    def __init__(self, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print("@@@", input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(),
                                      initializer='ones',
                                      trainable=True)
        super(MaskLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.multiply(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


#        # Custom loss layer
#        class CustomMaskLayer(Layer):
#            def __init__(self, **kwargs):
#                self.is_placeholder = True
#                super(CustomMaskLayer, self).__init__(**kwargs)
#
#            def vae_loss(self, pred, true, label):
#                xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
#                kl_loss = - 0.5 * tf.sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
#                return tf.mean(xent_loss + kl_loss)
#
#            def call(self, inputs):
#                y_pred = inputs[0]
#                y_true = inputs[1]
#                label = inputs[2]
#                loss = self.vae_loss(y_pred, y_true, label)
#                self.add_loss(loss, inputs=inputs)
#                # We won't actually use the output.
#                return y_pred
#
#        mask_aug = CustomMaskLayer()([pred_aug, input_mask])