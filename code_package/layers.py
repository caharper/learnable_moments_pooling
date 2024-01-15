import tensorflow as tf


class GlobalVariancePooling1D(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__()
        self.axis = axis

    def call(self, inputs):
        mu = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        return tf.reduce_mean((inputs - mu) ** 2, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"axis": self.axis,}
        )
        return config
