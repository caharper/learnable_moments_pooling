import tensorflow as tf
from code_package.layers import GlobalVariancePooling1D


class XVector(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__()

        self.axis = axis

        data_format = "channels_first"
        if axis == 1:
            data_format = "channels_last"
        self.gap = tf.keras.layers.GlobalAveragePooling1D(
            data_format=data_format, name="gap"
        )
        self.gvp = GlobalVariancePooling1D(axis=axis, name="gvp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):

        gap = self.gap(inputs)
        gvp = self.gvp(inputs)
        return self.concat([gap, gvp])

    def get_config(self):
        config = super().get_config()
        config.update(
            {"axis": self.axis,}
        )
        return config
