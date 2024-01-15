import tensorflow as tf

from learnable_moments_pooling.fixed_stats import (
    GlobalCentralMoment1D,
    GlobalStandardizedMoment1D,
)

from learnable_moments_pooling.learnable_stats import (
    LearnableRawMoment1D,
    LearnableCentralMoment1D,
    LearnableStandardizedMoment1D,
)


class DVector(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis

        data_format = "channels_first"
        if axis == 1:
            data_format = "channels_last"
        self.gap = tf.keras.layers.GlobalAveragePooling1D(
            data_format=data_format, name="gap"
        )

    def call(self, inputs):
        gap = self.gap(inputs)
        return gap

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config


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
        self.gvp = GlobalCentralMoment1D(power=2, axis=axis, name="gvp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        gap = self.gap(inputs)
        gvp = self.gvp(inputs)
        return self.concat([gap, gvp])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config


class SkewVector(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__()

        self.axis = axis

        data_format = "channels_first"
        if axis == 1:
            data_format = "channels_last"
        self.gap = tf.keras.layers.GlobalAveragePooling1D(
            data_format=data_format, name="gap"
        )
        self.gvp = GlobalCentralMoment1D(power=2, axis=axis, name="gvp")
        self.gsp = GlobalStandardizedMoment1D(power=3, axis=axis, name="gsp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        gap = self.gap(inputs)
        gvp = self.gvp(inputs)
        gsp = self.gsp(inputs)
        return self.concat([gap, gvp, gsp])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config


class KurtosisVector(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__()

        self.axis = axis

        data_format = "channels_first"
        if axis == 1:
            data_format = "channels_last"
        self.gap = tf.keras.layers.GlobalAveragePooling1D(
            data_format=data_format, name="gap"
        )
        self.gvp = GlobalCentralMoment1D(power=2, axis=axis, name="gvp")
        self.gsp = GlobalStandardizedMoment1D(power=3, axis=axis, name="gsp")
        self.gkp = GlobalStandardizedMoment1D(power=4, axis=axis, name="gkp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        gap = self.gap(inputs)
        gvp = self.gvp(inputs)
        gsp = self.gsp(inputs)
        gkp = self.gkp(inputs)
        return self.concat([gap, gvp, gsp, gkp])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config


class MixedXVector(tf.keras.layers.Layer):
    def __init__(self, shared=True, axis=1, **kwargs):
        super().__init__()

        self.axis = axis
        self.shared = shared

        self.raw = LearnableRawMoment1D(shared=shared, axis=axis, name="grp")
        self.gvp = GlobalCentralMoment1D(power=2, axis=axis, name="gvp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        grp = self.raw(inputs)
        gvp = self.gvp(inputs)
        return self.concat([grp, gvp])

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "shared": self.shared})
        return config


class MixedHOMVector(tf.keras.layers.Layer):
    def __init__(self, shared=True, axis=1, **kwargs):
        super().__init__()

        self.axis = axis
        self.shared = shared

        self.raw = LearnableRawMoment1D(shared=shared, axis=axis, name="grp")
        self.gvp = GlobalCentralMoment1D(power=2, axis=axis, name="gvp")
        self.gsp = GlobalStandardizedMoment1D(power=3, axis=axis, name="gsp")
        self.gkp = GlobalStandardizedMoment1D(power=4, axis=axis, name="gkp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        grp = self.raw(inputs)
        gvp = self.gvp(inputs)
        gsp = self.gsp(inputs)
        gkp = self.gkp(inputs)
        return self.concat([grp, gvp, gsp, gkp])

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "shared": self.shared})
        return config


class LearnedXVector(tf.keras.layers.Layer):
    def __init__(self, shared=True, axis=1, **kwargs):
        super().__init__()

        self.axis = axis
        self.shared = shared

        self.raw = LearnableRawMoment1D(shared=shared, axis=axis, name="grp")
        self.central = LearnableCentralMoment1D(shared=shared, axis=axis, name="gcp")
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        grp = self.raw(inputs)
        gcp = self.central(inputs)
        return self.concat([grp, gcp])

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "shared": self.shared})
        return config


class LearnedHOMVector(tf.keras.layers.Layer):
    def __init__(self, shared=True, axis=1, **kwargs):
        super().__init__()

        self.axis = axis
        self.shared = shared

        self.raw = LearnableRawMoment1D(shared=shared, axis=axis, name="grp")
        self.central = LearnableCentralMoment1D(shared=shared, axis=axis, name="gcp")
        self.standardized = LearnableStandardizedMoment1D(
            shared=shared, axis=axis, name="gsp"
        )
        self.concat = tf.keras.layers.Concatenate(name="stat_concat")

    def call(self, inputs):
        grp = self.raw(inputs)
        gcp = self.central(inputs)
        gsp = self.standardized(inputs)
        return self.concat([grp, gcp, gsp])

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "shared": self.shared})
        return config
