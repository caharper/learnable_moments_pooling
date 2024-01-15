import tensorflow as tf


class GlobalRawMoment1D(tf.keras.layers.Layer):
    def __init__(self, power=1, axis=1, maxlen=None, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(power, int):
            raise ValueError(f"Power expects an integer, got {type(power)}, {power}")

        self.power = power
        self.axis = axis
        self.maxlen = maxlen

    def call(self, inputs, input_lengths=None):
        if input_lengths is None:
            return tf.reduce_mean(inputs**self.power, axis=self.axis)

        if self.maxlen is None:
            raise ValueError("maxlen must be specified if input_lengths is not None")

        if self.axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        # Mask the outputs
        mask = tf.sequence_mask(
            tf.squeeze(input_lengths, axis=1), maxlen=self.maxlen, dtype=tf.float32
        )
        inputs = inputs * mask[..., tf.newaxis]
        pow_sum = tf.reduce_sum(inputs**self.power, axis=self.axis)
        return pow_sum / tf.cast(
            input_lengths, tf.float32
        )  # (batch_size, channels) / (batch_size, 1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power": self.power,
                "axis": self.axis,
                "maxlen": self.maxlen,
            }
        )
        return config


class GlobalCentralMoment1D(tf.keras.layers.Layer):
    def __init__(self, power=2, axis=1, maxlen=None, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(power, int):
            raise ValueError(f"Power expects an integer, got {type(power)}")

        self.power = power
        self.axis = axis
        self.maxlen = maxlen

    def compute_masked_mean(self, inputs, input_lengths, keepdims=False):
        if keepdims:
            input_lengths = input_lengths[..., tf.newaxis]
        mu = tf.reduce_sum(inputs, axis=self.axis, keepdims=keepdims) / tf.cast(
            input_lengths, tf.float32
        )

        return mu

    def call(self, inputs, input_lengths=None):
        if input_lengths is None:
            mu = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
            return tf.reduce_mean((inputs - mu) ** self.power, axis=self.axis)

        if self.maxlen is None:
            raise ValueError("maxlen must be specified if input_lengths is not None")

        if self.axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        # Mask the outputs
        mask = tf.sequence_mask(
            tf.squeeze(input_lengths, axis=1), maxlen=self.maxlen, dtype=tf.float32
        )
        inputs = inputs * mask[..., tf.newaxis]

        mu = self.compute_masked_mean(inputs, input_lengths, keepdims=True)
        powered = (inputs - mu) ** self.power
        out = self.compute_masked_mean(
            powered * mask[..., tf.newaxis], input_lengths, keepdims=False
        )
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power": self.power,
                "axis": self.axis,
                "maxlen": self.maxlen,
            }
        )
        return config


class GlobalStandardizedMoment1D(tf.keras.layers.Layer):
    def __init__(self, power=3, axis=1, maxlen=None, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(power, int):
            raise ValueError(f"Power expects an integer, got {type(power)}")

        self.power = power
        self.axis = axis
        self.maxlen = maxlen

    def compute_masked_mean(self, inputs, input_lengths, keepdims=False):
        if keepdims:
            input_lengths = input_lengths[..., tf.newaxis]
        mu = tf.reduce_sum(inputs, axis=self.axis, keepdims=keepdims) / tf.cast(
            input_lengths, tf.float32
        )

        return mu

    def compute_masked_std(self, inputs, input_lengths, keepdims=False):
        mu = self.compute_masked_mean(inputs, input_lengths, keepdims=True)

        mask = tf.sequence_mask(
            tf.squeeze(input_lengths, axis=1), maxlen=self.maxlen, dtype=tf.float32
        )

        powered = ((inputs - mu) ** 2) * mask[..., tf.newaxis]

        var = self.compute_masked_mean(powered, input_lengths, keepdims=keepdims)
        return tf.maximum(tf.sqrt(var), 1e-8)

    def call(self, inputs, input_lengths=None):
        if input_lengths is None:
            mu = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
            # Avoid division by 0
            std = tf.maximum(
                tf.math.reduce_std(inputs, axis=self.axis, keepdims=True), 1e-8
            )
            return tf.reduce_mean(((inputs - mu) / std) ** self.power, axis=self.axis)

        if self.maxlen is None:
            raise ValueError("maxlen must be specified if input_lengths is not None")

        if self.axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        # Mask the outputs
        mask = tf.sequence_mask(
            tf.squeeze(input_lengths, axis=1), maxlen=self.maxlen, dtype=tf.float32
        )
        inputs = inputs * mask[..., tf.newaxis]

        mu = self.compute_masked_mean(inputs, input_lengths, keepdims=True)
        std = self.compute_masked_std(inputs, input_lengths, keepdims=True)
        powered = ((inputs - mu) / std) ** self.power
        out = self.compute_masked_mean(
            powered * mask[..., tf.newaxis], input_lengths, keepdims=False
        )
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "power": self.power,
                "axis": self.axis,
                "maxlen": self.maxlen,
            }
        )
        return config
