import tensorflow as tf
from learnable_moments_pooling.learnable_stats.constraints import ClipTo


def compute_adaptive_span_mask(
    thresholds: tf.float32, ramp_softness: tf.float32, positions: tf.Tensor
):
    n_thresholds = tf.shape(thresholds)[0]
    positions = tf.repeat(positions[tf.newaxis], repeats=n_thresholds, axis=0)
    output = (1.0 / ramp_softness) * (ramp_softness + thresholds - positions)
    output = tf.cast(tf.clip_by_value(output, 0.0, 1.0), dtype=tf.float32)
    return output


class LearnableMomentBase1D(tf.keras.layers.Layer):
    def __init__(
        self,
        shared=True,
        axis=1,
        channels_last=True,
        initializer=None,
        min_pow=None,
        max_pow=None,
        include_min=None,
        include_max=None,
        maxlen=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shared = shared
        self.axis = axis
        self.channels_last = channels_last
        self.initializer = initializer
        self.min_pow = min_pow
        self.max_pow = max_pow
        self.include_min = include_min
        self.include_max = include_max
        self.maxlen = maxlen

    def build(self, input_shape):
        _, length, c = input_shape
        self.maxlen = length
        if self.shared:
            weight_shape = (1,)
        else:
            weight_shape = (c,)

        self.pow = self.add_weight(
            name="power",
            shape=weight_shape,
            initializer=self.initializer,
            trainable=True,
            constraint=ClipTo(
                min_value=self.min_pow,
                max_value=self.max_pow,
                include_low=self.include_min,
                include_high=self.include_max,
            ),
        )

    @tf.function
    def get_int_power_sums(self):
        pow_list = compute_adaptive_span_mask(
            self.pow[..., tf.newaxis] - 0.99,
            1.0,
            tf.range(int(self.min_pow), int(self.max_pow) + 1, dtype=tf.float32),
        )

        # Get the moment positions to keep (all non-negative so this works)
        integer_moments = tf.stop_gradient(
            tf.cast(tf.math.count_nonzero(pow_list, axis=1), tf.float32)
            - 1.0
            + tf.cast(self.min_pow, tf.float32)
        )
        pow_sum = tf.reduce_sum(pow_list, axis=1, keepdims=False)[
            tf.newaxis, tf.newaxis
        ]

        return integer_moments, pow_sum

    def call(self, inputs):
        raise "call must be defined when subclassing " + "LearnableMomentBase1D"

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "shared": self.shared,
                "axis": self.axis,
                "channels_last": self.channels_last,
                "initializer": self.initializer,
                "min_pow": self.min_pow,
                "max_pow": self.max_pow,
                "include_min": self.include_min,
                "include_max": self.include_max,
                "maxlen": self.maxlen,
            }
        )
        return config
