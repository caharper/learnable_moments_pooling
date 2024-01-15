import tensorflow as tf
from learnable_moments_pooling.learnable_stats.constraints import ClipTo
from learnable_moments_pooling.learnable_stats.base import (
    LearnableMomentBase1D,
    compute_adaptive_span_mask,
)
from learnable_moments_pooling.fixed_stats import (
    raw_moment,
    central_moment,
    standardized_moment,
)


def get_int_value(weight, min_val, max_val):
    mask_list = compute_adaptive_span_mask(
        weight, 1.0, tf.range(max_val + 1, dtype=tf.float32)
    )
    mask_list = tf.stop_gradient(tf.where(tf.cast(mask_list, tf.float32) > 0.0)[:, 0])
    int_val = tf.cast(tf.reduce_max(mask_list), tf.float32) + tf.cast(
        min_val, dtype=tf.float32
    )
    return int_val


class LearnableRawMoment1D(LearnableMomentBase1D):
    def __init__(
        self,
        initializer=tf.keras.initializers.RandomUniform(1.0, 1.0),
        min_pow=0,
        max_pow=6,
        include_min=False,
        **kwargs,
    ):
        super().__init__(
            initializer=initializer,
            min_pow=min_pow,
            max_pow=max_pow,
            include_min=include_min,
            **kwargs,
        )

    def call(self, inputs, input_lengths=None):
        # ln(·) is only defined for positive numbers
        inputs = tf.nn.relu(inputs) + 1e-6

        if input_lengths is None:
            return tf.reduce_mean(inputs**self.pow, axis=self.axis)

        if self.axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        self.add_metric(tf.reduce_mean(self.pow), name=f"{self.name}_avg_pow")

        output = raw_moment(
            inputs,
            self.pow,
            axis=self.axis,
            keepdims=False,
            input_lengths=input_lengths,
        )

        return output


class LearnableCentralMoment1D(LearnableMomentBase1D):
    def __init__(
        self,
        initializer=tf.keras.initializers.RandomUniform(2.0, 2.0),
        min_pow=2.0,
        max_pow=5.0,
        **kwargs,
    ):
        super().__init__(
            initializer=initializer, min_pow=min_pow, max_pow=max_pow, **kwargs
        )

    def call(self, inputs, input_lengths=None):
        # Get the integer power
        int_power, power_sum = self.get_int_power_sums()

        # Connect gradient
        inputs = inputs + power_sum

        self.add_metric(tf.reduce_mean(self.pow), name=f"{self.name}_avg_pow")

        # Compute the central moment
        output = central_moment(
            inputs,
            int_power,
            axis=self.axis,
            keepdims=False,
            input_lengths=input_lengths,
        )
        return output


class LearnableStandardizedMoment1D(LearnableMomentBase1D):
    def __init__(
        self,
        initializer=tf.keras.initializers.RandomUniform(3.0, 3.0),
        min_pow=3.0,
        max_pow=6.0,
        **kwargs,
    ):
        super().__init__(
            initializer=initializer, min_pow=min_pow, max_pow=max_pow, **kwargs
        )

    def call(self, inputs, input_lengths=None):
        # Get the integer power
        int_power, power_sum = self.get_int_power_sums()

        # Connect gradient
        inputs = inputs + power_sum

        self.add_metric(tf.reduce_mean(self.pow), name=f"{self.name}_avg_pow")

        # Compute the standardized moment
        return standardized_moment(
            inputs,
            int_power,
            axis=self.axis,
            keepdims=False,
            input_lengths=input_lengths,
        )
