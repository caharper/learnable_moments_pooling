import tensorflow as tf


class ClipTo(tf.keras.constraints.Constraint):
    """Ensures weights are never < 1"""

    def __init__(
        self,
        min_value=None,
        max_value=None,
        include_low=True,
        include_high=True,
        epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_value = min_value
        self.max_value = max_value
        self.include_low = include_low
        self.include_high = include_high
        self.epsilon = epsilon

    def __call__(self, w):
        clip_min = self.min_value
        clip_max = self.max_value

        # Add clip values if not supplied
        if self.min_value is None:
            clip_min = tf.reduce_min(w)
        if self.max_value is None:
            clip_max = tf.reduce_max(w)

        if not self.include_low:
            clip_min += self.epsilon
        if not self.include_high:
            clip_max -= self.epsilon

        return tf.clip_by_value(w, clip_value_min=clip_min, clip_value_max=clip_max)
