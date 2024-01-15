import tensorflow as tf


def raw_moment(inputs, pow, axis=1, keepdims=False, input_lengths=None):
    if input_lengths is not None:
        if axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        _, maxlen, channels = tf.keras.backend.int_shape(inputs)

        # Mask the outputs
        mask = tf.sequence_mask(
            tf.squeeze(input_lengths, axis=1),
            maxlen=maxlen,
            dtype=tf.bool,
        )[..., tf.newaxis]
        mask = tf.repeat(mask, repeats=channels, axis=-1)

        ragged_mean = tf.reduce_mean(
            tf.ragged.boolean_mask(inputs**pow, mask=mask),
            axis=axis,
            keepdims=keepdims,
        )

        if keepdims:
            return tf.ensure_shape(ragged_mean.to_tensor(), [None, 1, channels])
        return tf.ensure_shape(ragged_mean.to_tensor(), [None, channels])

    return tf.reduce_mean(inputs**pow, axis=axis, keepdims=keepdims)


def central_moment(inputs, pow, axis=1, keepdims=False, input_lengths=None):
    if input_lengths is not None:
        if axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        mu = raw_moment(
            inputs, pow=1, axis=axis, keepdims=True, input_lengths=input_lengths
        )
        inputs = inputs - mu

        return raw_moment(
            inputs, pow=pow, axis=axis, keepdims=keepdims, input_lengths=input_lengths
        )

    inputs = inputs - tf.reduce_mean(inputs, axis=axis, keepdims=True)
    return tf.reduce_mean(inputs**pow, axis=axis, keepdims=keepdims)


def standardized_moment(inputs, pow, axis=1, keepdims=False, input_lengths=None):
    if input_lengths is not None:
        if axis != 1:
            raise ValueError("axis must be 1 if input_lengths is not None")

        mu = raw_moment(
            inputs, pow=1, axis=axis, keepdims=True, input_lengths=input_lengths
        )
        var = central_moment(
            inputs, pow=2, axis=axis, keepdims=True, input_lengths=input_lengths
        )
        # Add perturbation to avoid numerical instability when taking the sqrt
        var = var + 1e-6
        std = tf.maximum(tf.sqrt(var), 1e-8)

        terms = (inputs - mu) / std

        return raw_moment(
            terms, pow=pow, axis=axis, keepdims=keepdims, input_lengths=input_lengths
        )

    mu = tf.reduce_mean(inputs, axis=axis, keepdims=True)
    std = tf.maximum(tf.math.reduce_std(inputs, axis=axis, keepdims=True), 1e-8)
    inputs = (inputs - mu) / std
    return tf.reduce_mean(inputs**pow, axis=axis, keepdims=keepdims)
