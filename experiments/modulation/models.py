import tensorflow as tf


def se_block(x, block_num, residual=False):
    *_, n_filters = tf.keras.backend.int_shape(x)
    se_shape = (1, n_filters)

    se = tf.keras.layers.GlobalAveragePooling1D(name=f"se_block_{block_num}_stat")(x)
    se = tf.keras.layers.Reshape(se_shape, name=f"se_block_{block_num}_reshape")(se)

    se = tf.keras.layers.Dense(
        n_filters // 2,
        activation=None,
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"se_block_{block_num}_dense_1",
    )(se)

    se = tf.keras.layers.ReLU()(se)
    se = tf.keras.layers.Dense(
        n_filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"se_block_{block_num}_dense_2",
    )(se)

    scale = tf.keras.layers.multiply([x, se], name=f"se_block_{block_num}_mult")

    if residual:
        scale = x + scale
    return scale


def create_model(
    filters,
    pooling_layer,
    use_se=True,
    use_act=True,
    residual=False,
    dilate=True,
    shared=None,
    input_shape=(1024, 2),
    n_classes=24,
):
    if dilate:
        dilation_rates = [1, 2, 3, 2, 2, 2, 1]
    else:
        dilation_rates = [1] * 7

    kernel_sizes = [7, 5, 7, 5, 3, 3, 3]

    inputs = tf.keras.layers.Input(shape=input_shape, name="input")
    net = inputs

    for i, (n_filters, dilation_rate, k_size) in enumerate(
        zip(filters, dilation_rates, kernel_sizes)
    ):
        net = tf.keras.layers.Conv1D(
            n_filters,
            k_size,
            dilation_rate=dilation_rate,
            padding="same",
            activation="relu",
            name=f"conv_{i+1}",
        )(net)

        if use_se:
            net = se_block(net, f"{i+1}", residual=residual)
    # Pooling
    if shared is not None:
        net = pooling_layer(name="stat_pool", shared=shared)(net)
    else:
        net = pooling_layer(name="stat_pool")(net)

    net = tf.keras.layers.Dense(128, name="dense_variant_1")(net)
    net = tf.keras.layers.Activation("selu", name="act_variant_1")(net)

    net = tf.keras.layers.Dense(128, name="dense_variant_2")(net)
    net = tf.keras.layers.Activation("selu", name="act_variant_2")(net)

    variant_output = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="variant"
    )(net)

    return tf.keras.models.Model(inputs, variant_output)
