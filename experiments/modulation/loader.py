import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def get_tf_batch(elements, include_snr=False):
    data = elements["rf_signal"]
    labels = elements["label"]
    if include_snr:
        snrs = elements["snr"]
        return data, labels, snrs

    return data, labels


def prepare_dataset(
    ds,
    batch_size=32,
    include_snr=False,
    shuffle=True,
    deterministic=False,
):
    if shuffle:
        ds = ds.shuffle(batch_size * 4, reshuffle_each_iteration=True)

    ds = (
        ds.batch(batch_size)
        .map(
            lambda x: get_tf_batch(x, include_snr=include_snr),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


def load(
    dataset_path,
    split,
    batch_size,
    include_snr,
    deterministic=False,
    take=None,
    shuffle=True,
):
    # No validation set with the current configuration
    if split not in ["train", "test"]:
        raise ValueError(f"expected split to be train, test, got {split}.")

    ds = tfds.load("radio_ml_2018.01", data_dir=dataset_path, split=split)

    ds = prepare_dataset(
        ds,
        batch_size=batch_size,
        include_snr=include_snr,
        shuffle=shuffle,
        deterministic=deterministic,
    )

    if take:
        ds = ds.take(take)

    return ds
