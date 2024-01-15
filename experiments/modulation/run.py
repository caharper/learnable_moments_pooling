import bocas
import os
from datetime import datetime
import tensorflow as tf
import termcolor
import resource
import loader
import pandas as pd
import numpy as np
import models
from layers import (
    XVector,
    DVector,
    SkewVector,
    KurtosisVector,
    MixedXVector,
    MixedHOMVector,
    LearnedHOMVector,
    LearnedXVector,
)
from metrics import (
    evaluate_varying_snrs_accs,
    evaluate_group_accs,
    evaluate_group_varying_snr_accs,
)
from callbacks import StatLogger
from learnable_moments_pooling.learnable_stats import (
    LearnableRawMoment1D,
    LearnableCentralMoment1D,
    LearnableStandardizedMoment1D,
)
from utils import get_n_trainable_weights

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

DATASET_PATH = "./dataset/radio_ml_2018.01"

pooling_classes = {
    "dvector": DVector,
    "xvector": XVector,
    "skewvector": SkewVector,
    "kurtosisvector": KurtosisVector,
    "learned-raw": LearnableRawMoment1D,
    "learned-int-central": LearnableCentralMoment1D,
    "learned-int-standardized": LearnableStandardizedMoment1D,
    "learned-int-xvector": LearnedXVector,
    "learned-int-hom-vector": LearnedHOMVector,
    "mixed-xvector": MixedXVector,
    "mixed-hom-vector": MixedHOMVector,
}


def get_eval_predictions(model, eval_ds):
    labels = []
    preds = []
    snrs = []

    for d, l, s in eval_ds:
        preds += list(np.argmax(model.predict(d), axis=1))
        labels += list(l.numpy())
        snrs += list(s.numpy())

    df = pd.DataFrame({"Label": labels, "Prediction": preds, "SNR": snrs})
    return df


def get_model(config):
    pooling_layer = pooling_classes.get(config.pooling_cls)

    if not pooling_layer:
        raise ValueError(f"Unexpected model type {config.pooling_cls}")

    if config.pooling_type == "Learned":
        return models.create_model(
            filters=config.filters,
            pooling_layer=pooling_layer,
            shared=config.shared_weights,
        )

    if pooling_layer:
        return models.create_model(filters=config.filters, pooling_layer=pooling_layer)


def get_name(config):
    now = datetime.now()
    return f"{config.pooling_cls}_{now.strftime('%m_%d_%y_%H_%M')}_{config.version}"


def run(config):
    name = get_name(config)
    version = config.version
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))
    termcolor.cprint(
        termcolor.colored(f"Training model: {name}", "green", attrs=["bold"])
    )
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))

    train_ds = loader.load(
        DATASET_PATH,
        split="train",
        batch_size=config.batch_size,
        include_snr=False,
        shuffle=True,
    )

    # For this experiment, we are using the testing set as the validation set
    val_ds = loader.load(
        DATASET_PATH,
        split="test",
        batch_size=config.batch_size,
        include_snr=False,
        shuffle=False,
    )
    eval_ds = loader.load(
        DATASET_PATH,
        split="test",
        batch_size=config.batch_size,
        include_snr=True,
        shuffle=False,
    )

    model = get_model(config)

    base_lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr, global_clipnorm=10.0)

    monitor = "loss"

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=2, name="top_2_acc", dtype=None
            ),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name="top_5_acc", dtype=None
            ),
        ],
    )

    n_trainable_params = get_n_trainable_weights(model)

    # Since we are using the validation set as the testing set, monitor loss
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join("./models", version, name, "ckpt"),
            save_best_only=True,
            monitor=monitor,
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("./models", version, name, "logs")
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=7,
            min_lr=1e-7,
            mode="min",
            min_delta=1e-3,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=12,
            restore_best_weights=True,
            min_delta=1e-3,
        ),
    ]

    if config.track_stats:
        stat_ds = loader.load(
            DATASET_PATH,
            split="test",
            batch_size=1,
            include_snr=False,
            shuffle=False,
            take=1000,
            deterministic=True,
        ).map(lambda x, y: x)

        stat_logger = StatLogger(
            eval_ds=stat_ds, layer_name=f"se_block_{len(config.filters)}_mult"
        )
        callbacks.append(stat_logger)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    metrics = model.evaluate(val_ds, return_dict=True, verbose=2)

    # Evaluation
    eval_pred_df = get_eval_predictions(model, eval_ds)
    snr_varying_metrics = evaluate_varying_snrs_accs(eval_pred_df)
    metrics["Max SNR Accuracy"] = np.max(list(snr_varying_metrics.values()))
    group_accs = evaluate_group_accs(eval_pred_df)
    group_varying_snr_accs = evaluate_group_varying_snr_accs(eval_pred_df)

    artifacts = [
        bocas.artifacts.KerasHistory(history, name="history"),
        bocas.artifacts.Metrics(metrics, name="metrics"),
        bocas.artifacts.Metrics(n_trainable_params, name="n_trainable_params"),
        bocas.artifacts.Metrics(snr_varying_metrics, name="snr_varying_acc"),
        bocas.artifacts.Metrics(group_accs, name="group_acc"),
        *[
            bocas.artifacts.Metrics(res, name=f"{g_name}_snr_varying_acc")
            for g_name, res in group_varying_snr_accs.items()
        ],
    ]

    if config.track_stats:
        artifacts.extend(
            [
                bocas.artifacts.Metrics(stat_logger.means, name="means"),
                bocas.artifacts.Metrics(stat_logger.variances, name="variances"),
                bocas.artifacts.Metrics(stat_logger.skews, name="skewnesses"),
                bocas.artifacts.Metrics(stat_logger.kurtosises, name="kurtosises"),
            ]
        )

    return bocas.Result(
        name=name,
        config=config,
        artifacts=artifacts,
    )
