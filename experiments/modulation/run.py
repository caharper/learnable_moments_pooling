import bocas
import os
from datetime import datetime
import tensorflow as tf
import termcolor
import resource
import loader
import models
from utils import get_n_trainable_weights

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


def get_model(config, n_classes):
    return models.create_model(size=config.size, n_classes=n_classes)


def get_name(config):
    now = datetime.now()
    return f"{config.size}_{now.strftime('%m_%d_%y_%H_%M')}_{config.version}"


def _configure_model(config, num_classes):
    model = get_model(config, num_classes)
    scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.initial_learning_rate,
        first_decay_steps=config.decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.05,
        name=None,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler, clipnorm=10.0)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
        run_eagerly=config.run_eagerly,
        jit_compile=config.jit_compile,
    )
    return model


def configure_model(config, num_classes):
    if config.computer == "superpod":
        mirrored_strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )

        with mirrored_strategy.scope():
            model = _configure_model(config, num_classes=num_classes)
    else:
        model = _configure_model(config, num_classes=num_classes)

    config.trainable_parameters = get_n_trainable_weights(model)
    return model


def run(config):
    # Check GPU status
    if config.computer in ["superpod", "lab", "superpod-test"]:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if config.computer == "superpod":
            assert len(gpus) == 8, "Did not find 8 GPUs!"

    name = get_name(config)
    version = config.version
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))
    termcolor.cprint(
        termcolor.colored(f"Training model: {name}", "green", attrs=["bold"])
    )
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))

    train_ds, info = loader.load(
        split="train",
        batch_size=config.batch_size,
        augment=True,
        return_info=True,
        data_dir=config.data_dir,
    )

    # Load datasets
    val_ds = loader.load(
        split="validation",
        batch_size=config.batch_size,
        augment=True,
        data_dir=config.data_dir,
    )
    eval_ds = loader.load(
        split="test",
        batch_size=config.batch_size,
        augment=False,
        data_dir=config.data_dir,
    )

    _, label_key = info.supervised_keys
    n_classes = info.features[label_key].num_classes

    model = configure_model(config, n_classes)

    # Example callbacks
    monitor = "val_loss"
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("./models", version, name, "logs"),
            profile_batch=(10, 11),
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join("./models", version, name, "backup")
        ),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor=monitor,
        #     factor=0.1,
        #     patience=8,
        #     min_lr=1e-7,
        #     mode="min",
        #     min_delta=1e-3,
        # ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=12,
            restore_best_weights=True,
            min_delta=1e-3,
        ),
    ]

    # Verbose=2 for Maneframe
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=config.verbose,
    )

    # Verbose=2 for Maneframe
    metrics = model.evaluate(eval_ds, return_dict=True, verbose=config.verbose)

    return bocas.Result(
        name=name,
        config=config,
        artifacts=[
            bocas.artifacts.KerasHistory(history, name="history"),
            bocas.artifacts.Metrics(metrics, name="metrics"),
        ],
    )
