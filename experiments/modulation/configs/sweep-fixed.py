import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Model info
    config.pooling_type = "Fixed"
    config.filters = [32, 48, 64, 72, 84, 96, 108]
    config.pooling_cls = bocas.Sweep(
        ["dvector", "xvector", "skewvector", "kurtosisvector"]
    )
    config.shared_weights = True

    # Training info
    config.batch_size = 128
    config.epochs = 100
    config.track_stats = True

    # Versioning
    config.version = "v0.2"

    return config
