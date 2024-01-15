import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Model info
    config.filters = [32, 48, 64, 72, 84, 96, 108]
    config.pooling_type = "Mixed"
    config.pooling_cls = bocas.Sweep(["mixed-xvector", "mixed-hom-vector"])
    config.shared_weights = bocas.Sweep([True, False])

    # Training info
    config.batch_size = 128
    config.epochs = 100
    config.track_stats = True

    # Versioning
    config.version = "v0.2"

    return config
