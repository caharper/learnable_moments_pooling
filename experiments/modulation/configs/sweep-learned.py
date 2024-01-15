import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Model info
    config.pooling_type = "Learned"
    config.filters = [32, 48, 64, 72, 84, 96, 108]
    config.pooling_cls = bocas.Sweep(
        ["learned-int-xvector", "learned-int-hom-vector", "learned-raw"]
    )
    config.shared_weights = bocas.Sweep([True, False])

    # Training info
    config.batch_size = 128
    config.epochs = 100
    config.track_stats = True

    # Versioning
    config.version = "v0.2"

    return config
