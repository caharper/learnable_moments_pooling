import bocas
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model_type = "Example"

    # Computer info
    config.computer = "superpod"  # or "local", "lab", "superpod-test", "superpod"

    # Dataset info
    if config.computer in ["superpod", "superpod-test"]:
        config.data_dir = "/clay-project-template/data"
    elif config.computer == "lab":
        raise NotImplementedError("Lab data dir not implemented yet.")
    elif config.computer == "local":
        config.data_dir = "/Volumes/harper1tb"
    else:
        raise ValueError(f"Unknown computer {config.computer}")

    config.size = bocas.Sweep([1, 2, 3])

    # bump whenever you change things in the script
    # (good for record keeping, filtering in final table production, etc)
    config.version = "v0.1"
    config.batch_size = 128
    config.epochs = 100

    config.verbose = 2
    config.run_eagerly = False
    config.jit_compile = False  # NOTE: tf.ragged is not supported by XLA

    return config
