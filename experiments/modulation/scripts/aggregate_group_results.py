import bocas
import numpy as np
import pandas as pd

group_order = ["Amplitude", "Phase", "Amplitude and Phase", "Frequency"]
results = bocas.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    config = result.config
    history = result.get("history").metrics
    metrics = result.get("group_acc").metrics

    # Only show most recent version
    if not config.version == "v0.2":
        continue

    cols = []
    col_heads = []

    col_heads += ["Pooling Type"]
    cols += [config.pooling_type]

    col_heads += ["Pooling Class"]
    cols += [config.pooling_cls]

    col_heads += ["Shared Weights"]
    cols += [config.shared_weights]

    col_heads += ["Batch Size", "Max Epochs"]
    cols += [config.batch_size, config.epochs]

    col_heads += ["Used Epochs"]
    cols += [len(history["loss"])]

    # Add group accuracies
    for g in group_order:
        col_heads += [g]
        cols += [metrics[g]]

    col_heads += ["Version"]
    cols += [config.version]

    cols = [cols]
    all_dfs.append(pd.DataFrame(cols, columns=col_heads))

df = pd.concat(all_dfs)
final_col_order = [
    "Pooling Type",
    "Pooling Class",
    "Shared Weights",
    *group_order,
    "Batch Size",
    "Max Epochs",
    "Used Epochs",
    "Version",
]
df = df[final_col_order]
result = df.to_markdown(index=False)

with open("results/group_macro_metrics.md", "w") as f:
    f.write("# RadioML 2018.01A Group Macro Results\n")
    f.write(result)
