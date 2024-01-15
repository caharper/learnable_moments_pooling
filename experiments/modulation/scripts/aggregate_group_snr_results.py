import bocas
import numpy as np
import pandas as pd

group_order = ["Amplitude", "Phase", "Amplitude and Phase", "Frequency"]
results = bocas.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    config = result.config

    # Only show most recent version
    if not config.version == "v0.2":
        continue

    for group in group_order:
        group_metrics = result.get(f"{group}_snr_varying_acc")

        cols = []
        col_heads = []

        col_heads += ["Pooling Type"]
        cols += [config.pooling_type]

        col_heads += ["Pooling Class"]
        cols += [config.pooling_cls]

        col_heads += ["Shared Weights"]
        cols += [config.shared_weights]

        group_name = group_metrics.name.split("_")[0]
        col_heads += ["Modulation Group"]
        cols += [group_name]

        col_heads += ["Version"]
        cols += [config.version]

        # Add snr accuracy columns
        for snr, acc in group_metrics.metrics.items():
            col_heads += [snr]
            cols += [acc]

        cols = [cols]
        all_dfs.append(pd.DataFrame(cols, columns=col_heads))

df = pd.concat(all_dfs)
result = df.to_markdown()

with open("results/group_snr_metrics.md", "w") as f:
    f.write("# RadioML 2018.01A Group SNR Results\n")
    f.write(result)
