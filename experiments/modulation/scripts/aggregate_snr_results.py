import bocas
import numpy as np
import pandas as pd

results = bocas.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    config = result.config
    metrics = result.get("metrics").metrics
    snr_metrics = result.get("snr_varying_acc").metrics

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

    col_heads += ["Version"]
    cols += [config.version]

    for snr, acc in snr_metrics.items():
        col_heads += [snr]
        cols += [acc]

    cols = [cols]
    all_dfs.append(pd.DataFrame(cols, columns=col_heads))

df = pd.concat(all_dfs)
result = df.to_markdown(index=False)

with open("results/snr_metrics.md", "w") as f:
    f.write("# RadioML 2018.01A SNR Results\n")
    f.write(result)
