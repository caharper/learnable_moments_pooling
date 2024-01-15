import bocas
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

results = bocas.Result.load_collection("artifacts/")
names = []

all_dfs = []
for result in results:
    config = result.config
    history = result.get("history").metrics

    # Only show most recent version
    if not config.version == "v0.2":
        continue

    means = result.get("means").metrics
    variances = result.get("variances").metrics
    skewnesses = result.get("skewnesses").metrics
    kurtosises = result.get("kurtosises").metrics

    to_numpy = lambda x: [a.numpy() for a in x]
    means = to_numpy(means)
    variances = to_numpy(variances)
    skewnesses = to_numpy(skewnesses)
    kurtosises = to_numpy(kurtosises)

    val_acc = np.array(history["val_acc"])

    acc = np.array(history["acc"])

    n_conv_layers = 0
    try:
        n_conv_layers = len(config.filters)
    except:
        n_conv_layers = len([32, 48, 64, 72, 84, 96, 108])

    if config.shared_weights:
        shared = "shared"
    else:
        shared = "not_shared"

    name = f"{config.pooling_type}-{config.pooling_cls}-{n_conv_layers}-{shared}"

    if name in names:
        count = 1
        tmp_name = name
        while tmp_name in names:
            tmp_name = name + f"---{count}"

        name = tmp_name

    names.append(name)

    df = pd.DataFrame(
        {
            "Type": [config.pooling_type] * len(acc),
            "Layers": [n_conv_layers] * len(acc),
            "Model": [name] * len(acc),
            "Pooling Class": [config.pooling_cls] * len(acc),
            "Validation Accuracy": val_acc,
            "Train Accuracy": acc,
            "Epoch": list(range(1, len(acc) + 1)),
            "Shared Weights": [config.shared_weights] * len(acc),
            "Version": [config.version] * len(acc),
            "Mean": means[1:],
            "Variance": variances[1:],
            "Skewness": skewnesses[1:],
            "Kurtosis": kurtosises[1:],
        }
    )

    all_dfs.append(df)

df = pd.concat(all_dfs)

line_colors = [
    "#696969",
    "#d3d3d3",
    "#556b2f",
    "#8b4513",
    "#228b22",
    "#483d8b",
    "#008b8b",
    "#000080",
    "#9acd32",
    "#7f007f",
    "#8fbc8f",
    "#b03060",
    "#ff0000",
    "#ff8c00",
    "#ffd700",
    "#00ff00",
    "#8a2be2",
    "#00ff7f",
    "#dc143c",
    "#00ffff",
    "#f4a460",
    "#0000ff",
    "#f08080",
    "#ff00ff",
    "#f0e68c",
    "#6495ed",
    "#dda0dd",
    "#90ee90",
    "#ff1493",
    "#7b68ee",
]

fig = px.line(
    df,
    x="Epoch",
    y="Validation Accuracy",
    color="Model",
    hover_data=["Type", "Pooling Class", "Shared Weights", "Version"],
    color_discrete_sequence=line_colors,
)
fig.write_image("plots/validation_acc_epoch.png", scale=3)


def make_combined_stat_figure(df, models, line_colors, stat):
    traces = []

    for model, line_color in zip(models, line_colors):
        sub_df = df[df["Model"] == model]

        val_acc_trace = go.Scatter(
            x=sub_df["Epoch"],
            y=sub_df["Validation Accuracy"],
            mode="lines",
            name=model,
            line=dict(color=line_color),
        )

        customdata = np.dstack(
            [
                np.array(sub_df["Type"]),
                sub_df["Pooling Class"],
                sub_df["Shared Weights"],
                sub_df["Version"],
                sub_df[stat],
            ]
        )
        customdata = np.array(customdata)
        customdata = customdata[0]

        val_acc_trace = go.Scatter(
            x=sub_df["Epoch"],
            y=sub_df["Validation Accuracy"],
            mode="lines",
            name=model,
            legendgroup=model,
            line=dict(color=line_color),
            hovertemplate="<b>Epoch:</b> %{x}<br>"
            + "<b>Validation Accuracy:</b> %{y:.2f}<br>"
            + "<b>Type:</b> %{customdata[0]}<br>"
            + "<b>Pooling Class:</b> %{customdata[1]}<br>"
            + "<b>Shared Weights:</b> %{customdata[2]}<br>"
            + "<b>Version:</b> %{customdata[3]}<br>"
            + "<b>Stat:</b> %{customdata[4]:.4f}<br>",
            customdata=customdata,
        )

        stat_trace = go.Scatter(
            x=sub_df["Epoch"],
            y=sub_df[stat],
            mode="lines",
            name=model,
            legendgroup=model,
            showlegend=False,
            line=dict(color=line_color, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>Epoch:</b> %{x}<br>"
            + "<b>Validation Accuracy:</b> %{y:.2f}<br>"
            + "<b>Type:</b> %{customdata[0]}<br>"
            + "<b>Pooling Class:</b> %{customdata[1]}<br>"
            + "<b>Shared Weights:</b> %{customdata[2]}<br>"
            + "<b>Version:</b> %{customdata[3]}<br>"
            + "<b>Stat:</b> %{customdata[4]:.4f}<br>",
            customdata=customdata,
        )

        traces.append(val_acc_trace)
        traces.append(stat_trace)

    layout = go.Layout(
        legend=dict(orientation="h", y=-0.2),  # adjust y position of legend
        margin=dict(l=50, r=50, t=50, b=50),  # adjust margins
        yaxis=dict(title="Validation Accuracy", titlefont=dict(size=20)),
        yaxis2=dict(
            title=f"Average {stat} of Activations",
            overlaying="y",
            side="right",
            titlefont=dict(size=20),
        ),
        xaxis=dict(title="Epoch", titlefont=dict(size=20)),
    )

    fig = go.Figure(layout=layout)
    fig.add_traces(traces)

    return fig


models = np.unique(df["Model"])

# Save statistics plots
fig = make_combined_stat_figure(df, models, line_colors, stat="Mean")
fig.write_image("plots/val_acc_v_mean.png", scale=3)

fig = make_combined_stat_figure(df, models, line_colors, stat="Variance")
fig.write_image("plots/val_acc_v_variance.png", scale=3)

fig = make_combined_stat_figure(df, models, line_colors, stat="Skewness")
fig.write_image("plots/val_acc_v_skewness.png", scale=3)

fig = make_combined_stat_figure(df, models, line_colors, stat="Kurtosis")
fig.write_image("plots/val_acc_v_kurtosis.png", scale=3)
