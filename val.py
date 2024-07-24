import matplotlib
import numpy as np
from sklearn.manifold import TSNE

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.DoubleTensor')


def predict(Phi, adapt_in, adapt_out, cfg, val_in=None, H = None, lam=0, a=None):
    if val_in is None:
        val_in = adapt_in
    with torch.no_grad():
        # Ordinary Least Squares on Adaptation Set to get a
        X = torch.from_numpy(adapt_in)
        Y = torch.from_numpy(adapt_out)

        if a is None:
            phi = Phi(X)
            phi_T = phi.transpose(0, 1)
            A = torch.inverse(torch.mm(phi_T, phi) + lam*torch.eye(cfg['not']))
            a = torch.mm(torch.mm(A, phi_T), Y)
        if a is not None:
            a = torch.tensor(a).clone().detach()

        # Compute NN predictions
        inputs = torch.from_numpy(val_in)
        pred = torch.mm(Phi(inputs), a)
        adapt = torch.mm(Phi(X), a)

        # Compute Adversarial Network Prediction
        temp = Phi(inputs)
        if H is None:
            return adapt.numpy(), pred.numpy(), a.numpy()
        else:
            h_out = H(temp)
            if cfg['classification_loss'] == 'cross-entropy':
                _softmax = nn.Softmax(dim=1)
                h_out = _softmax(h_out)
            h_out = h_out.numpy()
            return adapt.numpy(), pred.numpy(), a.numpy(), h_out


def val(x: np.ndarray, y: np.ndarray, Phi, H, cfg):
    """
    Computes Loss Statistics
    max_loss: MSE Loss between residual forces and zero
    avg_loss: MSE Loss between residual forces and average output
    adapt_loss: MSE Loss between output of Phi Network and a adapted over the entire dataset
    """
    criterion = nn.MSELoss()
    with torch.no_grad():
        max_loss = criterion(torch.from_numpy(y), 0.0*torch.from_numpy(y)).item()
        avg_loss = criterion(torch.from_numpy(y), torch.from_numpy(np.ones((len(y), 1))) * np.mean(y, axis=0)[np.newaxis, :]).item()
        _, pred, _, _ = predict(Phi=Phi, H=H, adapt_in=x, adapt_out=y, cfg=cfg)
        adapt_loss = criterion(torch.from_numpy(y), torch.from_numpy(pred)).item()
    return {"max_loss": max_loss, "avg_loss": avg_loss, "adapt_loss": adapt_loss}


def plot_errors(t: np.ndarray, x: np.ndarray, y: np.ndarray, Phi, idx_val_start: int, idx_val_end:int, cfg: dict, title, lam=0, a=None):
    """"
    Visualize adaptation and validation
    """
    t -= t[0]
    adapt_in = x[:idx_val_start, :]
    adapt_out = y[:idx_val_start, :]
    val_in = x[idx_val_start:idx_val_end, :]
    if a is not None:
        y_adapt, y_val, _ = predict(Phi=Phi, adapt_in=adapt_in, adapt_out=adapt_out, a=a, cfg=cfg)
    else:
        y_adapt, y_val, a = predict(Phi=Phi, adapt_in=adapt_in, adapt_out=adapt_out, val_in=val_in, cfg=cfg)

    # fig = make_subplots(rows=1, cols=3, subplot_titles=("$F_{r_{X}}$", "$F_{r_{Y}}$", "$F_{r_{z}}$"), start_cell="top-left")
    fig = make_subplots(rows=1, cols=3, subplot_titles=("X", "Y", "Z"), start_cell="top-left")
    for i in range(2):
        fig = fig.add_trace(go.Scatter(x=t[:idx_val_end], y=y[:idx_val_end, i], mode="lines", line={"color": "grey"},
                                       name="Ground Truth", legendgroup="group1", showlegend=False), row=1, col=i + 1)
        # fig = fig.add_trace(go.Scatter(x=t[:idx_val_start], y=y_adapt[:, i], mode="lines", line={"color": "red"},
        #                                name="Adaptation", legendgroup="group2", showlegend=False), row=1, col=i + 1)
        fig = fig.add_trace(go.Scatter(x=t[:idx_val_end], y=y_val[:, i], mode="lines", line={"color": "blue"},
                                       name="Validation", legendgroup="group3", showlegend=False), row=1, col=i + 1)
        fig.layout.annotations[i].update(y=1.02)
    fig = fig.add_trace(go.Scatter(x=t[:idx_val_end], y=y[:idx_val_end, i+1], mode="lines", line={"color": "grey"},
                                   name="Ground Truth",  legendgroup="group1"), row=1, col=i + 2)
    # fig = fig.add_trace(go.Scatter(x=t[:idx_val_start], y=y_adapt[:, i+1], mode="lines", line={"color": "red"},
    #                                name="Adaptation", legendgroup="group2"), row=1, col=i + 2)
    fig = fig.add_trace(go.Scatter(x=t[:idx_val_end], y=y_val[:, i+1], mode="lines", line={"color": "blue"},
                                   name="Validation", legendgroup="group3"), row=1, col=i + 2)

    fig.layout.annotations[i+1].update(y=1.02)
    fig.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))
    fig.update_annotations(font=dict(size=16))

    fig['layout']["title"] = title
    fig['layout']['xaxis']['title'] = 'Time [s]'
    fig['layout']['xaxis2']['title'] = 'Time [s]'
    fig['layout']['xaxis3']['title'] = 'Time [s]'
    fig['layout']['yaxis']['title'] = 'Residual Force [N]'
    return fig


def plot_class(t: np.ndarray, X: np.ndarray, Y: np.ndarray, Phi, H, idx_val_start: int, idx_val_end: int, cfg):
    """"
    Visualize Classification
    """
    t -= t[0]
    adapt_in = X[:idx_val_start, :]
    adapt_out = Y[:idx_val_start, :]
    val_in = X[idx_val_start:idx_val_end, :]
    _, _, _, h_out = predict(Phi=Phi, H=H, adapt_in=adapt_in, adapt_out=adapt_out, val_in=val_in, cfg=cfg)
    if cfg['classification_loss'] == "cross-entropy":
        fig = go.Figure()
        colors = plt.cm.RdYlGn(np.linspace(1, 0, cfg['nc']))
        for i in range(cfg['nc']):
            fig.add_trace(go.Scatter(x=t[idx_val_start:idx_val_end], y=h_out[:, i], mode="lines", name=str(i+1),
                                     legendgroup=str(i),
                                     line={"color": 'rgb(' + str(255*colors[i][0]) + ',' + str(255*colors[i][1]) + ',' +
                                                    str(255*colors[i][2]) + ')'}))

        V = np.linalg.norm(X[:, 0:3], axis=1)[idx_val_start:idx_val_end]
        lines = [[0, next(x[0] for x in enumerate(np.arange(cfg["nc"])) if x[1] > V[0])]]
        for idx, v in enumerate(V):
            i = next(x[0] for x in enumerate(np.arange(cfg["nc"])) if x[1] > V[idx])
            if i != lines[-1][1]:
                lines.append([idx, i])
        lines.append([len(V), None])

        for idx in range(len(lines[:-1])):
            fig.add_trace(go.Scatter(x=t[lines[idx][0] + idx_val_start:lines[idx+1][0] + idx_val_start],
                                     y=[1]*(lines[idx+1][0] - lines[idx][0]), mode="lines", showlegend=False,
                                     legendgroup=str(lines[idx][1] - 1),
                                     line={"color": 'rgb(' + str(255*colors[lines[idx][1] - 1][0]) + ',' +
                                                    str(255*colors[lines[idx][1]-1][1]) + ',' +
                                                    str(255*colors[lines[idx][1]-1][2]) + ')'}))
        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Classification Prediction",
            legend_title="Speed Condition",
        )
        # fig.update_yaxes(rangemode="tozero")
        return fig


def plot_dist(data: list, cfg: dict):
    colors = plt.cm.RdYlGn(np.linspace(1, 0, cfg['nc']))
    bars = []
    for i in range(len(data)):
        bars.append(go.Bar(x=[i+1], y=[len(data[i].X)], name=str(i+1),
                           marker={"color": 'rgb(' + str(255*colors[i][0]) + ',' + str(255*colors[i][1]) + ',' +
                                                    str(255*colors[i][2]) + ')'}))
    fig = go.Figure(data=bars)
    fig.update_layout(
        xaxis_title="Sub-Dataset",
        yaxis_title="Datapoints",
        legend_title="Speed Condition",
    )
    return fig


def plot_a(log_a: list, cfg: dict, n: int = 3):
    colors = plt.cm.RdYlGn(np.linspace(1, 0, cfg['nc']))

    label = np.array(log_a, dtype=object)[-cfg["nc"] * n:, 0]
    x = np.array([a.reshape(3*cfg["not"]) for a in np.array(log_a, dtype=object)[-cfg["nc"] * n:, 1]])

    n_components = 2
    tsne = TSNE(n_components, learning_rate='auto', init='pca', perplexity=5)
    out = tsne.fit_transform(x)

    fig = go.Figure()
    for i in range(cfg['nc']):
        show_legend = True
        for idx, l in enumerate(label):
            if i == l:
                fig.add_trace(go.Scatter(x=[out[idx, 0]], y=[out[idx, 1]], mode='markers', name=str(l+1),
                                         legendgroup=f"group{l}", showlegend=show_legend, marker={"size": 12,
                "color": 'rgb(' + str(255*colors[l][0]) + ',' + str(255*colors[l][1]) + ',' + str(255*colors[l][2]) + ')'}))
                show_legend = False
    fig.update_layout(legend_title="Speed Condition")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig = plt.figure()
    colors = ["green", "orange", "red"]
    labels = ["no damage", "slight damage", "significant damage"]

    for i in range(cfg['nc']):
        r = 0
        for idx, l in enumerate(label):
            if l == i and r == 0:
                plt.scatter(x=[out[idx, 0]], y=[out[idx, 1]], color=colors[l])
                r += 1
    plt.legend(labels, title="Condition")
    for idx, l in enumerate(label):
        plt.scatter(x=[out[idx, 0]], y=[out[idx, 1]], color=colors[l])
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.title(r"t-SNE Plot of Training Adaptation Coefficients $A^{*}$ with $\alpha=0$", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return fig