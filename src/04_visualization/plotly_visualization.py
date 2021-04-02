import sys
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy import interpolate

sys.path += ['src/01_data_processing', 'src/02_modelling', 'src/03_modell_evaluation','src/00_utils']

import spectrogram as spec
import train_test_split as splt
import train_model_autoencoder as train
import naming
import eval_model_autoencoder as eval


def make_mel_trace(mel, colorbar_len=0.45, colorbar_y=1.01):
    """
    Make heatmap trace of mel spectrogram
    """
    mel_trace = dict(visible=True,
                     type='heatmap',
                     x = np.array(range(mel.shape[0])),
                     y = np.array(range(mel.shape[1])),
                     z = mel,
                     colorscale='inferno',
                     colorbar=dict(len=colorbar_len,
                                   y=colorbar_y,
                                   yanchor='top',
                                   thickness=10))
    return mel_trace


def make_invisible_error_traces(timewise_recon_error, times, thresh):
    """
    Make invisible traces showing error over time and threshold.
    """
    above_thresh = timewise_recon_error.copy()
    below_thresh = timewise_recon_error.copy()
    above_thresh[above_thresh<thresh] = np.nan
    below_thresh[below_thresh>thresh] = np.nan

    text_trace = dict(visible=False,
                      type='scatter',
                      x=[9],
                      y=[thresh+0.005],
                      mode='text',
                      text='threshold',
                      textposition='top center',
                      showlegend=False)

    thresh_trace = dict(visible=False,
                        type='scatter',
                        x=[0,10],
                        y=[thresh,thresh],
                        marker=dict(color='black'),
                        mode='lines',
                        showlegend=False)

    above_trace = dict(visible=False,
                       type='scatter',
                       x=times,
                       y=above_thresh,
                       marker=dict(color='red'),
                       mode='lines',
                       showlegend=False)

    below_trace = dict(visible=False,
                       type='scatter',
                       x=times,
                       y=below_thresh,
                       marker=dict(color='green'),
                       mode='lines',
                       showlegend=False)

    return text_trace, thresh_trace, above_trace, below_trace


def make_sliders(thresh_range, active_step):
    """
    Make to select error trace based on threshold.
    """
    steps = []
    for i, thresh in enumerate(thresh_range):
        step = dict(label=round(thresh,2), method="update",
                     args=[{"visible": [True] + [False] * 4 * len(thresh_range)}])
        step["args"][0]["visible"][1+i*4] = True
        step["args"][0]["visible"][1+i*4+1] = True
        step["args"][0]["visible"][1+i*4+2] = True
        step["args"][0]["visible"][1+i*4+3] = True
        steps.append(step)

    sliders = [dict(currentvalue=dict(visible=False),
                    active=active_step,
                    steps=steps)]
    return sliders

def make_figure_layout(fig, sliders, mel):
    """
    Make layout for figure with two subplots for mel spectrogram and error over time.
    """
    fig.update_layout(
      height=800,
      width=600,
      xaxis1 = dict(
        tickmode = 'array',
        tickvals = np.linspace(0,mel.shape[1]-1,6),
        ticktext = [0,2,4,6,8,10]),
      yaxis1 = dict(
        tickmode = 'array',
        tickvals = np.linspace(0,mel.shape[0],6),
        ticktext = [0,512,1024,2048,4096,8000],
        title='Hz'),
      yaxis2 = dict(range=[0,0.2]),
      sliders=sliders,
    )


def make_eval_visualisation(mel_file, model, scaler, dim, step, thresh_range, active_step=1, as_images=True):
    """
    Call functions in this module to make a visualization for a given mel spectrogram file.
    """
    times, timewise_recon_error = eval.reco_loss_over_time(model=model,
                                                           scaler=scaler,
                                                           mel_file=mel_file,
                                                           dim=dim,
                                                           step=step,
                                                           as_images=as_images)
    # Interpolate linearly between point for plotting
    f = interpolate.interp1d(times, timewise_recon_error)
    times = np.arange(times[0], times[-1], 0.005)
    timewise_recon_error = f(times)

    # Generate figure with two subplots
    fig = make_subplots(rows=2, cols=1, vertical_spacing = 0.1,
                        subplot_titles=('spectrogram', 'reconstruction error'),
                        shared_xaxes=False)

    # Add spectrogram as visible trace to first row
    mel = np.load(mel_file)
    mel_trace = make_mel_trace(mel)
    fig.add_trace(mel_trace,row=1,col=1)

    # Add invsisible error traces to the second row
    for thresh in thresh_range:
        invisible_error_traces = make_invisible_error_traces(timewise_recon_error, times, thresh)
        for trace in invisible_error_traces:
            fig.add_trace(trace, row=2, col=1)

    # Make active step visible
    for data in fig.data[1+4*active_step:1+4*(active_step+1)]:
        data.visible = True

    # Make sliders to navigate between error traces in second row
    sliders = make_sliders(thresh_range, active_step)

    # Make figure layout and show
    make_figure_layout(fig, sliders, mel)
    fig.show()

