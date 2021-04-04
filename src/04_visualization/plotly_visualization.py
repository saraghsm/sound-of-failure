import sys
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy import interpolate

sys.path += ['src/01_data_processing', 'src/02_modelling', 'src/03_modell_evaluation', 'src/00_utils']

import spectrogram as spec
import train_test_split as splt
import train_model_autoencoder as train
import naming
import eval_model_autoencoder as eval


def make_mel_trace(mel, colorbar_len=0.3, colorbar_y=1.01):
    """
    Make heatmap trace of mel spectrogram
    """
    mel_trace = dict(visible=True,
                     type='heatmap',
                     x=np.array(range(mel.shape[0])),
                     y=np.array(range(mel.shape[1])),
                     z=mel,
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
    above_thresh[above_thresh < thresh] = np.nan
    below_thresh[below_thresh > thresh] = np.nan

    thresh_trace = dict(visible=False,
                        type='scatter',
                        x=[0, 10],
                        y=[thresh, thresh],
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

    return above_trace, below_trace, thresh_trace


def make_ref_thresh_trace_error(ref_thresh, thresh_range):
    """
    Make dashed horizontal line for reference threshold
    """
    assert thresh_range[0] <= ref_thresh <= thresh_range[-1], 'reference threshold outside of threshold range'
    ref_step = np.abs(ref_thresh - thresh_range).argmin()
    ref_thresh = thresh_range[ref_step]
    ref_thresh_trace = dict(visible=True,
                            type='scatter',
                            x=[0, 10],
                            y=[ref_thresh, ref_thresh],
                            marker=dict(color='black'),
                            mode='lines',
                            line=dict(color='black', dash='dash', width=1),
                            showlegend=False)
    return ref_thresh_trace, ref_step


def make_mean_error_trace(mean_recon_error):
    """
    Sert marker for mean error of sample file
    """
    mean_error_trace = dict(visible=True,
                            type='scatter',
                            x=[mean_recon_error],
                            y=[5],
                            mode='markers',
                            marker_symbol='x-thin',
                            marker=dict(size=8,
                                        color='black',
                                        line=dict(width=2, color='black')),
                            name='mean error of<br>sample + percentile',
                            showlegend=True)
    return mean_error_trace


def make_hist_traces(reco_loss_train, ref_thresh, thresh_range):
    """
    Make visible traces for histogram
    """
    # Histogram trace
    hist_trace = dict(visible=True,
                      type='histogram',
                      x=reco_loss_train,
                      marker=dict(color='green'),
                      histnorm='probability density',
                      opacity=0.3,
                      showlegend=False)

    # Probability distribution trace
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(reco_loss_train.reshape(-1, 1))
    errors = np.arange(0, 0.1, 0.0001)
    prob_density = np.exp(kde.score_samples(errors.reshape(-1, 1)))
    dist_trace = dict(visible=True,
                      type='scatter',
                      x=errors,
                      y=prob_density,
                      mode='lines',
                      line=dict(color='green', width=1),
                      showlegend=False)

    # Ref threshold trace
    assert thresh_range[0] <= ref_thresh <= thresh_range[-1], 'reference threshold outside of threshold range'
    ref_step = np.abs(ref_thresh - thresh_range).argmin()
    ref_thresh = thresh_range[ref_step]
    ref_thresh_trace = dict(visible=True,
                            type='scatter',
                            x=[ref_thresh, ref_thresh],
                            y=[0, 100],
                            mode='lines',
                            line=dict(color='black', dash='dash', width=1),
                            name='recommended<br>threshold',
                            showlegend=True)

    return hist_trace, dist_trace, ref_thresh_trace


def make_sliders(thresh_range, active_step, num_visible, num_invisible):
    """
    Make slider to select error trace based on threshold.
    """
    steps = []
    imgs_per_step = int(num_invisible / len(thresh_range))
    for i, thr in enumerate(thresh_range):
        # import pdb; pdb.set_trace()
        step = dict(label=round(thr, 2), method="update",
                    args=[{"visible": [True] * num_visible + [False] * num_invisible}])
        for j in range(imgs_per_step):
            step["args"][0]["visible"][num_visible + i * imgs_per_step + j] = True
        steps.append(step)

    sliders = [dict(currentvalue=dict(visible=False),
                    active=active_step,
                    steps=steps)]
    return sliders


def make_figure_layout(fig, sliders, mel, thresh_range, width=600, height=1000):
    """
    Make layout for figure with three subplots for mel spectrogram, error over time and training error distribution.
    """
    fig.update_layout(
        height=height,
        width=width,
        xaxis1=dict(
            tickmode='array',
            tickvals=np.linspace(0, mel.shape[1] - 1, 6),
            ticktext=[0, 2, 4, 6, 8, 10]),
        yaxis1=dict(
            tickmode='array',
            tickvals=np.linspace(0, mel.shape[0], 6),
            ticktext=[0, 512, 1024, 2048, 4096, 8000],
            title='Hz'),
        yaxis2=dict(range=[0, 0.2]),
        xaxis3=dict(range=[thresh_range[0], thresh_range[-1]]),
        yaxis3=dict(range=[0, 100]),
        sliders=sliders,
        legend=dict(
            traceorder='reversed',
            font=dict(size=10),
            yanchor="top",
            y=0.275,
            xanchor="right",
            x=0.99))


def make_eval_visualisation(mel_file,
                            model,
                            scaler,
                            reco_loss_train,
                            dim, step,
                            thresh_range,
                            ref_thresh,
                            width=600,
                            height=1000,
                            status_bar_width=0.025,
                            as_images=True):
    """
    Call functions in this module to make a visualization for a given mel spectrogram file.
    """
    times, timewise_recon_error = eval.reco_loss_over_time(model=model,
                                                           scaler=scaler,
                                                           mel_file=mel_file,
                                                           dim=dim,
                                                           step=step,
                                                           as_images=as_images)
    mean_recon_error = timewise_recon_error.mean()

    # Interpolate linearly between point for plotting
    f = interpolate.interp1d(times, timewise_recon_error)
    times = np.arange(times[0], times[-1], 0.005)
    timewise_recon_error = f(times)

    # Generate figure with two subplots
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.05,
                        subplot_titles=(
                        'spectrogram', 'reconstruction error over time', 'mean error distribution training'),
                        shared_xaxes=False)

    ########################
    # VISIBLE TRACES FIRST #
    ########################
    num_visible = 0

    # First row: Spectrogram
    mel = np.load(mel_file)
    mel_trace = make_mel_trace(mel)
    fig.add_trace(mel_trace, row=1, col=1)
    num_visible += 1

    # Second row: Reference threshold
    ref_thresh_trace, ref_step = make_ref_thresh_trace_error(ref_thresh, thresh_range)
    fig.add_trace(ref_thresh_trace, row=2, col=1)
    active_step = ref_step
    num_visible += 1

    # Third row: mean error
    mean_error_trace = make_mean_error_trace(mean_recon_error)
    fig.add_trace(mean_error_trace, row=3, col=1)
    num_visible += 1

    # Third row: histogram and distribution
    hist_traces = make_hist_traces(reco_loss_train, ref_thresh, thresh_range)
    for trace in hist_traces:
        fig.add_trace(trace, row=3, col=1)
        num_visible += 1

    # Third row: percentile label
    x_range = thresh_range[-1] - thresh_range[0]
    xlo = mean_recon_error - 0.055*x_range
    xhi = mean_recon_error + 0.055*x_range
    ylo = 9
    yhi = 15
    percentage_box_trace = dict(visible=True,
                                showlegend=False,
                                type='scatter',
                                mode='lines',
                                x=[xlo, xlo, xhi, xhi, xlo],
                                y=[ylo, yhi, yhi, ylo, ylo],
                                fill='toself',
                                fillcolor='white',
                                line=dict(width=0))

    percentage = str(round(sum(sorted(reco_loss_train) < mean_recon_error) / len(reco_loss_train) * 100, 2))
    percentage_text_trace = dict(visible=True, type='scatter',
                                 x=[(xhi + xlo) / 2], y=[(yhi + ylo) / 2],
                                 mode='text',
                                 text=percentage + '%',
                                 textposition='middle center',
                                 showlegend=False)
    fig.add_trace(percentage_box_trace, row=3, col=1)
    fig.add_trace(percentage_text_trace, row=3, col=1)
    num_visible += 1

    # Second row: status box
    xlo = 7.5
    xhi = 9.5
    yhi = 0.185
    ylo = yhi-status_bar_width
    status_box_trace = dict(visible=True,
                            showlegend=False,
                            type='scatter',
                            mode='lines',
                            x=[xlo, xlo, xhi, xhi, xlo],
                            y=[ylo, yhi, yhi, ylo, ylo],
                            fill='toself',
                            fillcolor='white',
                            line=dict(width=0))
    fig.add_trace(status_box_trace, row=2, col=1)
    num_visible += 1

    status_text_trace = dict(visible=True,
                             type='scatter',
                             x=[xlo + 0.1 * (xhi - xlo)],
                             y=[ylo + 0.5 * (yhi - ylo)],
                             mode='text',
                             text='Status',
                             textposition='middle right',
                             showlegend=False)
    fig.add_trace(status_text_trace, row=2, col=1)
    num_visible += 2

    #########################
    # INVISIBLE TRACES LAST #
    #########################
    num_invisible = 0

    # Add invsisible error traces to the second row
    for thresh in thresh_range:
        # Second row: colored error traces and horizontal threshold line
        invisible_error_traces = make_invisible_error_traces(timewise_recon_error, times, thresh)
        for trace in invisible_error_traces:
            fig.add_trace(trace, row=2, col=1)
            num_invisible += 1
        # Second row: status
        if mean_recon_error > thresh:
            color = 'red'
        else:
            color = 'green'
        status_trace = dict(visible=False,
                            type='scatter',
                            x=[xlo + 0.8 * (xhi - xlo)],
                            y=[ylo + 0.5 * (yhi - ylo)],
                            mode='markers',
                            marker=dict(size=18,
                                        color=color,
                                        line=dict(width=0)),
                            showlegend=False)
        fig.add_trace(status_trace, row=2, col=1)
        num_invisible += 1

        # Third row: vertical threshold line
        invisible_hist_trace = dict(visible=False, type='scatter',
                                    x=[thresh, thresh], y=[0, 100],
                                    mode='lines', line=dict(color='black', width=2),
                                    name='threshold', showlegend=True)
        fig.add_trace(invisible_hist_trace, row=3, col=1)
        num_invisible += 1

    ############################
    # MAKE ACTIVE STEP VISIBLE #
    ############################
    imgs_per_step = int(num_invisible / len(thresh_range))
    for data in fig.data[num_visible + imgs_per_step * active_step: num_visible + imgs_per_step * (active_step + 1)]:
        data.visible = True

    #################################
    # SLIDERS TO CONTROL VISIBILITY #
    #################################
    sliders = make_sliders(thresh_range, active_step, num_visible, num_invisible)

    # Make figure layout and show
    make_figure_layout(fig, sliders, mel, thresh_range, width, height)
    return fig
