import base64
from datetime import datetime
import os
from pathlib import Path
from random import choice
from urllib.parse import quote as urlquote

import dash
import dash_core_components as dcc
import dash_html_components as html
# ------------------------------------------------------------
# Useful functions
# ------------------------------------------------------------
import numpy
import pandas
import toml
from dash.dependencies import Input, Output, State
from flask import Flask, send_from_directory
from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from numpy.random import random

from src.data.datamanager import DataManager
from src.data.dataset import Activity, MotionSense
from src.data.window import KerasSequence, WindowSequence
from src.visual.plotter import Plotter, VecData, make_line_plot_new

datasets = toml.load("config.toml")["dataset"]
WISDM_PATH = Path(datasets["wisdm"])
MOTION_SENSE_PATH = Path(datasets["motion-sense"])
dataset = MotionSense(MOTION_SENSE_PATH)
datamanager = DataManager(dataset)
all_data = datamanager.read()
window_length = 100
windows = datamanager.create_windows(
    set(Activity),
    window_length,
    shuffle=True,
    seed=1,
    columns=[
        "xaccel_norm",
        "yaccel_norm",
        "zaccel_norm",
        "xrot_norm",
        "yrot_norm",
        "zrot_norm",
    ],
)  # , bypass_raw="6a3bf91cfb")
# windows = datamanager.create_windows(set(Activity), 100, shuffle=False, seed=1, columns=['xaccel_norm', 'yaccel_norm', 'zaccel_norm', 'xrot_norm', 'yrot_norm', 'zrot_norm', "activity"])#, bypass_raw="6a3bf91cfb")
seq = windows.to_keras_sequence(32)


# ------------------------------------------------------------
# Loading GANS
# ------------------------------------------------------------


classifier = load_model("models/classifiers/cnn_classifier.h5")

# acgan_4_gen = load_model("models/generators/acgan_4_generator.h5")
# acgan_4_disc = load_model("models/discriminators/acgan_4_discriminator.h5")

# acgan_7_gen = load_model("models/generators/acgan_7_generator.h5")
# acgan_7_disc = load_model("models/discriminators/acgan_7_discriminator.h5")

# acgan_10_gen = load_model("models/generators/acgan_10_generator.h5")
# acgan_10_disc = load_model("models/discriminators/acgan_10_discriminator.h5")

acgan_12_gen = load_model("models/generators/acgan_12_generator.h5")
acgan_12_disc = load_model("models/discriminators/acgan_12_discriminator.h5")

# acgan_13_gen = load_model("models/generators/acgan_13_generator.h5")
# acgan_13_disc = load_model("models/discriminators/acgan_13_discriminator.h5")

# acgan_16_gen = load_model("models/generators/acgan_16_generator.h5")
# acgan_16_disc = load_model("models/discriminators/acgan_16_discriminator.h5")


n_latents = 200
num_classes = len(dataset.ACTIVITIES.values())


def generate_labels(n_samples: int, num_classes: int, onehot: bool) -> numpy.ndarray:
    if onehot:
        return numpy.eye(num_classes)[numpy.random.choice(num_classes, n_samples)]
    return numpy.random.randint(0, num_classes, (n_samples, 1))


def smooth_positive_labels(y):
    return y - 0.3 + (random(y.shape) * 0.5)


def smooth_negative_labels(y):
    return y + random(y.shape) * 0.3


def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = numpy.random.choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y


# generate points in latent space as input for the generator
# Gaussian values with a mean close to zero and a
# standard deviation close to 1, e.g. a standard Gaussian distribution.
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = numpy.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input


def create_ground_values(n_samples: int):
    return (
        numpy.clip(
            noisy_labels(smooth_positive_labels(numpy.ones((n_samples, 1))), 0.05),
            0,
            1.2,
        ),
        numpy.clip(
            noisy_labels(smooth_negative_labels(numpy.zeros((n_samples, 1))), 0.05),
            0,
            1.2,
        ),
    )


def generate_and_prepare_data(generator, n_samples_each=300):

    latents = generate_latent_points(n_latents, n_samples_each * num_classes)
    Y_gen = numpy.array([[i] * n_samples_each for i in range(num_classes)]).flatten()
    X_gen = generator.predict([latents, Y_gen])

    window_sample = datamanager.create_windows(
        set(Activity),
        window_length,
        shuffle=True,
        columns=[
            "xaccel_norm",
            "yaccel_norm",
            "zaccel_norm",
            "xrot_norm",
            "yrot_norm",
            "zrot_norm",
        ],
    )
    seq_sample = windows.to_keras_sequence(n_samples_each * num_classes)

    X_real = seq_sample[0][0]
    Y_real = seq_sample[0][1]

    return X_real, Y_real, X_gen, Y_gen


def generate_fooling_data(generator, discriminator):
    X_real, Y_real, X_gen, Y_gen = generate_and_prepare_data(generator)
    Y_pred_fakereal, _ = discriminator.predict(X_gen)
    Y_gen_fooled = numpy.delete(Y_gen, numpy.where(Y_pred_fakereal < 0.5), axis=0)
    X_gen_fooled = numpy.delete(X_gen, numpy.where(Y_pred_fakereal < 0.5), axis=0)
    return X_gen_fooled, Y_gen_fooled


def plot_fooling_sample(activity, generator, discriminator):
    X_gen_fooled, Y_gen_fooled = generate_fooling_data(generator, discriminator)


def plot_real(cls):
    real_data = all_data.loc[all_data["activity"] == cls].iloc[0:window_length]
    acc_cols = ("xaccel_norm", "yaccel_norm", "zaccel_norm")
    plotter = Plotter(VecData(real_data, acc_cols, 100), dataset.FREQUENCY)
    return plotter.make_line_plot(cls)


def plot_fake(generator, discriminator, cls):
    cls_code = sorted(dataset.ACTIVITIES.values()).index(cls)
    X_gen_fooled, Y_gen_fooled = generate_fooling_data(generator, discriminator)
    fake_data = X_gen_fooled[numpy.where(Y_gen_fooled == cls_code)][0]

    acc_cols = ("xaccel_norm", "yaccel_norm", "zaccel_norm")
    df = pandas.DataFrame(
        data=fake_data, columns=[*acc_cols, "xrot_norm", "yrot_norm", "zrot_norm"]
    )
    plotter = Plotter(VecData(df, acc_cols, 100), dataset.FREQUENCY)
    return plotter.make_line_plot(cls)


counters = {name: 0 for name in sorted(dataset.ACTIVITIES.values())}
X_gen_fooled, Y_gen_fooled = generate_fooling_data(acgan_12_gen, acgan_12_disc)


def plot(generator, discriminator, cls):
    cls_code = sorted(dataset.ACTIVITIES.values()).index(cls)
    if counters[cls] >= len(Y_gen_fooled == cls_code):
        counters[cls] = 0
    fake_data = X_gen_fooled[numpy.where(Y_gen_fooled == cls_code)][counters[cls]]

    elected = None
    i = 0
    while elected is None and i < len(seq):
        real_data = seq[i][0]
        real_classes = seq[i][1].argmax(axis=1)
        elected = numpy.delete(real_data, numpy.where(real_classes != cls_code), axis=0)

    if counters[cls] >= len(elected):
        counters[cls] = 0
    real_data = elected[counters[cls]]
    return make_line_plot_new(
        fake_data * 30, real_data * 20, cls
    )  # , make_2d_animations_new(fake_data * 20, real_data * 20, cls)


def animate_real(cls):
    real_data = all_data.loc[all_data["activity"] == cls].iloc[0:window_length]
    acc_cols = ("xaccel_norm", "yaccel_norm", "zaccel_norm")
    plotter = Plotter(VecData(real_data, acc_cols, 1000), dataset.FREQUENCY)
    return plotter.make_2d_animations(cls)


def animate_fake(generator, discriminator, cls):
    cls_code = sorted(dataset.ACTIVITIES.values()).index(cls)
    fake_data = X_gen_fooled[numpy.where(Y_gen_fooled == cls_code)][counters[cls]]

    acc_cols = ("xaccel_norm", "yaccel_norm", "zaccel_norm")
    df = pandas.DataFrame(
        data=fake_data, columns=[*acc_cols, "xrot_norm", "yrot_norm", "zrot_norm"]
    )
    plotter = Plotter(VecData(df, acc_cols, 1000), dataset.FREQUENCY)
    return plotter.make_2d_animations(cls)


def plot_real_new(cls):
    choice(windows)
    real_data = all_data.loc[all_data["activity"] == cls].iloc[0:window_length]
    acc_cols = ("xaccel_norm", "yaccel_norm", "zaccel_norm")
    plotter = Plotter(VecData(real_data, acc_cols, 100), dataset.FREQUENCY)
    return plotter.make_line_plot(cls)


UPLOAD_DIRECTORY = "generated"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)


# @server.route("/download/<path:path>")
# def download(path):
#     """Serve a file from the upload directory."""
#     return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


def generate_table(dataframe, max_rows=10):
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
            html.Tbody(
                [
                    html.Tr(
                        [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]
                    )
                    for i in range(min(len(dataframe), max_rows))
                ]
            ),
        ]
    )

@server.route(f"/{UPLOAD_DIRECTORY}/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = f"/{UPLOAD_DIRECTORY}/{filename}"
    return html.A(filename, href=location)


app.layout = html.Div(
    [
        html.H1("Fake User Activity Generator"),
        html.Div(
            [
                dcc.Dropdown(
                    id="dd-activity-selector",
                    options=[
                        {"label": "Walk", "value": "wlk"},
                        {"label": "Jog", "value": "jog"},
                        {"label": "Stand up", "value": "std"},
                        {"label": "Sit down", "value": "sit"},
                        {"label": "Upstairs", "value": "ups"},
                        {"label": "Downstairs", "value": "dws"},
                    ],
                    style={"width": "200px"},
                ),
                html.Button(id="btn-generate", n_clicks=0, children="Generate"),
            ],
        ),
        dcc.Graph(id="plot", figure=plot(acgan_12_gen, acgan_12_disc, "wlk")),
        dcc.Graph(
            id="animation",
            figure=animate_fake(acgan_12_gen, acgan_12_disc, "wlk"),
        ),
        html.Button(id="btn-save", n_clicks=0, children="Save Action"),
        html.P(id="save-result", children="No saved"),
        html.Div(id="hidden-div", style={"display":"none"})

        # generate_table(windows[0][0]),
        # html.H2("Upload"),
        # dcc.Upload(
        #     id="upload-data",
        #     children=html.Div(["Drag and drop or click to select a file to upload."]),
        #     style={
        #         "width": "100%",
        #         "height": "60px",
        #         "lineHeight": "60px",
        #         "borderWidth": "1px",
        #         "borderStyle": "dashed",
        #         "borderRadius": "5px",
        #         "textAlign": "center",
        #         "margin": "10px",
        #     },
        #     multiple=True,
        # ),
        # html.H2("File List"),
        # html.Ul(id="file-list"),
        # seq.shape
    ],
    # style={'columnCount': 2}
)


# def save_file(name, content):
#     """Decode and store a file uploaded with Plotly Dash."""
#     data = content.encode("utf8").split(b";base64,")[1]
#     with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
#         fp.write(base64.decodebytes(data))


# def uploaded_files():
#     """List the files in the upload directory."""
#     files = []
#     for filename in os.listdir(UPLOAD_DIRECTORY):
#         path = os.path.join(UPLOAD_DIRECTORY, filename)
#         if os.path.isfile(path):
#             files.append(filename)
#     return files


# def file_download_link(filename):
#     """Create a Plotly Dash 'A' element that downloads a file from the app."""
#     location = "/saved/{}".format(urlquote(filename))
#     return html.A(filename, href=location)


# @app.callback(
#     Output("file-list", "children"),
#     [Input("upload-data", "filename"), Input("upload-data", "contents")],
# )
# def update_output(uploaded_filenames, uploaded_file_contents):
#     """Save uploaded files and regenerate the file list."""

#     if uploaded_filenames is not None and uploaded_file_contents is not None:
#         for name, data in zip(uploaded_filenames, uploaded_file_contents):
#             save_file(name, data)

#     files = uploaded_files()
#     if len(files) == 0:
#         return [html.Li("No files yet!")]
#     else:
#         return [html.Li(file_download_link(filename)) for filename in files]


@app.callback(
    Output("save-result", "children"),
    [Input("btn-save", "n_clicks")],
    [State("dd-activity-selector", "value")],
)
def save_gen(n_clicks, activity):
    if activity:
        fname = f"{activity}_{str(datetime.now())}"
        with open(f'{UPLOAD_DIRECTORY}/{fname}.csv', 'wb') as f:
            cls_code = sorted(dataset.ACTIVITIES.values()).index(activity)
            # numpy.save(f, X_gen_fooled[numpy.where(Y_gen_fooled == cls_code)][counters[activity]])
            numpy.savetxt(f, 20 * X_gen_fooled[numpy.where(Y_gen_fooled == cls_code)][counters[activity]], delimiter=',', fmt='%f', header="x_acc,y_acc,z_acc,x_rot,y_rot,z_rot", comments='')
        return f"SAVED {fname}"
    return ""


@app.callback(
    [Output("plot", "figure"), Output("animation", "figure")],
    [Input("btn-generate", "n_clicks")],
    [State("dd-activity-selector", "value")],
)
def update_output(n_clicks, activity):

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate("cancel the callback")

    if activity:
        counters[activity] += 1

    return [
        plot(acgan_12_gen, acgan_12_disc, activity or "wlk"),
        animate_fake(acgan_12_gen, acgan_12_disc, activity or "wlk"),
    ]


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)  # import dash
# import dash_core_components as dcc
# import dash_html_components as html

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div([
#     html.Label('Dropdown'),
#     dcc.Dropdown(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value='MTL'
#     ),

#     html.Label('Multi-Select Dropdown'),
#     dcc.Dropdown(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value=['MTL', 'SF'],
#         multi=True
#     ),

#     html.Label('Radio Items'),
#     dcc.RadioItems(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value='MTL'
#     ),

#     html.Label('Checkboxes'),
#     dcc.Checklist(
#         options=[
#             {'label': 'New York City', 'value': 'NYC'},
#             {'label': u'Montréal', 'value': 'MTL'},
#             {'label': 'San Francisco', 'value': 'SF'}
#         ],
#         value=['MTL', 'SF']
#     ),

#     html.Label('Text Input'),
#     dcc.Input(value='MTL', type='text'),

#     html.Label('Slider'),
#     dcc.Slider(
#         min=0,
#         max=9,
#         marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
#         value=5,
#     ),
# ], style={'columnCount': 2})

# if __name__ == '__main__':
#     app.run_server(debug=True)
