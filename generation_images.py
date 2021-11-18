import tensorflow as tf
from tensorflow import keras

from plotly.subplots import make_subplots
import plotly.express as px

from dcgan_creation import BATCH_SIZE, noise_dim


def function_generate_image(type_of_images: str, number_of_img: int):
    model = keras.models.load_model('models/generator_' + str(type_of_images) + '_model.h5')
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    generated_images = model(noise, training=False)

    number_of_img = int(number_of_img)
    nb_rows = 2
    nb_col = int(number_of_img / 2)

    fig = make_subplots(rows=nb_rows, cols=nb_col)

    for i in range(number_of_img):
        fig.add_trace(
            px.imshow((generated_images[i, :, :, :] * 127.5 + 127.5) / 255, color_continuous_scale='gray').data[0],
            row=int(i / nb_col) + 1, col=i % nb_col + 1)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(255, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

    return fig
