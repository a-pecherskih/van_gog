import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    # if request.method == 'POST':
    #     print(request.get_json())
    # f = request.files['file']
    # filename = f.filename
    #
    # path = 'static/images/' + filename
    # f.save(os.path.join(path))
    #
    # result_filename = neiro(path)
    #
    # return render_template('index.html', filename=result_filename)

    return render_template('index.html')


@app.route("/neiro", methods=['POST'])
def neiro():
    file = request.files['file']
    img_style_filename = request.form.get("hidden_img_style")
    iteration = int(request.form.get("iteration"))

    filename = file.filename
    path = 'static/images/' + filename
    file.save(os.path.join(path))

    img_style_path = 'static/style_images/' + img_style_filename

    start_time = time.time()
    result_filepath = runNeiro(path, img_style_path, iteration)
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    return result_filepath


def runNeiro(original_filepath, img_style_path, iteration):
    # img = load_image(original_filename)
    # img_style = load_image(original_filename)

    img = Image.open(original_filepath)
    img_style = Image.open(img_style_path)

    # Сеть VGG19. Нужно преобразовать в формат, который воспринимает эта сеть, используется функция preprocess_input (переводит )

    x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
    x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

    # Определяем вспомогательные коллекции с именами слоев, которые будем выделять из сети vgg19
    content_layers = ['block5_conv2']  # для контента последний сверточный слой у этой сети

    # Потери по стилям определять
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Загружаем сеть vgg19. include_top - не будем использовать полносвязанные нейронную сеть на ее конце, а веса предобученные по коллекции imagenet на 10млн изображения
    # trainable - веса нельзя менять
    # Подавать на вход изображение и брать на выход ее слоев вычесленные карты признаков

    vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Слои выше выделяем block1_conv1 - уровень 1

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs  # объединяем все коллекции между собой

    # print(vgg.input)
    # for m in model_outputs:
    #     print(m)

    model = keras.models.Model(vgg.input,
                               model_outputs)  # Указываем сход у сети и наши выходы (один вход и множество выходов)
    for layer in model.layers:
        layer.trainable = False

    # print(model.summary())

    # Вспомогательные переменные
    num_iterations = 50  # число итераций
    num_iterations = iteration  # число итераций из запроса
    content_weight = 1e3  # альфа
    style_weight = 1e-2  # бетта

    style_features, content_features = get_feature_representations(model, x_style, x_img, num_style_layers)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = np.copy(x_img)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
    iter_count = 1
    best_loss, best_img = float('inf'), None
    loss_weights = (style_weight, content_weight)

    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'num_content_layers': num_content_layers,
        'num_style_layers': num_style_layers,
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    imgs = []

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            all_loss = compute_loss(**cfg)

        loss, style_score, content_score = all_loss
        grads = tape.gradient(loss, init_image)

        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

            # Use the .numpy() method to get the concrete numpy array
            plot_img = deprocess_img(init_image.numpy())
            imgs.append(plot_img)
            print('Iteration: {}'.format(i))

    image = Image.fromarray(best_img.astype('uint8'), 'RGB')
    filename = "result.jpg"
    result_filepath = "static/result_images/" + filename
    image.save(result_filepath)

    return result_filepath


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


# Вспомогательная функция, чтобы преобразовать BGR -> RGB, чтобы видеть полученный результат
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)  # Убираем нулевую ось через метод squeeze
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")  # Проверяем что осталось 3 оси
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Добавляем средние значения к соответствующим цветовым компонентам
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # Меняем местами цветовые компоненты

    x = np.clip(x, 0, 255).astype('uint8')  # Отбрасываем все что < 0 и > 255
    return x


# Возврат карты функций для стилей и для контента
def get_feature_representations(model, x_style, x_img, num_style_layers):
    # batch compute content and style features
    style_outputs = model(x_style)
    content_outputs = model(x_img)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


# Вычисление потерь по контенту
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))  # квадрат разности


# Вычисляет матрицу грамма
def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])  # кол-во каналов
    a = tf.reshape(input_tensor, [-1, channels])  # первая разномерность как получится, а вторая - число каналов
    n = tf.shape(a)[0]  #
    gram = tf.matmul(a, a, transpose_a=True)  #
    return gram / tf.cast(n, tf.float32)


# Вычисление стиля для строго определенной нейронной сети
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)  # матрица граммов для формируемого изображения

    return tf.reduce_mean(tf.square(gram_style - gram_target))


# Вычисление всех потерь
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, num_content_layers,
                 num_style_layers):
    style_weight, content_weight = loss_weights  # style_weight - альфа, content_weight - бетта

    model_outputs = model(init_image)  # init_image, формируемое изображение, пропускаем через нейронную сеть

    style_output_features = model_outputs[:num_style_layers]  # карты признаков для стилей
    content_output_features = model_outputs[num_style_layers:]  # карты признаков для контента

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


if __name__ == "__main__":
    app.run()
