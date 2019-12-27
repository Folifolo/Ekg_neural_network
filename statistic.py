import json

import numpy as np
import tensorflow as tf

ECG_LEN = 5000


def open_json(link):
    f = open(link, 'r')
    data = json.load(f)
    return data


def create_mask(start, end):
    result = np.ones((1, ECG_LEN, 1))
    result[:, start:end, :] = np.zeros((1, end - start, 1))

    return result


def data_generator(patient):
    v6_data = data[patient]["Leads"]["v6"]["Signal"]
    v6_data = np.array(v6_data)
    v6_data = np.expand_dims(v6_data, axis=0)

    while True:
        yield v6_data


def build_model(latent_pool_size, x):
    ae_filters = {
        "conv1": tf.Variable(tf.truncated_normal([100, 1, 10], stddev=0.1)),
        "b_conv1": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
        "conv2": tf.Variable(tf.truncated_normal([80, 10, 10], stddev=0.1)),
        "b_conv2": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
        "conv3": tf.Variable(tf.truncated_normal([50, 10, 5], stddev=0.1)),
        "b_conv3": tf.Variable(tf.truncated_normal([5], stddev=0.1)),

        "deconv3": tf.Variable(tf.truncated_normal([50, 10, 5], stddev=0.1)),
        "b_deconv3": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
        "deconv4": tf.Variable(tf.truncated_normal([80, 10, 10], stddev=0.1)),
        "b_deconv4": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
        "deconv5": tf.Variable(tf.truncated_normal([100, 1, 10], stddev=0.1)),
        "b_deconv5": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
    }

    x_1 = tf.nn.conv1d(x, ae_filters["conv1"], stride=1, padding="SAME", use_cudnn_on_gpu=True) \
          + ae_filters["b_conv1"]
    x_1_mp = tf.layers.max_pooling1d(x_1, pool_size=2, strides=2, padding='SAME')
    x1_relu = tf.nn.leaky_relu(x_1_mp, alpha=0.2)

    x_2 = tf.nn.conv1d(x1_relu, ae_filters["conv2"], stride=1, padding="SAME", use_cudnn_on_gpu=True) \
          + ae_filters["b_conv2"]
    x_2_mp = tf.layers.max_pooling1d(x_2, pool_size=latent_pool_size, strides=latent_pool_size, padding='SAME')
    x2_relu = tf.nn.leaky_relu(x_2_mp, alpha=0.2)

    x_3 = tf.nn.conv1d(x2_relu, ae_filters["conv3"], stride=1, padding="SAME", use_cudnn_on_gpu=True) \
          + ae_filters["b_conv3"]
    x3_mp = tf.layers.max_pooling1d(x_3, pool_size=2, strides=2, padding='SAME')
    x3_relu = tf.nn.leaky_relu(x3_mp, alpha=0.2)

    x_dec1 = tf.contrib.nn.conv1d_transpose(x3_relu, ae_filters["deconv3"],
                                            [1, ECG_LEN // 2 // latent_pool_size + 1, 10],
                                            strides=2, padding="SAME")
    x_dec1_relu = tf.nn.leaky_relu(x_dec1, alpha=0.2)

    x_dec2 = tf.contrib.nn.conv1d_transpose(x_dec1_relu, ae_filters["deconv4"],
                                            [1, ECG_LEN // 2, 10],
                                            strides=latent_pool_size, padding="SAME")
    x_dec2_relu = tf.nn.leaky_relu(x_dec2, alpha=0.2)

    x_dec3 = tf.contrib.nn.conv1d_transpose(x_dec2_relu, ae_filters["deconv5"],
                                            [1, ECG_LEN, 1],
                                            strides=2, padding="SAME")
    x_dec3_relu = tf.nn.leaky_relu(x_dec3, alpha=0.2)

    return x_dec3_relu


def fit(model_output, mask_start, mask_end, epochs, patient, x):
    vars = tf.trainable_variables()
    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 1

    mask = create_mask(mask_start, mask_end)
    inversed_mask = np.ones(mask.shape) - mask

    error_function = tf.reduce_mean(tf.square(model_output - x) * mask) + loss_l2
    error_function_outside = tf.reduce_mean(tf.square(model_output - x) * mask)
    error_function_inside = tf.reduce_mean(tf.square(model_output - x) * inversed_mask)

    optimizer = tf.train.AdagradOptimizer(0.01).minimize(error_function)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)

    error_inside_list = []
    error_outside_list = []

    generator = data_generator(patient)

    error_inside = 0
    error_outside = 0

    for epoch in range(epochs):
        epoch_x = next(generator)
        _, _, error_inside, error_outside = sess.run(
            [optimizer, error_function, error_function_inside, error_function_outside], feed_dict={x_ph: epoch_x})

        error_inside_list.append(error_inside)
        error_outside_list.append(error_outside)

    return error_inside, error_outside


if __name__ == "__main__":
    raw_dataset_path = "data_1078.json"
    data = open_json(raw_dataset_path)

    config = tf.ConfigProto(device_count={'GPU': 1})
    epochs = 4000
    start_mask = 600
    end_mask = 4700
    patients = ["50483780"]  # ,'50483780']

    x_ph = tf.placeholder(tf.float32, [None, ECG_LEN])
    x = tf.reshape(x_ph, [-1, ECG_LEN, 1])

    for patient in patients:
        result_table = np.zeros((10, 4))
        i = 0
        for param in [3, 7]:
            for iteration in range(10):
                print(i)
                i += 1
                model_output = build_model(param, x)

                error_inside, error_outside = fit(model_output, start_mask, end_mask, epochs, patient, x)
                print(patient)
                print(error_inside, error_outside)

                if param == 3:
                    result_table[iteration, 0] = error_inside
                    result_table[iteration, 1] = error_outside
                else:
                    result_table[iteration, 2] = error_inside
                    result_table[iteration, 3] = error_outside

        np.savetxt('4100_'+str(patient) + '.csv', result_table, delimiter=';')
