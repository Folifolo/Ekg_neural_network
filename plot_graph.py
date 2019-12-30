import json
import random as rd

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

raw_dataset_path = "data_1078.json"  # файл с датасетом


def Openf(link):
    # function for opening some json files
    f = open(link, 'r')
    data = json.load(f)
    return data


def Create_mask(size, batch_size):
    SIZE = 5000
    RES = []
    start = rd.randint(0, SIZE - size)
    res0 = np.zeros(size)
    res1 = np.ones(start)
    RES_0 = np.append(np.append(res1, res0), np.ones(SIZE - size - start))
    for i in range(batch_size):
        RES.append(RES_0)
    RESULT = np.reshape(np.array(RES), (1, SIZE, 1))
    return (RESULT, start)


def Create_mask_static(start, size, batch_size):
    SIZE = 5000
    RES = []
    res0 = np.zeros(size)
    res1 = np.ones(start)
    RES_0 = np.append(np.append(res1, res0), np.ones(SIZE - size - start))
    for i in range(batch_size):
        RES.append(RES_0)
    RESULT = np.reshape(np.array(RES), (1, SIZE, 1))
    return (RESULT, start)


data = Openf(raw_dataset_path)



# Leads1= data["50488354"]["Leads"]

def MyGeteratorData(batch_size, patient):
    f = True
    v6Data = data[patient]["Leads"]["v6"]["Signal"]
    while f:
        RES = []
        for i in range(batch_size):
            res = v6Data.copy()
            RES.append(res)
        yield np.array(RES)


def get_mask_inverse(mask, SIZE):
    mask = np.reshape(mask[0], size_of_data)
    res = []
    for i in mask:
        if i == 0:
            res.append(1.0)
        else:
            res.append(0.0)
    return np.reshape(np.array(res), (1, SIZE, 1))


#########################################
## MODEL
#########################################
size_of_data = 5000
batch_size = 1  # how many images to use together for training
# tf.disable_eager_execution()
x_plac = tf.placeholder(tf.float32, [None, size_of_data])  # input data
x = tf.reshape(x_plac, [-1, size_of_data, 1])
learning_rate = 0.01
l2_rate = 1

ae_filters = {
    "conv1": tf.Variable(tf.truncated_normal([100, 1, 10], stddev=0.1)),
    "b_conv1": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
    "conv2": tf.Variable(tf.truncated_normal([80, 10, 10], stddev=0.1)),
    "b_conv2": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
    "conv3": tf.Variable(tf.truncated_normal([50, 10, 5], stddev=0.1)),
    "b_conv3": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
    "conv4": tf.Variable(tf.truncated_normal([100, 5, 5], stddev=0.1)),
    "b_conv4": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
    "conv5": tf.Variable(tf.truncated_normal([5, 10, 10], stddev=0.1)),
    "b_conv5": tf.Variable(tf.truncated_normal([10], stddev=0.1)),

    "deconv1": tf.Variable(tf.truncated_normal([5, 10, 10], stddev=0.1)),
    "b_deconv1": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
    "deconv2": tf.Variable(tf.truncated_normal([100, 5, 5], stddev=0.1)),
    "b_deconv2": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
    "deconv3": tf.Variable(tf.truncated_normal([50, 10, 5], stddev=0.1)),
    "b_deconv3": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
    "deconv4": tf.Variable(tf.truncated_normal([80, 10, 10], stddev=0.1)),
    "b_deconv4": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
    "deconv5": tf.Variable(tf.truncated_normal([100, 1, 10], stddev=0.1)),
    "b_deconv5": tf.Variable(tf.truncated_normal([10], stddev=0.1)),

}

latent_pool_size = 3


def set_architecture(latent_pool_size):
    #############################################
    ##LAYERS
    #############################################
    x_1 = tf.nn.conv1d(x, ae_filters["conv1"], stride=1, padding="SAME", use_cudnn_on_gpu=True) + ae_filters["b_conv1"]
    x_1_mp = tf.layers.max_pooling1d(x_1, pool_size=2, strides=2, padding='SAME')
    x1_relu = tf.nn.leaky_relu(x_1_mp, alpha=0.2)
    x_2 = tf.nn.conv1d(x1_relu, ae_filters["conv2"], stride=1, padding="SAME", use_cudnn_on_gpu=True) + ae_filters[
        "b_conv2"]
    x_2_mp = tf.layers.max_pooling1d(x_2, pool_size=latent_pool_size, strides=latent_pool_size, padding='SAME')
    x2_relu = tf.nn.leaky_relu(x_2_mp, alpha=0.2)
    x_3 = tf.nn.conv1d(x2_relu, ae_filters["conv3"], stride=1, padding="SAME", use_cudnn_on_gpu=True) + ae_filters[
        "b_conv3"]
    x3_mp = tf.layers.max_pooling1d(x_3, pool_size=2, strides=2, padding='SAME')
    x3_relu = tf.nn.leaky_relu(x3_mp, alpha=0.2)

    x_dec1 = tf.contrib.nn.conv1d_transpose(x3_relu, ae_filters["deconv3"],
                                            [batch_size, size_of_data // 2 // latent_pool_size + 1, 10], strides=2,
                                            padding="SAME")
    x_dec1_relu = tf.nn.leaky_relu(x_dec1, alpha=0.2)
    x_dec2 = tf.contrib.nn.conv1d_transpose(x_dec1_relu, ae_filters["deconv4"], [batch_size, 2500, 10],
                                            strides=latent_pool_size, padding="SAME")
    x_dec2_relu = tf.nn.leaky_relu(x_dec2, alpha=0.2)
    x_dec3 = tf.contrib.nn.conv1d_transpose(x_dec2_relu, ae_filters["deconv5"], [batch_size, size_of_data, 1],
                                            strides=2, padding="SAME")
    x_dec3_relu = tf.nn.leaky_relu(x_dec3, alpha=0.2)
    return x_1, x_dec3_relu


x_1, x_dec3_relu = set_architecture(7)

config = tf.ConfigProto(device_count={'GPU': 1})  # swich between GPU and CPU
vars = tf.trainable_variables()
loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_rate

size_of_mask = 4500
mask, start_mask = Create_mask_static(200, size_of_mask, batch_size)
mask_inverse = get_mask_inverse(mask, size_of_data)
func_of_error = tf.reduce_mean(tf.square(x_dec3_relu - x) * mask) + loss_l2
func_of_error_without_l2 = tf.reduce_mean(tf.square(x_dec3_relu - x) * mask)
func_of_error_inside = tf.reduce_mean(tf.square(x_dec3_relu - x) * mask_inverse)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(func_of_error)

init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)
# ae_filters_copy = sess.run(ae_filters.copy()) # copy for comparing
# defining batch size, number of epochs and learning rate
epox = [-2, -1]
error = [100000000000, 100000000]
error_inside_arr = []
error_without_l2_on_epoch_arr = []
N = 0
patient = '50679809'
generator = MyGeteratorData(batch_size, patient)
# for epoch in range(hm_epochs):
count = 1
while (abs(error[-1] - error[-2]) > 0.000000001 and N < 4000):
    epoch_loss_current = 0  # initializing error as 0
    epoch_loss_current_inside = 0
    error_without_l2_on_epoch = 0
    for i in range(count):
        epoch_x = next(generator)  # epoch_x теперь должен содержать картинки
        _, c, error_inside, error_without_l2 = sess.run(
            [optimizer, func_of_error, func_of_error_inside, func_of_error_without_l2], feed_dict={x_plac: epoch_x})
        epoch_loss_current += c
        epoch_loss_current_inside += error_inside / count
        error_without_l2_on_epoch += error_without_l2 / count
    # print('Epoch', N, '/', hm_epochs, 'inside_loss:',epoch_loss_current_inside,'\t' ,' outside_loss ',error_without_l2_on_epoch )
    # print('Epoch', N, '/', hm_epochs, 'loss:',epoch_loss_current/count)
    error_inside_arr.append(epoch_loss_current_inside)
    error_without_l2_on_epoch_arr.append(error_without_l2_on_epoch)
    error.append(epoch_loss_current / count)
    epox.append(N)
    N = N + 1
print(' done')


data1 = next(MyGeteratorData(batch_size, patient))
gridspec.GridSpec(3, 2)
plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=1)
plt.title("loss out of zone " + str(error_without_l2_on_epoch) + ' ' + "loss inside " + str(epoch_loss_current_inside))
plt.plot(range(size_of_data), data1[0])
data_output, Jakob = sess.run([x_dec3_relu, x_1], feed_dict={x_plac: data1})
print("Jakob_shape", np.shape(Jakob))
plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=1)
data_output = np.reshape(data_output, [batch_size, size_of_data])

plt.plot(range(size_of_data), data_output[0])
plt.plot(range(start_mask, size_of_mask + start_mask), np.zeros(size_of_mask) - 300, 'ro', markersize=3)
plt.subplot2grid((3, 2), (2, 0))
plt.xlabel(r'$epoch$', fontsize=15, horizontalalignment='right', x=1)
plt.ylabel(r'$Error$', fontsize=15, horizontalalignment='right', y=1)
plt.plot(error_inside_arr[50:], "g", label="Error inside", color='g')
plt.plot(error_without_l2_on_epoch_arr[50:], "g", label="Error outside ", color='b')
plt.legend()
plt.subplot2grid((3, 2), (2, 1))
plt.xlabel(r'$epoch$', fontsize=15, horizontalalignment='right', x=1)
plt.ylabel(r'$Error$', fontsize=15, horizontalalignment='right', y=1)
plt.plot(error_inside_arr[:50], "g", label="Error inside", color='g')
plt.plot(error_without_l2_on_epoch_arr[:50], "g", label="Error outside ", color='b')
plt.legend()
plt.show()
plt.clf()
