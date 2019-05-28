import matplotlib.pyplot as plt
import numpy as np
import json
import random as rd
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.parallel_for.gradients import jacobian
raw_dataset_path="ecg_data_200\\ecg_data_200.json" #файл с датасетом
def Openf(link):
    # function for opening some json files
    f = open(link,'r')
    data = json.load(f)
    return data


def Create_mask(size, batch_size):
	SIZE = 5000
	RES = []
	start = rd.randint(0, SIZE-size)
	res0 = np.zeros(size)
	res1 = np.ones(start)
	RES_0 = np.append(np.append(res1,res0), np.ones(SIZE - size - start))
	for i in range(batch_size):
		RES.append(RES_0)
	RESULT = np.reshape(np.array(RES), (1,SIZE,1))
	return (RESULT,start)

def Create_mask_static(start,size, batch_size):
	SIZE = 5000
	RES = []
	res0 = np.zeros(size)
	res1 = np.ones(start)
	RES_0 = np.append(np.append(res1,res0), np.ones(SIZE - size - start))
	for i in range(batch_size):
		RES.append(RES_0)
	RESULT = np.reshape(np.array(RES), (1,SIZE,1))
	return (RESULT,start)

def Graph_Jakobian(Jakobian):
	# а - количество тензорных ядер
	# b - размер маски 
	# c - размер тензоного ядра
	a,b,c = np.shape(Jakobian)
	arr_y = []
	for i in range(a):
		res = 0
		for j in range(b):
			for k in range(c):
				res +=abs(Jakobian[i][j][k])
		arr_y.append(res)
	return arr_y


data = Openf(raw_dataset_path)
Leads1= data["50519553"]["Leads"]

def MyGeteratorData(batch_size):
	f = True
	# Leads1 = data[str(rd.sample(data.keys(), 1)[0])]["Leads"]
	v6Data = Leads1["v6"]["Signal"]
	while f:
		RES = []
		for i in range(batch_size):
			res = v6Data.copy()
			RES.append(res)
		yield np.array(RES)



#########################################
## MODEL
#########################################
size_of_data = 5000
batch_size = 1  # how many images to use together for training
x_plac = tf.placeholder(tf.float32, [None, size_of_data])  # input data
x = tf.reshape(x_plac, [-1,size_of_data,1])
learning_rate = 0.01
l2_rate = 1



ae_filters = {
"conv1": tf.Variable(tf.truncated_normal([90, 1, 10], stddev=0.1)),
"b_conv1": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
"conv2": tf.Variable(tf.truncated_normal([150,10,10], stddev=0.1)),
"b_conv2": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
"conv3": tf.Variable(tf.truncated_normal([50,10,5],stddev= 0.1)),
"b_conv3": tf.Variable(tf.truncated_normal([5],stddev= 0.1)),
"conv4": tf.Variable(tf.truncated_normal([100,5,5],stddev= 0.1)),
"b_conv4": tf.Variable(tf.truncated_normal([5],stddev= 0.1)),
"conv5": tf.Variable(tf.truncated_normal([5,10,10],stddev= 0.1)),
"b_conv5": tf.Variable(tf.truncated_normal([10],stddev= 0.1)),

"deconv1": tf.Variable(tf.truncated_normal([5,10,10], stddev=0.1)),
"b_deconv1": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
"deconv2": tf.Variable(tf.truncated_normal([100, 5, 5], stddev=0.1)),
"b_deconv2": tf.Variable(tf.truncated_normal([5], stddev=0.1)),
"deconv3": tf.Variable(tf.truncated_normal([50,10,5], stddev=0.1)),
"b_deconv3":tf.Variable(tf.truncated_normal([5], stddev=0.1)),
"deconv4": tf.Variable(tf.truncated_normal([150, 10, 10], stddev=0.1)),
"b_deconv4": tf.Variable(tf.truncated_normal([10], stddev=0.1)),
"deconv5": tf.Variable(tf.truncated_normal([90,1,10], stddev=0.1)),
"b_deconv5":tf.Variable(tf.truncated_normal([10],stddev=0.1)),

}

#############################################
##LAYERS
#############################################
x_1 = tf.nn.conv1d(x, ae_filters["conv1"], stride=1, padding="SAME",use_cudnn_on_gpu=True) + ae_filters["b_conv1"]
x_1_mp = tf.layers.max_pooling1d(x_1,pool_size = 2,strides = 2,padding='SAME')
x1_relu = tf.nn.leaky_relu(x_1_mp,alpha = 0.2)
# x_1_relu = tf.nn.sigmoid(x_1_mp)
# x_1_relu = tf.keras.layers.LeakyReLU(x_1_mp)
x_2 = tf.nn.conv1d(x1_relu, ae_filters["conv2"], stride=1, padding="SAME",use_cudnn_on_gpu=True) + ae_filters["b_conv2"]
x_2_mp = tf.layers.max_pooling1d(x_2,pool_size = 35,strides = 35,padding='SAME')
x2_relu = tf.nn.leaky_relu(x_2_mp,alpha = 0.2)
# x2_relu = tf.keras.layers.LeakyReLU(x_2_mp,alpha = 0.2)
x_3 = tf.nn.conv1d(x2_relu, ae_filters["conv3"], stride=1, padding="SAME",use_cudnn_on_gpu=True) + ae_filters["b_conv3"]
x3_mp = tf.layers.max_pooling1d(x_3,pool_size = 2,strides = 2,padding='SAME')
x3_relu = tf.nn.leaky_relu(x3_mp,alpha = 0.2)
# x3_relu = tf.keras.layers.LeakyReLU(x_3_mp,alpha = 0.2)
# x_4 = tf.nn.conv1d(x3_relu, ae_filters["conv4"], stride=1, padding="SAME",use_cudnn_on_gpu=True) + ae_filters["b_conv4"]
# x4_relu = tf.nn.leaky_relu(x_4,alpha = 0.2)
# x_5 = tf.nn.conv1d(x4_relu, ae_filters["conv5"], stride=1, padding="SAME",use_cudnn_on_gpu=True) + ae_filters["b_conv5"]
# x5_relu = tf.nn.leaky_relu(x_5,alpha = 0.2)


x_dec1 = tf.contrib.nn.conv1d_transpose(x3_relu, ae_filters["deconv3"], [batch_size,72,10], stride=2, padding="SAME")
x_dec1_relu = tf.nn.leaky_relu(x_dec1,alpha = 0.2)
# x_dec1_relu = tf.keras.layers.LeakyReLU(x_dec1,alpha = 0.2)
x_dec2 = tf.contrib.nn.conv1d_transpose(x_dec1_relu, ae_filters["deconv4"], [batch_size,2500,10], stride=35, padding="SAME")
x_dec2_relu = tf.nn.leaky_relu(x_dec2,alpha = 0.2)
# x_dec2_relu = tf.keras.layers.LeakyReLU(x_dec2,alpha = 0.2)
x_dec3 = tf.contrib.nn.conv1d_transpose(x_dec2_relu, ae_filters["deconv5"], [batch_size,size_of_data,1], stride=2, padding="SAME")
x_dec3_relu = tf.nn.leaky_relu(x_dec3,alpha = 0.2)
# x_dec3_relu = tf.keras.layers.LeakyReLU(x_dec3,alpha = 0.2)
# x_dec4 = tf.contrib.nn.conv1d_transpose(x_dec3_relu, ae_filters["deconv5"], [batch_size,size_of_data,1], stride=1, padding="SAME")
# x_dec4_relu = tf.nn.leaky_relu(x_dec4,alpha = 0.2)
# x_dec5 = tf.contrib.nn.conv1d_transpose(x_dec4_relu, ae_filters["deconv5"], [batch_size,size_of_data,1], stride=1, padding="SAME")
# x_dec5_relu = tf.nn.leaky_relu(x_dec5,alpha = 0.2)




# J=tf.gradients(x_dec3_relu[0][0][0],ae_filters["conv1"])



############################################
##REGULARIZATION
############################################

vars   = tf.trainable_variables() 
loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])*l2_rate

for i  in range(1):
#################################################
##FUNC of ERROR
##################################################
	size_of_mask = 400
	mask,start_mask = Create_mask_static(560,size_of_mask,batch_size)
	func_of_error = tf.reduce_mean(tf.square(x_dec3_relu -x)*mask) + loss_l2
	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(func_of_error)

	config = tf.ConfigProto(device_count = {'GPU': 1})  # swich between GPU and CPU 

	################################################
	# LEARNING
	################################################
	init = tf.global_variables_initializer()
	sess = tf.Session(config=config)
	sess.run(init)
	ae_filters_copy = sess.run(ae_filters.copy()) # copy for comparing
	# defining batch size, number of epochs and learning rate
	hm_epochs =4500   # how many times to go through the entire dataset
	count = 1 # total request in one epoch
	epox = [-2,-1]
	error = [100000000000,100000000]
	N=0
	# for epoch in range(hm_epochs):
	while (abs(error[-1]-error[-2])>0.000000000000001 and N<hm_epochs):
		epoch_loss_current = 0    # initializing error as 0
		for i in range(count):
			epoch_x = next(MyGeteratorData(batch_size)) #epoch_x теперь должен содержать картинки
			_, c = sess.run([optimizer, func_of_error],feed_dict={x_plac: epoch_x})
			epoch_loss_current += c
		print('Epoch', N, '/', hm_epochs, 'loss:',epoch_loss_current/count)
		error.append(epoch_loss_current/count)
		epox.append(N)
		N = N + 1
	#################################################################
	##SAVER
	#################################################################
	# saver = tf.train.Saver()
	# save_path = saver.save(sess, "AutoincoderSave3/model_autoincoderCNN.ckpt")
	# print("Model saved in path: %s" % save_path)

	def model_summary():
		model_vars = tf.trainable_variables()
		slim.model_analyzer.analyze_vars(model_vars, print_info=True)


	#####################################
	##VISUALIZATION
	#####################################
	model_summary()
	print("___________________________________________________")

	data1 = next(MyGeteratorData(batch_size))
	plt.subplot(3, 1, 1)
	plt.plot(range(size_of_data), data1[0])
	data_output= sess.run([x_dec3_relu], feed_dict={x_plac: data1})
	plt.subplot(3, 1, 2)
	data_output = np.reshape(data_output, [batch_size, size_of_data])
	plt.plot(range(size_of_data), data_output[0])
	plt.plot(range(start_mask,size_of_mask+start_mask), np.zeros(size_of_mask)-300,'ro',markersize = 3)
	plt.subplot(3,1,3)
	plt.xlabel(r'$epoch$', fontsize=15, horizontalalignment='right' , x=1)
	plt.ylabel(r'$Error$', fontsize=15, horizontalalignment='right', y=1)
	plt.plot(epox[2:-1], error[2:-1],"g", label = "Error")
	plt.legend()
	plt.savefig("ecg_photo/vv" +"_"+str(size_of_mask)+ ".png")
	plt.show()
	plt.clf()


	###########################################
	##GET Jakobian
	###########################################

	J_Jak = np.zeros((10,1,90)) # the first fictive array for concatenate 
	for i in range(start_mask,start_mask+size_of_mask):
		dfi_dx = tf.gradients(x_dec3_relu[0][i][0],ae_filters["conv1"])
		J = sess.run([dfi_dx], feed_dict={x_plac: data1})
		J = np.transpose(J[0][0])
		J_Jak = np.append(J_Jak,J, axis = 1)
		print(i-start_mask,"\t/",size_of_mask)
	y_arr = Graph_Jakobian(Jakobian =J_Jak)
	print("Y_ARR",y_arr)
	plt.plot(range(1,11),y_arr)
	plt.show()



