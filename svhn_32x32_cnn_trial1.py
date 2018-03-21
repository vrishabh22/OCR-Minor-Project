import h5py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import timedelta
import time
import math
from sklearn.metrics import confusion_matrix
from random import shuffle

tf.reset_default_graph()
log_path = "logs_cnn2a/"

plt.rcParams['figure.figsize'] = (16.0,4.0)

filter_size_1 = 5
filters_num_1 = 32

filter_size_2 = 5
filters_num_2 = 64

fully_connected_layer_size = 256

file = h5py.File('data/SVHN_32x32_Grayscale.h5','r')

train_X = file['Training Set Images'][:]
train_y = file['Training Set Labels'][:]
test_X = file['Test Set Images'][:]
test_y = file['Test Set Labels'][:]
val_X = file['Validation Set Images'][:]
val_y = file['Validation Set Labels'][:]

file.close()

print("-----------------------------------")
print("Dimensions of Dataset : ")
print("Training Set : Images : ",train_X.shape," Labels : ",train_y.shape)
s = train_X.shape[0]
print("Test Set : Images : ",test_X.shape," Labels : ",test_y.shape)
print("Validation Set : Images : ",val_X.shape," Labels : ",val_y.shape)

image_size = train_X.shape[1]
channels_num = train_X.shape[-1]
classes_num = train_y.shape[1]

train_mean = np.mean(train_X,axis=0)
train_standard_dev = np.std(train_X,axis=0)

train_X = (train_X - train_mean) / train_standard_dev
test_X = (test_X - train_mean) / train_standard_dev
val_X = (val_X - train_mean) / train_standard_dev

def plot_images(imgs,n,true_classes,pred_classes=None):
	for i in range(n):
		plt.imshow(imgs[i,:,:,0],cmap=plt.cm.binary)
		if pred_classes is None:
			plt.title("True : {}".format(np.argmax(true_classes[i])))
		else:
			plt.title("True : {} Predicted : {}".format(np.argmax(true_classes[i]),pred_classes[i]))
		plt.show()

# plot_images(train_X,5,train_y)

def convolution_weight_var(layer_name,shape):
	with tf.name_scope('Conv_Weights'):
		return tf.get_variable(layer_name,shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())

def fully_connected_weight_var(layer_name,shape):
	with tf.name_scope('FC_Weights'):
		return tf.get_variable(layer_name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())

def bias_var(shape):
	with tf.name_scope('Biases'):
		return tf.Variable(tf.constant(0.0, shape=shape))	

def convolution_layer(input,layer_name,input_channels_num,filter_size,filters_num,use_pooling=True):
	shape = [filter_size,filter_size,input_channels_num,filters_num]
	weights = convolution_weight_var(layer_name,shape=shape)
	biases = bias_var(shape=[filters_num])
	with tf.name_scope('CONV'):
		layer = tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
		layer += biases
	if use_pooling:
		with tf.name_scope('POOL'):
			layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	with tf.name_scope('RELU'):
		layer = tf.nn.relu(layer)
	return layer,weights

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	features_num = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer,[-1,features_num])
	return layer_flat,features_num

def fully_connected_layer(input,layer_name,inputs_num,outputs_num,use_relu=True):
	weights = fully_connected_weight_var(layer_name,shape=[inputs_num,outputs_num])
	biases = bias_var(shape=[outputs_num])
	with tf.name_scope('FC'):
		layer = tf.matmul(input,weights)+biases
	if use_relu:
		with tf.name_scope('RELU'):
			layer = tf.nn.relu(layer)
	return layer

x = tf.placeholder(tf.float32, shape=(None,image_size,image_size,channels_num),name='x')
y_true = tf.placeholder(tf.float32, shape=[None,10],name='y_true')
y_true_class = tf.argmax(y_true, dimension=1)
dropout_keep_prob = tf.placeholder(tf.float32,name='keep_prob')

# Our Architecture
# INPUT -> [CONV -> RELU -> POOL -> CONV -> RELU -> POOL] -> (FLATTEN) -> FC -> RELU -> FC -> SOFTMAX
convolution_l_1,weights_convolution_l_1 = convolution_layer(input=x,layer_name="Convolution_1",input_channels_num=channels_num,filter_size=filter_size_1,filters_num=filters_num_1,use_pooling=True)
print("-------------------------------------------")
print("Output of First CONV-RELU_POOL")
print(convolution_l_1)

convolution_l_2,weights_convolution_l_2 = convolution_layer(input=convolution_l_1,layer_name="Convolution_2",input_channels_num=filters_num_1,filter_size=filter_size_2,filters_num=filters_num_2,use_pooling=True)
print("-------------------------------------------")
print("Output of Second CONV-RELU_POOL")
print(convolution_l_2)

# with tf.name_scope('DROPOUT'):
# 	dropout = tf.nn.dropout(convolution_l_2,dropout_keep_prob)
# print("-------------------------------------------")
# print("Output of Dropout")
# print(dropout)

layer_flat,features_num = flatten_layer(convolution_l_2)
print("-------------------------------------------")
print("Output of Flattening")
print(layer_flat)

fully_connected_l_1 = fully_connected_layer(input=layer_flat,layer_name="Fully_Connected_1",inputs_num=features_num,outputs_num=fully_connected_layer_size,use_relu=True)
print("-------------------------------------------")
print("Output of First Fully Connected Layer")
print(fully_connected_l_1)

fully_connected_l_2 = fully_connected_layer(input=fully_connected_l_1,layer_name="Fully_Connected_2",inputs_num=fully_connected_layer_size,outputs_num=classes_num,use_relu=False)
print("-------------------------------------------")
print("Output of Second Fully Connected Layer")
print(fully_connected_l_2)

with tf.name_scope('SOFTMAX'):
	y_pred = tf.nn.softmax(fully_connected_l_2)
y_pred_class = tf.argmax(y_pred,dimension=1)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fully_connected_l_2,labels=y_true)
# cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('CROSS_ENTROPY'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fully_connected_l_2,labels=y_true))

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05,global_step,10000,0.96,staircase=True)
with tf.name_scope('TRAIN'):
	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost,global_step=global_step)

correct_pred = tf.equal(y_pred_class,y_true_class)
with tf.name_scope('ACC'):
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

tf.summary.scalar("COST",cost)
tf.summary.scalar("ACCURACY",accuracy)

summary_op = tf.summary.merge_all()

session = tf.Session()
session.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())

saver = tf.train.Saver()
save_directory = 'checkpoints/'
save_path = os.path.join(save_directory,'SVHN_32x32_Grayscale_2a')

def next_batch(batch_size) :
	global count 
	if(count+batch_size>len(train_X)):
		count=0

	batch_X=train_X[count:count+batch_size,:]
	batch_y=train_y[count:count+batch_size,:]
	

	count+=batch_size

	return batch_X,batch_y

batch_size = 64
dropout = 0.5
count = 0
iterations_total = 0

print("---------------------------------")
print("\n\n")

def optimize(iterations_num,display_step,epoch_num):
    global iterations_total
    start_time = time.time()
    for i in range(iterations_total,iterations_total+iterations_num):
    	batch_X,batch_true_y = next_batch(batch_size)
        feed_dict_train = {x: batch_X, y_true: batch_true_y, dropout_keep_prob: dropout}
        _,summary = session.run([optimizer,summary_op], feed_dict=feed_dict_train)
        writer.add_summary(summary,epoch_num*iterations_num+i)
        if (i-iterations_total) % display_step == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            print "Step: {}".format(i-iterations_total)
            print "Minibatch Accuracy: %.4f" % (acc)
            validation_acc = session.run(accuracy, {x: val_X, y_true: val_y, dropout_keep_prob: 1.0})
            print "Validation Accuracy: %.4f" %(validation_acc)
            print("\n")
    iterations_total += iterations_num
    end_time = time.time()
    time_difference = end_time - start_time
    test_acc = session.run(accuracy, {x: test_X, y_true: test_y, dropout_keep_prob: 1.0})
    print("Test Accuracy: %.4f" %(test_acc))
    print("Time Used: "+str(timedelta(seconds=int(round(time_difference)))))
    print("\n")

print("\n-------------------------------------------------------")
print("Checking out the model")
z1 = (s/64)+1
z2 = ((s/64)+1)/5
print("For {} iterations displaying accuracies after every {} iterations:\n".format(z1,z2))
epochs_num = 12
for e in range(epochs_num):
	iterations_total = 0
	print("------------------------------------\n\nEpoch {}\n".format(e+1))
	optimize(iterations_num=z1,display_step=z2,epoch_num=e+1)

print("\n")
# print("------------------------------------------------------")
# print("For 50000 iterations displaying accuracies after every 1000 iterations:\n")
# optimize(iterations_num=50000,display_step=1000)
# print("\n")
# print("------------------------------------------------------")

saver.save(sess=session,save_path=save_path)

test_predictions = session.run(y_pred_class, {x: test_X, y_true: test_y, dropout_keep_prob: 1.0})

wrong = test_predictions != np.argmax(test_y,axis=1)
right = test_predictions == np.argmax(test_y,axis=1)

print("----------------------------------------")
print("Number of Predictions on Test Set : {}".format(test_predictions.size))
print("Number of Rightly Classified Images : {}".format(right.sum()))
print("Number of Wrongly Classified Images : {}".format(wrong.sum()))

# print("---------------------------------------")
# print("Showing Predictions of 5 images")
# plot_images(test_X,5,test_y,test_predictions)

print("-------------------------------------------------")
print("Accuracy of Test Set: {}".format(float(np.sum(right))/(test_predictions.size)))

plt.figure(figsize=(12,8))

confusion_mat = confusion_matrix(y_true=np.argmax(test_y,axis=1),y_pred=test_predictions)
confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1) [:,np.newaxis] * 100
print("-------------------------------------")
print("Confusion Matrix:")
# print(map(float,confusion_mat))
cm = []
for row in confusion_mat:
	r = map(float,row)
	r_2 = [ round(elem,3) for elem in r ]
	cm.append(r_2)
print(cm)