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
log_path = "logs_cnn1b/"

plt.rcParams['figure.figsize'] = (16.0,4.0)

# filter_size_1 = 5
# filters_num_1 = 32

# filter_size_2 = 5
# filters_num_2 = 64

# fully_connected_layer_size = 256

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

x = tf.placeholder(tf.float32, shape=(None,image_size,image_size,channels_num),name='x')
y_true = tf.placeholder(tf.float32, shape=[None,10],name='y_true')
y_true_class = tf.argmax(y_true, dimension=1)
dropout_keep_prob = tf.placeholder(tf.float32,name='keep_prob')


with tf.variable_scope('Hidden_1'):
	convolution_l_1 = tf.layers.conv2d(x,filters=48,kernel_size=[5,5],padding='same')
	norm_l_1 = tf.layers.batch_normalization(convolution_l_1)
	activation_l_1 = tf.nn.relu(norm_l_1)
	pool_l_1 = tf.layers.max_pooling2d(activation_l_1,pool_size=[2,2],strides=2,padding='same')
	dropout_l_1 = tf.layers.dropout(pool_l_1,rate=dropout_keep_prob)
	output_l_1 = dropout_l_1
	print("-------------------------------------------")
	print("Output of First Hidden Layer")
	print(output_l_1)

with tf.variable_scope('Hidden_2'):
	convolution_l_2 = tf.layers.conv2d(output_l_1,filters=64,kernel_size=[5,5],padding='same')
	norm_l_2 = tf.layers.batch_normalization(convolution_l_2)
	activation_l_2 = tf.nn.relu(norm_l_2)
	pool_l_2 = tf.layers.max_pooling2d(activation_l_2,pool_size=[2,2],strides=1,padding='same')
	dropout_l_2 = tf.layers.dropout(pool_l_2,rate=dropout_keep_prob)
	output_l_2 = dropout_l_2
	print("-------------------------------------------")
	print("Output of Second Hidden Layer")
	print(output_l_2)


with tf.variable_scope('Hidden_3'):
	convolution_l_3 = tf.layers.conv2d(output_l_2,filters=128,kernel_size=[5,5],padding='same')
	norm_l_3 = tf.layers.batch_normalization(convolution_l_3)
	activation_l_3 = tf.nn.relu(norm_l_3)
	pool_l_3 = tf.layers.max_pooling2d(activation_l_3,pool_size=[2,2],strides=2,padding='same')
	dropout_l_3 = tf.layers.dropout(pool_l_3,rate=dropout_keep_prob)
	output_l_3 = dropout_l_3
	print("-------------------------------------------")
	print("Output of Third Hidden Layer")
	print(output_l_3)

with tf.variable_scope('Hidden_4'):
	convolution_l_4 = tf.layers.conv2d(output_l_3,filters=160,kernel_size=[5,5],padding='same')
	norm_l_4 = tf.layers.batch_normalization(convolution_l_4)
	activation_l_4 = tf.nn.relu(norm_l_4)
	pool_l_4 = tf.layers.max_pooling2d(activation_l_4,pool_size=[2,2],strides=1,padding='same')
	dropout_l_4 = tf.layers.dropout(pool_l_4,rate=dropout_keep_prob)
	output_l_4 = dropout_l_4
	print("-------------------------------------------")
	print("Output of Fourth Hidden Layer")
	print(output_l_1)

with tf.variable_scope('Hidden_5'):
	convolution_l_5 = tf.layers.conv2d(output_l_4,filters=192,kernel_size=[5,5],padding='same')
	norm_l_5 = tf.layers.batch_normalization(convolution_l_5)
	activation_l_5 = tf.nn.relu(norm_l_5)
	pool_l_5 = tf.layers.max_pooling2d(activation_l_5,pool_size=[2,2],strides=2,padding='same')
	dropout_l_5 = tf.layers.dropout(pool_l_5,rate=dropout_keep_prob)
	output_l_5 = dropout_l_5
	print("-------------------------------------------")
	print("Output of Fifth Hidden Layer")
	print(output_l_5)

with tf.variable_scope('Hidden_6'):
	convolution_l_6 = tf.layers.conv2d(output_l_5,filters=192,kernel_size=[5,5],padding='same')
	norm_l_6 = tf.layers.batch_normalization(convolution_l_6)
	activation_l_6 = tf.nn.relu(norm_l_6)
	pool_l_6 = tf.layers.max_pooling2d(activation_l_6,pool_size=[2,2],strides=1,padding='same')
	dropout_l_6 = tf.layers.dropout(pool_l_6,rate=dropout_keep_prob)
	output_l_6 = dropout_l_6
	print("-------------------------------------------")
	print("Output of First Hidden Layer")
	print(output_l_6)

# with tf.variable_scope('Hidden_7')
# 	convolution_l_7 = tf.layers.conv2d(output_l_6,filters=192,kernel_size=[5,5],padding='same')
# 	norm_l_7 = tf.layers.batch_normalization(convolution_l_7)
# 	activation_l_7 = tf.nn.relu(norm_l_7)
# 	pool_l_7 = tf.layers.max_pooling2d(activation_l_7,pool_size=[2,2],strides=2,padding='same')
# 	dropout_l_7 = tf.layers.dropout(pool_l_7,rate=dropout_keep_prob)
# 	output_l_7 = dropout_l_7
# 	print("-------------------------------------------")
# 	print("Output of First Hidden Layer")
# 	print(output_l_7)

# with tf.variable_scope('Hidden_8')
# 	convolution_l_8 = tf.layers.conv2d(output_l_7,filters=192,kernel_size=[5,5],padding='same')
# 	norm_l_8 = tf.layers.batch_normalization(convolution_l_8)
# 	activation_l_8 = tf.nn.relu(norm_l_8)
# 	pool_l_8 = tf.layers.max_pooling2d(activation_l_8,pool_size=[2,2],strides=1,padding='same')
# 	dropout_l_8 = tf.layers.dropout(pool_l_8,rate=dropout_keep_prob)
# 	output_l_8 = dropout_l_8
# 	print("-------------------------------------------")
# 	print("Output of First Hidden Layer")
# 	print(output_l_8)

flatten = tf.reshape(output_l_6, [-1, 4 * 4 * 192])

with tf.variable_scope('Hidden_7'):
    dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
    output_7 = dense

with tf.variable_scope('Hidden_8'):
    dense = tf.layers.dense(output_7, units=3072, activation=tf.nn.relu)
    output_8 = dense

with tf.variable_scope('Digit'):
    dense = tf.layers.dense(output_8, units=10)
    digit = dense


with tf.name_scope('SOFTMAX'):
	y_pred = tf.nn.softmax(digit)

y_pred_class = tf.argmax(y_pred,dimension=1)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit,labels=y_true)
# cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('CROSS_ENTROPY'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit,labels=y_true))

global_step = tf.Variable(0,name='global_step',trainable=False)
learning_rate = tf.train.exponential_decay(1e-2,global_step,10000,0.9,staircase=True)

with tf.name_scope('TRAIN'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

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
save_path = os.path.join(save_directory,'SVHN_32x32_Grayscale_1b')

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

# def optimize(iterations_num,display_step,epoch_num):
#     global iterations_total
#     start_time = time.time()
#     for i in range(iterations_total,iterations_total+iterations_num):
#     	batch_X,batch_true_y = next_batch(batch_size)
#         feed_dict_train = {x: batch_X, y_true: batch_true_y, dropout_keep_prob: dropout}
#         _,summary = session.run([optimizer,summary_op], feed_dict=feed_dict_train)
#         writer.add_summary(summary,epoch_num*iterations_num+i)
#         if (i-iterations_total) % display_step == 0:
#             acc = session.run(accuracy, feed_dict=feed_dict_train)
#             print "Step: {}".format(i-iterations_total)
#             print "Minibatch Accuracy: %.4f" % (acc)
#             validation_acc = session.run(accuracy, {x: val_X, y_true: val_y, dropout_keep_prob: 1.0})
#             print "Validation Accuracy: %.4f" %(validation_acc)
#             print("\n")
#     iterations_total += iterations_num
#     end_time = time.time()
#     time_difference = end_time - start_time
#     test_acc = session.run(accuracy, {x: test_X, y_true: test_y, dropout_keep_prob: 1.0})
#     print("Test Accuracy: %.4f" %(test_acc))
#     print("Time Used: "+str(timedelta(seconds=int(round(time_difference)))))
#     print("\n")

# print("Checking out the model")
# print("For 5000 iterations displaying accuracies after every 500 iterations:\n")
# optimize(iterations_num=5000,display_step=500)
# print("\n")

# print("Checking out the model")
# z1 = (s/64)+1
# z2 = ((s/64)+1)/20
# print("For {} iterations displaying accuracies after every {} iterations:\n".format(z1,z2))
# epochs_num = 1
# for e in range(epochs_num):
# 	iterations_total = 0
# 	print("------------------------------------\n\nEpoch {}\n".format(e+1))
# 	optimize(iterations_num=z1,display_step=z2,epoch_num=e)
# print("\n")

# saver.save(sess=session,save_path=save_path)
saver.restore(sess=session,save_path=save_path)

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