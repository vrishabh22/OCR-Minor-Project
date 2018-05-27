from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

plt.rcParams['figure.figsize'] = (16.0, 4.0)
h5f = h5py.File('data/SVHN_multi_grey.h5','r')

X_train = h5f['train_dataset'][:]
y_train = h5f['train_labels'][:]
X_val = h5f['valid_dataset'][:]
y_val = h5f['valid_labels'][:]
X_test = h5f['test_dataset'][:]
y_test = h5f['test_labels'][:]

h5f.close()

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

_,img_height, img_width, num_channels = X_train.shape

num_digits, num_labels = y_train.shape[1], len(np.unique(y_train))

X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])

print('Training set', X_train.shape, y_train.shape)


from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

def subtract_mean(a):
    for i in range(a.shape[0]):
        a[i] -= a[i].mean()
    return a

X_train = subtract_mean(X_train)
X_test = subtract_mean(X_test)
X_val = subtract_mean(X_val)

def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2*nrows))
    rs = np.random.choice(images.shape[0], nrows*ncols)
    for i, ax in zip(rs, axes.flat): 
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)
        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number) 
        ax.imshow(images[i,:,:,0], cmap='binary')
        ax.set_title(title)   
        ax.set_xticks([]); ax.set_yticks([])
        
        
plot_images(X_train, 2, 8, y_train)
plt.show()

def init_conv_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def init_fc_weights(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def init_biases(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


    #--------------------------------------MAIN MODEL----------------------------------
def conv_layer(input_tensor,    
                filter_size,   
                in_channels,    
                num_filters,    
                layer_name,     
                pooling,pooling_stride):       
    
    with tf.variable_scope(layer_name) as scope:
        shape = [filter_size, filter_size, in_channels, num_filters]
        weights = init_conv_weights(shape, layer_name)
        biases = init_biases([num_filters])
        tf.summary.histogram(layer_name + '/weights_6layers', weights)
        activations = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], 'SAME') + biases
        activations = tf.layers.batch_normalization(activations)
        activations = tf.nn.relu(activations)
        if pooling_stride:
            activations = tf.nn.max_pool(activations, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
        elif pooling:
            activations = tf.nn.max_pool(activations, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        return activations

def flatten_tensor(input_tensor):
    input_tensor_shape = input_tensor.get_shape()
    num_activations = input_tensor_shape[1:4].num_elements()
    input_tensor_flat = tf.reshape(input_tensor, [-1, num_activations])
    return input_tensor_flat, num_activations

def fc_layer(input_tensor,          
             input_dim,     
             output_dim,    
             layer_name,    
             relu=False):         

    with tf.variable_scope(layer_name):
        weights = init_fc_weights([input_dim, output_dim], layer_name + '/weights_6layers')
        biases = init_biases([output_dim])
        tf.summary.histogram(layer_name + '/weights_6layers', weights)
        activations = tf.matmul(input_tensor, weights) + biases
        if relu:
            activations = tf.nn.relu(activations)
        return activations

filter_size1 = filter_size2 = 5          
num_filters1 = num_filters2 = 32        

filter_size3 = filter_size4 = 5          
num_filters3 = num_filters4 = 64

filter_size5 = filter_size6 = 5          
num_filters5 = num_filters6 = 128 

filter_size7 = filter_size8 = 5          
num_filters7 = num_filters8 = 256  
fc1_size = fc2_size= 256


with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, num_channels), name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, num_digits], name='y_')

with tf.variable_scope("dropout"):
    p_keep_3 = tf.placeholder(tf.float32)
    tf.summary.scalar('input_keep_probability', p_keep_3)


conv_1 = conv_layer(x, filter_size1, num_channels, num_filters1, "conv1", pooling=True,pooling_stride=False)
drop_block1 = tf.nn.dropout(conv_1, p_keep_3)
conv_2 = conv_layer(drop_block1, filter_size2, num_filters1, num_filters2, "conv2", pooling=False,pooling_stride=True)
drop_block2 = tf.nn.dropout(conv_2, p_keep_3) 

conv_3 = conv_layer(drop_block2, filter_size3, num_filters2, num_filters3, "conv3", pooling=True,pooling_stride=False)
drop_block3 = tf.nn.dropout(conv_3, p_keep_3)
conv_4 = conv_layer(drop_block3, filter_size4, num_filters3, num_filters4, "conv4", pooling=False,pooling_stride=True)
drop_block4 = tf.nn.dropout(conv_4, p_keep_3) 

conv_5 = conv_layer(drop_block4, filter_size5, num_filters4, num_filters5, "conv5", pooling=True,pooling_stride=False)
drop_block5 = tf.nn.dropout(conv_5, p_keep_3)
conv_6 = conv_layer(drop_block5, filter_size6, num_filters5, num_filters6, "conv6", pooling=False,pooling_stride=True)
drop_block6 = tf.nn.dropout(conv_6, p_keep_3)

conv_7 = conv_layer(drop_block6, filter_size7, num_filters6, num_filters7, "conv7", pooling=True,pooling_stride=False)
drop_block7 = tf.nn.dropout(conv_7, p_keep_3)
conv_8 = conv_layer(drop_block7, filter_size8, num_filters7, num_filters8, "conv8", pooling=False,pooling_stride=True)

flat_tensor, num_activations = flatten_tensor(tf.nn.dropout(conv_8, p_keep_3)) 

fc_1 = fc_layer(flat_tensor, num_activations, fc1_size, 'fc1', relu=True)
fc_2 = fc_layer(fc_1, fc1_size, fc2_size, 'fc2', relu=True)

logits_1 = fc_layer(fc_2, fc2_size, num_labels, 'softmax1')
logits_2 = fc_layer(fc_2, fc2_size, num_labels, 'softmax2')
logits_3 = fc_layer(fc_2, fc2_size, num_labels, 'softmax3')
logits_4 = fc_layer(fc_2, fc2_size, num_labels, 'softmax4')
logits_5 = fc_layer(fc_2, fc2_size, num_labels, 'softmax5')

y_pred = tf.stack([logits_1, logits_2, logits_3, logits_4, logits_5])
y_pred_cls = tf.transpose(tf.argmax(y_pred, dimension=2))



with tf.variable_scope("loss"):
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=y_[:, 0]))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=y_[:, 1]))
    loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=y_[:, 2]))
    loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=y_[:, 3]))
    loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=y_[:, 4]))
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    tf.summary.scalar('loss', loss)

with tf.variable_scope('optimizer'):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1e-3, global_step, 7500, 0.5, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.variable_scope('accuracy'):
    correct_prediction = tf.reduce_min(tf.cast(tf.equal(y_pred_cls, y_), tf.float32), 1)
    accuracy = tf.reduce_mean(correct_prediction) * 100.0
    tf.summary.scalar('accuracy', accuracy)

session = tf.Session()
saver = tf.train.Saver()
save_path = os.path.join('checkpoints_6layers_copy3/', 'svhn_multi_v5_6layers')

try:
    print("Restoring last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir='checkpoints_6layers_copy3')
    print(last_chk_path)
    saver.restore(session, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)

except:
    print("Failed to restore checkpoint - initializing variables")
    session.run(tf.global_variables_initializer())

LOG_DIR = 'logs_6layers_copy3/'
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(LOG_DIR,graph=tf.get_default_graph())
batch_size = 128 
d3 = 0.85

def feed_dict(step=0):
    offset = (step * batch_size) % (y_train.shape[0] - batch_size)
    xs, ys = X_train[offset:offset + batch_size], y_train[offset:offset+batch_size]
    return {x: xs, y_: ys, p_keep_3: d3}


def evaluate_batch(test, batch_size):
    cumulative_accuracy = 0.0
    n_images = y_test.shape[0] if test else y_val.shape[0]
    n_batches = n_images // batch_size + 1

    for i in range(n_batches):
        offset = i * batch_size
        if test:
            xs, ys = X_test[offset:offset+batch_size], y_test[offset:offset+batch_size]
        else:
            xs, ys = X_val[offset:offset+batch_size], y_val[offset:offset+batch_size]
        cumulative_accuracy += session.run(accuracy,
                {x: xs, y_: ys, p_keep_3: 1.})
    return cumulative_accuracy / (0.0 + n_batches)

def optimize(num_iterations, display_step):
    start_time = time.time()
    for step in range(num_iterations):
        summary, i, _ = session.run([merged, global_step, optimizer], feed_dict(step))
        train_writer.add_summary(summary, i)
        if (i % display_step == 0) or (step == num_iterations - 1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict(step))
            print("Minibatch accuracy at step %d: %.4f" % (i, batch_acc))
            run_time = time.time() - start_time
            print("\nTime usage: " + str(timedelta(seconds=int(round(run_time)))))

            test_acc = evaluate_batch(test=True, batch_size=512)
            print("Test accuracy: %.4f" % test_acc)

            saver.save(session, save_path=save_path, global_step=global_step)
            print('Model saved in file: {}'.format(save_path))

num_iter=int(X_train.shape[0]/batch_size)+1
# for i in range(10):
    # print(i)
optimize(num_iterations=6000, display_step=200)

