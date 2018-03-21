# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

plt.rcParams['figure.figsize'] = (16.0, 4.0)

def load_test_train_extra(dataset_path):
	data = loadmat(dataset_path)
	return data['X'],data['y']

train_X,train_y = load_test_train_extra('data/train_32x32.mat')
test_X,test_y = load_test_train_extra('data/test_32x32.mat')
extra_X,extra_y = load_test_train_extra('data/extra_32x32.mat')

print("-----------------------------------------")
print("Dimensions of Images in Loaded Dataset")
print("Training Set Dimensions : ",train_X.shape,train_y.shape)
print("Test Set Dimensions : ", test_X.shape,test_y.shape)
print("Extra Set Dimensions : ",extra_X.shape,extra_y.shape)

train_X, train_y = train_X.transpose((3,0,1,2)),train_y[:,0]
test_X,test_y = test_X.transpose((3,0,1,2)),test_y[:,0]
extra_X,extra_y = extra_X.transpose((3,0,1,2)),extra_y[:,0]

print("------------------------------------------")
print("Dimensions of Images in form (image idx,row,column,color channel)")
print("Training Set Dimensions : ",train_X.shape,train_y.shape)
print("Test Set Dimensions : ", test_X.shape,test_y.shape)
print("Extra Set Dimensions : ",extra_X.shape,extra_y.shape)
total_num_images = train_X.shape[0] + test_X.shape[0] + extra_X.shape[0]
print("Total number of images : ",total_num_images)

# print("-------------------------------------------")
# print("Showing first 5 images in dataset (test) : ")

def show_first_five_images(img,labels):
    for i in range(5):
        if img[i].shape == (32, 32, 3):
            plt.imshow(img[i])
        else:
            plt.imshow(img[i,:,:,0],cmap=plt.cm.binary)
        plt.title(labels[i])
        plt.show()

# show_first_five_images(train_X,train_y)
# show_first_five_images(test_X,test_y)
# show_first_five_images(extra_X,extra_y)

print("-------------------------------------------")
print("Class labels in the dataset : ")
print(np.unique(train_y))
print("Distribution in Train Set : ")
print dict((x,train_y.tolist().count(x)) for x in set(train_y))
print("Distribution in Test Set : ")
print dict((x,test_y.tolist().count(x)) for x in set(test_y))
print("Distribution in Extra Set : ")
print dict((x,extra_y.tolist().count(x)) for x in set(extra_y))

print("-----------------------------------------------")
train_y[train_y == 10] = 0
test_y[test_y == 10] = 0
extra_y[extra_y == 10] = 0
print("After changing label 10 to 0, unique class labels are : ")
print(np.unique(train_y))

print("-------------------------------------------------")

def balanced_set(y,num_per_class):
	balanced_st = []
	for lab in np.unique(y):
		imgs = np.where(y==lab)[0]
		random_set = np.random.choice(imgs,size=num_per_class,replace=False)
		balanced_st += random_set.tolist();
	return balanced_st

samples_from_extra = balanced_set(extra_y,200)
samples_from_train = balanced_set(train_y,400)

val_X, val_y = np.copy(extra_X[samples_from_extra]),np.copy(extra_y[samples_from_extra])

extra_X = np.delete(extra_X,samples_from_extra,axis=0)
extra_y = np.delete(extra_y,samples_from_extra,axis=0)

val_X = np.concatenate([val_X,np.copy(train_X[samples_from_train])])
val_y = np.concatenate([val_y,np.copy(train_y[samples_from_train])])

train_X = np.delete(train_X,samples_from_train,axis=0)
train_y = np.delete(train_y,samples_from_train,axis=0)

train_X = np.concatenate([train_X,extra_X])
train_y = np.concatenate([train_y,extra_y])

print("After creating a balanced validation set : ")
print("Training Set Dimensions : ",train_X.shape,train_y.shape)
print("Test Set Dimensions : ", test_X.shape,test_y.shape)
print("Validation Set Dimensions : ",val_X.shape,val_y.shape)
total_num_images = train_X.shape[0] + test_X.shape[0] + val_X.shape[0]
print("Total number of images : ",total_num_images)

print("-----------------------------------------------------")

suffixes = ['B','KB','MB','GB']

def get_size(n_bytes):
    if n_bytes == 0: return '0 B'
    i = 0
    while n_bytes >= 1024:
        n_bytes /= 1024.
        i += 1
    f = ('%.2f' % n_bytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def rgb_to_gray(images):
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)

train_gray = rgb_to_gray(train_X).astype(np.float32)
test_gray = rgb_to_gray(test_X).astype(np.float32)
val_gray = rgb_to_gray(val_X).astype(np.float32)

print("Dimensions before and after conversion to grayscale")
print("Training Set : ",train_X.shape,train_gray.shape)
print("Test Set : ",test_X.shape,test_gray.shape)
print("Validation Set : ",val_X.shape,val_gray.shape)
print('\n')

print("Datatypes before and after conversion to grayscale")
print("Training Set : ",train_X.dtype,train_gray.dtype)
print("Test Set : ",test_X.dtype,test_gray.dtype)
print("Validation Set : ",val_X.dtype,val_gray.dtype)
print('\n')

print("Dataset Sizes before and after conversion to grayscale")
print("Training Set : ",get_size(train_X.nbytes),get_size(train_gray.nbytes))
print("Test Set : ",get_size(test_X.nbytes),get_size(test_gray.nbytes))
print("Validation Set : ",get_size(val_X.nbytes),get_size(val_gray.nbytes))
print('\n')

# print("-------------------------------------------------")
# print("Displaying 5 Images after RGB to Gray scale conversion (in test)")
# show_first_five_images(train_gray,train_y)
# show_first_five_images(test_gray,test_y)
# show_first_five_images(val_gray,val_y)

encoding = OneHotEncoder().fit(train_y.reshape(-1, 1))

train_y = encoding.transform(train_y.reshape(-1, 1)).toarray()
test_y = encoding.transform(test_y.reshape(-1, 1)).toarray()
val_y = encoding.transform(val_y.reshape(-1, 1)).toarray()

print("---------------------------------------------------")
print("After One Hot Encoding of labels, dimension of labels are : ")
print("Training set", train_y.shape)
print("Test set", test_y.shape)
print("Validation set", val_y.shape)

print("----------------------------------------------------")
print("Storing all the RGB Images")
file = h5py.File('data/SVHN_32x32_RGB.h5','w')

file.create_dataset('Training Set Images', data=train_X)
file.create_dataset('Test Set Images', data=test_X)
file.create_dataset('Validation Set Images', data=val_X)
file.create_dataset('Training Set Labels', data=train_y)
file.create_dataset('Test Set Labels', data=test_y)
file.create_dataset('Validation Set Labels', data=val_y)

file.close()
print("Storing RGB Images Done")
print("------------------------------------------------------")
print("Storing all the Grayscale Images")

file = h5py.File('data/SVHN_32x32_Grayscale.h5','w')

file.create_dataset('Training Set Images', data=train_gray)
file.create_dataset('Test Set Images', data=test_gray)
file.create_dataset('Validation Set Images', data=val_gray)
file.create_dataset('Training Set Labels', data=train_y)
file.create_dataset('Test Set Labels', data=test_y)
file.create_dataset('Validation Set Labels', data=val_y)

file.close()
print("Storing the Grayscale Images Done")
print("---------------------------------------------------")
print("Preprocessing Done")