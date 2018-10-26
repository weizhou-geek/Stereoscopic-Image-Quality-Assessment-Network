__author__ = 'weizhou'


import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, ELU, MaxPooling2D, Flatten, Dense, Dropout, normalization
from keras.optimizers import SGD, rmsprop, adam
from keras import backend as K
import os
import scipy.io as sio


os.environ["CUDA_VISIBLE_DEVICES"]="0"

#left image
left_image = Input(shape=(32, 32, 3))
#conv1
left_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_left')(left_image)
left_elu1 = ELU()(left_conv1)
left_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(left_elu1)
#conv2
left_conv2 = Conv2D(32, (3, 3), padding='same', name='conv2_left')(left_pool1)
left_elu2 = ELU()(left_conv2)
left_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_left')(left_elu2)
#conv3
left_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_left')(left_pool2)
left_elu3 = ELU()(left_conv3)
#conv4
left_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_left')(left_elu3)
left_elu4 = ELU()(left_conv4)
#conv5
left_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_left')(left_elu4)
left_elu5 = ELU()(left_conv5)
left_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_left')(left_elu5)
#fc6
left_flat6 = Flatten()(left_pool5)
left_fc6 = Dense(512)(left_flat6)
left_elu6 = ELU()(left_fc6)
left_drop6 = Dropout(0.35)(left_elu6)
#fc7
left_fc7 = Dense(512)(left_drop6)
left_elu7 = ELU()(left_fc7)
left_drop7 = Dropout(0.5)(left_elu7)


#right image
right_image = Input(shape=(32, 32, 3))
#conv1
right_conv1 = Conv2D(32, (3, 3), padding='same', name='conv1_right')(right_image)
right_elu1 = ELU()(right_conv1)
right_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_right')(right_elu1)
#conv2
right_conv2 = Conv2D(32, (3, 3), padding='same', name='conv2_right')(right_pool1)
right_elu2 = ELU()(right_conv2)
right_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_right')(right_elu2)
#conv3
right_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_right')(right_pool2)
right_elu3 = ELU()(right_conv3)
#conv4
right_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_right')(right_elu3)
right_elu4 = ELU()(right_conv4)
#conv5
right_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_right')(right_elu4)
right_elu5 = ELU()(right_conv5)
right_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_right')(right_elu5)
#fc6
right_flat6 = Flatten()(right_pool5)
right_fc6 = Dense(512)(right_flat6)
right_elu6 = ELU()(right_fc6)
right_drop6 = Dropout(0.35)(right_elu6)
#fc7
right_fc7 = Dense(512)(right_drop6)
right_elu7 = ELU()(right_fc7)
right_drop7 = Dropout(0.5)(right_elu7)


#concatenate1
add_conv2 = keras.layers.add([left_conv2, right_conv2])
subtract_conv2 = keras.layers.subtract([left_conv2, right_conv2])
fusion1_conv2 = keras.layers.merge([add_conv2, subtract_conv2], mode='concat')
fusion1_elu2 = ELU()(fusion1_conv2)
fusion1_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_fusion1')(fusion1_elu2)
#conv3
fusion1_conv3 = Conv2D(64, (3, 3), padding='same', name='conv3_fusion1')(fusion1_pool2)
fusion1_elu3 = ELU()(fusion1_conv3)
#conv4
fusion1_conv4 = Conv2D(64, (3, 3), padding='same', name='conv4_fusion1')(fusion1_elu3)
fusion1_elu4 = ELU()(fusion1_conv4)
#conv5
fusion1_conv5 = Conv2D(128, (3, 3), padding='same', name='conv5_fusion1')(fusion1_elu4)
fusion1_elu5 = ELU()(fusion1_conv5)
fusion1_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_fusion1')(fusion1_elu5)
#fc6
fusion1_flat6 = Flatten()(fusion1_pool5)
fusion1_fc6 = Dense(512)(fusion1_flat6)
fusion1_elu6 = ELU()(fusion1_fc6)
fusion1_drop6 = Dropout(0.35)(fusion1_elu6)
#fc7
fusion1_fc7 = Dense(512)(fusion1_drop6)
fusion1_elu7 = ELU()(fusion1_fc7)
fusion1_drop7 = Dropout(0.5)(fusion1_elu7)


#concatenate2
add_conv5 = keras.layers.add([left_conv5, right_conv5])
subtract_conv5 = keras.layers.subtract([left_conv5, right_conv5])
fusion2_conv5 = keras.layers.merge([add_conv5, subtract_conv5], mode='concat')
fusion2_elu5 = ELU()(fusion2_conv5)
fusion2_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_fusion2')(fusion2_elu5)
#fc6
fusion2_flat6 = Flatten()(fusion2_pool5)
fusion2_fc6 = Dense(512)(fusion2_flat6)
fusion2_elu6 = ELU()(fusion2_fc6)
fusion2_drop6 = Dropout(0.35)(fusion2_elu6)
#fc7
fusion2_fc7 = Dense(512)(fusion2_drop6)
fusion2_elu7 = ELU()(fusion2_fc7)
fusion2_drop7 = Dropout(0.5)(fusion2_elu7)


#concatenate3
fusion3_drop7 = keras.layers.merge([left_drop7, right_drop7, fusion1_drop7, fusion2_drop7], mode='concat')
#fc8
fusion3_fc8 = Dense(1024)(fusion3_drop7)
#fc9
predictions = Dense(1)(fusion3_fc8)

model_all = Model(input=[left_image, right_image], output=predictions, name='all_model')
model_all.summary()


X_testLeft = np.load('./test_image_left_LIVE2.npy')
X_testRight = np.load('./test_image_right_LIVE2.npy')
X_testLeft = X_testLeft.astype('float32')
X_testRight = X_testRight.astype('float32')
X_testLeft /= 255
X_testRight /= 255
X_testLeft -= np.mean(X_testLeft, axis=0)
X_testRight -= np.mean(X_testRight, axis=0)
X_testLeft /= np.std(X_testLeft, axis=0)
X_testRight /= np.std(X_testRight, axis=0)

model_all.load_weights('./LIVE2_model_300.hdf5')
y_predict = model_all.predict([X_testLeft, X_testRight], batch_size=10)
sio.savemat('./predictScore_testLIVE2_model_300.mat', {'score': y_predict})
print('complete saving')
