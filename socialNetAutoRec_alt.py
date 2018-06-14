from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import regularizers
import keras.backend as K
import tensorflow as tf
import numpy as np
import random
from math import sqrt

def autoRec_loss(y_true,y_pred):
    zero = K.constant(0.0, dtype='float32')
    mask = K.not_equal(y_true, zero)
    return K.sum(K.square(tf.boolean_mask(y_true - y_pred, mask)), axis=-1)

        
def split_train_test(data):
    copyData = np.copy(data)
    sliceX = round(len(data)/10) # fatias de 10%
    test = np.full((sliceX,len(data[0,:])),0.0)
    #tira 10% pra teste. 10% dos individuos retirados COMPLETAMENTE. Preciso ainda separar em entrada e saida
    for i in range(sliceX):
        index = random.randrange(len(copyData))
        test[i]= copyData[index]
        np.delete(copyData,index)

    train = copyData#new_data#copyData
    return [train, test]

#quebra o conjunto de validacao entre entrada e saida
def validation_split(val, percentage):
    val_entry = np.copy(val)
    val_expect = np.copy(val)

    for i in range(len(val)):
        entryObservations = np.nonzero(val[i,:]) # espero q sejam os indicies
        length = len(entryObservations[0])
        sliceX = round(length*percentage)

        for j in range(sliceX):
            index = random.randrange(length)
            val_entry[i,entryObservations[0][index]] = 0

    return [val_entry, val_expect]


def test_split(val, value):
    val_entry = np.copy(val)
    val_expect = np.copy(val)

    for i in range(len(val)):
        entryObservations = np.nonzero(val[i,:]) # espero q sejam os indicies
        length = len(entryObservations[0])
        sliceX = value
        if length != 0:
            for j in range(sliceX):
                index = random.randrange(length)
                val_entry[i,entryObservations[0][index]] = 0

    return [val_entry, val_expect]


def test_accuracy(y_pred,y_true):
    mask = np.nonzero(y_true)
    y_pred_rectified = y_pred[mask]
    y_true_rectified = y_true[mask]
    return sqrt(sum((y_pred_rectified - y_true_rectified)**2)/len(y_true_rectified))



user_best_friend = np.genfromtxt('../lastFM/fooData/best_friend_index.dat',
                     dtype=None,
                     delimiter=' ')

user_artist_data_zeros = np.genfromtxt('../lastFM/fooData/foo_with_zeros.dat',
                     dtype=None,
                     delimiter=' ')

input_size= len(user_artist_data_zeros)
input_dim = len(user_artist_data_zeros[0,:])
output_dim = len(user_artist_data_zeros[0,:])

#print(input_dim)
#print(input_size)

encoding_dim = 50 

train_1 = user_artist_data_zeros
train_2 = np.full((len(user_artist_data_zeros), len(user_artist_data_zeros[0,:])), 0.0)


for i in range(len(user_best_friend)):
    index = int(user_best_friend[i])
    train_2[i,:] = train_1[index,:]

#print (len(train_1))        

input_data = Input(shape=(input_dim,))
input_best_friend = Input(shape=(input_dim,))
concatLayer = concatenate([input_data, input_best_friend])
encoded = Dense(encoding_dim, activation='sigmoid',activity_regularizer=regularizers.l2(1e-3))(concatLayer)
decoded = Dense(output_dim, activation='linear', activity_regularizer=regularizers.l2(1e-3))(encoded)

        #mapping
autoencoder = Model(inputs=[input_data, input_best_friend], outputs=decoded)

autoencoder.compile(optimizer='adadelta', loss=autoRec_loss) #'binary_crossentropy'


x_train= [train_1, train_2]
#x_test = test       

autoencoder.fit(x_train, train_1,
                epochs=100,
                shuffle=True,
                batch_size=1,
                validation_split=0.1
                )

#
[x_test_proto, y_test] = test_split(train_1, 2)

x_test = np.append(x_test_proto, train_2)

y_test_pred = autoencoder.predict([x_test_proto, train_2])

print(test_accuracy(y_test,y_test_pred))
