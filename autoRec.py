from keras.layers import Input, Dense
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
    #new_data= np.delete(copyData, test)

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
            #length-=1

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


list_of_files = [('../lastFM/fooData/foo.dat'), ('../lastFM/fooData/foo_friends.dat')]



for file in list_of_files:
    data = np.genfromtxt(file,
                     dtype=None,
                     delimiter=' ')


    if file == '../lastFM/fooData/foo.dat':


        input_size= len(data)
        input_dim = len(data[0,:])

        #print(input_dim)
        #print(input_size)

        encoding_dim = 50 #size of the encoding representation. Must tune this up



        input_data = Input(shape=(input_dim,))


        [train, test] = split_train_test(data)

    

        encoded = Dense(encoding_dim, activation='sigmoid',activity_regularizer=regularizers.l2(1e-3))(input_data)
        decoded = Dense(input_dim, activation='linear', activity_regularizer=regularizers.l2(1e-3))(encoded)


        #mapping
        autoencoder = Model(input_data, decoded)


        # ver documentação keras
        autoencoder.compile(optimizer='adadelta', loss=autoRec_loss) #'binary_crossentropy'


        x_train= train
        x_test = test       

        autoencoder.fit(x_train, x_train,
                        epochs=100,
                        shuffle=True,
                        batch_size=1,
                        validation_split=0.1
                        )

# encode and decode some digits
# note that we take them from the *test* set

        
        [x_test, y_test] = test_split(test, 2)
        y_test_pred = autoencoder.predict(x_test)

        print(test_accuracy(y_test,y_test_pred))
        
        #autoencoder.evaluate(x_test, y_test)

        #print(encoded_data)

        #print(max(x_test[0,:]))
        #print(max(x_test[1,:]))

        #print(max(y_test_pred[0,:]))

        #y_testinho = y_test[0,:]
        #pred_test = y_test_pred[0,:]



        #mask = np.nonzero(y_test[0,:])

        #print(y_testinho[mask])
        #print(pred_test[mask])
