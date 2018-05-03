from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import keras.backend as K
import tensorflow as tf
import numpy as np
import random

def autoRec_loss(y_true,y_pred):
	zero = K.constant(0, dtype='float32')
	where = K.not_equal(y_true, zero)
	y_true_rectified = tf.boolean_mask(y_true, where)
	y_pred_rectified = tf.boolean_mask(y_pred, where)
	return K.sum((y_true - y_pred)**2)


def split_train_test(data):
    copyData = data
    sliceX = round(len(data)/10) # fatias de 10%
    test = np.full((sliceX,len(data[0,:])),0.0)

    #tira 10% pra teste. 10% dos individuos retirados COMPLETAMENTE. Preciso ainda separar em entrada e saida
    for i in range(sliceX):
        index = random.randrange(len(copyData))
        test[i-1]= copyData[index]
        np.delete(copyData,index)
        
    train = copyData

    return [train, test]

#quebra o conjunto de validacao entre entrada e saida
def validation_split(val, percentage):
    val_entry = val
    val_expect = val

    for i in range(len(val)):
        entryObservations = np.nonzero(val[i,:]) # espero q sejam os indicies
        length = len(entryObservations[0])
        sliceX = round(length*percentage)

        for j in range(sliceX):
            index = random.randrange(length)
            print(j)
            val_entry[i,entryObservations[0][index]] = 0
            #np.delete(val_entry[i,:], entryObservations[index]) #remove uma observacao aleatoria
            #entryObservations= np.delete(entryObservations,index)
            length-=1

    return [val_entry, val_expect]

list_of_files = [('foo.dat'), ('foo_friends.dat')]

#datalist = [(pylab.loadtxt(filename), label) for filename, label in list_of_files ]


for file in list_of_files:
    data = np.genfromtxt(file,
                     #skip_header=1,
                     #skip_footer=1,
                     #names=True,
                     dtype=None,
                     delimiter=' ')
    #print(data)

    if file == 'foo.dat':
        #data = np.asarray(data[0])
        #print(data)
        input_size= len(data)
        input_dim = len(data[0,:])

        #print('sou lindo ')
        print(input_dim)



        encoding_dim = 20 #size of the encoding representation. Must tune this up

        #input_dim = 20000 #ver depois

        input_data = Input(shape=(input_dim,))

        #split
        [train, test] = split_train_test(data)

        [x_test, y_test] = validation_split(test, 0.2)
        
