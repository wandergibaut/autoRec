from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import keras.backend as K
import tensorflow as tf
import numpy as np
import random

def autoRec_loss(y_true,y_pred):
	zero = K.constant(0.0, dtype='float32')
	mask = K.not_equal(y_true, zero)
	return K.sum(K.square(tf.boolean_mask(y_true - y_pred, mask)), axis=-1)
        #y_true_rectified = tf.boolean_mask(tf.convert_to_tensor(y_true, dtype='float32'), mask)
	#y_pred_rectified = tf.boolean_mask(tf.convert_to_tensor(y_pred, dtype='float32'), mask)
	#return K.reduced_sum(K.square(y_true_rectified - y_pred_rectified), axis=-1)
	
	

        
def split_train_test(data):
    copyData = data
    sliceX = round(len(data)/10) # fatias de 10%
    test = np.full((sliceX,len(data[0,:])),0.0)
    #tira 10% pra teste. 10% dos individuos retirados COMPLETAMENTE. Preciso ainda separar em entrada e saida
    for i in range(sliceX):
        index = random.randrange(len(copyData))
        test[i]= copyData[index]
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
            val_entry[i,entryObservations[0][index]] = 0
            length-=1

    return [val_entry, val_expect]

def test_accuracy(y_pred,y_true):
    zero = K.constant(0, dtype='float32')
    where = K.not_equal(y_true, zero)
    y_true_rectified = tf.boolean_mask(y_true, where)
    y_pred_rectified = tf.boolean_mask(y_pred, where)
    return K.sum((y_true - y_pred)**2)



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
        print(input_size)



        encoding_dim = 500 #size of the encoding representation. Must tune this up

        #input_dim = 20000 #ver depois

        input_data = Input(shape=(input_dim,))

        #split
        [train, test] = split_train_test(data)

        print (len(train))        
        print (len(test))


        #encoded representation
        #encoded = Dense(round(input_dim/100), activation='sigmoid')(input_data)
        #encoded = Dense(round(input_dim/100), activation='sigmoid')(encoded)
        encoded = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(1e-5))(input_data)
        #encoded = Dense(encoding_dim, activation='sigmoid', activity_regularizer=regularizers.l1(1e-5))(encoded)

        #decoded = Dense(round(input_dim/100), activation='sigmoid')(encoded)
        #decoded = Dense(round(input_dim/10), activation='sigmoid')(decoded)
        decoded = Dense(input_dim, activation='linear')(encoded)
        #decoded = Dense(input_dim, activation='linear', activity_regularizer=regularizers.l1(1e-5))(decoded)

        #, kernel_regularizer=regularizers.l2(.1)

        #mapping
        autoencoder = Model(input_data, decoded)


        # ver documentação keras
        autoencoder.compile(optimizer='adadelta', loss=autoRec_loss) #'binary_crossentropy'

        ##
        #dados do treino, dividir em teste e treino corretamente
        #x_train= data
        #x_test = data

        x_train= train
        x_test = test       

        autoencoder.fit(x_train, x_train,
                        epochs=5,
                        shuffle=True,
                        batch_size=1,#256,
                        validation_split=0.1)
                        #validation_data=(val, val))

# encode and decode some digits
# note that we take them from the *test* set

        
        [x_test, y_test] = validation_split(test, 0.2)

        y_test_pred = autoencoder.predict(x_test)
        #decoded_data = decoder.predict(encoded_data)

        autoencoder.evaluate(x_test, y_test)

        #print(encoded_data)

        print(max(x_test[0,:]))
        #print(max(x_test[1,:]))

        print(max(y_test_pred[0,:]))
        #print(max(encoded_data[1,:]))

        #print(sum(x_test - encoded_data)/len(encoded_data[0,:]))
        #print('teste de precisao')

        print(test_accuracy(y_test,y_test_pred))





#evaluation here


#rmse




# use Matplotlib (don't ask)
#import matplotlib.pyplot as plt

#n = 10  # how many digits we will display
#plt.figure(figsize=(20, 4))
#for i in range(n):
    # display original
#    ax = plt.subplot(2, n, i + 1)
#    plt.imshow(x_test[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)

    # display reconstruction
#    ax = plt.subplot(2, n, i + 1 + n)
#    plt.imshow(decoded_imgs[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()
