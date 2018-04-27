from google.colab import files

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import keras.backend as K

import numpy as np

def autoRec_loss(y_true,y_pred):
    y_true_rectified = y_true[np.nonzero(y_true)]
    y_pred_rectified = y_pred[np.nonzero(y_true)] #sim ta certo

    return sum((y_true_rectified - y_pred_rectified)**2)

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

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

        #encoded representation


        encoded = Dense(round(input_dim/10), activation='sigmoid', kernel_regularizer=regularizers.l2(.1))(input_data)
        encoded = Dense(round(input_dim/100), activation='sigmoid', kernel_regularizer=regularizers.l2(.1))(encoded)
        encoded = Dense(encoding_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(.1))(encoded)

        decoded = Dense(round(input_dim/100), activation='sigmoid', kernel_regularizer=regularizers.l2(.1))(encoded)
        decoded = Dense(round(input_dim/10), activation='sigmoid', kernel_regularizer=regularizers.l2(.1))(decoded)
        decoded = Dense(input_dim, activation='linear', kernel_regularizer=regularizers.l2(.1))(decoded)



        #mapping
        autoencoder = Model(input_data, decoded)




        # this model maps an input to its encoded representation
        #encoder = Model(input_data, encoded)


        # create a placeholder for an encoded (32-dimensional) input
        #encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        #decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        #decoder = Model(encoded_input, decoder_layer(encoded_input))


        # ver documentação keras
        autoencoder.compile(optimizer='adadelta', loss=autoRec_loss) #'binary_crossentropy'

        ##
        #dados do treino, dividir em teste e treino corretamente
        x_train= data
        x_test = data

        autoencoder.fit(x_train, x_train,
                        epochs=5,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
        encoded_data = autoencoder.predict(x_test)
        #decoded_data = decoder.predict(encoded_data)

        print(encoded_data)

        print(max(x_test[0,:]))
        print(max(x_test[1,:]))

        print(max(encoded_data[0,:]))
        print(max(encoded_data[1,:]))

        print(sum(x_test - encoded_data)/len(encoded_data[0,:]))

