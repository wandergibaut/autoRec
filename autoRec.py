from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 20 #size of the encoding representation. Must tune this up

input_dim = 20000 #ver depois


input_data = Input(shape=(input_dim,))

#encoded representation

encoded = Dense(input_dim/4, activation='relu')(input_data)
encoded = Dense(input_dim/8, activation='relu')(encoded)
encoded = Dense(input_dim/16, activation='relu')(encoded)

decoded = Dense(input_dim/8, activation='relu')(encoded)
decoded = Dense(input_dim/4, activation='relu')(decoded)
decoded = Dense(input_dim, activation='identity')(decoded)



#mapping
autoencoder = Model(input_data, decoded)




# this model maps an input to its encoded representation
encoder = Model(input_data, encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


# ver documentação keras
autoencoder.compile(optimizer='',loss='')

##
#dados do treino
(x_train, _), (x_test, _) = foo


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)













# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()