from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, BatchNormalization, MaxPool2D, Dropout, Input, Softmax, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential

def model_autoencoder(input_shape = (64, 64, 3)):

    #Encoder
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=64, padding='same', kernel_initializer='he_normal', name='enc_1st_conv')(inputs)
    conv1 = ReLU(name='enc_1st_relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='enc_1st_maxpool')(conv1)
    conv2 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=128, padding='same', kernel_initializer='he_normal', name='enc_2nd_conv')(pool1)
    conv2 = ReLU(name='enc_2nd_relu')(conv2)
    pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='enc_2nd_maxpool')(conv2)
    conv3 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=256, padding='same', kernel_initializer='he_normal', name='enc_3rd_conv')(pool2)
    conv3 = ReLU(name='enc_3rd_relu')(conv3)

    latent = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='enc_3rd_maxpool')(conv3)

    #Decoder
    dec = Conv2DTranspose(kernel_size=(5,5), strides=(2,2), filters= 256, padding = 'same', kernel_initializer='he_normal', name='dec_3rd_conv', activation='relu')(latent)
    dec = Conv2DTranspose(kernel_size=(5,5), strides=(2,2), filters= 128, padding = 'same', kernel_initializer='he_normal', name='dec_2nd_conv', activation='relu')(dec)
    dec = Conv2DTranspose(kernel_size=(5,5), strides=(2,2), filters= 64, padding = 'same', kernel_initializer='he_normal', name='dec_1st_conv', activation='relu')(dec)
    
    out = Conv2D(kernel_size=(5,5), strides=(1,1), filters= input_shape[2], padding = 'same', kernel_initializer='he_normal', name='dec_out', activation='relu')(dec)

    entire_model = Model(inputs= inputs, outputs = out)
    return entire_model

