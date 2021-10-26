from tensorflow.keras.layers import Dense, Flatten, Conv2D, ReLU, BatchNormalization, MaxPool2D, Dropout, Input, Softmax
from tensorflow.keras.models import Model, Sequential


#1. (모델1: 취약승객) man(s0), woman (s1) 
#2. (모델2: OOP + phoning**)  safe driving(c0), 
#                           ,phoning(c1) 휴대폰,
#                           ,close(c5), far(c6), behind(c7)
#3. (모델3: 벨트/미벨트) 벨트(b0), 노벨트(b1)

<<<<<<< HEAD
def model_cnn(input_shape = (128, 128, 3), classifier = None, base_trainable = True):
=======
def model_cnn(input_shape = (120, 160, 3), classifier = None, base_trainable = True):
>>>>>>> bd39c2de29960e4ce9ae688fdb4e8717f3c80088

    inputs = Input(shape=input_shape)
    #base_conv = base_conv_net(input_shape=input_shape, base_trainable= base_trainable)(inputs)

    conv1 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=64, padding='same', kernel_initializer='he_normal', name='1st_conv')(inputs)
    conv1 = ReLU(name='1st_relu')(conv1)
    pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='1st_maxpool')(conv1)


    conv2 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=128, padding='same', kernel_initializer='he_normal', name='2nd_conv')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = ReLU(name='2nd_relu')(conv2)
    # conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='2nd_maxpool')(conv2)

    conv3 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=256, padding='same', kernel_initializer='he_normal', name='3rd_conv')(pool2)
    conv3 = ReLU(name='3rd_relu')(conv3)
    pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name='3rd_maxpool')(conv3)

    dropout = Dropout(0.5)(pool3)
    flat = Flatten()(dropout)

    fc1 = Dense(128, activation='relu', kernel_initializer='he_normal', name='fc1')(flat)
    dropout = Dropout(0.5)(fc1)
    
    # 멀티 아웃풋 헤드
<<<<<<< HEAD
    if (classifier=="Belt"):
        out_belt= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_belt')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_belt, name='belt_classifier')

    elif (classifier=="Weak"):
        out_weak= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_weak')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_weak, name='weak_classifier')
    
    elif (classifier=="OOP"):
        out_oop= Dense(5, kernel_initializer='he_normal', name='out_oop', activation='softmax')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_oop, name='oop_classifier')

    elif (classifier=="Mask"):
        out_mask= Dense(2, activation='softmax', kernel_initializer='he_normal', name='out_mask')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_mask, name='mask_classifier')
=======
    

    if (classifier=="Belt"):
        out_belt= Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='out_belt')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_belt)

    elif (classifier=="Weak"):
        out_weak= Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='out_week')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_weak)
    
    elif (classifier=="OOP"):
        out_oop= Dense(5, kernel_initializer='he_normal', name='out_oop', activation='softmax')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_oop)

    elif (classifier=="Mask"):
        out_mask= Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='out_mask')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_mask)
    
    else:
        out_belt= Dense(2, activation='sigmoid', kernel_initializer='he_normal', name='out_belt')(dropout)
        entire_model = Model(inputs=inputs, outputs=out_belt)
>>>>>>> bd39c2de29960e4ce9ae688fdb4e8717f3c80088

    return entire_model


def base_conv_net(input_shape=(120,160,3), base_trainable = True):

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=32, padding='same', kernel_initializer='he_normal', name='1st_conv')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Dropout(0.5)(conv1)
    pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv1)

    conv2 = Conv2D(kernel_size=(5,5), strides=(1,1), filters=48, padding='same', kernel_initializer='he_normal', name='2nd_conv')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv2)
    
    conv_net = Model(inputs = inputs, outputs = pool2)
    conv_net.trainable= base_trainable

    return conv_net