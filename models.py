from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers,initializers

class Mish(Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def conv2d_norm(x, filters, kernel_size=(3, 3), padding='same', groups=1, strides=(1, 1), activation=None, regularizer = None, norm = 'bn',name=None):

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, groups=groups,use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'zeros', kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    if norm == 'bn':
        x = BatchNormalization()(x)
    elif norm == 'ln':
        x = LayerNormalization()(x)
    # x = BatchNormalization(axis = 3, scale = True)(x)

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

class MLPattention(layers.Layer):
    def __init__(self, height, width, channels, *args, **kwargs):
        super(MLPattention, self).__init__(*args, **kwargs)
        self.hvscan1 = Conv1D(width * channels, 1, strides=1, padding='valid', groups=channels,use_bias=True, activation='relu', kernel_initializer = initializers.RandomNormal(stddev=0.0000001), bias_initializer = 'ones')
        self.hvscan2 = Conv1D(height * channels, 1, strides=1, padding='valid', groups=channels,use_bias=True, activation='sigmoid', kernel_initializer = initializers.RandomNormal(stddev=0.0000001), bias_initializer = 'ones')
        self.bn = BatchNormalization()

def MAEFNet(pretrained_weights = None,input_size = (256,256,1)):
    kn=24

    inputs = Input(input_size)
    conv1 = conv2d_norm(inputs, kn, 7, 'same', 1, 2, activation='relu')
 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv2d_norm(pool1, kn*2, activation='relu')
    conv2 = conv2d_norm(conv2, kn*2, activation='relu')

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv2d_norm(pool2, kn*4, activation='relu')
    conv3 = conv2d_norm(conv3, kn*4, activation='relu')

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv2d_norm(pool3, kn*8, activation='relu')
    conv4 = conv2d_norm(conv4, kn*8, kernel_size = (3, 3), activation='relu')
 
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv2d_norm(pool4, kn*8, kernel_size = (3, 3), activation='relu')

    up6 = conv2d_norm((UpSampling2D(size = (2,2))(conv5)), kn*8, 2, activation = 'relu', padding = 'same')
    merge6 = conv4 + up6
    conv6 = conv2d_norm(merge6, kn*8, activation='relu')
    conv6 = conv2d_norm(conv6, kn*4, activation='relu')

    up7 = conv2d_norm((UpSampling2D(size = (2,2))(conv6)), kn*4, 2, activation = 'relu', padding = 'same')
    merge7 = conv3 + up7
    conv7 = conv2d_norm(merge7, kn*4, activation='relu')
    conv7 = conv2d_norm(conv7, kn*2, activation='relu')

    up8 = conv2d_norm((UpSampling2D(size = (2,2), interpolation='bilinear')(conv7)), kn*2, 2, activation = 'relu', padding = 'same')
    merge8 = conv2 + up8
    conv8 = conv2d_norm(merge8, kn*2, activation='relu')
    conv8 = conv2d_norm(conv8, kn, activation='relu')

    up9 = conv2d_norm((UpSampling2D(size = (4,4), interpolation='bilinear')(conv8)), kn, 2, activation = 'relu', padding = 'same')
    conv9 = conv2d_norm(up9, kn, activation='relu')
    conv9 = conv2d_norm(conv9, kn, activation='relu')
    conv9 = conv2d_norm(conv9, 3, activation='relu')

    inputs2 = concatenate([inputs,conv9], axis = 3)
    conv10 = conv2d_norm(inputs2, kn, 7, 'same', 1, 2, activation='relu')
 
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
    merge11 = pool10 + conv8
    conv11 = conv2d_norm(merge11, kn*2, activation='relu')
    conv11 = conv2d_norm(conv11, kn*2, activation='relu')

    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    merge12 = pool11 + conv7
    conv12 = conv2d_norm(merge12, kn*4, activation='relu')
    conv12 = conv2d_norm(conv12, kn*4, activation='relu')

    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    merge13 = pool12 + conv6
    conv13 = conv2d_norm(merge13, kn*8, activation='relu')
    conv13 = conv2d_norm(conv13, kn*8, kernel_size = (3, 3), activation='relu')


    pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
    merge14 = pool13 + conv5
    conv14 = conv2d_norm(merge14, kn*16, kernel_size = (3,3), padding = 'same', activation='relu')
    _,H,W,C = conv14.shape
    conv14_w = MLPattention(H, W, C)(conv14)
    conv14 = conv14 * conv14_w
    conv14 = BatchNormalization()(conv14)

    up15 = conv2d_norm((UpSampling2D(size = (2,2))(conv14)), kn*8, 2, activation = 'relu', padding = 'same')
    merge16 = conv13 + up15
    conv16 = conv2d_norm(merge16, kn*8, activation='relu')
    conv16 = conv2d_norm(conv16, kn*8, activation='relu')

    up16 = conv2d_norm((UpSampling2D(size = (2,2))(conv16)), kn*4, 2, activation = 'relu', padding = 'same')
    merge17 = conv12 + up16
    conv17 = conv2d_norm(merge17, kn*4, activation='relu')
    conv17 = conv2d_norm(conv17, kn*4, activation='relu')

    up17 = conv2d_norm((UpSampling2D(size = (2,2), interpolation='bilinear')(conv17)), kn*2, 2, activation = 'relu', padding = 'same')
    merge18 = conv11 + up17
    conv18 = conv2d_norm(merge18, kn*2, activation='relu')
    conv18 = conv2d_norm(conv18, kn*2, activation='relu')

    up18 = conv2d_norm((UpSampling2D(size = (4,4), interpolation='bilinear')(conv18)), kn, 2, activation = 'relu', padding = 'same')
    conv19 = conv2d_norm(up18, kn, activation='relu')
    conv19 = conv2d_norm(conv19, kn, activation='relu')
    conv19 = conv2d_norm(conv19, 2, activation='relu')
    conv20 = conv2d_norm(conv19, 1, activation='sigmoid')
    
    model = Model(inputs = inputs, outputs = conv20)

    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model