import tensorflow as tf
from tensorflow.keras.layers import *
import keras
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import Model

# Inverted Residual Block

def inverted_residual_block(inputs, num_filters, strides=1, expansion_ratio=1):
    # point-wise conv
    x = Conv2D(filters=expansion_ratio*inputs.shape[-1],
                 kernel_size=1,
                 padding='same',
                 use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    
    # depth-wise conv
    x = DepthwiseConv2D(kernel_size=3,
                          strides=strides,
                          padding='same',
                          use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    
    # point-wise conv
    x = Conv2D(filters=num_filters,
                 kernel_size=1,
                 padding='same',
                 use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # Residual Connection
    if strides == 1 and (inputs.shape == x.shape):
        return Add()([inputs, x])
    return x


# Spatial Attention

class SpatialAttentionModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3):
        '''
        paper: https://arxiv.org/abs/1807.06521
        code: https://gist.github.com/innat/99888fa8065ecbf3ae2b297e5c10db70
        '''
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, 
                                            use_bias=False, 
                                            kernel_initializer='he_normal',
                                            strides=1, padding='same', 
                                            activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, 
                                            use_bias=False, 
                                            kernel_initializer='he_normal',
                                            strides=1, padding='same', 
                                            activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(16, kernel_size=kernel_size, 
                                            use_bias=False, 
                                            kernel_initializer='he_normal',
                                            strides=1, padding='same', 
                                            activation=tf.nn.relu)
        self.conv4 = tf.keras.layers.Conv2D(1, kernel_size=kernel_size,  
                                            use_bias=False,
                                            kernel_initializer='he_normal',
                                            strides=1, padding='same', 
                                            activation=tf.math.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs,  axis=3)
        x = tf.stack([avg_out, max_out], axis=3) 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)

# Channel Attention

class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, ratio=8):
        '''
        paper: https://arxiv.org/abs/1807.06521
        code: https://gist.github.com/innat/99888fa8065ecbf3ae2b297e5c10db70
        '''
        super(ChannelAttentionModule, self).__init__()
        self.ratio = ratio
        self.gapavg = tf.keras.layers.GlobalAveragePooling2D()
        self.gmpmax = tf.keras.layers.GlobalMaxPooling2D()
        
    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(input_shape[-1]//self.ratio, 
                                            kernel_size=1, 
                                            strides=1, padding='same',
                                            use_bias=True, activation=tf.nn.relu)
    
        self.conv2 = tf.keras.layers.Conv2D(input_shape[-1], 
                                            kernel_size=1, 
                                            strides=1, padding='same',
                                            use_bias=True, activation=tf.nn.relu)
        super(ChannelAttentionModule, self).build(input_shape)

    def call(self, inputs):
        # compute gap and gmp pooling 
        gapavg = self.gapavg(inputs)
        gmpmax = self.gmpmax(inputs)
        gapavg = tf.keras.layers.Reshape((1, 1, gapavg.shape[1]))(gapavg)   
        gmpmax = tf.keras.layers.Reshape((1, 1, gmpmax.shape[1]))(gmpmax)   
        # forward passing to the respected layers
        gapavg_out = self.conv2(self.conv1(gapavg))
        gmpmax_out = self.conv2(self.conv1(gmpmax))
        return tf.math.sigmoid(gapavg_out + gmpmax_out)
    
    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[3]
        return (input_shape[0], output_len)

# Squeeze and Excitation Block 

def se_block(tensor, ratio):
    nb_channel = K.int_shape(tensor)[-1]

    x = GlobalAveragePooling2D()(tensor)
    x = Dense(nb_channel // ratio, activation='relu')(x)
    x = Dense(nb_channel, activation='sigmoid')(x)

    x = Multiply()([tensor, x])
    return x

# Pixel Attention

def pixel_attention(x, nf):
    # Assuming input_features shape: (batch_size, height, width, channels)
    
    # Apply convolution to capture spatial dependencies
    conv = tf.keras.layers.Conv2D(nf, 3, padding='same', activation='relu')(x)
    
    # Apply convolution to obtain attention scores
    attention_scores = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(conv)
    
    # Multiply attention scores with input features
    weighted_features = tf.multiply(x, attention_scores)
    
    return weighted_features

# Proposed CSP Attention

def csp_module(x, nf, level):
    """
    CSP Module with unique naming for Concatenate layers.

    Parameters:
    x - Input tensor.
    nf - Number of filters (or other parameter relevant to the module).
    level - Level index to ensure unique naming of layers.

    Returns:
    x_out - Output tensor after applying CSP operations.
    """
    x_pa = pixel_attention(x, nf)
    x_ca = ChannelAttentionModule()(x)
    x_sa = SpatialAttentionModule()(x)
    x_casa = Multiply()([x_sa, x_ca])
    
    # Assign a unique name to the Concatenate layer using the level index
    x_out = Concatenate(name=f'visualized_layer_level_{level}')([x_casa, x_pa])
    return x_out

# CSP-SegNet Model

# Input Layer
inputs = tf.keras.layers.Input((256,256,3))

# Downsampling
x = SeparableConv2D(32, (3,3), padding="same")(inputs)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)

x1 = inverted_residual_block(x, num_filters=32)

x1_csp = csp_module(x1, 32, 1) 

x = MaxPooling2D((2,2))(x1)

x = SeparableConv2D(64, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)

x2 = inverted_residual_block(x, num_filters=64)

x2_csp = csp_module(x2, 64, 2) 

x = MaxPooling2D((2,2))(x2)

x = SeparableConv2D(128, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)

x3 = inverted_residual_block(x, num_filters=128)

x3_csp = csp_module(x3, 128, 3) 

x = MaxPooling2D((2,2))(x3)

x = SeparableConv2D(256, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)

x4 = inverted_residual_block(x, num_filters=256)

x4_csp = csp_module(x4, 256, 4)

x = MaxPooling2D((2,2))(x4)

# Upsampling

x = UpSampling2D(interpolation="bilinear")(x)
x = Concatenate()([x, x4_csp])

x = SeparableConv2D(256, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)


x = SeparableConv2D(256, (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = UpSampling2D(interpolation="bilinear")(x)
x = Concatenate()([x, x3_csp])

x = SeparableConv2D(128, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)


x = SeparableConv2D(128, (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = UpSampling2D(interpolation="bilinear")(x)
x = Concatenate()([x, x2_csp])

x = SeparableConv2D(64, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)


x = SeparableConv2D(64, (1,1), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = UpSampling2D(interpolation="bilinear")(x)
x = Concatenate()([x, x1_csp])

x = SeparableConv2D(32, (3,3), padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = se_block(x, 16)


x = SeparableConv2D(3, (1,1), activation='softmax')(x)
model = keras.Model(inputs, x)
