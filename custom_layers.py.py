import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, LayerNormalization, ReLU
from tensorflow.keras.activations import sigmoid
class GatedGLU(Layer):
    """
    Gated Gated Linear Unit (GLU) 模块。
    """
    def __init__(self, filters, kernel_size=1, activation='gelu', use_bias=True, **kwargs):
        """
        初始化门控 GLU 模块。
        :param filters: 输出通道数。
        :param kernel_size: 卷积核大小，默认为 1x1 卷积。
        :param activation: 激活函数，默认为 GELU。
        :param use_bias: 是否使用偏置，默认为 True。
        """
        super(GatedGLU, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

        # 定义两个分支：一个用于值（value），一个用于门控（gate）
        self.value_branch = Conv1D(filters=filters, kernel_size=kernel_size, use_bias=use_bias)
        self.gate_branch = Conv1D(filters=filters, kernel_size=kernel_size, use_bias=use_bias)

        # 定义激活函数
        if activation == 'gelu':
            self.act_fn = tf.nn.gelu
        elif activation == 'relu':
            self.act_fn = ReLU()
        elif activation == 'sigmoid':
            self.act_fn = Sigmoid()
        else:
            raise ValueError("Unsupported activation function. Please use 'gelu', 'relu', or 'sigmoid'.")

    def call(self, inputs):
        """
        前向传播。
        :param inputs: 输入张量，形状为 (batch_size, height, width, channels)。
        :return: 输出张量，形状与输入相同。
        """
        # 计算值分支和门控分支
        value = self.value_branch(inputs)
        gate = self.gate_branch(inputs)

        # 应用激活函数
        gate = self.act_fn(gate)

        # 门控操作：逐元素相乘
        output = value * gate

        return output

    def get_config(self):
        config = super(GatedGLU, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'use_bias': self.use_bias
        })
        return config

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

class GeMWeightedFusion(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # 3 个权重 + 1 个 p
        self.alpha = self.add_weight(name='alpha',
                                     shape=(3,),
                                     initializer='ones',
                                     trainable=True)
        self.p_raw = self.add_weight(name='p',
                                     shape=(1,),
                                     initializer='ones',
                                     trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: list [x1,x2,x3]  (B, C) - 注意现在是二维
        x1, x2, x3 = inputs
        
        # 保证 p > 0
        p = tf.nn.softplus(self.p_raw) + 1e-6

        # 权重归一化
        w = tf.nn.softmax(self.alpha)           # (3,)
        w = tf.reshape(w, [3, 1])               # (3,1) 便于广播

        # 堆叠以便广播：stack → (3, B, C)
        stack = tf.stack([x1, x2, x3], axis=0)  # (3, B, C)

        # GeM 融合
        stack = tf.pow(tf.maximum(stack, 1e-6), p)   # 避免负值/0
        fused = tf.reduce_sum(stack * w[:, :, tf.newaxis], axis=0)  # (B, C)
        fused = tf.pow(fused, 1.0 / p)

        return fused
    
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Dense, GRU
from tensorflow.keras.models import Model
import tensorflow as tf

def residual_shrinkage_block_1d(incoming, nb_blocks, out_channels, downsample=False, downsample_strides=2, activation='relu', kernel_size=31,batch_norm=True, bias=True, weights_init='variance_scaling', bias_init='zeros', regularizer='l2', weight_decay=0.0001, trainable=True, name="ResidualShrinkageBlock"):
  residual = incoming
  in_channels = incoming.shape[-1]
  for i in range(nb_blocks):
    identity = residual
    if downsample:
      downsample_strides = 2
    else:
      downsample_strides = 1
    
    if batch_norm:
      residual = BatchNormalization()(residual)
    residual = Activation(activation)(residual)
    residual = Conv1D(out_channels, kernel_size=kernel_size, strides=downsample_strides, padding='same', kernel_initializer=weights_init, use_bias=bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(residual)
    
    if batch_norm:
      residual = BatchNormalization()(residual)
    residual = Activation(activation)(residual)
    residual = Conv1D(out_channels, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=weights_init, use_bias=bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(residual)
    
    # Thresholding
    abs_mean = tf.reduce_mean(tf.abs(residual), axis=1, keepdims=True)
    scales = Dense(out_channels // 4, activation='linear', kernel_regularizer=regularizer, kernel_initializer=weights_init)(abs_mean)
    scales = BatchNormalization()(scales)
    scales = Activation('relu')(scales)
    scales = Dense(out_channels, activation='linear', kernel_regularizer=regularizer, kernel_initializer=weights_init)(scales)
    scales = tf.sigmoid(scales)
    thres = abs_mean * scales
    residual = tf.sign(residual) * tf.maximum(tf.abs(residual) - thres, 0)
    
    # Downsampling and projection
    if downsample_strides > 1:
      identity = Conv1D(out_channels, 1, strides=downsample_strides, padding='same', kernel_initializer=weights_init, use_bias=bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(identity)
    
    if in_channels != out_channels or downsample:
      identity = Conv1D(out_channels, 1, strides=1, padding='same', kernel_initializer=weights_init, use_bias=bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(identity)
    
    residual = residual + identity

  return residual
