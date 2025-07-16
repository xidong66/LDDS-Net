from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dense
from tensorflow.keras.models import Model
from custom_layers import GatedGLU, GeMWeightedFusion, ResidualShrinkageBlock
def LDDSN(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # 初始卷积层
    x = Conv1D(32, 31, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GatedGLU(filters=32, kernel_size=31, activation='gelu')(x)
    
    # 残差块 (需要实现residual_shrinkage_block_1d)
    x1 = residual_shrinkage_block_1d(x, nb_blocks=1, out_channels=64, downsample=True)
    x2 = residual_shrinkage_block_1d(x1, nb_blocks=1, out_channels=128, downsample=True)
    x3 = residual_shrinkage_block_1d(x2, nb_blocks=1, out_channels=256, downsample=True)
    
    x1=GlobalAveragePooling1D()(x1)
    x2=GlobalAveragePooling1D()(x2)
    x3=GlobalAveragePooling1D()(x3)
    x1 = Dense(256, activation='relu')(x1)
    x2 = Dense(256, activation='relu')(x2)
    x3 = Dense(256, activation='relu')(x3)
    # 假设 x1,x2,x3 是 (B,T,C) 的 3 个特征图
    fused = GeMWeightedFusion()([x1, x2, x3])   # (B,T,C)
    #fused = GlobalAveragePooling1D()(fused)     # (B,C)
# 之后可再 Dense / 分类
    # 使用自适应GeM融合三層特征
      # (B, 256)
    
    outputs = Dense(num_classes, activation='softmax')(fused)
    
    return Model(inputs, outputs)

# Example usage:
model =LDDSN((4500, 1), 4)
model.summary()
