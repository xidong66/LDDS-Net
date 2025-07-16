from tensorflow.keras.callbacks import EarlyStopping
from utils import AdjustLearningRateCallback  # 原回调工具

import tensorflow as tf

def weighted_crossentropy(weights):
    def loss(y_true, y_pred):
        # 计算未加权的交叉熵损失
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        # 应用类别权重
        weighted_loss = unweighted_loss * weights
        # 返回加权损失
        return weighted_loss
    return loss
class_weights = [2.0, 1.0, 1.5, 1.0] 
def train_model(model, X_train, y_train, X_val, y_val, config):
    callbacks = [
        AdjustLearningRateCallback(factor=0.1, patience=config['lr_patience']),
        EarlyStopping(patience=config['early_stop_patience'])
    ]
    
    model.compile(
        optimizer='Adam',
        loss=weighted_crossentropy(config['class_weights']),
        metrics=['accuracy']
    )
    
    return model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )