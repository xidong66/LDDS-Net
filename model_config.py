class ModelConfig:
    INPUT_SHAPE = (4500, 1)
    NUM_CLASSES = 4
    CLASS_WEIGHTS = [2.0, 1.0, 1.5, 1.0] 
    TRAIN_PARAMS = {
        'batch_size': 256,
        'epochs': 80,
        'lr_patience': 2,
        'early_stop_patience': 7
    }