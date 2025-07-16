from configs.model_config import ModelConfig
from data_loader import load_data, preprocess_data
from models.lddsn_model import build_lddsn
from trainers import train_model
from evaluators import evaluate_and_visualize

if __name__ == "__main__":
    # 加载配置
    config = ModelConfig()
    
    # 数据准备
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("database/cinc2017denoise.npz")
    y_train, y_val, y_test = preprocess_data(y_train, y_val, y_test)
    
    # 构建模型
    model = build_lddsn(config.INPUT_SHAPE, config.NUM_CLASSES)
    model.summary()
    
    # 训练
    history = train_model(model, X_train, y_train, X_val, y_val, config.TRAIN_PARAMS)
    
    # 评估
    evaluate_and_visualize(model, X_test, y_test)