from utils import evaluate_model, plot_confusion_matrix2
import numpy as np
def evaluate_and_visualize(model, X_test, y_test):
    # 性能评估
    evaluate_model(model, X_test, y_test)
    
    # 混淆矩阵
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    plot_confusion_matrix2(np.argmax(y_test, axis=1), y_pred, 
                          classes=['A', 'N', 'O', '~'])
    
    #