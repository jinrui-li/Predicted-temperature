import os
import tensorflow as tf

# 检查是否有可用的 GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 设置 GPU 为可见设备
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("使用 GPU 进行运算")
    except RuntimeError as e:
        print(f"GPU 设置出错: {e}")
        # 如果设置 GPU 出错，则回退到 CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("回退到 CPU 进行运算")
else:
    # 如果没有可用的 GPU，则使用 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("没有可用的 GPU，使用 CPU 进行运算")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Layer
from tensorflow.keras.callbacks import EarlyStopping
import math

from sklearn.feature_selection import SelectKBest, f_regression
import keras_tuner as kt

# 自定义注意力层
class Attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        return (input_shape[0], input_shape[-1])

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']  # 只保留 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("数据基本信息：")
        df.info()

        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n数据包含缺失值，进行处理...")
            df = df.interpolate(method='linear')  # 线性插值填充缺失值

        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

# 数据预处理
def preprocess_data(df, target_col='tempmax'):
    # 提取特征和目标变量
    date_col = pd.to_datetime(df['date'])  # 将日期转换为datetime类型
    df_features = df.drop(['name', 'date'], axis=1)

    # 数据平滑：移动平均
    df_features = df_features.rolling(window=3).mean().dropna()
    date_col = date_col[2:]  # 调整日期列以匹配平滑后的数据

    # 特征选择
    selector = SelectKBest(score_func=f_regression, k=5)
    X_selected = selector.fit_transform(df_features, df_features[target_col])
    feature_names = df_features.columns[selector.get_support()].tolist()  # 转换为列表

    # 确保 target_col 在特征列表中
    if target_col not in feature_names:
        feature_names.append(target_col)
        # 重新选择特征
        X_selected = df_features[feature_names].values

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_selected)

    # 确定目标变量索引
    target_index = feature_names.index(target_col)

    return scaled_data, scaler, target_index, date_col, feature_names

# 创建序列数据
def create_sequences(data, target_index, time_steps=30, pred_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps - pred_steps + 1):
        X.append(data[i:(i + time_steps), :])
        y.append(data[(i + time_steps):(i + time_steps + pred_steps), target_index])
    return np.array(X), np.array(y)

# 构建双向LSTM模型，添加注意力机制
def build_lstm_model(hp):
    input_shape = (30, 5)  # 假设输入形状
    output_units = 3  # 假设输出单元数
    model = Sequential()
    model.add(Input(shape=input_shape))  # 使用 Input 层指定输入形状
    # 调整第一个LSTM层的神经元数量
    hp_units1 = hp.Int('units1', min_value=64, max_value=256, step=32)
    model.add(Bidirectional(LSTM(hp_units1, return_sequences=True)))
    # 调整Dropout层的丢弃率
    hp_dropout1 = hp.Float('dropout1', min_value=0.1, max_value=0.3, step=0.05)
    model.add(Dropout(hp_dropout1))
    model.add(Attention(return_sequences=True))  # 添加注意力层
    # 调整第二个LSTM层的神经元数量
    hp_units2 = hp.Int('units2', min_value=32, max_value=128, step=16)
    model.add(Bidirectional(LSTM(hp_units2, return_sequences=False)))
    # 调整Dropout层的丢弃率
    hp_dropout2 = hp.Float('dropout2', min_value=0.1, max_value=0.3, step=0.05)
    model.add(Dropout(hp_dropout2))
    model.add(Dense(32))
    model.add(Dense(output_units))

    # 调整学习率
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    from tensorflow.keras.optimizers import Adam  # 从 tensorflow.keras 导入 Adam
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
    return model

# 反标准化预测结果
def inverse_transform_predictions(predictions, scaler, target_index, feature_names):
    pred_transformed = []
    for i in range(len(predictions)):
        dummy_array = np.zeros((1, len(feature_names)))
        for j in range(len(predictions[i])):
            dummy_array[0, target_index] = predictions[i][j]
            pred_transformed.append(scaler.inverse_transform(dummy_array)[0, target_index])
    return np.array(pred_transformed).reshape(len(predictions), -1)

# 评估模型
def evaluate_model(y_actual, y_pred):
    mae = mean_absolute_error(y_actual[:, 0], y_pred[:, 0])
    rmse = math.sqrt(mean_squared_error(y_actual[:, 0], y_pred[:, 0]))
    r2 = r2_score(y_actual[:, 0], y_pred[:, 0])
    return mae, rmse, r2

# 主函数
def main():
    file_path = 'zhengzhou 2017-09-01 to 2025-05-01.csv'

    # 加载数据
    df = load_data(file_path)
    if df is None:
        return

    # 数据预处理
    scaled_data, scaler, target_index, date_col, feature_names = preprocess_data(df)

    # 设置参数
    time_steps = 30  # 历史天数
    pred_steps = 3  # 预测天数

    # 创建序列数据
    X, y = create_sequences(scaled_data, target_index, time_steps, pred_steps)

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 调整数据形状以适应SVM
    X_train_svm = X_train.reshape(X_train.shape[0], -1)
    X_test_svm = X_test.reshape(X_test.shape[0], -1)

    # 构建SVM模型
    svm_model = SVR()
    svm_model.fit(X_train_svm, y_train[:, 0])

    # 训练SVM模型
    svm_predict = svm_model.predict(X_test_svm)
    svm_predict = svm_predict.reshape(-1, 1)

    # 反标准化SVM预测结果
    svm_predict = inverse_transform_predictions(svm_predict, scaler, target_index, feature_names)

    # 使用Keras Tuner进行LSTM模型调优
    tuner = kt.Hyperband(
        build_lstm_model,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        directory='hyperband_dir',
        project_name='temperature_prediction'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_train, y_train,
                 epochs=50,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[early_stopping],
                 verbose=1)

    # 获取最佳超参数
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"最佳第一个LSTM层神经元数量: {best_hps.get('units1')}")
    print(f"最佳第一个Dropout层丢弃率: {best_hps.get('dropout1')}")
    print(f"最佳第二个LSTM层神经元数量: {best_hps.get('units2')}")
    print(f"最佳第二个Dropout层丢弃率: {best_hps.get('dropout2')}")
    print(f"最佳学习率: {best_hps.get('learning_rate')}")

    # 使用最佳超参数构建LSTM模型
    lstm_model = tuner.hypermodel.build(best_hps)

    # 训练双向LSTM模型
    history = lstm_model.fit(X_train, y_train,
                             epochs=50,
                             batch_size=32,
                             validation_split=0.2,
                             callbacks=[early_stopping],
                             verbose=1)

    # 评估双向LSTM模型
    lstm_predict = lstm_model.predict(X_test)
    lstm_predict = inverse_transform_predictions(lstm_predict, scaler, target_index, feature_names)

    # 反标准化实际值
    y_test_actual = inverse_transform_predictions(y_test, scaler, target_index, feature_names)

    # 评估模型
    svm_mae, svm_rmse, svm_r2 = evaluate_model(y_test_actual, svm_predict)
    lstm_mae, lstm_rmse, lstm_r2 = evaluate_model(y_test_actual, lstm_predict)

    # 打印评估结果
    print(f'SVM MAE: {svm_mae:.2f}')
    print(f'SVM RMSE: {svm_rmse:.2f}')
    print(f'SVM R²: {svm_r2:.2f}')

    print(f'LSTM MAE: {lstm_mae:.2f}')
    print(f'LSTM RMSE: {lstm_rmse:.2f}')
    print(f'LSTM R²: {lstm_r2:.2f}')

    # 结果对比分析表
    results = pd.DataFrame({
        'Model': ['SVM', 'LSTM'],
        'MAE': [svm_mae, lstm_mae],
        'RMSE': [svm_rmse, lstm_rmse],
        'R² (R-Squared)': [svm_r2, lstm_r2]
    })
    print(results)

    # 绘制改进的时间序列图
    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制实际值和预测值
    ax.plot(date_col[-len(y_test_actual):], y_test_actual[:, 0], label='实际值', color='black', linewidth=2)
    ax.plot(date_col[-len(svm_predict):], svm_predict[:, 0], label='SVM预测值', color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(date_col[-len(lstm_predict):], lstm_predict[:, 0], label='LSTM预测值', color='red', linewidth=1.5,
            alpha=0.7)

    # 设置标题和标签
    ax.set_title('气温预测时间序列图', fontsize=16)
    ax.set_xlabel('日期', fontsize=14)
    ax.set_ylabel('温度 (°C)', fontsize=14)

    # 设置X轴日期格式和定位器
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 旋转X轴标签并增加间距
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax.legend(fontsize=12)

    # 优化布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('time_series_plot.png', dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()