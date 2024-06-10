import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lassonet import LassoNetClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = [
    "ID", "Diagnosis", "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", 
    "Smoothness_mean", "Compactness_mean", "Concavity_mean", "Concave_points_mean", 
    "Symmetry_mean", "Fractal_dimension_mean", "Radius_se", "Texture_se", "Perimeter_se", 
    "Area_se", "Smoothness_se", "Compactness_se", "Concavity_se", "Concave_points_se", 
    "Symmetry_se", "Fractal_dimension_se", "Radius_worst", "Texture_worst", "Perimeter_worst", 
    "Area_worst", "Smoothness_worst", "Compactness_worst", "Concavity_worst", 
    "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst"
]
data = pd.read_csv(url, names=column_names)
# 数据预处理
X = data.iloc[:, 2:]
breakpoint()
y = data['Diagnosis'].map({'M': 1, 'B': 0})

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练LassoNet模型
model = LassoNetClassifier(hidden_dims=(100,))
path = model.path(X_train, y_train, return_state_dicts=True)

importance = model.feature_importances_.reshape(30)
# 归一化特征重要性
importance = importance.numpy()  # 将 Tensor 转换为 NumPy 数组
importance = (importance - importance.min()) / (importance.max() - importance.min())
# 排序特征重要性
indices = np.argsort(importance)
# 画图
features = column_names[2:]
plt.figure(figsize=(12, 10))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Normalized Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # 反转 y 轴，使最重要的特征在顶部
plt.tight_layout()  # 自动调整子图参数，使图形布局更紧凑
plt.savefig("uci_breast_cancer-importance.png")
breakpoint()


n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_test, y_pred))
    lambda_.append(save.lambda_)


fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, accuracy, ".-")
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, accuracy, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("classification accuracy")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("uci_breast_cancer-classification-training.png")
