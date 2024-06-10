import pandas as pd
import numpy as np
import os
import zipfile
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lassonet import LassoNetClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 下载并解压数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
zip_path = "UCI_HAR_Dataset.zip"
if not os.path.exists(zip_path):
    urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("UCI_HAR_Dataset")

# 加载特征名称
features_path = "UCI_HAR_Dataset/UCI HAR Dataset/features.txt"
features_df = pd.read_csv(features_path, delim_whitespace=True, header=None)
feature_names = features_df[1].tolist()

# 加载训练数据
train_data_path = "UCI_HAR_Dataset/UCI HAR Dataset/train/X_train.txt"
train_labels_path = "UCI_HAR_Dataset/UCI HAR Dataset/train/y_train.txt"
X_train = pd.read_csv(train_data_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(train_labels_path, delim_whitespace=True, header=None)

# 加载测试数据
test_data_path = "UCI_HAR_Dataset/UCI HAR Dataset/test/X_test.txt"
test_labels_path = "UCI_HAR_Dataset/UCI HAR Dataset/test/y_test.txt"
X_test = pd.read_csv(test_data_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(test_labels_path, delim_whitespace=True, header=None)

# 数据预处理
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练 LassoNet 模型
model = LassoNetClassifier(hidden_dims=(100,))
path = model.path(X_train, y_train, return_state_dicts=True)

# 提取特征重要性
importance = model.feature_importances_

# 归一化特征重要性
importance = (importance - importance.min()) / (importance.max() - importance.min())

# 排序特征重要性
indices = np.argsort(importance)[-50:]

# 画图
plt.figure(figsize=(12, 10))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Normalized Importance')
plt.title('Top 50 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.savefig("uci_human_activity_recognition-importance.png")

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

plt.savefig("uci_human_activity_recognition-training.png")