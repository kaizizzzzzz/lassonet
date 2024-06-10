import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from lassonet import LassoNetClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载特定的indel文件夹中的所有CSV文件
data_files = {
    'train': 'ProteinGym_indels/A0A1J4YT16_9PROT_Davidi_2020.csv'
}
dataset = load_dataset('ICML2022/ProteinGym', data_files=data_files)
# breakpoint()
# 将数据集转换为 pandas DataFrame
df = pd.concat([pd.DataFrame(dataset[split]) for split in dataset])

# 查看数据集的前几行
print(df.head())

# 定义20种标准氨基酸
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# 创建一个独热编码器并进行拟合
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(list(amino_acids)).reshape(-1, 1))

def encode_sequence(sequence):
    # 这里假设你的序列编码逻辑
    encoded = [ord(char) for char in sequence]
    return encoded

def pad_sequence(encoded_sequence, max_length):
    return np.pad(encoded_sequence, (0, max_length - len(encoded_sequence)), mode='constant')

# 假设最大长度为目标长度
max_length = max(df['mutant'].apply(len))
# breakpoint()
# 将序列编码为数值特征
X_encoded = np.array([pad_sequence(encode_sequence(row['mutant']), max_length) for _, row in df.iterrows()])
y_dms_score = df['DMS_score'].values
y_label = df['DMS_score_bin'].values

# 转换为 torch tensor
X_encoded = torch.tensor(X_encoded, dtype=torch.float)
y_dms_score = torch.tensor(y_dms_score, dtype=torch.float)
y_label = torch.tensor(y_label, dtype=torch.long)

# 拆分数据集
X_train, X_test, y_train_dms_score, y_test_dms_score, y_train_label, y_test_label = train_test_split(
    X_encoded, y_dms_score, y_label, test_size=0.2, random_state=42)

print("Train and test sets created successfully.")

model = LassoNetClassifier(
    hidden_dims=(100,100),
    M=30,
)

path = model.path(X_train, y_train_label, return_state_dicts=True)

n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_test_label.cpu(), y_pred.cpu()))
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

plt.savefig("proteingym-classification-training.png")

