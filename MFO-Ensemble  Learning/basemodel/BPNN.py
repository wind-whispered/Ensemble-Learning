import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 加载数据
print("Loading data...")
file_path = r"../data/data_with_predictions.xlsx"
data = pd.read_excel(file_path)
print("Data loaded.")

# 数据预处理
print("Preprocessing data...")
# 定义特征列
feature_columns = ['Type_school', 'grade', 'age', 'days_missed', 'sex', 'sore_throat', 'doctor_visit',
                   'cough', 'fever', 'vomiting', 'diarrhea', 'conjunctival_swelling', 'runny_nose',
                   'rash', 'pneumonia', 'abdominal_pain', 'stuffy_nose', 'abnormal_amygdala',
                   'headache', 'fatigue', 'sneezing', 'earache']

X = data[feature_columns]
y = data['cause']

# 编码分类变量
categorical_features = ['Type_school', 'sex', 'sore_throat', 'doctor_visit', 'cough', 'fever',
                        'vomiting', 'diarrhea', 'conjunctival_swelling', 'runny_nose', 'rash',
                        'pneumonia', 'abdominal_pain', 'stuffy_nose', 'abnormal_amygdala',
                        'headache', 'fatigue', 'sneezing', 'earache']

# 对分类特征进行one-hot编码，使用prefix参数确保列名的一致性
X = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features)

# 将所有数据转换为float32类型
X = X.astype(np.float32)

# 编码目标变量
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Data preprocessing completed.")

# 数据集划分
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Data split completed.")

# 将数据转换为 PyTorch 的 Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# 定义反向传播神经网络
class BPNNModel(nn.Module):
    def __init__(self, input_size):
        super(BPNNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 13)  # 输入层到第一隐层
        self.layer2 = nn.Linear(13, 8)  # 第一隐层到第二隐层
        self.layer3 = nn.Linear(8, 2)  # 第二隐层到输出层

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# 初始化模型、损失函数和优化器
input_size = X_train.shape[1]
model = BPNNModel(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
print("Training the BPNN model...")
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Model training completed.")

# 测试模型并计算性能指标
print("Making predictions on the test set...")
model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    _, y_pred_tensor = torch.max(outputs_test, 1)
    y_pred = y_pred_tensor.numpy()
    y_prob_tensor = torch.softmax(outputs_test, dim=1)
    y_prob = y_prob_tensor.numpy()

# 计算性能指标
print("Calculating performance metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# 将性能指标保存到 Excel 文件
metrics_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1score]
})
metrics_filename = '../result/BPNN/model_metrics.xlsx'
metrics_df.to_excel(metrics_filename, index=False)
print(f"Performance metrics saved to {metrics_filename}")

# 绘制混淆矩阵
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化（真实标签总数）
labels = np.array([f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_percent.flatten())]).reshape(cm.shape)
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={"size": 14})
plt.ylabel('True labels', fontsize=16)
plt.xlabel('Predicted labels', fontsize=16)
class_labels = le.inverse_transform([0, 1])
plt.xticks([0.5, 1.5], class_labels, fontsize=16)
plt.yticks([0.5, 1.5], class_labels, fontsize=16)
plt.title('Confusion Matrix-BPNN', fontsize=16)
cm_filename = '../result/BPNN/confusion_matrix.pdf'
plt.savefig(cm_filename, format='pdf')
plt.close()
print(f"Confusion matrix saved to {cm_filename}")

# 绘制 ROC 曲线
# print("Plotting ROC curves...")
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(2):
#     fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# 
# plt.figure(figsize=(8, 6))
# colors = ['darkorange', 'navy']
# for i, color in zip(range(2), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label=f'ROC curve for {class_labels[i]} (area = {roc_auc[i]:0.2f})')
# 
# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=16)
# plt.ylabel('True Positive Rate', fontsize=16)
# plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
# plt.legend(loc="lower right", fontsize=12)
# roc_filename = '../result/BPNN/roc_curves.pdf'
# plt.savefig(roc_filename, format='pdf')
# plt.close()
# print(f"ROC curves saved to {roc_filename}")

fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Ensemble ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve -BPNN', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
roc_filename = './result/BPNN/roc_curves.pdf'
plt.savefig(roc_filename, format='pdf')
plt.close()
print(f"ROC curves saved to {roc_filename}")


# 计算特征重要性（通过置换重要性）
print("Calculating feature importance...")
baseline_accuracy = accuracy  # 基线准确率
n_repeats = 10  # 置换的重复次数
feature_importances = {}

# 创建原始特征到编码后特征的映射
feature_to_columns = {}
for feature in feature_columns:
    if feature in categorical_features:
        # 获取所有以 'feature_' 开头的列名
        columns = [col for col in X.columns if col.startswith(feature + '_')]
    else:
        columns = [feature]
    feature_to_columns[feature] = columns

# 针对22个特征逐一计算
for feature in tqdm(feature_columns, desc="Calculating Feature Importances"):
    score_drops = []
    for _ in range(n_repeats):
        # 对该特征的所有列进行整体置换
        X_test_permuted = X_test.copy()
        for col in feature_to_columns[feature]:
            X_test_permuted[col] = np.random.permutation(X_test_permuted[col].values)
        X_test_permuted_tensor = torch.tensor(X_test_permuted.values, dtype=torch.float32)

        # 测试置换后的数据集性能
        with torch.no_grad():
            outputs_test_permuted = model(X_test_permuted_tensor)
            _, y_pred_permuted = torch.max(outputs_test_permuted, 1)
            permuted_accuracy = accuracy_score(y_test, y_pred_permuted.numpy())

        # 计算性能下降（基线准确率 - 置换后准确率）
        score_drops.append(baseline_accuracy - permuted_accuracy)

    # 平均性能下降作为特征的重要性评分
    feature_importances[feature] = np.mean(score_drops)

# 将变量贡献度保存到 Excel 文件
importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
importance_filename = '../result/BPNN/feature_importance.xlsx'
importance_df.to_excel(importance_filename, index=False)
print(f"Feature importances saved to {importance_filename}")

# 绘制变量贡献度图
print("Plotting feature importance...")
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], color='blue')
plt.xlabel("Permutation Importance (Decrease in Accuracy)", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.title("Feature Importance", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
importance_plot_filename = '../result/BPNN/feature_importance.pdf'
plt.savefig(importance_plot_filename, format='pdf', bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to {importance_plot_filename}")

print("All tasks completed.")
