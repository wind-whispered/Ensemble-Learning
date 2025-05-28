import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
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

# 编码目标变量
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Data preprocessing completed.")

# 数据集划分
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Data split completed.")

# 初始化并训练随机森林分类器
print("Training the Random Forest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
with tqdm(total=1, desc="Training Progress") as pbar:
    rf_model.fit(X_train, y_train)
    pbar.update(1)
print("Model training completed.")

# 预测测试集
print("Making predictions on the test set...")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)

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
metrics_filename = 'result/Random_Forest/model_metrics.xlsx'
metrics_df.to_excel(metrics_filename, index=False)
print(f"Performance metrics saved to {metrics_filename}")

# confusion matrix
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化（真实标签总数）
labels = np.array([f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_percent.flatten())]).reshape(cm.shape)
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={"size": 30})
plt.ylabel('True Labels', fontsize=30)
plt.xlabel('Predicted Labels', fontsize=30)
class_labels = le.inverse_transform([0, 1])
plt.xticks([0.5, 1.5], class_labels, fontsize=30)
plt.yticks([0.5, 1.5], class_labels, fontsize=30)
plt.title('Confusion Matrix - RF', fontsize=30)
plt.tight_layout()
cm_filename = './result/Random_Forest/confusion_matrix.pdf'
plt.savefig(cm_filename, format='pdf')
plt.close()
print(f"Confusion matrix saved to {cm_filename}")

# 绘制 ROC 曲线
print("Plotting ROC curves...")
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Ensemble ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=30)
plt.ylabel('True Positive Rate', fontsize=30)
plt.title('ROC Curve - RF', fontsize=30)
plt.legend(loc="lower right", fontsize=30)
plt.grid(alpha=0.3)
roc_filename = './result/Random_Forest/roc_curve.pdf'
plt.savefig(roc_filename, format='pdf')
plt.close()
print(f"ROC curves saved to {roc_filename}")

# 计算变量贡献度（通过置换重要性）
print("Calculating feature importance...")

# 创建原始特征到编码后特征的映射
feature_to_columns = {}
for feature in feature_columns:
    if feature in categorical_features:
        # 由于get_dummies使用了prefix参数，列名将以'feature_value'的形式出现
        columns = [col for col in X.columns if col.startswith(feature + '_')]
    else:
        columns = [feature]
    feature_to_columns[feature] = columns

# 计算基线准确率
baseline_score = rf_model.score(X_test, y_test)

# 初始化变量贡献度字典
feature_importances = {}
n_repeats = 10  # 重复次数

for feature in tqdm(feature_columns, desc="Calculating Feature Importances"):
    importances = []
    for _ in range(n_repeats):
        X_test_permuted = X_test.copy()
        # 对该特征的所有列进行置换
        for col in feature_to_columns[feature]:
            X_test_permuted[col] = np.random.permutation(X_test_permuted[col].values)
        # 计算置换后的准确率
        permuted_score = rf_model.score(X_test_permuted, y_test)
        # 计算重要性（基线准确率 - 置换后准确率）
        importance = baseline_score - permuted_score
        importances.append(importance)
    # 计算重要性的平均值
    feature_importances[feature] = np.mean(importances)

# 将变量贡献度保存到 Excel 文件
importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
importance_filename = 'result/Random_Forest/feature_importance.xlsx'
importance_df.to_excel(importance_filename, index=False)
print(f"Feature importances saved to {importance_filename}")

# 绘制变量贡献度图
print("Plotting feature importance...")
# 对重要性值进行排序
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], color='blue')
plt.xlabel("Permutation Importance (Decrease in Accuracy)", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.title("Feature Importance", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
importance_plot_filename = 'result/Random_Forest/feature_importance.pdf'
plt.savefig(importance_plot_filename, format='pdf', bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to {importance_plot_filename}")

print("All tasks completed.")
