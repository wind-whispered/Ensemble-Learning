import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from optimization.MFO import optimize_weights_mfo
import os
import warnings

warnings.filterwarnings('ignore')

# ========= Step 1: 加载数据 ==========
file_path = r"data/data_with_predictions.xlsx"
print("Loading data...")
data = pd.read_excel(file_path)

feature_columns = ['Type_school', 'grade', 'age', 'days_missed', 'sex', 'sore_throat', 'doctor_visit',
                   'cough', 'fever', 'vomiting', 'diarrhea', 'conjunctival_swelling', 'runny_nose',
                   'rash', 'pneumonia', 'abdominal_pain', 'stuffy_nose', 'abnormal_amygdala',
                   'headache', 'fatigue', 'sneezing', 'earache']

X = data[feature_columns]
y = data['cause']

categorical_features = ['Type_school', 'sex', 'sore_throat', 'doctor_visit', 'cough', 'fever',
                        'vomiting', 'diarrhea', 'conjunctival_swelling', 'runny_nose', 'rash',
                        'pneumonia', 'abdominal_pain', 'stuffy_nose', 'abnormal_amygdala',
                        'headache', 'fatigue', 'sneezing', 'earache']
X = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ========= Step 2: 定义模型 ==========
models = {
    'SVM': SVC(probability=True, random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    'DT': DecisionTreeClassifier(random_state=42),
    'NB': GaussianNB(),
    'BP': MLPClassifier(hidden_layer_sizes=(13, 8), activation='relu', solver='adam',
                        max_iter=200, random_state=42)
}

print("Training base models...")
for name, model in models.items():
    model.fit(X_train, y_train)

probs_all = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}


# ========= Step 3: 交叉熵损失函数 ==========
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ========= Step 4: 遍历组合并训练 ==========
result_base = "./result/集成模型/"
os.makedirs(result_base, exist_ok=True)

model_keys = list(models.keys())
print("Start model combinations training...")

for r in range(2, len(model_keys) + 1):
    for comb in combinations(model_keys, r):
        comb_name = "_".join(comb)
        print(f"\nProcessing combination: {comb_name}")

        probs_comb = [probs_all[k] for k in comb]
        dim = len(probs_comb)

        # 目标函数：使用当前组合的预测概率
        def objective(weights):
            weights = np.clip(weights, 0, 1)
            weights /= np.sum(weights)
            ensemble_probs = sum(w * p for w, p in zip(weights, probs_comb))
            return cross_entropy_loss(y_test, ensemble_probs)

        best_weights = optimize_weights_mfo(objective, dim=dim, flame_min=2*dim)

        # 保存结果
        ensemble_probs = sum(w * p for w, p in zip(best_weights, probs_comb))
        ensemble_pred = (ensemble_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred)
        rec = recall_score(y_test, ensemble_pred)
        f1 = f1_score(y_test, ensemble_pred)
        fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
        roc_auc = auc(fpr, tpr)

        # ========== 保存目录 ==========
        save_dir = os.path.join(result_base, f"{comb_name}")
        os.makedirs(save_dir, exist_ok=True)

        # 保存指标
        pd.DataFrame({
            'Accuracy': [acc],
            'Precision': [prec],
            'Recall': [rec],
            'F1 Score': [f1],
            'AUC': [roc_auc],
            'Weights': [list(best_weights)]
        }).to_excel(os.path.join(save_dir, "metrics.xlsx"), index=False)

        # 混淆矩阵
        cm = confusion_matrix(y_test, ensemble_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化（真实标签总数）
        labels = np.array([f"{v}\n({p:.1%})" for v, p in zip(cm.flatten(), cm_percent.flatten())]).reshape(cm.shape)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={"size": 30})
        plt.ylabel('True Labels', fontsize=30)
        plt.xlabel('Predicted Labels', fontsize=30)
        class_labels = le.inverse_transform([0, 1])
        plt.xticks([0.5, 1.5], class_labels, fontsize=30)
        plt.yticks([0.5, 1.5], class_labels, fontsize=30)
        plt.title(f'Confusion Matrix - {comb_name}', fontsize=30)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.pdf"), format='pdf')
        plt.close()

        # ROC 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'Ensemble ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate', fontsize=30)
        plt.ylabel('True Positive Rate', fontsize=30)
        # plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
        plt.title(f'ROC Curve - {comb_name}', fontsize=30)
        plt.legend(loc="lower right", fontsize=30)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, "roc_curve.pdf"), format='pdf')
        plt.close()

print("All combinations processed.")