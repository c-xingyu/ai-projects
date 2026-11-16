from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 2. 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 训练决策树
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. 预测
y_pred = clf.predict(X_test)

# 5. 评估
acc = accuracy_score(y_test, y_pred)
print(f"决策树准确率: {acc:.4f}")

print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred))

# 6. 混淆矩阵热力图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Digits Confusion Matrix (Decision Tree)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('day14_digits_confusion.png', dpi=150)
plt.show()