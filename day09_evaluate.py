import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# ========== 空1：导入混淆矩阵和分类报告的工具 ==========
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("iris.csv")
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ========== 空2：计算并打印混淆矩阵 ==========
# 提示：混淆矩阵函数需要两个参数，一个是真实标签，一个是预测标签
cm = confusion_matrix(y_test, y_pred)
print("=== 混淆矩阵 ===")
print(cm)

# ========== 空3：打印分类报告 ==========
# 提示：classification_report 同样需要真实标签和预测标签
print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred))

# ========== 空4（进阶）：把混淆矩阵画成热力图 ==========
# 提示：用 seaborn.heatmap，参考你 Day 6 的代码
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
# ========== 空5：加标题和标签 ==========
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('day09_confusion_matrix.png', dpi=150)
plt.show()