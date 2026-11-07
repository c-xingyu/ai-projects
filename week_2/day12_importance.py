import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# ========== 空1：获取特征重要性 ==========
# 提示：DecisionTreeClassifier 训练后有一个属性叫 feature_importances_
importance = clf.feature_importances_

# 画图
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
plt.bar(features, importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Decision Tree Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('day12_importance.png', dpi=150)
plt.show()

for name, imp in zip(features, importance):
    print(f"{name}: {imp:.4f}")