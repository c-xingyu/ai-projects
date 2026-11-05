import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

clf = DecisionTreeClassifier(random_state=42)

# ========== 空1：5折交叉验证 ==========
# 提示：cross_val_score(模型, X, y, cv=几折)
scores = cross_val_score(clf, X, y, cv=5)#

print("=== 5折交叉验证结果 ===")
print(f"每折准确率: {scores}")
print(f"平均准确率: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")

# ========== 空2：对比不同模型稳定性 ==========
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=200)
}

print("\n=== 模型稳定性对比 ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)#
    print(f"{name}: 均值={scores.mean():.4f}, 标准差={scores.std():.4f}")