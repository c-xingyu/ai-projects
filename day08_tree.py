import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ========== 1. 准备数据 ==========
df = pd.read_csv("iris.csv")
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 2. 创建模型 ==========
# DecisionTreeClassifier = 决策树分类器
# 它会自动从训练数据中学习"如果花瓣长度>2.5，那么是versicolor"这样的规则
clf = DecisionTreeClassifier(random_state=42)

# ========== 3. 训练模型 ==========
# fit = "拟合"，也就是让模型看数据、学规律
clf.fit(X_train, y_train)
print("模型训练完成！")

# ========== 4. 预测测试集 ==========
# predict = 用学好的模型去猜测试集的标签
y_pred = clf.predict(X_test)

# ========== 5. 评估准确率 ==========
# accuracy_score = 猜对了多少 / 总共多少
acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {acc:.4f} ({acc*100:.2f}%)")

# ========== 额外：看看预测对了哪些 ==========
print("\n=== 预测结果对比 ===")
print("实际标签:  ", list(y_test))
print("预测标签:  ", list(y_pred))