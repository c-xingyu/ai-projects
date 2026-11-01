import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

# ========== 任务1：准备特征X和标签y ==========
# X = 特征（4个数值列），y = 标签（species种类）
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# ========== 任务2：划分数据集 ==========
# test_size=0.2 表示 20% 做测试集，80% 做训练集
# random_state=42 保证每次运行划分结果一样（可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 任务3：验证划分结果 ==========
print("=== 数据集划分结果 ===")
print(f"总样本数: {len(X)}")
print(f"训练集样本数: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"测试集样本数: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

print("\n=== 训练集前5行 ===")
print(X_train.head())
print(y_train.head())

print("\n=== 测试集前5行 ===")
print(X_test.head())
print(y_test.head())

print("\n=== 训练集中各类别数量 ===")
print(y_train.value_counts())

print("\n=== 测试集中各类别数量 ===")
print(y_test.value_counts())