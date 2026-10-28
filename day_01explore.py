import pandas as pd

# 读取数据
df = pd.read_csv("iris.csv")

# 看数据长什么样
print("=== 数据形状（行数, 列数）===")
print(df.shape)

print("\n=== 前5行 ===")
print(df.head())

print("\n=== 基础统计 ===")
print(df.describe())

print("\n=== 每种花有多少朵 ===")
print(df['species'].value_counts())