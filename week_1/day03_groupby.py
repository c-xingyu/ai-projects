import pandas as pd

df = pd.read_csv("iris.csv")

# 任务1：按花的种类(species)分组，计算每组的平均值
# 你的代码：
group_mean=df.groupby('species').mean()
#groupby函数：
#语法：F.groupby('xxxx'),按照xx分组
print("=== 按种类分组均值 ===")
print(group_mean)

# 任务2：按花的种类分组，计算每组的最大值
group_max=df.groupby('species').max()
print("\n=== 按种类分组最大值 ===")
print(group_max)

# 任务3：进阶——只计算 sepal_length 和 petal_length 两列的均值
two_col_mean=df.groupby('species')[['sepal_length','petal_length']].mean()
print("\n=== 只算两列的均值 ===")
print(two_col_mean)