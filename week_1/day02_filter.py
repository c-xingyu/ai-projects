import pandas as pd

df = pd.read_csv("iris.csv")

# ========== 任务1：筛选 + 选列 ==========
# 要求：找出萼片长度(sepal_length)大于5.0的花，只显示前3列
# 提示：
#   筛选语法：df[df['列名'] > 数值]
#   选多列：df[['列1', '列2', '列3']]
#   两个操作可以连在一起写：df[条件][['列1','列2','列3']]

filtered = df[df['sepal_length']>5.0][['sepal_length','sepal_width','petal_length']]
print("=== 萼片长度>5.0的花（前3列）===")
print(filtered.head(10))  # 只打印前10行，省得刷屏

# ========== 任务2：取前N大 ==========
# 要求：找出花瓣长度(petal_length)最大的前5朵花
# 提示：搜 "pandas nlargest"

top5 = df.nlargest(5,'petal_length')
print("\n=== 花瓣长度最大的前5朵 ===")
print(top5)