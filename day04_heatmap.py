import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# 去掉 species 列，只保留4个数值列，计算相关系数
numeric_df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
corr = numeric_df.corr()

# 画热力图
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Iris Feature Correlation Heatmap')
plt.savefig('day06_heatmap.png', dpi=150)
plt.show()