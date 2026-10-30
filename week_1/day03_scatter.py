import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# 画散点图：萼片长度 vs 萼片宽度
plt.scatter(df['sepal_length'], df['sepal_width'])

# 加标签和标题
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris: Sepal Length vs Width')

# 保存为图片（GitHub上能看到）
plt.savefig('day04_scatter.png', dpi=150)

# 显示出来
plt.show()