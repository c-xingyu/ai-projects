import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# 先按种类把数据分成三份（用Day2学过的筛选）
setosa = df[df['species'] == 'setosa']
versicolor = df[df['species'] == 'versicolor']
virginica = df[df['species'] == 'virginica']

# 分别画三种颜色
plt.scatter(setosa['sepal_length'], setosa['sepal_width'], 
            color='red', label='setosa')
plt.scatter(versicolor['sepal_length'], versicolor['sepal_width'], 
            color='green', label='versicolor')
plt.scatter(virginica['sepal_length'], virginica['sepal_width'], 
            color='blue', label='virginica')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris: Sepal Length vs Width by Species')
plt.legend()  # 显示图例，自动用上面写的 label
plt.savefig('day05_color.png', dpi=150)
plt.show()