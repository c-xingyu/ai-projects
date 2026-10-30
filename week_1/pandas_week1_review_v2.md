# Pandas 与数据可视化 Week 1 学习复盘

> 项目：鸢尾花数据集探索（ai-projects）
> 目标：从零开始，掌握数据读取、筛选、分组、可视化全流程

---

## 一、本周实际文件清单

按你本地的文件名整理：

| 实际文件名 | 对应天数 | 技能点 |
|---|---|---|
| `day01explore.py` | Day 1 | 数据初探：读取、形状、统计、计数 |
| `day02_filter.py` | Day 2 | 筛选与排序：布尔索引、选多列、TopN |
| `day03_groupby.py` | Day 3 | 分组统计：按种类聚合、指定列计算 |
| `day03_scatter.py` | Day 4 | 基础可视化：matplotlib 散点图 |
| `day03_color.py` | Day 5 | 进阶可视化：多色散点图 + 图例 |
| `day04_heatmap.py` | Day 6 | 热力图：特征相关性分析（seaborn） |

---

## 二、每日代码逐行讲解

### Day 1：认识数据（day01explore.py）

```python
import pandas as pd

df = pd.read_csv("iris.csv")
# 把 CSV 文件读入内存，生成 DataFrame（二维表格）
# df 里有 150 行 × 5 列的数据

print(df.shape)           # (150, 5)：150朵花，5个特征列
print(df.head())          # 前5行，快速检查数据格式
print(df.describe())      # 数值列统计：均值、标准差、最值、四分位数
print(df['species'].value_counts())  # 类别计数：三种花各50朵
```

**核心概念**：DataFrame 是 pandas 的核心，理解成**带行列标签的 Excel 表格**。`df['列名']` 取出的是 Series（一列数据）。

---

### Day 2：筛选与排序（day02_filter.py）

#### 布尔索引 + 选列

```python
filtered = df[df['sepal_length'] > 5.0][['sepal_length', 'sepal_width', 'petal_length']]
```

拆解三步：

1. **`df['sepal_length'] > 5.0`** → 生成 150 个 True/False，这叫**布尔索引**
2. **`df[...]`** → 只保留 True 的行，False 的行被过滤
3. **`[['列1','列2','列3']]`** → 外层是 pandas 列选择，内层是 Python 列表

**关键区别**：
- `df['列名']` → 取一列（Series）
- `df[['列1','列2']]` → 取多列（DataFrame）

#### TopN 排序

```python
df.nlargest(5, 'petal_length')
# 按 petal_length 降序，取前 5 行
# 等价：df.sort_values('petal_length', ascending=False).head(5)
```

---

### Day 3：分组统计（day03_groupby.py）

```python
df.groupby('species').mean()
```

**底层逻辑：Split → Apply → Combine**

1. **拆分**：按 `species` 把 150 行拆成 3 组（setosa/versicolor/virginica 各 50 行）
2. **应用**：每组调用 `.mean()` 算平均值
3. **合并**：三组结果拼成新表，索引是 species

**只算指定列**：
```python
df.groupby('species')[['sepal_length', 'petal_length']].mean()
# groupby 后加 [[...]]，表示"只对这些列做聚合"
```

---

### Day 4：基础散点图（day03_scatter.py）

```python
import matplotlib.pyplot as plt

plt.scatter(df['sepal_length'], df['sepal_width'])
# scatter：散点图，x轴、y轴各传一列数据

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris: Sepal Length vs Width')

plt.savefig('day03_scatter.png', dpi=150)
# 保存图片，dpi 控制清晰度
# 必须在 plt.show() 之前！

plt.show()
# 弹出窗口显示
```

**标准流程**：画图 → 加装饰（标题/标签） → 保存 → 显示

**坑**：`savefig` 在 `show()` 之后调用，保存的是空白图。

---

### Day 5：多色散点图（day03_color.py）

```python
# 手动按类别拆成三份
setosa = df[df['species'] == 'setosa']
versicolor = df[df['species'] == 'versicolor']
virginica = df[df['species'] == 'virginica']

# 分别画三次，每次不同颜色
plt.scatter(setosa['sepal_length'], setosa['sepal_width'],
            color='red', label='setosa')
plt.scatter(versicolor['sepal_length'], versicolor['sepal_width'],
            color='green', label='versicolor')
plt.scatter(virginica['sepal_length'], virginica['sepal_width'],
            color='blue', label='virginica')

plt.legend()  # 根据 label 自动生成图例
```

**为什么要拆三次？** matplotlib 不会自动按类别上色，"你要什么颜色，就画什么颜色"。

**观察结论**：红色（setosa）集中在左上，与其他两种花明显分离。这说明只用萼片两个维度，setosa 就很容易被分类器识别。

---

### Day 6：热力图——特征相关性分析（day04_heatmap.py）

这是本周最后一个任务，也是**从"描述数据"迈向"理解数据"**的关键一步。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

# 只保留4个数值列，去掉类别列 species
numeric_df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# 计算相关系数矩阵
corr = numeric_df.corr()
# corr 是一个 4×4 的表格，每个值在 -1 到 1 之间
# 1 表示完全正相关，-1 表示完全负相关，0 表示无关

# 用 seaborn 画热力图
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
# annot=True：在每个格子里显示具体数字
# cmap='coolwarm'：颜色方案，红=正相关，蓝=负相关
# fmt='.2f'：数字保留两位小数

plt.title('Iris Feature Correlation Heatmap')
plt.savefig('day04_heatmap.png', dpi=150)
plt.show()
```

#### 这张图在说什么？

**相关系数解读**：
- `petal_length` 和 `petal_width`：约 **0.96**（深红色）→ 高度正相关
  - 花瓣越长，几乎必然越宽
  - **启示**：如果做特征选择，这两个保留一个就够了，否则信息冗余
  
- `sepal_length` 和 `petal_length`：约 **0.87**（红色）→ 强正相关
  - 萼片长的花，花瓣也倾向于长

- `sepal_width` 和 `petal_length`：约 **-0.43**（浅蓝）→ 弱负相关
  - 萼片越宽，花瓣倾向于越短（但不明显）

**复试时能说的话**：
> "我通过相关性热力图发现，petal_length 和 petal_width 的相关系数高达 0.96，这说明它们携带的信息高度重叠。在后续做机器学习特征选择时，我可能会考虑降维或去掉其中一个，以减少多重共线性对模型的影响。"

---

## 三、核心概念速查表

| 概念 | 一句话解释 |
|---|---|
| DataFrame | pandas 二维表格，带行索引和列名 |
| Series | pandas 一维数组，类似带标签的列表 |
| 布尔索引 | `df[条件]`，用 True/False 筛选行 |
| 链式操作 | `df[条件][['列1','列2']]`，先筛选行，再选列 |
| groupby | 先按某列分组，再对每组做聚合计算 |
| scatter | 散点图，看两个连续变量的关系 |
| savefig vs show | savefig 存文件，show 弹窗口；savefig 要在 show 之前 |
| label + legend | label 给散点命名，legend 在角落显示图例 |
| corr() | 计算各列之间的皮尔逊相关系数 |
| heatmap | 用颜色深浅可视化矩阵数值，适合看相关性 |

---

## 四、常见错误清单

| 错误现象 | 原因 | 解决 |
|---|---|---|
| `git status` 报 "not a git repository" | 当前文件夹没有 `.git` | `cd` 到有 `.git` 的目录 |
| `git add` 报 "did not match any files" | 路径层级写错 | `pwd` 确认位置 |
| `git push` 分支不匹配 | 本地是 `main`，推送写了 `master` | `git push -u origin main` |
| 改了代码运行没变化 | 运行的是 pip 安装的原版 | 直接 `python 文件名.py` |
| `savefig` 空白 | `savefig` 在 `show()` 之后 | 调整顺序 |
| `No module named seaborn` | 没装 seaborn | `pip install seaborn` |

---

## 五、Week 1 到 Week 2 的过渡

Week 1 你完成了 **EDA（探索性数据分析）**：
- 知道数据长什么样（head/shape）
- 知道数据分布（describe）
- 知道不同类别的差异（groupby）
- 知道特征之间的关系（scatter/heatmap）

Week 2 你将进入 **机器学习建模**：
- 把数据分成训练集和测试集
- 用 sklearn 训练分类模型
- 评估模型好坏
- 对比不同模型的效果

**复试时的叙事逻辑**：
> "我首先对数据做了探索性分析，发现 setosa 在萼片维度上可分离，且花瓣长度和宽度高度相关。基于此，我构建了决策树等分类模型..."

这就是从"理解数据"到"构建模型"的完整故事线。

---

*生成时间：2026-10-30*
*对应项目：https://github.com/c-xingyu/ai-projects*
