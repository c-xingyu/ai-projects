# Pandas 与数据可视化 Week 1 学习复盘

> 项目：鸢尾花数据集探索（ai-projects）
> 目标：从零开始，掌握数据读取、筛选、分组、可视化全流程

---

## 一、本周学习地图

| 天数 | 文件 | 技能点 | 核心操作 |
|:---:|---|---|---|
| Day 1 | `day01_explore.py` | 数据初探 | `read_csv`, `shape`, `head`, `describe`, `value_counts` |
| Day 2 | `day02_filter.py` | 筛选与排序 | 布尔索引 `df[条件]`, 选多列 `df[[...]]`, `nlargest` |
| Day 3 | `day03_groupby.py` | 分组统计 | `groupby('列').mean()`, `groupby('列')[['col1','col2']].max()` |
| Day 4 | `day04_scatter.py` | 基础可视化 | `plt.scatter`, `xlabel/ylabel/title`, `savefig`, `show` |
| Day 5 | `day04_color.py` | 进阶可视化 | 按类别拆分数据、多色散点图、`plt.legend()` |

---

## 二、每日代码逐行讲解

### Day 1：认识数据

```python
import pandas as pd

df = pd.read_csv("iris.csv")
# 作用：把 CSV 文件读入内存，生成 DataFrame（二维表格）
# df 变量里现在有 150 行 × 5 列的数据

print(df.shape)
# 输出：(150, 5)
# 150 朵花，5 列（sepal_length, sepal_width, petal_length, petal_width, species）

print(df.head())
# 输出前 5 行，快速确认数据格式是否正确

print(df.describe())
# 只对数值列做统计：count（非空个数）、mean（均值）、std（标准差）、min/max、四分位数
# 作用：一眼看出数据的分布范围和异常值

print(df['species'].value_counts())
# 对类别列计数，输出每种花各有多少朵（setosa/versicolor/virginica 各 50）
```

**核心概念**：DataFrame 是 pandas 的核心数据结构，你可以把它理解成**带行列标签的 Excel 表格**。`df['列名']` 取出的是一列数据，类型叫 Series。

---

### Day 2：筛选与排序

#### 任务 1：按条件筛选 + 只选特定列

```python
filtered = df[df['sepal_length'] > 5.0][['sepal_length', 'sepal_width', 'petal_length']]
```

拆解为两步理解：

**第一步：生成布尔索引**
```python
df['sepal_length'] > 5.0
# 返回一个由 True/False 组成的 Series，长度 = 150
# 每一行对应一个判断结果
```

**第二步：用布尔索引做遮罩**
```python
df[df['sepal_length'] > 5.0]
# 只保留 True 对应的行，False 的行被过滤掉
# 结果是一个新的 DataFrame，行数变少（约 118 行）
```

**第三步：选列**
```python
[['sepal_length', 'sepal_width', 'petal_length']]
# 外层 [] 是 pandas 的列选择语法
# 内层 [] 是 Python 列表，列出你想保留的列名
# 注意：选单列用 df['列名']，选多列用 df[['列1','列2']]
```

#### 任务 2：取前 N 大

```python
top5 = df.nlargest(5, 'petal_length')
# 等价写法：df.sort_values('petal_length', ascending=False).head(5)
# 按 petal_length 降序排列，取前 5 行
```

---

### Day 3：分组统计（Split - Apply - Combine）

```python
group_mean = df.groupby('species').mean()
```

pandas 的 `groupby` 遵循经典的 **"拆分-应用-合并"** 三步模型：

1. **拆分（Split）**：按 `species` 列的值把 150 行拆成 3 组
   - setosa 组：50 行
   - versicolor 组：50 行
   - virginica 组：50 行

2. **应用（Apply）**：对每组分别调用 `.mean()`，计算各列平均值

3. **合并（Combine）**：把三组结果拼成一张新表，索引是 `species`

**进阶写法**：只计算指定列
```python
df.groupby('species')[['sepal_length', 'petal_length']].mean()
# 方括号 [[...]] 放在 groupby 之后
# 含义：先按 species 分组，但只对这两列做聚合计算
```

---

### Day 4：基础散点图（matplotlib 流程）

```python
import matplotlib.pyplot as plt

plt.scatter(df['sepal_length'], df['sepal_width'])
# scatter：散点图，第一个参数是 x 轴数据，第二个是 y 轴数据

plt.xlabel('Sepal Length (cm)')   # x 轴标签
plt.ylabel('Sepal Width (cm)')    # y 轴标签
plt.title('Iris: Sepal Length vs Width')  # 图标题

plt.savefig('day04_scatter.png', dpi=150)
# 保存为图片文件，dpi=150 控制清晰度
# 必须在 plt.show() 之前调用！

plt.show()
# 弹出窗口显示图像
# 注意：show() 会清空当前画布，如果在 savefig 之后调用，保存的图仍是正常的
#（但如果在 show 之后再 savefig，保存的会是空白图）
```

**matplotlib 绘图的标准流程**：
1. 画图（scatter/plot/bar...）
2. 加装饰（标题、标签、图例）
3. 保存（savefig）
4. 显示（show）

---

### Day 5：多色散点图（按类别上色）

```python
# 第一步：把数据按类别拆成三份
setosa = df[df['species'] == 'setosa']
versicolor = df[df['species'] == 'versicolor']
virginica = df[df['species'] == 'virginica']

# 第二步：分别画三次 scatter，每次指定不同颜色
plt.scatter(setosa['sepal_length'], setosa['sepal_width'],
            color='red', label='setosa')
plt.scatter(versicolor['sepal_length'], versicolor['sepal_width'],
            color='green', label='versicolor')
plt.scatter(virginica['sepal_length'], virginica['sepal_width'],
            color='blue', label='virginica')

plt.legend()  # 根据 label 参数自动生成图例
```

**为什么要拆成三次画？**

matplotlib 不会自动识别 DataFrame 中的类别列并按颜色分开。它的设计理念是"你要什么，就画什么"。所以手动拆分数据、分三次调用 scatter、指定不同颜色，是最直观的做法。

**从这张图能观察到什么？**
- 红色点（setosa）集中在左上方（萼片短但宽）
- 绿色和蓝色点有重叠，说明 versicolor 和 virginica 在"萼片"这两个维度上不太好区分
- 这说明：**如果只用 sepal 的两个特征做分类，setosa 很容易分开，但另外两种花会混淆**

---

## 三、核心概念速查表

| 概念 | 一句话解释 |
|---|---|
| DataFrame | pandas 的二维表格，带行索引和列名 |
| Series | pandas 的一维数组，类似带标签的列表 |
| 布尔索引 | `df[条件]`，用 True/False 筛选行 |
| 链式操作 | `df[条件][['列1','列2']]`，先筛选行，再选列 |
| groupby | 先按某列分组，再对每组做聚合计算 |
| scatter | 散点图，看两个连续变量的关系 |
| savefig vs show | savefig 存文件，show 弹窗口；savefig 要在 show 之前 |
| label + legend | label 给散点命名，legend 在角落显示图例 |

---

## 四、常见错误清单（你踩过的坑）

| 错误现象 | 原因 | 解决 |
|---|---|---|
| `git status` 报 "not a git repository" | 当前文件夹没有 `.git` | `cd` 到有 `.git` 的目录 |
| `git add` 报 "did not match any files" | 路径写错，或当前目录层级不对 | 用 `pwd` 确认位置，路径从当前位置开始写 |
| `git push` 报 "master does not match" | 本地分支是 `main`，推送写了 `master` | `git push -u origin main` |
| 改了代码运行没变化 | 运行的是 pip 安装的原版，不是当前文件 | 直接运行 `python 文件名.py` |
| `savefig` 保存空白图 | `savefig` 在 `show()` 之后调用 | 调整顺序：先 savefig，再 show |

---

## 五、下一步预告（Day 5 开始——从数据可视化进入机器学习）

Week 2 你将学习 **scikit-learn（sklearn）**，用代码实现真正的机器学习流程：

| 天数 | 内容 |
|---|---|
| Day 5 | 数据集划分：训练集 vs 测试集（`train_test_split`） |
| Day 6 | 第一个模型：决策树分类器（`DecisionTreeClassifier`） |
| Day 7 | 模型评估：准确率、混淆矩阵 |
| Day 8 | 模型对比：决策树 vs KNN vs 逻辑回归 |
| Day 9 | 交叉验证（Cross Validation） |
| Day 10 | 特征重要性分析 |

**Week 2 结束后，你将拥有一个完整的机器学习项目**，可以写进简历和复试自述中。

---

## 六、给跨考生的面试话术储备

如果被问到"这个项目你做了什么"，可以这样说：

> "这是我学习数据分析时的入门项目。我用 pandas 对经典的鸢尾花数据集做了完整的探索性分析：包括数据清洗与筛选、分组统计各花种的特征均值、用 matplotlib 和 seaborn 绘制了散点图和热力图。通过可视化我发现 setosa 在萼片维度上与其他两种花有明显分离，这让我理解了**为什么简单的分类模型也能在这个数据集上取得高准确率**。整个项目用 Git 管理，代码和可视化结果都开源在 GitHub 上。"

---

*生成时间：2026-10-30*
*对应项目：https://github.com/c-xingyu/ai-projects*
