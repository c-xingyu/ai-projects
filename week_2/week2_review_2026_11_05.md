# Week 2 学习复盘：sklearn 机器学习 + Linux 基础

> 时间：2026.11.03 ~ 2026.11.05
> 双线作战：考研 AI 项目（线 A）+ 具身智能实习准备（线 B）

---

## 线 A：sklearn 机器学习入门（鸢尾花项目 Week 2）

### Day 7：数据集划分——train_test_split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**核心理解**：
- `test_size=0.2`：20% 做测试集，80% 做训练集
- `random_state=42`：固定随机种子，保证每次划分结果可复现
- 训练集 120 朵，测试集 30 朵，三种花分布均匀（40/41/39）

**运行结果**：
```
总样本数: 150
训练集样本数: 120 (80%)
测试集样本数: 30 (20%)
训练集中各类别: versicolor 41, setosa 40, virginica 39
```

---

### Day 8：第一个机器学习模型——决策树

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)          # 训练：让模型学规律
y_pred = clf.predict(X_test)       # 预测：用学好的模型猜测试集
acc = accuracy_score(y_test, y_pred)
```

**核心理解**：
- `fit()` = 学习过程，模型从训练数据中提取规则
- `predict()` = 推理过程，模型对新数据做判断
- 决策树通过"如果花瓣长度 > 2.5，那么是 versicolor"这样的规则做分类

**运行结果**：准确率 **100%**（30/30 全对）

---

### Day 9：模型评估——混淆矩阵与分类报告

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

**核心理解**：
- **混淆矩阵**：行是真实标签，列是预测标签，对角线是猜对的数量
- **Precision**：模型说"这是 A"的时候，有多少真的是 A
- **Recall**：所有真正的 A，模型找出了多少
- **F1-score**：Precision 和 Recall 的调和平均

**运行结果**：
```
混淆矩阵:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

分类报告:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11
```

**可视化**：用 seaborn heatmap 画出蓝色调的混淆矩阵热力图。

---

### Day 10：模型对比——决策树 vs KNN vs 逻辑回归

```python
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(max_iter=200)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
```

**核心理解**：
- **同一套数据、同一套划分**，用不同模型跑，公平对比
- 用字典 `results` 收集结果，最后用 `max(results, key=results.get)` 找出最佳模型

**运行结果**：
```
DecisionTree: 1.0000
KNN: 1.0000
LogisticRegression: 1.0000
最佳模型: DecisionTree (1.0000)
```

> 注：鸢尾花数据集本身很简单，三个模型都能满分。换复杂数据集后差距会显现。

---

### Day 11：交叉验证——排除"划分运气"

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5)
print(f"每折准确率: {scores}")
print(f"平均准确率: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")
```

**核心理解**：
- **K 折交叉验证**：把数据分成 K 份，轮流拿一份做测试、其余做训练，跑 K 次
- **均值**：模型的平均表现水平
- **标准差**：模型的稳定性，越小越稳定
- 交叉验证比单次划分更可靠，排除了"运气好分到简单测试集"的偏差

**运行结果**：
```
DecisionTree: 均值=0.9533, 标准差=0.0340
KNN: 均值=0.9733, 标准差=0.0249
LogisticRegression: 均值=0.9733, 标准差=0.0249
```

**结论**：KNN 和逻辑回归不仅平均分更高，标准差也更小，说明它们在这个数据集上**更稳定**。

---

### Day 12：特征重要性——理解决策树在看什么

```python
importance = clf.feature_importances_

plt.bar(features, importance)
for name, imp in zip(features, importance):
    print(f"{name}: {imp:.4f}")
```

**核心理解**：
- `feature_importances_` 是决策树训练后自动计算的：每个特征对分类的贡献度
- 值越大，说明这个特征越重要
- 特征重要性可以指导**特征选择**：去掉重要性低的特征，简化模型

**运行结果（柱状图）**：
- `petal_length` 和 `petal_width` 的柱子最高
- `sepal_width` 几乎为 0

**结论**：决策树主要靠**花瓣尺寸**做判断，萼片宽度几乎没用。这和 Day 6 热力图看到的 "petal_length 与 petal_width 高度相关" 互相印证。

---

### Week 2 sklearn 核心流程总结

```
数据准备 (X, y)
    ↓
数据划分 (train_test_split)
    ↓
创建模型 (DecisionTree / KNN / LogisticRegression)
    ↓
训练模型 (model.fit)
    ↓
预测评估 (model.predict → accuracy_score)
    ↓
深度评估 (confusion_matrix / classification_report)
    ↓
稳定性验证 (cross_val_score)
    ↓
可解释性分析 (feature_importances_)
```

这是机器学习的**标准完整流程**，复试时讲项目按这个结构讲。

---

## 线 B：Linux 基础（具身智能实习准备）

### 第一组：文件与目录操作

| 命令 | 作用 | 实际运行结果 |
|:---|:---|:---|
| `pwd` | 显示当前路径 | `/home/cxing` |
| `ls` | 列出文件 | 显示当前目录内容 |
| `ls -l` | 详细列表（权限/大小/日期） | `-rw-r--r-- 1 cxing cxing ...` |
| `cd ~` | 回到家目录 | 路径变 `~` |
| `mkdir test` | 创建目录 | 成功 |
| `cd test` | 进入目录 | 路径变 `~/test` |
| `touch hello.py` | 创建空文件 | 成功 |
| `nano hello.py` | 文本编辑器 | 写入 `print("hello")` 并保存 |
| `python3 hello.py` | 运行 Python 脚本 | 输出 `hello` |
| `rm hello.py` | 删除文件 | 成功 |
| `cd ..` | 回到上级目录 | 路径变 `~` |
| `rmdir test` | 删除空目录 | 成功 |

**踩过的坑**：`cd~` 没空格 → `Command not found`。记住：**命令和参数之间永远有空格**。

---

### 第二组：复制、移动、查看

| 命令 | 作用 | 实际运行结果 |
|:---|:---|:---|
| `cp src/sensor.py src/sensor_backup.py` | 复制文件 | 成功，目录下多了 backup 文件 |
| `mv src/sensor_backup.py src/old_sensor.py` | 移动/重命名 | 文件名改变 |
| `echo 'print("sensor data")' > src/sensor.py` | 往文件写入内容 | 文件内有了 Python 代码 |
| `cat src/sensor.py` | 显示文件全部内容 | 输出 `print("sensor data")` |
| `head -1 src/sensor.py` | 看前 1 行 | 同上（只有一行） |
| `tail -1 src/sensor.py` | 看最后 1 行 | 同上 |
| `du -sh src/sensor.py` | 查看文件大小 | `4.0K` |
| `wc -l src/sensor.py` | 统计行数 | `1` |
| `find . -name "*.py"` | 查找所有 .py 文件 | `./src/sensor.py ./src/old_sensor.py` |
| `rm -rf robot_project` | 递归删除整个目录 | 成功清理 |

**踩过的坑**：
- 拼写错误：`robot_porject` → `No such file or directory`
- 顺序错误：先 `mv` 再 `cp`，导致源文件已被改名
- `wc -lsrc/sensor.py` 没空格 → `unexpected argument`

---

### 第三组：权限管理（实习高频考点）

| 命令 | 作用 | 运行结果 |
|:---|:---|:---|
| `touch run.sh` | 创建脚本 | 成功 |
| `ls -l run.sh` | 查看权限 | `-rw-r--r--`（无执行权限） |
| `./run.sh` | 尝试运行 | `Permission denied` ❌ |
| `chmod +x run.sh` | 添加执行权限 | 成功 |
| `ls -l run.sh` | 再次查看权限 | `-rwxr-xr-x`（绿色，有 x） ✅ |
| `./run.sh` | 再次运行 | 输出 `running` ✅ |
| `chmod 755 run.sh` | 数字权限表示法 | `rwxr-xr-x` |

**权限字符含义**：
```
-rwxr-xr-x
 |   |   |
 |   |   └── 其他用户权限: r-x (读+执行)
 |   └────── 组用户权限: r-x (读+执行)
 └────────── 文件所有者权限: rwx (读+写+执行)

r = read (4)    w = write (2)    x = execute (1)
755 = 7(rwx) + 5(r-x) + 5(r-x)
```

**踩过的坑**：
- `chomd` 拼写错误 → `Command not found`（正确是 `chmod`）
- `cd~` 没空格 → `Command not found`（第三次踩）

---

### Linux 常见错误清单（本周汇总）

| 错误现象 | 原因 | 正确写法 |
|:---|:---|:---|
| `cd~` not found | 命令和参数之间没空格 | `cd ~` |
| `chomd` not found | 拼写错误 | `chmod` |
| `cp src/xx` No such file | 源文件不存在或拼写错 | 先 `ls` 确认文件名 |
| `wc -lsrc` unexpected argument | 参数和路径黏在一起 | `wc -l src/xx.py` |
| `./run.sh` Permission denied | 没有执行权限 | 先 `chmod +x run.sh` |
| `robot_porject` No such file | 拼写错误 | 仔细核对文件夹名 |

---

## 双线进度总览

### 本周完成清单

| 线 | 任务 | 状态 |
|:---|:---|:---|
| **线 A** | Day 7 数据划分 | ✅ |
| | Day 8 决策树模型 | ✅ |
| | Day 9 混淆矩阵与分类报告 | ✅ |
| | Day 10 三种模型对比 | ✅ |
| | Day 11 交叉验证 | ✅ |
| | Day 12 特征重要性 | ✅ |
| **线 B** | Linux 第一组：文件目录 | ✅ |
| | Linux 第二组：复制移动查看 | ✅ |
| | Linux 第三组：权限管理 | ✅ |

### GitHub 提交记录

- `ai-projects` 仓库：Day 7~12 全部 push，累计 **12+ commits**
- `free-python-games` 仓库：复刻声明已添加
- Linux 练习：本地 WSL 环境已搭建，三组合命令全部跑通

---

## 下周预告

| 线 | 内容 |
|:---|:---|
| **线 A** | MNIST 手写数字分类（新数据集，sklearn 基线） |
| **线 B** | Linux 进程管理 + 网络工具（`ps` `top` `ping`），然后进入 **Docker** |

---

*复盘时间：2026-11-05*
*对应项目：https://github.com/c-xingyu/ai-projects*
