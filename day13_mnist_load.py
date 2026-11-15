from sklearn.datasets import load_digits

print("正在加载 digits 数据集（本地内置，无需下载）...")
digits = load_digits()
X, y = digits.data, digits.target

print(f"\n=== 手写数字数据集概况 ===")
print(f"图片总数: {X.shape[0]}")
print(f"每张图片像素数: {X.shape[1]} ({8}x{8}={8*8})")
print(f"像素值范围: {X.min():.0f} ~ {X.max():.0f}")
print(f"数字类别: {sorted(set(y))}")
print(f"\n第一张图片像素值（前20个）: {X[0][:20]}")
print(f"第一张图片的真实标签: {y[0]}")