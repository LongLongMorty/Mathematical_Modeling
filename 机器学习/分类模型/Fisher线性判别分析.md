#### Fisher线性判别分析

Fisher线性判别分析（Fisher's Linear Discriminant Analysis, LDA）是一种分类技术，由罗纳德·费舍尔（Ronald Fisher）在1936年提出。它是一种监督学习算法，主要用于两类或多类分类问题。LDA的核心思想是寻找一个线性组合的特征，使得不同类别的数据在这个新的特征空间中能够被最佳地分离。

Fisher线性判别分析的主要步骤包括：

1. **计算类内散度矩阵（Within-class Scatter Matrix）**：这个矩阵表示同一类别内的数据点相对于其类别均值的分散程度。
2. **计算类间散度矩阵（Between-class Scatter Matrix）**：这个矩阵表示不同类别之间的数据点相对于各自类别均值的分散程度。
3. **求解特征向量和特征值**：通过计算类内散度矩阵的逆与类间散度矩阵的乘积，得到的矩阵的特征向量和特征值。
4. **选择主要成分**：根据特征值的大小选择最重要的特征向量，这些特征向量定义了一个新的空间，使得在这个空间中不同类别的数据最大程度地分离。
5. **进行数据投影**：将原始数据投影到选择的特征向量上，从而实现数据的降维和分类。

Fisher线性判别分析的主要优点是能够在保持类别分离的前提下对数据进行降维，从而提高分类器的效率和效果。然而，它也有一些局限性，比如假设数据是线性可分的，以及假设不同类别的数据具有相同的协方差结构。在实际应用中，如果这些假设不成立，LDA的性能可能会受到影响。



例子：

生物学家W.L Grogan和W.W.Wirth试图将两种蠓Apf和Af进行鉴别，给出了9只Af和6只Apf蠓虫的触角长度和翅膀长度的数据，已知Af是宝贵的传粉益虫，Apf是某种疾病的载体，要求建立一种模型，正确区分两类蠓虫。
已知6只Apf蠓虫(Apf midges)和9只Af蠓虫(Af midges)的触长、翅长数据表见表1(Talbe 1)和表2(Table 2)所示。
蠓的二分类问题：
 问题1: 试给出该问题的Fisher分类器；

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
apf_midges = np.array([
    [1.14, 1.78],
    [1.20, 1.86],
    [1.30, 1.96],
    [1.26, 2.00],
    [1.28, 2.00],
    [1.18, 1.96]
])

af_midges = np.array([
    [1.24, 1.72],
    [1.38, 1.64],
    [1.36, 1.40],
    [1.74, 1.70],
    [1.38, 1.82],
    [1.48, 1.82],
    [1.38, 1.90],
    [1.54, 1.82],
    [1.56, 2.08]
])

# Step 1: 计算均值向量
mean_apf = np.mean(apf_midges, axis=0)
mean_af = np.mean(af_midges, axis=0)

# Step 2: 计算类内散布矩阵
S_W = np.zeros((2,2))
for midge in apf_midges:
    midge, mv = midge.reshape(2,1), mean_apf.reshape(2,1)
    S_W += (midge - mv).dot((midge - mv).T)

for midge in af_midges:
    midge, mv = midge.reshape(2,1), mean_af.reshape(2,1)
    S_W += (midge - mv).dot((midge - mv).T)

# Step 3: 计算类间散布矩阵
mean_diff = (mean_apf - mean_af).reshape(2,1)
S_B = len(apf_midges) * mean_diff.dot(mean_diff.T) + len(af_midges) * mean_diff.dot(mean_diff.T)

# Step 4: 计算特征值和特征向量
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Step 5: 对特征向量进行排序
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# 提取最大特征向量
w = eigen_pairs[0][1].reshape(2,1)

# Step 6: 将数据投影到新的轴上
apf_projected = apf_midges.dot(w)
af_projected = af_midges.dot(w)

# Optional: Visualization
plt.scatter(apf_projected, np.zeros_like(apf_projected), color='red', label='Apf')
plt.scatter(af_projected, np.zeros_like(af_projected), color='blue', label='Af')
plt.xlabel('Projection onto the discriminant')
plt.legend()
plt.show()
```



 问题2: 有三个待识别的模式样本，它们分别是(1.24,1.80)T ,(1.28,1.84)T,(1.40,2.04)T，试问这三个样本属于哪一种蠓。

```python
# New samples
new_samples = np.array([
    [1.24, 1.80],
    [1.28, 1.84],
    [1.40, 2.04]
])

# Project new samples
new_samples_projected = new_samples.dot(w)

# Optional: Add these projections to the existing plot for visualization
plt.scatter(apf_projected, np.zeros_like(apf_projected), color='red', label='Apf')
plt.scatter(af_projected, np.zeros_like(af_projected), color='blue', label='Af')
plt.scatter(new_samples_projected, np.zeros_like(new_samples_projected), color='green', marker='x', label='New Samples')
plt.xlabel('Projection onto the discriminant')
plt.legend()
plt.show()

# Determine the class of new samples
# This is a simple approach and assumes equal class priors and equal cost of misclassification
for i, sample in enumerate(new_samples_projected):
    distance_to_apf = np.abs(sample - np.mean(apf_projected))
    distance_to_af = np.abs(sample - np.mean(af_projected))
    if distance_to_apf < distance_to_af:
        print(f"Sample {i+1} ({new_samples[i]}) is classified as Apf")
    else:
        print(f"Sample {i+1} ({new_samples[i]}) is classified as Af")

```

