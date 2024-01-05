### 主成分分析法（PCA）

#### 1、主成分分析的原理

主成分分析是利用降维的思想，在损失很少信息的前提下把多个指标转化为几个综合指标的多元统计方法。通常把转化生成的综合指标称之为主成分，其中每个主成分都是原始变量的线性组合，且各个主成分之间互不相关，这就使得主成分比原始变量具有某些更优越的性能。这样在研究复杂问题时就可以只考虑少数几个主成分而不至于损失太多信息，从而更容易抓住主要矛盾，揭示事物内部变量之间的规律性，同时使问题得到简化，提高分析效率。主成分分析正是研究如何通过原来变量的少数几个线性组合来解释原来变量绝大多数信息的一种多元统计方法。



#### 2、主成分分析的步骤

    1. 根据研究问题选取初始分析变量；

        2. 根据初始变量特性判断由协方差阵求主成分还是由相关阵求主成分（数据标准化的话需要用系数相关矩阵，数据未标准化则用协方差阵）；

        3. 求协差阵或相关阵的特征根与相应标准特征向量；

        4. 判断是否存在明显的多重共线性，若存在，则回到第一步；

        5. 主成分分析的适合性检验

        6. 得到主成分的表达式并确定主成分个数，选取主成分；

        7. 结合主成分对研究问题进行分析并深入研究。



一组数据是否可以用主成分分析，必须做适合性检验。可以用球形检验和KMO统计量检验。

（1）球形检验（Bartlett)

 球形检验的假设：

​            H0：相关系数矩阵为单位阵（即变量不相关）

​            H1：相关系数矩阵不是单位阵（即变量间有相关关系）

在Python中，可以使用`scipy`库的`stats`模块来进行Bartlett的球状性检验。

```python
import pandas as pd
from scipy.stats import bartlett

# 这里假设df中的每一列都是要进行PCA的变量

# Bartlett的球状性检验
chi_square_value, p_value = bartlett(*df.values.T)

# 输出检验结果
print(f"Bartlett's test chi-squared value: {chi_square_value}")
print(f"Bartlett's test p-value: {p_value}")

# 如果p值小于显著性水平（通常为0.05），则拒绝原假设，认为数据适合做PCA
if p_value < 0.05:
    print('Variables are correlated, suitable for PCA.')
else:
    print('Variables are not sufficiently correlated, PCA might not be suitable.')
```

```
Bartlett's test chi-squared value: 258.201527090366
Bartlett's test p-value: 4.970755673743075e-52
Variables are correlated, suitable for PCA.
```

2）KMO（Kaiser-Meyer-Olkin)统计量

​    KMO统计量比较样本相关系数与样本偏相关系数，它用于检验样本是否适于作主成分分析。

​    KMO的值在0,1之间，该值越大，则样本数据越适合作主成分分析和因子分析。一般要求该值大于0.5，方可作主成分分析或者相关分析。

​    Kaiser在1974年给出了经验原则：

​         

```
             0.9以上       适合性很好

             0.8~0.9        适合性良好

             0.7~0.8        适合性中等

             0.6~0.7        适合性一般

             0.5~0.6        适合性不好

             0.5以下       不能接受的        
```

Kaiser-Meyer-Olkin (KMO) 测试是一种度量变量之间的偏相关是否足够低，以至于可以进行主成分分析（PCA）或因子分析的指标。简而言之，KMO 测试评估数据是否适合进行降维。

KMO 测试的值介于 0 和 1 之间。KMO 值越接近 1，变量之间的偏相关越小，因此每个变量对其他变量的解释越少，这意味着进行 PCA 是合适的。一般认为 KMO 值大于 0.6 的数据集适合进行 PCA。

KMO 测试并不是强制的步骤，但它可以作为决定是否进行 PCA 的有用指标。如果 KMO 测试的结果显示数据不适合进行 PCA，那么进行 PCA 可能得不到有意义或可靠的结果。

在 Python 中，你可以使用 `factor_analyzer` 库来进行 KMO 测试。

```python
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo

# 计算 KMO 值
kmo_all, kmo_model = calculate_kmo(df)

print(f"KMO: {kmo_model}")

# 判断 KMO 值是否适合进行 PCA
if kmo_model < 0.6:
    print('KMO test indicates that PCA might not be suitable.')
else:
    print('KMO test indicates that PCA is suitable.')

```

```
KMO: 0.8353019591410404
KMO test indicates that PCA is suitable.
```

3.求相关矩阵

标准化：

```python
from sklearn import preprocessing
df = preprocessing.scale(df)
```

求相关系数矩阵:

```python
covX = np.around(np.corrcoef(df.T),decimals=3)
covX
```

4.求解特征值和特征向量

```python
featValue, featVec=  np.linalg.eig(covX.T)  #求解系数相关矩阵的特征值和特征向量
featValue, featVec
```

5.对特征值进行排序并输出 降序

```python
featValue = sorted(featValue)[::-1]
featValue
```

6.求特征值的贡献度

```python
gx = featValue/np.sum(featValue)
gx
```

7.求特征值的累计贡献度

```python
lg = np.cumsum(gx)
lg
```



上述代码个人不是很喜欢，可以直接用sklearn库里面的PCA直接写：

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 数据标准化（PCA之前重要的一步，因为PCA受到数据规模的影响）
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 创建PCA对象，不指定n_components
pca = PCA()

# 进行PCA
df_reduced = pca.fit_transform(df_scaled)

# 计算保留95%信息（方差）所需的最小特征数
total_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(total_variance >= 0.95)[0][0] + 1

print(f"保留95%的信息需要的最小特征数：{n_components}")

# 你也可以只保留这些主成分
df_reduced = pd.DataFrame(df_reduced[:, :n_components])

# 查看降维后的数据
print(df_reduced)

```

```
保留95%的信息需要的最小特征数：5
```

一开始是八个特征，现在是5个就能体现95%的特征，非常完美。