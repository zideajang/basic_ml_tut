$$ W = (X^TX)^{-1}X^TY$$

如果是两个参数和一个截断，也就是三个特征值，我们在等式两边除以矩阵是行不通，代替除以矩阵我们可以通过乘以逆矩阵来避免矩阵除法。值得注意的是只有方阵可逆。


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```
读取数据
```
data = np.genfromtxt(r"data/data.csv",delimiter=',')
print(data)

```

```
[[100.    4.    9.3]
 [ 50.    3.    4.8]
 [100.    4.    8.9]
 [100.    2.    6.5]
```

切分数据为
