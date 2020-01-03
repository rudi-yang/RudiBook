# python
```python
#coding=utf-8
import json

# 编码问题
license_plat = "测试数据"
driver_name = "测试数据".decode('utf-8')[0].encode('utf-8')

# ssl 连接限制
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# json dump乱码
json.dumps(data, ensure_ascii=False)
json.dumps(data, ensure_ascii=False).decode('utf8').encode('gb2312')


# 在转eval之前先进行.decode('unicode-escape'）
data = eval(data.decode('unicode-escape'))

# 生成随机数
import random

# 利用Python中的randomw.sample()函数
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素。
# 表示从[A,B)间随机生成N个数，结果以列表返回123
# 注意：区间为[A,B）左闭右开
resultList = random.sample(range(A, B), N)


```

# pandas
```python
import pandas as pd
import numpy as np
import seaborn as sns

# 显示行列设置
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)

# 读入中文文件
data = pd.read_csv("data.csv", encoding='gbk')

# 去除Unname:0
data = pd.read_csv("data.csv", index_col=0)

# 显示数据精度设置
np.set_printoptions(suppress=True)
pd.set_option("display.float_format", lambda x: "%.3f" % x)  

# 数据相关性分析
corr = data.corr()
sns.heatmap(corr, cmap='Blues', annot=True)

sns.pairplot(data[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# pretty format
from tabulate import tabulate
import pandas as pd

df = pd.DataFrame({'col_two': [0.0001, 1e-005, 1e-006, 1e-007],
                   'column_3': ['ABCD', 'ABCD', 'long string', 'ABCD'],
                   "fdsaljfklsdjfkljsadfk": [1, 2, 3, 4],
                   "12312": [1, 2, 3, 4],
                   "jfdkfjkdjk": [1, 2, 3, 4],
                   "ekekekek": [1, 2, 3, 4],
                   "fdsafdsalfjdskljflk": [3, 3, 3, 3]})
print(tabulate(df, headers='keys', tablefmt='psql'))


# Pandas 数据乱序
df.sample(frac=1)
df.sample(frac=1).reset_index(drop=True)

from sklearn.utils import shuffle
df = shuffle(df)


# pandas 分组取最大
import pandas as pd

df = pd.DataFrame({'Sp': ['a', 'b', 'c', 'd', 'e', 'f'], 'Mt': ['s1', 's1', 's2', 's2', 's2', 's3'], 'Value': [1, 2, 3, 4, 5, 6], 'Count': [3, 2, 5, 10, 10, 6]})
print(df)

check = df.groupby('Mt').apply(lambda t: t[t.Count == t.Count.max()])
print(check)

idx = df.groupby('Mt')['Count'].idxmax()
print(df.iloc[idx])

check = df.iloc[df.groupby(['Mt']).apply(lambda x:x['Count'].idxmax())]
print(check)

# pandas 多进程
from joblib import Parallel,delayed
import multiprocessing

def fun_avg(name,data):
    temp=data[data["age"]==data["age"].max()]
    return temp


def applyParallel(dfGrouped,func):
    ret=Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name,group)for name,group in dfGrouped)
    return pd.concat(ret)


result=applyParallel(data.groupby('id'),fun_avg)
print(result)

```


# matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams["font.family"] = 'Arial Unicode MS'


# 回归线
plt.scatter(data["complain"], data["prediction_prob"])
reg = LinearRegression().fit(np.reshape(np.array(data["prediction_prob"]), [-1, 1]), data["complain"])
pred = reg.predict(np.reshape(np.array(data["prediction_prob"]), [-1, 1]))
plt.plot(data["prediction_prob"], pred, linewidth=2, label=u'回归线')
```

# 其他

pip 下载缓慢 更换其他源头
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gevent
```