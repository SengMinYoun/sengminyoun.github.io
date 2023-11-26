---
layout: single
title: '03 Preprocessing Techniques'
categories: Machine_Learning_Lab
tag: [python, sklearn, machinelearning]
toc: true
author_profile: false
sidebar:
    nav: 'docs'
search: true
sidebar:
    nav: "counts"
use_math: true
---

<div class ="notice--success">
This series will deal with utilizing machine learning libraries. It is intended as a refresher on the topics. 
</div>

## Label Encoder ## 

We can use Label Encoder to turn categorical values into integers. We have to be cautious though. Turning categorical values can be useful if we are indicating whether something is 'true' or 'false' using integer values. 

```python
import pandas as pd

df = pd.DataFrame({'A': ['a', 'b', 'c', 'a','b'],
                   'B': [1, 2, 3, 1, 0]})
```

![image-20231126204556207](/images/2023-11-26-Preprocessing Techniques/image-20231126204556207.png)

```python
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
le.fit(df['A'])

le.classes_
```

![image-20231126204634220](/images/2023-11-26-Preprocessing Techniques/image-20231126204634220.png)

```python
df['le_A'] = le.transform(df['A'])
```

![image-20231126204718663](/images/2023-11-26-Preprocessing Techniques/image-20231126204718663.png)

## Scaling ## 

### Min-Max Scaling ###

$X' = \frac{x - min(x)}{max(x) - min(x)}$

```python
df = pd.DataFrame({
    'A': [10, 20, -10, 0, 25],
    'B': [1, 2, 3, 1, 0]
})
```

![image-20231126205304811](/images/2023-11-26-Preprocessing Techniques/image-20231126205304811.png)

```python
from sklearn.preprocessing import MinMaxScaler 

mms = MinMaxScaler()
mms.fit(df)
```

![image-20231126205342826](/images/2023-11-26-Preprocessing Techniques/image-20231126205342826.png)

```python
df_mms = mms.transform(df)
df_mms.reshape(-1)
```

![image-20231126205525262](/images/2023-11-26-Preprocessing Techniques/image-20231126205525262.png)

### Standard Scaler (Z-score)

$ Z = \frac{x - \mu}{\sigma}$

```python
from sklearn.preprocessing import StandardScaler 

ss = StandardScaler()
ss.fit(df)
```

```python
df_ss = ss.transform(df)
```

### Robust Scaler ###

$X' = \frac{x_i - Q_2}{Q_3 - Q_1}$

```python
df = pd.DataFrame({
    'A' : [-0.1, 0., 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 5.0]
})

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

mm = MinMaxScaler()
ss = StandardScaler()
rs = RobustScaler() 

df_scaler = df.copy()
df_scaler['MinMax'] = mm.fit_transform(df)
df_scaler['Standard'] = ss.fit_transform(df)
df_scaler['Robust'] = rs.fit_transform(df)

df_scaler
```

![image-20231126210323332](/images/2023-11-26-Preprocessing Techniques/image-20231126210323332.png)

```python 
import seaborn as sns
import matplotlib.pyplot as plt 

sns.set_theme(style='whitegrid')

plt.figure(figsize=(18, 6))
sns.boxplot(data=df_scaler, orient='h')
```

![image-20231126210414499](/images/2023-11-26-Preprocessing Techniques/image-20231126210414499.png)

* Min-max & standard scalers are affected by outliers 
* The robust scaler turns the median into 0 and therefore isn't affected by outliers