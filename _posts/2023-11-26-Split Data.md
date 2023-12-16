---

layout: single
title: 'Split Data'
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

## Splitting Data ##

```python
from sklearn.datasets import load_iris
import pandas as pd 
iris = load_iris()

from sklearn.model_selection import train_test_split
features = iris.data[:, 2:]
labels = iris.target
```

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
stratify = labels, random_state=13)
```

<br>By checking for unique values, we can see that stratifying worked. 

```python
import numpy as np 
np.unique(y_test, return_counts=True)
```

![image-20231126191823947](/images/2023-11-26-Split Data/image-20231126191823947.png)

## Decision Tree Classifier ##

Now we can create and train a model. We will start with a Decision Tree Classifier. 

```python 
from sklearn.tree import DecisionTreeClassifier 

iris_tree = DecisionTreeClassifier(max_depth=2, random_state=13)
iris_tree.fit(X_train, y_train)
```

![image-20231126190327621](/images/2023-11-26-Split Data/image-20231126190327621.png)

Depth determines how much the model branches out. We have to limit the max_depth as having excessive depth could result in high variance. 

```python
from sklearn.metrics import accuracy_score

y_pred_tr = iris_tree.predict(X_train)
accuracy_score(y_train, y_pred_tr)
```

<br>

Accuracy is **one** metric to measure the accuracy of binary classification models. We will delve deeper into other concepts later. 

Here is our tree visualized: 

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(iris_tree);
```

![image-20231126190302790](/images/2023-11-26-Split Data/image-20231126190302790.png)

We also try visualizing our decision regions. 

```python
from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(14,8))
plot_decision_regions(X=X_train, y=y_train, clf=iris_tree, legend=2)
plt.show()
```

![image-20231126190413685](/images/2023-11-26-Split Data/image-20231126190413685.png)

There are several wrong predictions.  However, the model isn't overly complex. We can't yet determine whether our model has high variance and therefore is prone to overfitting. 

```python
scatter_highlight_kwargs = {'s':150, 'label':'Test data', 'alpha':0.9}
scatter_kwargs = {'s':150, 'edgecolor':None, 'alpha':0.9}

plt.figure(figsize=(12,8))
plot_decision_regions(X=features, y=labels, 
                      X_highlight=X_test, clf=iris_tree,
                      legend=2, scatter_highlight_kwargs=scatter_highlight_kwargs,
                      scatter_kwargs=scatter_kwargs,
                      contourf_kwargs={"alpha":0.2}
                      )
```

![image-20231126192139321](/images/2023-11-26-Split Data/image-20231126192139321.png)

### Making Predictions ### 

```python
features = iris.data 
labels = iris.target

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size = 0.2, 
                                                    stratify=labels, 
                                                    random_state=13)
iris_tree = DecisionTreeClassifier(max_depth=2, random_state = 13)
iris_tree.fit(X_train, y_train)

plt.figure(figsize=(12,8))
plot_tree(iris_tree);
```

![image-20231126192334733](/images/2023-11-26-Split Data/image-20231126192334733.png)

```python
test_data = np.array([[4.3, 2., 1.2, 1.0]])
iris_tree.predict(test_data)

iris_tree.predict_proba(test_data)
```

![image-20231126192422377](/images/2023-11-26-Split Data/image-20231126192422377.png)
