---

layout: single
title: '02 Playing with the Titanic Data'
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

## Did Dicpario Really Have To Die? ##

### Table Break down

* pclass -> class of ticket 
* survived -> whether the passenger survived or not 
* sex -> sex 
* sibsp -> the number of siblings or spouses 
* parch -> the number of parents or children 
* fare -> the amount paid for the ticket 
* boat -> the serial number of the boat if survived 

### EDA ### 

```python
import pandas as pd 

titanic_url = "https://raw.githubusercontent.com/PinkWink/ML_tutorial/master/dataset/titanic.xls"
titanic = pd.read_excel(titanic_url)
titanic.head()
```

![image-20231126195158135](/images/2023-11-26-Titanic Data/image-20231126195158135.png)

```python
f, ax = plt.subplots(1,2, figsize=(16,8 )) 

titanic["survived"].value_counts().plot.pie(ax=ax[0],
                                            autopct='%1.1f%%', 
                                            shadow=True,
                                            explode = [0, 0.05])
ax[0].set_title("Pieplot - survived")
ax[0].set_ylabel("")

sns.countplot(titanic,x="survived", ax=ax[1])
ax[1].set_title("Count plot - survived")

plt.show()
```

![image-20231126195250316](/images/2023-11-26-Titanic Data/image-20231126195250316.png)

Survival rate based on gender? 

```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.countplot(titanic, x="sex", ax=ax[0])
ax[0].set_title("Count of Passengers of Sex")
ax[0].set_ylabel('')

sns.countplot(titanic, hue='survived', x='sex', ax=ax[1])
ax[1].set_title("Sex: Survived and Unsurvived")

plt.show()
```

![image-20231126195340340](/images/2023-11-26-Titanic Data/image-20231126195340340.png)

```python 
pd.crosstab(titanic['pclass'], titanic['survived'], margins=True)
```

![image-20231126195515417](/images/2023-11-26-Titanic Data/image-20231126195515417.png)

* The survival rate of first class passengers is high 
* The survival rate of women is also very high
* Does that mean there were lots of women in the first class

```python
grid = sns.FacetGrid(titanic, 
                     row='pclass', 
                     col="sex", 
                     height=4, 
                     aspect=2)
grid.map(plt.hist, "age", alpha=0.8, bins=20)
grid.add_legend();
```

![image-20231126195648015](/images/2023-11-26-Titanic Data/image-20231126195648015.png)

```python
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10,8))

sns.histplot(x='age', color='r',
             data=titanic, kde=True,
             element='step')
```

![image-20231126195811314](/images/2023-11-26-Titanic Data/image-20231126195811314.png)

Check out a cool little function in pandas called 'cut'

```python
titanic['age_cat'] = pd.cut(titanic['age'], 
                            bins=[0,7,15,30,60,100],
                            include_lowest=True,
                            labels=['baby', 'teen', 'young',
                                    'adult', 'old'])
titanic.head()
```

![image-20231126200318979](/images/2023-11-26-Titanic Data/image-20231126200318979.png)

```python
fig, axes = plt.subplots(1,2, figsize=(14,6))
sns.set_theme(style="darkgrid", palette="pastel")
women = titanic[titanic["sex"] == 'female']
men = titanic[titanic["sex"] == "male"]

ax = sns.histplot(women[women['survived'] == 1]['age'],bins=20, label="survived", ax=axes[0])
ax = sns.histplot(women[women['survived'] == 0]['age'], bins = 40, label="passed", ax=axes[0])
ax.legend();
ax = sns.histplot(men[men['survived'] == 1]['age'],bins=20, label="survived", ax=axes[1])
ax = sns.histplot(men[men['survived'] == 0]['age'], bins = 40, label="passed", ax=axes[1])
ax.legend();
```

![image-20231126200415241](/images/2023-11-26-Titanic Data/image-20231126200415241.png)

We'll do a regex revision on a later post

```python
import re 

title = []
for idx, dataset in titanic.iterrows(): 
    tmp = dataset['name']
    title.append(re.search('\,\s\w+(\s\w+)?\.', tmp).group()[2:-1])
titanic['title'] = title
titanic.head()
```

![image-20231126200524602](/images/2023-11-26-Titanic Data/image-20231126200524602.png)

```python
titanic['title'].unique()
titanic['title'] = titanic['title'].replace('Mlle', 'Miss')
titanic['title'] = titanic['title'].replace('Ms', 'Miss')
titanic['title'] = titanic['title'].replace('Mme', 'Miss')
rare_f = ['Dona', 'Lady', 'the Countess']
rare_m = ['Capt', 'Col', 'Don', 'Major', 'Rev', 'Sir', "Dr", "Master", "Jonkheer"]

for each in rare_f: 
    titanic['title'] = titanic['title'].replace(each, 'rare_f')

for each in rare_m: 
    titanic['title'] = titanic['title'].replace(each, 'rare_m')

titanic['title'].unique()

```

![image-20231126200654777](/images/2023-11-26-Titanic Data/image-20231126200654777.png)

```python
a = titanic[['title', 'survived']].groupby(['title'], as_index=False).mean()

```

![image-20231126200729775](/images/2023-11-26-Titanic Data/image-20231126200729775.png)

### Survivor Prediction ### 

```python
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
le.fit(titanic['sex'])
```

LabelEncoder turns categorical values into integers

```python
le.classes_
```

![image-20231126200921208](/images/2023-11-26-Titanic Data/image-20231126200921208.png)

```python
titanic['gender'] = le.transform(titanic['sex'])
titanic.head()
```

We will just discard null values 

```python
titanic = titanic[titanic['age'].notnull()]
titanic = titanic[titanic['fare'].notnull()]
titanic.info()
```

![image-20231126201034507](/images/2023-11-26-Titanic Data/image-20231126201034507.png)

```python
from sklearn.model_selection import train_test_split

x = titanic[['pclass', 'age', 'sibsp', 'parch', 'fare', 'gender']]
y = titanic['survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    random_state=13)
```

```python
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 

dt = DecisionTreeClassifier(max_depth=4, random_state = 13)
dt.fit(x_train, y_train)
```

![image-20231126201217471](/images/2023-11-26-Titanic Data/image-20231126201217471.png)

```python
pred = dt.predict(x_test)
print(accuracy_score(y_test, pred))
```

![image-20231126201302390](/images/2023-11-26-Titanic Data/image-20231126201302390.png)

### Dicaprio's Survival Rate ###

* We will assume Dicaprio was in third class, was eighteen years of age, had no siblings on board, had no parents on board, paid 5 for the fare and was a male

```python
import numpy as np 

dicaprio = np.array([[3,18,0,0,5,1]])
print(f"Dicaprio: {dt.predict_proba(dicaprio)[0,1]}")

print(dt.predict_proba(dicaprio))
```

![image-20231126201823710](/images/2023-11-26-Titanic Data/image-20231126201823710.png)