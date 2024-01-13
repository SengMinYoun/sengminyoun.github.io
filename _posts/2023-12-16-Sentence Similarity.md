---
layout: single
title:  "Sentence Similarity"
categories: Machine_Learning_Lab
tag: [python, machinelearning]
toc: true
author_profile: false
sidebar:
    nav: 'docs'
search: true
sidebar:
    nav: "counts"
use_math: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


### 문장의 유사도


#### 문장을 벡터로 표현할 수 있다면 거리를 구할 수 있다


### CountVectorize



```python
from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer(min_df = 1)
```

* 단순 거리 측정



```python
contents = [
    '상처받은 아이들은 너무 일찍 커버려', 
    '내가 상처받은 거 아는 사람 불편해',
    '잘 사는 사람들은 좋은 사람 되기 쉬워',
    '아무 일도 아니야 괜찮아'
]
```


```python
from konlpy.tag import Okt
t = Okt()
```


```python
contents_tokens = [t.morphs(row) for row in contents]
contents_tokens
```

<pre>
[['상처', '받은', '아이', '들', '은', '너무', '일찍', '커버', '려'],
 ['내', '가', '상처', '받은', '거', '아는', '사람', '불편해'],
 ['잘', '사는', '사람', '들', '은', '좋은', '사람', '되기', '쉬워'],
 ['아무', '일도', '아니야', '괜찮아']]
</pre>
* 형태소로 띄어쓰기



```python
contents_to_vectorize = []

for content in contents_tokens: 
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word 
    
    contents_to_vectorize.append(sentence)

contents_to_vectorize
```

<pre>
[' 상처 받은 아이 들 은 너무 일찍 커버 려',
 ' 내 가 상처 받은 거 아는 사람 불편해',
 ' 잘 사는 사람 들 은 좋은 사람 되기 쉬워',
 ' 아무 일도 아니야 괜찮아']
</pre>

```python
X = vectorizer.fit_transform(contents_to_vectorize)
X
```

<pre>
<4x17 sparse matrix of type '<class 'numpy.int64'>'
	with 20 stored elements in Compressed Sparse Row format>
</pre>

```python
num_samples, num_features = X.shape
num_samples, num_features 
```

<pre>
(4, 17)
</pre>
네 개의 문장에 전체 말뭉치의 단어가 17개였다



```python
vectorizer.get_feature_names_out()
```

<pre>
array(['괜찮아', '너무', '되기', '받은', '불편해', '사는', '사람', '상처', '쉬워', '아는',
       '아니야', '아무', '아이', '일도', '일찍', '좋은', '커버'], dtype=object)
</pre>

```python
X.toarray().transpose()
```

<pre>
array([[0, 0, 0, 1],
       [1, 0, 0, 0],
       [0, 0, 1, 0],
       [1, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 2, 0],
       [1, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [1, 0, 0, 0],
       [0, 0, 0, 1],
       [1, 0, 0, 0],
       [0, 0, 1, 0],
       [1, 0, 0, 0]], dtype=int64)
</pre>

```python
new_post = ['상처받기 싫어 괜찮아']
new_post_tokens = [t.morphs(row) for row in new_post]

new_post_to_vectorize = []

for content in new_post_tokens:
    sentence = ''
    for word in content: 
        sentence = sentence + ' ' + word
    
    new_post_to_vectorize.append(sentence)

new_post_to_vectorize
```

<pre>
[' 상처 받기 싫어 괜찮아']
</pre>

```python
new_post_vec = vectorizer.transform(new_post_to_vectorize)
new_post_vec.toarray()
```

<pre>
array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)
</pre>

```python
import scipy as sp 

def dist_raw(v1, v2):
    delta = v1 - v2 
    return sp.linalg.norm(delta.toarray())
```


```python
dist = [dist_raw(each, new_post_vec) for each in X]
dist
```

<pre>
[2.449489742783178, 2.23606797749979, 3.1622776601683795, 2.0]
</pre>

```python
print("Best post is ", dist.index(min(dist)), ',dist = ', min(dist))
print('Test post is -->', new_post)
print('Best dist post is -->', contents[dist.index(min(dist))])
```

<pre>
Best post is  3 ,dist =  2.0
Test post is --> ['상처받기 싫어 괜찮아']
Best dist post is --> 아무 일도 아니야 괜찮아
</pre>

```python
for i in range(0, len(contents)):
    print(X.getrow(i).toarray())

print('-' * 40)
print(new_post_vec.toarray())
```

<pre>
[[0 1 0 1 0 0 0 1 0 0 0 0 1 0 1 0 1]]
[[0 0 0 1 1 0 1 1 0 1 0 0 0 0 0 0 0]]
[[0 0 1 0 0 1 2 0 1 0 0 0 0 0 0 1 0]]
[[1 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0]]
----------------------------------------
[[1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]]
</pre>
### Tf-idf vectorization



```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 1, decode_error= 'ignore')
```


```python
X = vectorizer.fit_transform(contents_to_vectorize)
X
```

<pre>
<4x17 sparse matrix of type '<class 'numpy.float64'>'
	with 20 stored elements in Compressed Sparse Row format>
</pre>

```python
X.shape
```

<pre>
(4, 17)
</pre>
* 가중치와 역가중치 반영



```python
X.toarray().transpose()
```

<pre>
array([[0.        , 0.        , 0.        , 0.5       ],
       [0.43671931, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.39264414, 0.        ],
       [0.34431452, 0.40104275, 0.        , 0.        ],
       [0.        , 0.50867187, 0.        , 0.        ],
       [0.        , 0.        , 0.39264414, 0.        ],
       [0.        , 0.40104275, 0.6191303 , 0.        ],
       [0.34431452, 0.40104275, 0.        , 0.        ],
       [0.        , 0.        , 0.39264414, 0.        ],
       [0.        , 0.50867187, 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.5       ],
       [0.        , 0.        , 0.        , 0.5       ],
       [0.43671931, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.5       ],
       [0.43671931, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.39264414, 0.        ],
       [0.43671931, 0.        , 0.        , 0.        ]])
</pre>

```python
new_post_vec = vectorizer.transform(new_post_to_vectorize)
new_post_vec.toarray()
```

<pre>
array([[0.78528828, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.6191303 , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ]])
</pre>

```python
def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarray())
```


```python
dist = [dist_norm(each, new_post_vec) for each in X]
dist
```

<pre>
[1.2544516324460193, 1.2261339938790283, 1.414213562373095, 1.1021396119773588]
</pre>