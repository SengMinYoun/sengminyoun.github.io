---
layout: single
title:  "Naive Bayes Sentiment Analysis"
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

<div class ="notice--success">
This series will deal with utilizing machine learning libraries. It is intended as a refresher on the topics. 
</div>

## 감성분석



```python
from nltk.tokenize import word_tokenize
import nltk
```


```python
train = [
    ("I like you", "pos"),
    ("I hate you", "neg"), 
    ("you like me", "neg"), 
    ("I like her", "pos")
]
```


```python
sentence = train[0][0]
word_tokenize(sentence)
```

<pre>
['I', 'like', 'you']
</pre>
* 말 뭉치 만들기



```python
all_words = set(
    word.lower() for sentence in train for word in word_tokenize(sentence[0])
)
all_words
```

<pre>
{'hate', 'her', 'i', 'like', 'me', 'you'}
</pre>

```python
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]
t
```

<pre>
[({'her': False,
   'like': True,
   'me': False,
   'hate': False,
   'you': True,
   'i': False},
  'pos'),
 ({'her': False,
   'like': False,
   'me': False,
   'hate': True,
   'you': True,
   'i': False},
  'neg'),
 ({'her': False,
   'like': True,
   'me': True,
   'hate': False,
   'you': True,
   'i': False},
  'neg'),
 ({'her': True,
   'like': True,
   'me': False,
   'hate': False,
   'you': False,
   'i': False},
  'pos')]
</pre>
* 말 뭉치 대비해서 단어가 있고 없음을 표기



```python
classifier = nltk.NaiveBayesClassifier.train(t)
```


```python
classifier.show_most_informative_features()
```

<pre>
Most Informative Features
                    hate = False             pos : neg    =      1.7 : 1.0
                     her = False             neg : pos    =      1.7 : 1.0
                    like = True              pos : neg    =      1.7 : 1.0
                      me = False             pos : neg    =      1.7 : 1.0
                     you = True              neg : pos    =      1.7 : 1.0
                       i = False             neg : pos    =      1.0 : 1.0
</pre>
like가 있을 때, positive할 확률이 1.7:1.0이다...



```python
test_sentence = "I like MeRui"
test_sent_features = {
    word.lower() : (word in word_tokenize(test_sentence.lower())) for word in all_words
}
test_sent_features
```

<pre>
{'her': False,
 'like': True,
 'me': False,
 'hate': False,
 'you': False,
 'i': True}
</pre>

```python
classifier.classify(test_sent_features)
```

<pre>
'pos'
</pre>
### 한글 감성분석



```python
from konlpy.tag import Okt
```


```python
post_tagger = Okt()
```


```python
train = [
    ("메리가 좋아", "pos"),
    ("고양이도 좋아", "pos"),
    ("난 수업이 지루해", "neg"), 
    ("메리는 이쁜 고양이야", "pos"),
    ("난 마치고 메리랑 놀거야", "pos")
]
```


```python
word_tokenize(train[0][0])
```

<pre>
['메리가', '좋아']
</pre>
* 말 뭉치 만들기



```python
all_words = set(
    word for sentence in train for word in word_tokenize(sentence[0])
)
all_words
```

<pre>
{'고양이도',
 '고양이야',
 '난',
 '놀거야',
 '마치고',
 '메리가',
 '메리는',
 '메리랑',
 '수업이',
 '이쁜',
 '좋아',
 '지루해'}
</pre>
* 메리가, 메리는 메리랑이 다 다른 단어가 됨



```python
t = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in train]
t
```

<pre>
[({'이쁜': False,
   '지루해': False,
   '메리랑': False,
   '수업이': False,
   '좋아': True,
   '마치고': False,
   '놀거야': False,
   '난': False,
   '고양이도': False,
   '고양이야': False,
   '메리는': False,
   '메리가': True},
  'pos'),
 ({'이쁜': False,
   '지루해': False,
   '메리랑': False,
   '수업이': False,
   '좋아': True,
   '마치고': False,
   '놀거야': False,
   '난': False,
   '고양이도': True,
   '고양이야': False,
   '메리는': False,
   '메리가': False},
  'pos'),
 ({'이쁜': False,
   '지루해': True,
   '메리랑': False,
   '수업이': True,
   '좋아': False,
   '마치고': False,
   '놀거야': False,
   '난': True,
   '고양이도': False,
   '고양이야': False,
   '메리는': False,
   '메리가': False},
  'neg'),
 ({'이쁜': True,
   '지루해': False,
   '메리랑': False,
   '수업이': False,
   '좋아': False,
   '마치고': False,
   '놀거야': False,
   '난': False,
   '고양이도': False,
   '고양이야': True,
   '메리는': True,
   '메리가': False},
  'pos'),
 ({'이쁜': False,
   '지루해': False,
   '메리랑': True,
   '수업이': False,
   '좋아': False,
   '마치고': True,
   '놀거야': True,
   '난': True,
   '고양이도': False,
   '고양이야': False,
   '메리는': False,
   '메리가': False},
  'pos')]
</pre>

```python
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()
```

<pre>
Most Informative Features
                       난 = True              neg : pos    =      2.5 : 1.0
                      좋아 = False             neg : pos    =      1.5 : 1.0
                    고양이도 = False             neg : pos    =      1.1 : 1.0
                    고양이야 = False             neg : pos    =      1.1 : 1.0
                     놀거야 = False             neg : pos    =      1.1 : 1.0
                     마치고 = False             neg : pos    =      1.1 : 1.0
                     메리가 = False             neg : pos    =      1.1 : 1.0
                     메리는 = False             neg : pos    =      1.1 : 1.0
                     메리랑 = False             neg : pos    =      1.1 : 1.0
                      이쁜 = False             neg : pos    =      1.1 : 1.0
</pre>
* 문제가 있어 보인다...negative가 전반적으로 높다.



```python
test_sentence = '난 수업이 마치면 메리랑 놀거야'

test_sent_features = {
    word : (word in word_tokenize(test_sentence.lower())) for word in all_words
}
test_sent_features
```

<pre>
{'이쁜': False,
 '지루해': False,
 '메리랑': True,
 '수업이': True,
 '좋아': False,
 '마치고': False,
 '놀거야': True,
 '난': True,
 '고양이도': False,
 '고양이야': False,
 '메리는': False,
 '메리가': False}
</pre>

```python
classifier.classify(test_sent_features)
```

<pre>
'neg'
</pre>
* Negative가 뜬다 ㅠㅠ

* 우리나라 말은 형태소 분석이 필수다!

* Lucy Park님의 추천대로 진행해보자 

* 예를 들어 '이'가 성이 될 수도 있고 명사가 될 수도 있잖아!


* 형태소 분석을 한 후 품사를 단어 뒤에 붙여 넣도록 하자



```python
def tokenize(doc):
    return["/".join(t) for t in post_tagger.pos(doc, norm=True, stem=True)]
```


```python
train_docs = [(tokenize(row[0]), row[1]) for row in train]
train_docs
```

<pre>
[(['메리/Noun', '가/Josa', '좋다/Adjective'], 'pos'),
 (['고양이/Noun', '도/Josa', '좋다/Adjective'], 'pos'),
 (['난/Noun', '수업/Noun', '이/Josa', '지루하다/Adjective'], 'neg'),
 (['메리/Noun', '는/Josa', '이쁘다/Adjective', '고양이/Noun', '야/Josa'], 'pos'),
 (['난/Noun', '마치/Noun', '고/Josa', '메리/Noun', '랑/Josa', '놀다/Verb'], 'pos')]
</pre>
* 풀어서 말뭉치 만들기



```python
tokens = [t for d in train_docs for t in d[0]]
tokens
```

<pre>
['메리/Noun',
 '가/Josa',
 '좋다/Adjective',
 '고양이/Noun',
 '도/Josa',
 '좋다/Adjective',
 '난/Noun',
 '수업/Noun',
 '이/Josa',
 '지루하다/Adjective',
 '메리/Noun',
 '는/Josa',
 '이쁘다/Adjective',
 '고양이/Noun',
 '야/Josa',
 '난/Noun',
 '마치/Noun',
 '고/Josa',
 '메리/Noun',
 '랑/Josa',
 '놀다/Verb']
</pre>

```python
def term_exists(doc):
    return {word : (word in set(doc)) for word in tokens}
```


```python
train_xy = [(term_exists(d), c) for d, c in train_docs]
train_xy
```

<pre>
[({'메리/Noun': True,
   '가/Josa': True,
   '좋다/Adjective': True,
   '고양이/Noun': False,
   '도/Josa': False,
   '난/Noun': False,
   '수업/Noun': False,
   '이/Josa': False,
   '지루하다/Adjective': False,
   '는/Josa': False,
   '이쁘다/Adjective': False,
   '야/Josa': False,
   '마치/Noun': False,
   '고/Josa': False,
   '랑/Josa': False,
   '놀다/Verb': False},
  'pos'),
 ({'메리/Noun': False,
   '가/Josa': False,
   '좋다/Adjective': True,
   '고양이/Noun': True,
   '도/Josa': True,
   '난/Noun': False,
   '수업/Noun': False,
   '이/Josa': False,
   '지루하다/Adjective': False,
   '는/Josa': False,
   '이쁘다/Adjective': False,
   '야/Josa': False,
   '마치/Noun': False,
   '고/Josa': False,
   '랑/Josa': False,
   '놀다/Verb': False},
  'pos'),
 ({'메리/Noun': False,
   '가/Josa': False,
   '좋다/Adjective': False,
   '고양이/Noun': False,
   '도/Josa': False,
   '난/Noun': True,
   '수업/Noun': True,
   '이/Josa': True,
   '지루하다/Adjective': True,
   '는/Josa': False,
   '이쁘다/Adjective': False,
   '야/Josa': False,
   '마치/Noun': False,
   '고/Josa': False,
   '랑/Josa': False,
   '놀다/Verb': False},
  'neg'),
 ({'메리/Noun': True,
   '가/Josa': False,
   '좋다/Adjective': False,
   '고양이/Noun': True,
   '도/Josa': False,
   '난/Noun': False,
   '수업/Noun': False,
   '이/Josa': False,
   '지루하다/Adjective': False,
   '는/Josa': True,
   '이쁘다/Adjective': True,
   '야/Josa': True,
   '마치/Noun': False,
   '고/Josa': False,
   '랑/Josa': False,
   '놀다/Verb': False},
  'pos'),
 ({'메리/Noun': True,
   '가/Josa': False,
   '좋다/Adjective': False,
   '고양이/Noun': False,
   '도/Josa': False,
   '난/Noun': True,
   '수업/Noun': False,
   '이/Josa': False,
   '지루하다/Adjective': False,
   '는/Josa': False,
   '이쁘다/Adjective': False,
   '야/Josa': False,
   '마치/Noun': True,
   '고/Josa': True,
   '랑/Josa': True,
   '놀다/Verb': True},
  'pos')]
</pre>

```python
classifier = nltk.NaiveBayesClassifier.train(train_xy)
classifier.show_most_informative_features()
```

<pre>
Most Informative Features
                  난/Noun = True              neg : pos    =      2.5 : 1.0
                 메리/Noun = False             neg : pos    =      2.5 : 1.0
                고양이/Noun = False             neg : pos    =      1.5 : 1.0
            좋다/Adjective = False             neg : pos    =      1.5 : 1.0
                  가/Josa = False             neg : pos    =      1.1 : 1.0
                  고/Josa = False             neg : pos    =      1.1 : 1.0
                 놀다/Verb = False             neg : pos    =      1.1 : 1.0
                  는/Josa = False             neg : pos    =      1.1 : 1.0
                  도/Josa = False             neg : pos    =      1.1 : 1.0
                  랑/Josa = False             neg : pos    =      1.1 : 1.0
</pre>

```python
test_sentence = '난 수업이 마치면 메리랑 놀거야'

test_docs = tokenize(test_sentence)
test_docs
```

<pre>
['난/Noun',
 '수업/Noun',
 '이/Josa',
 '마치/Noun',
 '면/Josa',
 '메리/Noun',
 '랑/Josa',
 '놀다/Verb']
</pre>

```python
test_setence_features = {word : (word in test_docs) for word in tokens}

test_setence_features
```

<pre>
{'메리/Noun': True,
 '가/Josa': False,
 '좋다/Adjective': False,
 '고양이/Noun': False,
 '도/Josa': False,
 '난/Noun': True,
 '수업/Noun': True,
 '이/Josa': True,
 '지루하다/Adjective': False,
 '는/Josa': False,
 '이쁘다/Adjective': False,
 '야/Josa': False,
 '마치/Noun': True,
 '고/Josa': False,
 '랑/Josa': True,
 '놀다/Verb': True}
</pre>

```python
classifier.classify(test_setence_features)
```

<pre>
'neg'
</pre>