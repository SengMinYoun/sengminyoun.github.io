---

layout: single
title: 'Text Preprocessing [Korean Alert]'
categories: Natural_Language_Processing
tag: [python, deeplearning, machinelearning, NLP]
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
This series will focus on NLP process of the Korean language!!! 
</div>

## Text Preprocessing ##

### Morphological Analysis (형태소 분석) ###

```python
from konlpy.tag import kkma, Hannanum, Komoran, Okt
```

*for the life of me I can't install MeCab*

```python
kkma = Kkma()
okt = Okt()
komoran = Komoran()
hannanum = Hannanum() 
```

### Tokenization (토큰화) ### 

* Dealing with symbols 
  * For symbols that are not alphanumeric we need to handle them separately 
  * It could be possible to remove all symbols. In such a case, however, it would be difficult to train the model for cases in which the symbols have special meaning 
* Tokenization methods for certain special words 
  * We need to figure out how to handle words that consist of multiple words but is in fact one word. For instance, United Kingdom
  * It is much more effective for the model if the user considers the characteristics of the words before parsing 

**We can use the nltk package to tokenize**

```python
import nltk 
nltk.download('punkt')
```

```python
sentence = 'Time is money friend'

from nltk.tokenize import word_tokenize

tokens = word_tokenize(sentence)
tokens
```

> ['Time', 'is', 'money', 'friend']

* With Korean it's not enough to simply tokenize words based on empty spaces due to particles and conjunctions 
* We can remove particles and conjunctions after tagging 

```python 
sentence = '언제나 현재에 집중할 수 있다면 행복할 것이다.'

print(f'Kkma 형태소 분석: {kkma.pos(sentence)}')
print(f'Hannanum 형태소 분석: {hannanum.pos(sentence)}')
print(f'Komoran 형태소 분석: {komoran.pos(sentence)}')
print(f'Okt 형태소 분석: {okt.pos(sentence)}')
```

> Kkma 형태소 분석: [('언제나', 'MAG'), ('현재', 'NNG'), ('에', 'JKM'), ('집중', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETD'), ('수', 'NNB'), ('있', 'VA'), ('다면', 'ECE'), ('행복', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETD'), ('것', 'NNB'), ('이', 'VCP'), ('다', 'EFN'), ('.', 'SF')] 
>
> Hannanum 형태소 분석: [('언제나', 'M'), ('현재', 'N'), ('에', 'J'), ('집중', 'N'), ('하', 'X'), ('ㄹ', 'E'), ('수', 'N'), ('있', 'P'), ('다면', 'E'), ('행복', 'N'), ('하', 'X'), ('ㄹ', 'E'), ('것', 'N'), ('이', 'J'), ('다', 'E'), ('.', 'S')] 
>
> Komoran 형태소 분석: [('언제나', 'MAG'), ('현재', 'NNG'), ('에', 'JKB'), ('집중', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM'), ('수', 'NNB'), ('있', 'VV'), ('다면', 'EC'), ('행복', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETM'), ('것', 'NNB'), ('이', 'VCP'), ('다', 'EF'), ('.', 'SF')] 
>
> Okt 형태소 분석: [('언제나', 'Adverb'), ('현재', 'Noun'), ('에', 'Josa'), ('집중', 'Noun'), ('할', 'Verb'), ('수', 'Noun'), ('있다면', 'Adjective'), ('행복할', 'Adjective'), ('것', 'Noun'), ('이다', 'Josa'), ('.', 'Punctuation')]

**Tokenizing without tagging**

```python 
print(f'Kkma 형태소 분석: {kkma.morphs(sentence)}')
print(f'Hannanum 형태소 분석: {hannanum.morphs(sentence)}')
print(f'Komoran 형태소 분석: {komoran.morphs(sentence)}')
print(f'Okt 형태소 분석: {okt.morphs(sentence)}')
```

>Kkma 형태소 분석: ['언제나', '현재', '에', '집중', '하', 'ㄹ', '수', '있', '다면', '행복', '하', 'ㄹ', '것', '이', '다', '.'] 
>
>Hannanum 형태소 분석: ['언제나', '현재', '에', '집중', '하', 'ㄹ', '수', '있', '다면', '행복', '하', 'ㄹ', '것', '이', '다', '.'] 
>
>Komoran 형태소 분석: ['언제나', '현재', '에', '집중', '하', 'ㄹ', '수', '있', '다면', '행복', '하', 'ㄹ', '것', '이', '다', '.'] 
>
>Okt 형태소 분석: ['언제나', '현재', '에', '집중', '할', '수', '있다면', '행복할', '것', '이다', '.']

**Just the nouns**

```python
print(f'Kkma 형태소 분석: {kkma.nouns(sentence)}')
print(f'Hannanum 형태소 분석: {hannanum.nouns(sentence)}')
print(f'Komoran 형태소 분석: {komoran.nouns(sentence)}')
print(f'Okt 형태소 분석: {okt.nouns(sentence)}')
```

> Kkma 형태소 분석: ['현재', '집중', '수', '행복'] 
>
> Hannanum 형태소 분석: ['현재', '집중', '수', '행복', '것']
>
> Komoran 형태소 분석: ['현재', '집중', '수', '행복', '것']
>
> Okt 형태소 분석: ['현재', '집중', '수', '것']

### Sentence Tokenization (문장 토큰화) ###

```python
sentences = 'The world is a beautiful book. \n But of little use to him who cannot read it.'
print(sentences)
tokens = [x for x in sentences.split('\n')]
tokens
```

> The world is a beautiful book.  
> But of little use to him who cannot read it.
>
> ['The world is a beautiful book. ', ' But of little use to him who cannot read it.']

**We can also us sent_tokenize()**

```python
from nltk.tokenize import sent_tokenize 

tokens = sent_tokenize(sentences)
tokens
```

> ['The world is a beautiful book.', 'But of little use to him who cannot read it.']

**We can use Kkma() to tokenize sentences**

```python
text = '진짜? 내일 뭐하지. 이렇게 애매모호한 문장도? 밥은 먹었어. 나는' 
print(kkma.sentences(text))
```

> ['진짜? 내일 뭐하지. 이렇게 애매모호한 문장도? 밥은 먹었어.', '나는']

**We can also use the kss library to tokenize Korean sentences**

```python
import kss 
print(kss.split_sentences(text))
```

>['진짜? 내일 뭐하지.', '이렇게 애매모호한 문장도?', '밥은 먹었어.', '나는']

### Tokenization Using Regex Expressions ### 



