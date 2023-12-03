---

layout: single
title: 'Text Preprocessing [Korean Alert]'
categories: Natural_Language_Processing_Lab
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

```python
from nltk.tokenize import RegexpTokenizer 

sentence  = "Where there\'s will, there\'s a way"

tokenizer = RegexpTokenizer("[\w]+")

tokens = tokenizer.tokenize(sentence)
tokens
```

> ['Where', 'there', 's', 'will', 'there', 's', 'a', 'way']

```python 
sentence = '안녕하세요 ㅋㅋ 저는 자연어 처리(Natural Language Processing)를ㄹ!! 배우고 있습니다.'

tokenizer = RegexpTokenizer('[가-힣]+') #like [a-z] for Korean
tokens = tokenizer.tokenize(sentence)
tokens
```

> ['안녕하세요', '저는', '자연어', '처리', '를', '배우고', '있습니다']

### TextBlob을 이용한 토큰화 ###

```python 
from textblob import TextBlob

eng = "Where there\'s will, there\'s a way"

blob = TextBlob(eng)
blob.words
```

>WordList(['Where', 'there', "'s", 'will', 'there', "'s", 'a', 'way'])

```python
kor = '성공의 비결은 단 한 가지, 잘할 수 있는 일에 광적으로 집중하는 것이다.'

blob = TextBlob(kor)
blob.words
```

>WordList(['성공의', '비결은', '단', '한', '가지', '잘할', '수', '있는', '일에', '광적으로', '집중하는', '것이다'])

### 케라스를 이용한 토큰화

```python 
from keras.preprocessing.text import text_to_word_sequence 

text_to_word_sequence(eng)
```

> ['where', "there's", 'will', "there's", 'a', 'way']

```python
text_to_word_sequence(kor)
```

> ['성공의', '비결은', '단', '한', '가지', '잘할', '수', '있는', '일에', '광적으로', '집중하는', '것이다']

### 기타 토큰나이저 

* WhiteSpaceTokenizer : 공백을 기준으로 토큰화
* WordPunktTokenizer : 텍스트를 알파벳 문자, 숫자, 알파벳 이외의 문자 리스트로 토큰화
*  MWETokenizer : MWE는 Multi-Word Expression의 약자로 'Republic of Korea'와 같이 여러 단어로 이뤄진 특정 그룹을 한 개체로 취급
*  TweetTokenizer : 트위터에서 사용되는 문장의 토큰화를 위해서 만들어졌으며, 문장 속 감성의 표현과 감정을 다룸

### n-gram 추출 

* n-gram은 n개의 어절이나 음절을 연쇄적으로 분류해 그 빈도를 분석
* $ n = 1 $ -> unigram, $ n = 2 $ -> bigram etc. 

```python 
from nltk import ngrams 

sentence = 'There is no royal road to learning'
bigram = list(ngrams(sentence.split(),2))
print(bigram)
```

> [('There', 'is'), ('is', 'no'), ('no', 'royal'), ('royal', 'road'), ('road', 'to'), ('to', 'learning')]

```python 
trigram = list(ngrams(sentence.split(), 3))
print(trigram)
```

> [('There', 'is', 'no'), ('is', 'no', 'royal'), ('no', 'royal', 'road'), ('royal', 'road', 'to'), ('road', 'to', 'learning')]

*Bigram is used most frequently*

### POS(parts-of-speech) Tagging 

POS Tags 

| Number | Tag | Description | 설명 |
| -- | -- | -- | -- |
| 1 | `CC` | Coordinating conjunction |
| 2 | `CD` | Cardinal number |
| 3 | `DT` | Determiner | 한정사
| 4 | `EX` | Existential there |
| 5 | `FW` | Foreign word | 외래어 |
| 6 | `IN` | Preposition or subordinating conjunction | 전치사 또는 종속 접속사 |
| 7 | `JJ` | Adjective | 형용사 |
| 8 | `JJR` | Adjective, comparative | 헝용사, 비교급 |
| 9 | `JJS` | Adjective, superlative | 형용사, 최상급 |
| 10 | `LS` | List item marker |
| 11 | `MD` | Modal |
| 12 | `NN` | Noun, singular or mass | 명사, 단수형 |
| 13 | `NNS` | Noun, plural | 명사, 복수형 |
| 14 | `NNP` | Proper noun, singular | 고유명사, 단수형 |
| 15 | `NNPS` | Proper noun, plural | 고유명사, 복수형 |
| 16 | `PDT` | Predeterminer | 전치한정사 |
| 17 | `POS` | Possessive ending | 소유형용사 |
| 18 | `PRP` | Personal pronoun | 인칭 대명사 |
| 19 | `PRP$` | Possessive pronoun | 소유 대명사 |
| 20 | `RB` | Adverb | 부사 |
| 21 | `RBR` | Adverb, comparative | 부사, 비교급 |
| 22 | `RBS` | Adverb, superlative | 부사, 최상급 |
| 23 | `RP` | Particle |
| 24 | `SYM` | Symbol | 기호
| 25 | `TO` | to |
| 26 | `UH` | Interjection | 감탄사 |
| 27 | `VB` | Verb, base form | 동사, 원형 |
| 28 | `VBD` | Verb, past tense | 동사, 과거형 |
| 29 | `VBG` | Verb, gerund or present participle | 동사, 현재분사 |
| 30 | `VBN` | Verb, past participle | 동사, 과거분사 |
| 31 | `VBP` | Verb, non-3rd person singular present | 동사, 비3인칭 단수 |
| 32 | `VBZ` | Verb, 3rd person singular present | 동사, 3인칭 단수 |
| 33 | `WDT` | Wh-determiner |
| 34 | `WP` | Wh-pronoun |
| 35 | `WP$` | Possessive wh-pronoun |
| 36 | `WRB` | Wh-adverb |

```python 
sentence = 'Think like a man of action and act like a man of thought.'

words = word_tokenize(sentence)

nltk.download('averaged_perceptron_tagger')

nltk.pos_tag(words)
```

> [('Think', 'VBP'), ('like', 'IN'), ('a', 'DT'), ('man', 'NN'), ('of', 'IN'), ('action', 'NN'), ('and', 'CC'), ('act', 'NN'), ('like', 'IN'), ('a', 'DT'), ('man', 'NN'), ('of', 'IN'), ('thought', 'NN'), ('.', '.')]

```python 
nltk.pos_tag(word_tokenize('A rolling stone gathers no moss'))
```

> [('A', 'DT'), ('rolling', 'VBG'), ('stone', 'NN'), ('gathers', 'NNS'), ('no', 'DT'), ('moss', 'NN')]

### Removing Stopwords 

* 영어의 전치사(on, in), 한국어의 조사(을, 를) 등은 분석에 필요하지 않은 경우가 많음
* 길이가 짧은 단어, 등장 빈도 수가 적은 단어들도 분석에 큰 영향을 주지 않음
* 일반적으로 사용되는 도구들은 해당 단어들을 제거해주지만 완벽하게 제거되지는 않음
* 사용자가 불용어 사전을 만들어 해당 단어들을 제거하는 것이 좋음
* **도구들이 걸러주지 않는 전치사, 조사 등을 불용어 사전을 만들어 불필요한 단어들을 제거**

``` python 
stop_words = 'on in the'
stop_words = stop_words.split(' ')
stop_words
```

```python 
sentence = 'singer on the stage'
sentence = sentence.split(' ')
nouns = []

for noun in sentence: 
    if noun not in stop_words:
        nouns.append(noun)
nouns
```

> ['singer', 'stage']

### Spelling Correction 

* 텍스트에 오탈자가 존재하는 경우가 있음

* 예를 들어, 단어 'apple'을 'aplpe'과 같이 철자 순서가 바뀌거나 spple 같이 철자가 틀릴 수 있음
* 사람이 적절한 추정을 통해 이해하는데는 문제가 없지만, 컴퓨터는 이러한 단어를 그대로 받아들여 처리가 필요
*  철자 교정 알고리즘은 이미 개발되어 워드 프로세서나 다양한 서비스에서 많이 적용됨

```python 
from autocorrect import Speller 

spell = Speller('en')
print(spell('peoplle'))
print(spell('peple'))
print(spell('eople'))
```

> people 
>
> people 
>
> people

```python 
s = word_tokenize('Thhe Earlly biirrd catchess the womm.')
print(s)

ss =' '.join([spell(s) for s in s])
ss
```

>['Thhe', 'Earlly', 'biirrd', 'catchess', 'the', 'womm', '.']
>
>'The Early bird catches the worm .'

### Singularize and Pluralize 

```python 
from textblob import TextBlob 

words = 'apples bananas oranges'
textblob = TextBlob(words)

print(textblob.words)
print(textblob.words.singularize())
```

> ['apples', 'bananas', 'oranges'] 
>
> ['apple', 'banana', 'orange']

```python 
words = 'car train airplane'
textblob = TextBlob(words)

print(textblob.words)
print(textblob.words.pluralize())
```

> ['car', 'train', 'airplane']
>
> ['cars', 'trains', 'airplanes']

### Stemming (어간 추출)

"어간(Stem)을 추출하는 작업을 어간 추출(stemming)이라고 합니다. 어간 추출은 형태학적 분석을 단순화한 버전이라고 볼 수도 있고, 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업이라고 볼 수도 있습니다. 이 작업은 섬세한 작업이 아니기 때문에 어간 추출 후에 나오는 결과 단어는 사전에 존재하지 않는 단어일 수도 있습니다. 예제를 보면 쉽게 이해할 수 있습니다. 어간 추출 알고리즘 중 하나인 포터 알고리즘(Porter Algorithm)에 아래의 문자열을 입력으로 넣는다고 해봅시다." - *딥러닝을 이용한 자연어 처리*

```python 
stemmer = nltk.stem.PorterStemmer()
```

```python 
stemmer.stem('application')
```

> 'applic'

```python 
stemmer.stem('beginner')
```

> 'beginn'

```python 
stemmer.stem('beginning')
```

> 'begin'

```python 
stemmer.stem('catches')
```

> 'catcch'

### Lemmatization (표제어 추출)

"표제어(Lemma)는 한글로는 '표제어' 또는 '기본 사전형 단어' 정도의 의미를 갖습니다. 표제어 추출은 단어들로부터 표제어를 찾아가는 과정입니다. 표제어 추출은 단어들이 다른 형태를 가지더라도, 그 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단합니다. 예를 들어서 am, are, is는 서로 다른 스펠링이지만 그 뿌리 단어는 be라고 볼 수 있습니다. 이때, 이 단어들의 표제어는 be라고 합니다.

표제어 추출을 하는 가장 섬세한 방법은 단어의 형태학적 파싱을 먼저 진행하는 것입니다. 형태소란 '의미를 가진 가장 작은 단위'를 뜻합니다. 그리고 형태학(morphology)이란 형태소로부터 단어들을 만들어가는 학문을 뜻합니다. 형태소의 종류로 어간(stem)과 접사(affix)가 존재합니다." - *딥러닝을 이용한 자연어 처리*

1. Stem (어간) : 단어의 의미를 담고 있는 단어의 핵심 부분
2. Affix (접사): 단어에 추가적인 의미를 주는 부분 

```python 
from nltk.stem.wordnet import WordNetLemmatizer 

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() 
```

```python 
lemmatizer.lemmatize('application')
```

> 'application'

```python 
lemmatizer.lemmatize('beginning')
```

> 'beginning'

```python
lemmatizer.lemmatize('catches')
```

> 'catch'

* You can also specify the part of seech 

```python 
lemmatizer.lemmatize('has', 'v')
```

> 'have'

```python
lemmatizer.lemmatize('am', 'v')
```

> 'be'

```python
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :',words)
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])
```

> 표제어 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting'] 
>
> 표제어 추출 후 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']

### Dependency Parsing (의존 구문 분석)

* Spacy 라이브러리를 이용해 문장을 token들로 구성된 document로 처리하고, 각 token에는 품사, 의존 관계, 개체명 정보 등이 태깅
  * token.text : token 문자열 
  * token.dep_ : token과 token의 지배소 간의 의존 관계 유형 
  * token.head : 지배소 token

```python 
import spacy 
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

doc = nlp('I am taking a course on natural language processing at my university')

for token in doc: 
    print(token.text, token.dep_, token.head.text)
```

> I nsubj taking 
>
> am aux taking 
>
> taking ROOT taking 
>
> a det course 
>
> course dobj taking 
>
> on prep course 
>
> natural amod language
>
> language compound processing 
>
> processing pobj on 
>
> at prep taking 
>
> my poss university
>
>  university pobj at

```python
displacy.render(doc, style='dep', jupyter=True)
```

![image-20231203191542673](/images/2023-11-29-Text Preprocessing/image-20231203191542673.png)

### Lexical Ambiguity

![image-20231203191703503](/images/2023-11-29-Text Preprocessing/image-20231203191703503.png)

```python
from nltk.wsd import lesk 
from nltk.tokenize import word_tokenize

s = 'I saw bats.'

print(word_tokenize(s))
print(lesk(word_tokenize(s), 'saw'))
print(lesk(word_tokenize(s), 'bats'))
```

> ['I', 'saw', 'bats', '.'] 
>
> Synset('saw.v.01')
>
> Synset('squash_racket.n.01')

