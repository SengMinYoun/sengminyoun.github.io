---

layout: single
title: 'The NLP Process [Korean Alert]'
categories: Natural Language Processing
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

## The NLP Process 

### What is NLP? ###

* Language refers to the language that we use in our day to day lives 
* NLP is a method of analyzing such language 
* NLP has a lot of applications: organizing text, sentiment analysis, summarization, translation, chatbots, voice recognition etc. 



### The NLP Process ###

![image-20231129135204444](/images/2023-11-29-The NLP Process/image-20231129135204444.png)

* 어휘 분석 (Lexical Analysis) : Discerning the structure of the words then analyzing the meaning as well as the parts of speech for the words
  * 형태소 분석 (Morphological Analysis): Breaking the words down to their most basic forms then analyzing the rules and restraints related to the words 
  * 품사 태깅 (Part-of-Speech Tagging): Tagging the words with their respective parts of speech in order to solve ambiguity and redundancy
* 구문 분석 (Syntactic Analysis): Analyzing the grammatical components of a natural language sentence 
  * 구구조 구문 분석 (Phrase Structure Parsing): 구구조문접에 기반한 구문분석 기술 
  * 의전 구문 분석 (Dependency Parsing): Analyzing the relationship between words in a sentence to analyze the grammatic structure of the sentence as a whole 
* 의미 분석 (Semantic Analysis): Interpreting the sentence based on its meaning 
  * 단어 의미 중의성 해소 (Word Sense Disambiguation): 문장 내 중의성을 가지는 어휘를 사전에 정의된 의미와 매칭하여 어희적 중의성 해결 
  * 의미역 분석 (Semantic Role): In order to analyze the actual meaning,  analyze the role of the words within a sentence and categorize them
  * ![image-20231129141357505](/images/2023-11-29-The NLP Process/image-20231129141357505.png)