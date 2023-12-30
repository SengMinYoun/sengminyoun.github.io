---
layout: single
title: 'Ecol-system'
categories: Projects
tag: [python, sklearn, machinelearning, deeplearning]
toc: true
author_profile: false
sidebar:
    nav: 'docs'
search: true
sidebar:
    nav: "counts"
use_math: true
---

## ECOL-System (Participation Award)

### Background

My submission for the 2023 Korea Sports Promotion Foundation Big Data Analysis for the Promotion of Daily Sports Participation. 

![image-20231230095957975](../../../../../images/projects/image-20231230095957975.png)

![image-20231230100021350](../../../../../images/projects/image-20231230100021350.png)

![image-20231230100104574](../../../../../images/projects/image-20231230100104574.png)

![image-20231230100122691](../../../../../images/projects/image-20231230100122691.png)

![image-20231230100139907](../../../../../images/projects/image-20231230100139907.png)

### Goals 

Estimate factors that contribute to people enjoying golf on a regular basis and cluster people based on the KSPO 100-Physical Fitness Test to suggest appropriate exercises related to golf. 

![image-20231230100413349](../../../../../images/projects/image-20231230100413349.png)

### The Run Down

The data was collected from the Korea Public Data Portal and the Culture Big Data Platform. For the regression analysis I used the 'Korea Free Time Activities Survey 2022' which is a comprehensive survey of what the participants do during their free time. It is an official survey conducted by the Ministry of Culture, Sports and Tourism. For the cluster analysis I used the 'KSPO 100-Physical Test' data set provided  by KSPO. It is the physical test results of all everyone who participated in KSPO's official fitness test. The test consists 

![image-20231230103712940](../../../../../images/projects/image-20231230103712940.png)

![image-20231230103754969](../../../../../images/projects/image-20231230103754969.png)

![image-20231230103821799](../../../../../images/projects/image-20231230103821799.png)

The number of participants that actually had golf experience was quite small. As the data was not balanced, in order to improve the model's AUC score, I used SMOTE to oversample. 

![image-20231230103844218](../../../../../images/projects/image-20231230103844218.png)

#### Preprocessing (Regression)

```python
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
```


```python
df = pd.read_spss('2022 국민여가활동조사 원시자료.SAV')

print(df.head())
```

<pre>
          ID Q1_1 Q1_2 Q1_3 Q1_4 Q1_5 Q1_6 Q1_7 Q1_8 Q1_9  ... DM5     DM6  \
0  1001005.0  미참여  미참여  미참여   참여   참여  미참여  미참여  미참여  미참여  ...  기혼     가구주   
1  1001008.0  미참여   참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  미혼  가구주 아님   
2  1001010.0  미참여  미참여  미참여  미참여  미참여  미참여   참여  미참여  미참여  ...  기혼     가구주   
3  1001018.0  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  기혼     가구주   
4  1001025.0  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  기혼     가구주   



            DM7        DM8  DM9 DM10 DM11 DM12 DM19 DM20  

0  고용원이 있는 자영업자   600만원 이상  대도시  수도권   서울  미등록  교대제  미시행  
1         상용근로자   600만원 이상  대도시  수도권   서울  미등록  전일제  미시행  
2         상용근로자  500~600만원  대도시  수도권   서울  미등록  전일제  미시행  
3  고용원이 없는 자영업자  300~400만원  대도시  수도권   서울  미등록  전일제  미시행  
4  고용원이 없는 자영업자  200~300만원  대도시  수도권   서울  미등록  전일제  미시행  

[5 rows x 372 columns]
</pre>

```python
df.isnull().values.any()
```

<pre>
True
</pre>



```python
df.info
```

<pre>
<bound method DataFrame.info of               ID Q1_1 Q1_2 Q1_3 Q1_4 Q1_5 Q1_6 Q1_7 Q1_8 Q1_9  ... DM5  \
0      1001005.0  미참여  미참여  미참여   참여   참여  미참여  미참여  미참여  미참여  ...  기혼   
1      1001008.0  미참여   참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  미혼   
2      1001010.0  미참여  미참여  미참여  미참여  미참여  미참여   참여  미참여  미참여  ...  기혼   
3      1001018.0  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  기혼   
4      1001025.0  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  기혼   
...          ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ...  ..   
10041  2477060.0  미참여  미참여  미참여  미참여  미참여  미참여   참여  미참여  미참여  ...  기혼   
10042  2477061.0  미참여  미참여  미참여  미참여  미참여  미참여   참여  미참여   참여  ...  기혼   
10043  2477062.0   참여  미참여  미참여  미참여  미참여  미참여   참여  미참여  미참여  ...  미혼   
10044  2477063.0  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  미참여  ...  기혼   
10045  2477064.0  미참여  미참여  미참여  미참여  미참여  미참여   참여  미참여  미참여  ...  기혼   



          DM6           DM7        DM8   DM9 DM10 DM11     DM12 DM19 DM20  

0         가구주  고용원이 있는 자영업자   600만원 이상   대도시  수도권   서울      미등록  교대제  미시행  
1      가구주 아님         상용근로자   600만원 이상   대도시  수도권   서울      미등록  전일제  미시행  
2         가구주         상용근로자  500~600만원   대도시  수도권   서울      미등록  전일제  미시행  
3         가구주  고용원이 없는 자영업자  300~400만원   대도시  수도권   서울      미등록  전일제  미시행  
4         가구주  고용원이 없는 자영업자  200~300만원   대도시  수도권   서울      미등록  전일제  미시행  
...       ...           ...        ...   ...  ...  ...      ...  ...  ...  
10041  가구주 아님         상용근로자  500~600만원  읍면지역  동남권   경남  해당사항 없음  전일제  미시행  
10042  가구주 아님         상용근로자   600만원 이상  읍면지역  동남권   경남  해당사항 없음  전일제  미시행  
10043     가구주         상용근로자  400~500만원  읍면지역  동남권   경남  해당사항 없음  전일제  미시행  
10044     가구주         상용근로자  500~600만원  읍면지역  동남권   경남  해당사항 없음  전일제  미시행  
10045  가구주 아님         임시근로자  500~600만원  읍면지역  동남권   경남  해당사항 없음  전일제  미시행  

[10046 rows x 372 columns]>
</pre>
칼럼 추출



```python
filtered1 = df[['Q1_24', 'Q2_2_1', 'Q2_4_1_1_N', 'Q2_5_1_N',
                'Q8', 'Q9_N', 'Q10_N','Q11_1_C', 'Q11_3_D',
                'Q12_1_C',
                'Q12_3_D', 'Q13_2_1', 'Q13_2_2',
                'Q13_4_1_N', 'Q13_4_2_N', 'Q21','Q32', 'Q33',
                'Q39', 'Q40_N', 'Q41_1_N', 'Q41_2_N', 'Q37_1',
                'Q44', 'Q44_4_N', 'DM1', 'DM2', 'DM3', 'DM4', 'DM5', 
                'DM11', 'DM12']].copy()
```

<br>한 번 이상 참여한 여가활동에 대한 응답이 '골프'면 골프경험자로 분류.

골프 경험자 = 1

골프 무경험자 = 0



```python
filtered1['experience'] = 0
```


```python
filtered1['experience'] = np.where(filtered1['Q1_24'] == '참여', 1, 0)
```


```python
le = LabelEncoder()
filtered1['Q1_24'] = le.fit_transform(filtered1['Q1_24'])
```

'Q8' NaN -> 1로 변경 (1 == 전혀 그렇지 않음)



```python
filtered1['Q8'].value_counts()
```

<pre>
5            1397
6            1330
보통이다          591
매우 그렇다        361
3             113
2              80
전혀 그렇지 않다      54
Name: Q8, dtype: int64
</pre>



```python
filtered1['Q8'].replace(['보통이다', '매우 그렇다', '전혀 그렇지 않다'],
                         [4, 7, 1], inplace=True)
```


```python
filtered1['Q8'].value_counts()
```

<pre>
5    1397
6    1330
4     591
7     361
3     113
2      80
1      54
Name: Q8, dtype: int64
</pre>



```python
filtered1['Q8'].fillna(1, inplace=True)
```


```python
filtered1.iloc[0:20]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q1_24</th>
      <th>Q2_2_1</th>
      <th>Q2_4_1_1_N</th>
      <th>Q2_5_1_N</th>
      <th>Q8</th>
      <th>Q9_N</th>
      <th>Q10_N</th>
      <th>Q11_1_C</th>
      <th>Q11_3_D</th>
      <th>Q12_1_C</th>
      <th>...</th>
      <th>Q44</th>
      <th>Q44_4_N</th>
      <th>DM1</th>
      <th>DM2</th>
      <th>DM3</th>
      <th>DM4</th>
      <th>DM5</th>
      <th>DM11</th>
      <th>DM12</th>
      <th>experience</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>10000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>수영</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>60.0</td>
      <td>여성</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>10.0</td>
      <td>여성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>30000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>11.0</td>
      <td>여성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>700000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>10.0</td>
      <td>여성</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>15000.0</td>
      <td>6</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>12.0</td>
      <td>남성</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>볼링, 탁구</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>50.0</td>
      <td>남성</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>친구(연인 포함)와 함께</td>
      <td>11.0</td>
      <td>10000.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>농구, 배구, 야구, 축구, 족구</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>40.0</td>
      <td>남성</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>8.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>40.0</td>
      <td>여성</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>8.0</td>
      <td>60000.0</td>
      <td>5</td>
      <td>200000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>9.0</td>
      <td>남성</td>
      <td>20대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>30000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>10.0</td>
      <td>여성</td>
      <td>40대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>150000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>8.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>수영</td>
      <td>무응답</td>
      <td>...</td>
      <td>하지 않았다</td>
      <td>NaN</td>
      <td>여성</td>
      <td>50대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>150000.0</td>
      <td>1</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>요가/필라테스/태보</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>40.0</td>
      <td>여성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>스포츠 경기 직접관람- 경기장방문관람(축구, 야구,농구, 배구 등)</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>45.0</td>
      <td>남성</td>
      <td>40대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>스포츠 경기 직접관람- 경기장방문관람(축구, 야구,농구, 배구 등)</td>
      <td>...</td>
      <td>하였다</td>
      <td>50.0</td>
      <td>남성</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하지 않았다</td>
      <td>NaN</td>
      <td>여성</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>20000.0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하지 않았다</td>
      <td>NaN</td>
      <td>여성</td>
      <td>15~19세</td>
      <td>중졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>300000.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>...</td>
      <td>하였다</td>
      <td>45.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>...</td>
      <td>하였다</td>
      <td>40.0</td>
      <td>여성</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 33 columns</p>

</div>


Q11_3_D + Q12_3_D + Q37_1 합쳐서 골프 희망자 칼럼 생성



```python
filtered1[filtered1['Q11_3_D']=='골프']
filtered1[filtered1['Q12_3_D']=='골프']
filtered1['experience_hope'] = np.where((filtered1['Q11_3_D'] == '골프') | (filtered1['Q12_3_D'] == '골프') | (filtered1['Q37_1'] == '골프'), 1, 0)
```


```python
filtered1.drop(['Q11_3_D', 'Q12_3_D', 'Q37_1'], axis= 1, inplace=True)
```


```python
filtered1.drop(['Q13_4_1_N', 'Q13_4_2_N'], axis= 1, inplace=True)
```

Q21 예 -> 1, 아니요 -> 0



```python
filtered1['Q21'] = np.where(filtered1['Q21'] == '예', 1, 0)
```

Q33 

빈값 -> 평균값으로 대체



```python
filtered1['Q33'].unique()
```

<pre>
['', '보통이다', '매우 좋음', '매우 나쁨']
Categories (4, object): ['', '매우 나쁨', '매우 좋음', '보통이다']
</pre>



```python
filtered1['Q33'].replace(['', '보통이다', '매우 좋음', '매우 나쁨'], [4, 4, 7, 1 ], inplace=True)
```

Q33 무응답이 너무 많아 컬럼 제거



```python
filtered1.drop('Q33', axis= 1, inplace=True)
```

Q41_1_N 결측치 확인



```python
filtered1[filtered1['Q41_1_N'].isnull()]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q1_24</th>
      <th>Q2_2_1</th>
      <th>Q2_4_1_1_N</th>
      <th>Q2_5_1_N</th>
      <th>Q8</th>
      <th>Q9_N</th>
      <th>Q10_N</th>
      <th>Q11_1_C</th>
      <th>Q12_1_C</th>
      <th>Q13_2_1</th>
      <th>...</th>
      <th>Q44_4_N</th>
      <th>DM1</th>
      <th>DM2</th>
      <th>DM3</th>
      <th>DM4</th>
      <th>DM5</th>
      <th>DM11</th>
      <th>DM12</th>
      <th>experience</th>
      <th>experience_hope</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 28 columns</p>

</div>



```python
filtered1[filtered1['Q44'].isnull()]
filtered1['Q44'].unique()
```

<pre>
['하였다', '하지 않았다']
Categories (2, object): ['하였다', '하지 않았다']
</pre>



```python
filtered1.iloc[0:50]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q1_24</th>
      <th>Q2_2_1</th>
      <th>Q2_4_1_1_N</th>
      <th>Q2_5_1_N</th>
      <th>Q8</th>
      <th>Q9_N</th>
      <th>Q10_N</th>
      <th>Q11_1_C</th>
      <th>Q12_1_C</th>
      <th>Q13_2_1</th>
      <th>...</th>
      <th>Q44_4_N</th>
      <th>DM1</th>
      <th>DM2</th>
      <th>DM3</th>
      <th>DM4</th>
      <th>DM5</th>
      <th>DM11</th>
      <th>DM12</th>
      <th>experience</th>
      <th>experience_hope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>10000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>60.0</td>
      <td>여성</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>10.0</td>
      <td>여성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>30000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>11.0</td>
      <td>여성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>700000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>10.0</td>
      <td>여성</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>15000.0</td>
      <td>6</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>12.0</td>
      <td>남성</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>3</td>
      <td>...</td>
      <td>50.0</td>
      <td>남성</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>친구(연인 포함)와 함께</td>
      <td>11.0</td>
      <td>10000.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>40.0</td>
      <td>남성</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>40.0</td>
      <td>여성</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>8.0</td>
      <td>60000.0</td>
      <td>5</td>
      <td>200000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>9.0</td>
      <td>남성</td>
      <td>20대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>30000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>10.0</td>
      <td>여성</td>
      <td>40대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>150000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>50대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>150000.0</td>
      <td>1</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>40.0</td>
      <td>여성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>스포츠 경기 직접관람- 경기장방문관람(축구, 야구,농구, 배구 등)</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>45.0</td>
      <td>남성</td>
      <td>40대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>스포츠 경기 직접관람- 경기장방문관람(축구, 야구,농구, 배구 등)</td>
      <td>보통이다</td>
      <td>...</td>
      <td>50.0</td>
      <td>남성</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>20000.0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>15~19세</td>
      <td>중졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>300000.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>5</td>
      <td>...</td>
      <td>45.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>40.0</td>
      <td>여성</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>150000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>40.0</td>
      <td>여성</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>200000.0</td>
      <td>300000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>14.0</td>
      <td>여성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>혼자서</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>20000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>NaN</td>
      <td>남성</td>
      <td>15~19세</td>
      <td>중졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>15000.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>140000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>10.0</td>
      <td>여성</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>4.0</td>
      <td>300000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>300000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>3</td>
      <td>...</td>
      <td>44.0</td>
      <td>여성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>4.0</td>
      <td>50000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>보통이다</td>
      <td>...</td>
      <td>50.0</td>
      <td>남성</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>1000.0</td>
      <td>7</td>
      <td>80000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>200.0</td>
      <td>6</td>
      <td>30000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>40대</td>
      <td>중졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>20000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>보통이다</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>여성</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>10000.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>여성</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>40000.0</td>
      <td>80000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>8.0</td>
      <td>여성</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>8.0</td>
      <td>남성</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>40대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>혼자서</td>
      <td>0.0</td>
      <td>2500.0</td>
      <td>6</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>4.0</td>
      <td>남성</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>80000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>11.0</td>
      <td>여성</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>9.0</td>
      <td>남성</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>20000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>48.0</td>
      <td>여성</td>
      <td>70대 이상</td>
      <td>초졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>15000.0</td>
      <td>80000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>6</td>
      <td>...</td>
      <td>NaN</td>
      <td>남성</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>30000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>5.0</td>
      <td>50000.0</td>
      <td>1</td>
      <td>150000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>53.0</td>
      <td>남성</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>200000.0</td>
      <td>250000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>20대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>150000.0</td>
      <td>20000.0</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>5</td>
      <td>...</td>
      <td>49.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
      <td>혼자서</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>80000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>5</td>
      <td>...</td>
      <td>23.0</td>
      <td>여성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>160000.0</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>45.0</td>
      <td>남성</td>
      <td>60대</td>
      <td>중졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>혼자서</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>60대</td>
      <td>초졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>30000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>NaN</td>
      <td>여성</td>
      <td>70대 이상</td>
      <td>초졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>보통이다</td>
      <td>...</td>
      <td>40.0</td>
      <td>남성</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 28 columns</p>

</div>



```python
filtered1.drop(['Q13_2_1', 'Q13_2_2'], axis=1, inplace=True)
```


```python
filtered1.columns
```

<pre>
Index(['Q1_24', 'Q2_2_1', 'Q2_4_1_1_N', 'Q2_5_1_N', 'Q8', 'Q9_N', 'Q10_N',
       'Q11_1_C', 'Q12_1_C', 'Q21', 'Q32', 'Q39', 'Q40_N', 'Q41_1_N',
       'Q41_2_N', 'Q44', 'Q44_4_N', 'DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM11',
       'DM12', 'experience', 'experience_hope'],
      dtype='object')
</pre>



```python
filtered1['Q44'].unique()
filtered1['Q44'] = np.where(filtered1['Q44'] =='하였다', 1, 0)
```

Q44_4_N 결측치 평균값으로 대체



```python
filtered1['Q44_4_N'].fillna((filtered1['Q44_4_N'].mean()), inplace=True)
```


```python
filtered1['DM1'].unique()
```

<pre>
['여성', '남성']
Categories (2, object): ['남성', '여성']
</pre>



```python
filtered1['DM1'] = np.where(filtered1['DM1'] =='남자', 1, 2)
```


```python
filtered1.iloc[0:50]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q1_24</th>
      <th>Q2_2_1</th>
      <th>Q2_4_1_1_N</th>
      <th>Q2_5_1_N</th>
      <th>Q8</th>
      <th>Q9_N</th>
      <th>Q10_N</th>
      <th>Q11_1_C</th>
      <th>Q12_1_C</th>
      <th>Q21</th>
      <th>...</th>
      <th>Q44_4_N</th>
      <th>DM1</th>
      <th>DM2</th>
      <th>DM3</th>
      <th>DM4</th>
      <th>DM5</th>
      <th>DM11</th>
      <th>DM12</th>
      <th>experience</th>
      <th>experience_hope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>10000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>1</td>
      <td>...</td>
      <td>60.000000</td>
      <td>2</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>10.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>30000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>11.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>700000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>10.000000</td>
      <td>2</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>15000.0</td>
      <td>6</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>12.000000</td>
      <td>2</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>50.000000</td>
      <td>2</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>친구(연인 포함)와 함께</td>
      <td>11.0</td>
      <td>10000.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>40.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>40.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>8.0</td>
      <td>60000.0</td>
      <td>5</td>
      <td>200000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>9.000000</td>
      <td>2</td>
      <td>20대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>30000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>10.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>150000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>50대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>150000.0</td>
      <td>1</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>40.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>스포츠 경기 직접관람- 경기장방문관람(축구, 야구,농구, 배구 등)</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>45.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>스포츠 경기 직접관람- 경기장방문관람(축구, 야구,농구, 배구 등)</td>
      <td>0</td>
      <td>...</td>
      <td>50.000000</td>
      <td>2</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>20000.0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>15~19세</td>
      <td>중졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>300000.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>0</td>
      <td>...</td>
      <td>45.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>40.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>150000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>40.000000</td>
      <td>2</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>200000.0</td>
      <td>300000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>14.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>혼자서</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>20000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>15~19세</td>
      <td>중졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>15000.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>140000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>10.000000</td>
      <td>2</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>4.0</td>
      <td>300000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>300000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>44.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>4.0</td>
      <td>50000.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>0</td>
      <td>...</td>
      <td>50.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>1000.0</td>
      <td>7</td>
      <td>80000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>1</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>200.0</td>
      <td>6</td>
      <td>30000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>중졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>20000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>온라인게임 경기관람(e-스포츠 경기 포함)</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>혼자서</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>10000.0</td>
      <td>5</td>
      <td>50000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>40000.0</td>
      <td>80000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>300000.0</td>
      <td>500000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>20대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>8.000000</td>
      <td>2</td>
      <td>50대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>40대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>혼자서</td>
      <td>0.0</td>
      <td>2500.0</td>
      <td>6</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>1</td>
      <td>...</td>
      <td>4.000000</td>
      <td>2</td>
      <td>60대</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>80000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>11.000000</td>
      <td>2</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>9.000000</td>
      <td>2</td>
      <td>70대 이상</td>
      <td>고졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>미등록</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>20000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>48.000000</td>
      <td>2</td>
      <td>70대 이상</td>
      <td>초졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>15000.0</td>
      <td>80000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>60대</td>
      <td>대졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>가족(친척 포함)과 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>30000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>5.0</td>
      <td>50000.0</td>
      <td>1</td>
      <td>150000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>53.000000</td>
      <td>2</td>
      <td>40대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>3.0</td>
      <td>30000.0</td>
      <td>1</td>
      <td>200000.0</td>
      <td>250000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>20대</td>
      <td>고졸</td>
      <td>3인 이상</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>150000.0</td>
      <td>20000.0</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>0</td>
      <td>...</td>
      <td>49.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0</td>
      <td>혼자서</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>80000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>23.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>3인 이상</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>160000.0</td>
      <td>스포츠 경기 간접관람- TV, DMB를 통한관람(축구, 야구, 농구, 배구 등)</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>45.000000</td>
      <td>2</td>
      <td>60대</td>
      <td>중졸</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>혼자서</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>60대</td>
      <td>초졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0</td>
      <td>혼자서</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>20000.0</td>
      <td>30000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>43.049928</td>
      <td>2</td>
      <td>70대 이상</td>
      <td>초졸이하</td>
      <td>2인</td>
      <td>기혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0</td>
      <td>친구(연인 포함)와 함께</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>100000.0</td>
      <td>150000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>40.000000</td>
      <td>2</td>
      <td>30대</td>
      <td>대졸이하</td>
      <td>1인</td>
      <td>미혼</td>
      <td>서울</td>
      <td>해당사항 없음</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 26 columns</p>

</div>



```python
filtered1['Q32'].unique()
```

<pre>
['6', '7', '8', '5', '9', '4', '3', '행복', '불행', '2']
Categories (10, object): ['2', '3', '4', '5', ..., '8', '9', '불행', '행복']
</pre>



```python
filtered1['Q32'].replace(['불행', '행복'], [1,10], inplace=True)
```


```python
filtered1['DM2'].unique()
filtered1['DM2'].replace(['60대', '30대', '70대 이상', '20대', '50대', '40대', '15~19세'], [65, 35, 75, 25, 55, 45, 17.5], inplace=True)
```


```python
filtered1['DM3'].unique()
```

<pre>
['대졸이하', '고졸', '중졸', '초졸이하']
Categories (4, object): ['고졸', '대졸이하', '중졸', '초졸이하']
</pre>



```python
filtered1['DM3'].replace(['대졸이하', '고졸', '중졸', '초졸이하'], [4,3,2,1], inplace=True)
```


```python
filtered1.drop(['DM4', 'Q39'], axis=1, inplace=True)
```


```python
filtered1['Q2_2_1'].replace(['가족(친척 포함)과 함께', '혼자서', '친구(연인 포함)와 함께', '동호회(종교단체 등 포함)회원과 함께', '직장 동료', '기타'],
                            [4, 1, 2, 5, 4, 1], inplace=True)
```


```python
filtered1['DM5'] = np.where(filtered1['DM5']=='기혼', 1, 0)
```


```python
filtered1['DM11'].value_counts()
```

<pre>
경기    1326
서울    1164
부산     690
경남     678
경북     620
인천     617
대구     580
충남     558
전북     523
전남     515
충북     482
강원     472
대전     460
광주     447
울산     393
제주     300
세종     221
Name: DM11, dtype: int64
</pre>



```python
filtered1['DM11'].replace(['경기', '서울', '부산', '경남', '경북', '인천', '대구', '충남',
                           '전북', '전남', '충북', '강원', '대전', '광주', '울산', '제주', '세종'],
                           [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], inplace=True)
```


```python
filtered1['DM12'] = np.where((filtered1['DM12']=='미등록') |(filtered1['DM12']=='장애등록'), 1,0 )
```


```python
filtered1.drop(['Q40_N', 'Q41_2_N'],axis=1, inplace=True)
```


```python
filtered1.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Q1_24</th>
      <th>Q2_2_1</th>
      <th>Q2_4_1_1_N</th>
      <th>Q2_5_1_N</th>
      <th>Q8</th>
      <th>Q9_N</th>
      <th>Q10_N</th>
      <th>Q11_1_C</th>
      <th>Q12_1_C</th>
      <th>Q21</th>
      <th>...</th>
      <th>Q44</th>
      <th>Q44_4_N</th>
      <th>DM1</th>
      <th>DM2</th>
      <th>DM3</th>
      <th>DM5</th>
      <th>DM11</th>
      <th>DM12</th>
      <th>experience</th>
      <th>experience_hope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>10000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>50000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>60.0</td>
      <td>2</td>
      <td>65.0</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10.0</td>
      <td>2</td>
      <td>35.0</td>
      <td>4</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>30000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11.0</td>
      <td>2</td>
      <td>35.0</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>700000.0</td>
      <td>200000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>10.0</td>
      <td>2</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>3.0</td>
      <td>15000.0</td>
      <td>6</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>무응답</td>
      <td>무응답</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12.0</td>
      <td>2</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>

</div>



```python
filtered1['Q11_1_C'].unique()
filtered1['Q12_1_C'].unique()
filtered1['watch_sports'] = np.where((filtered1['Q11_1_C']=='무응답') |(filtered1['Q12_1_C']=='무응답'), 0,1 )
filtered1.drop(['Q11_1_C','Q12_1_C'], axis=1, inplace=True)
```

Q1_24 -> 골프참여 여부

Q2_2_1 -> 지난 1년간 가장 많이 참여한 여가활동 동반자(1순위)

Q2_4_1_1_N -> 지난 1년간 가장 많이 참여한 여가활동 1회 여가 소요시간 - 시간(1순위)

Q2_5_1_N -> 지난 1년간 가장 많이 참여한 여가활동 1회 기준 비용(1순위)

Q8 -> 여가활동을 위한 별도의 지식,정보 습득여부

Q9_N -> 지난 1년간 여가활동을 위한 지출액(월 평균)

Q10_N -> 적절하다고 생각하는 여가비용(월 평균)

Q21 -> 지난 1년간 여가활동을 위한 동호회 참여 여부

Q32 -> 현재 행복 수준 - 10점 척도

Q41_1_N -> 젠체 동거 가구원 수

Q44 -> 지난 1주일 간 경제활동 여부

Q44_4_N -> 주당 평근 근무 시간 

watch_sports -> 스포프 관람여부

DM1 -> 성별

DM2 -> 연령

DM3 -> 학력

DM5 -> 혼인상태

DM11 -> 지역(시도)

DM12 -> 장애여부




```python
filtered1.columns
```

<pre>
Index(['Q1_24', 'Q2_2_1', 'Q2_4_1_1_N', 'Q2_5_1_N', 'Q8', 'Q9_N', 'Q10_N',
       'Q21', 'Q32', 'Q41_1_N', 'Q44', 'Q44_4_N', 'DM1', 'DM2', 'DM3', 'DM5',
       'DM11', 'DM12', 'experience', 'experience_hope', 'watch_sports'],
      dtype='object')
</pre>



```python
column_list = ['tried_golf','whom_with', 'time_per', 'cost_per','research',
               'spent_last_year', 'amount_appropriate','club_participate',
               'cur_happiness', 'family', 'wk_econ_act', 'avg_wk_workhr',
               'sex','age','edu', 'marrital', 'province', 'disabled', 'exp_golf',
               'want_golf', 'watch_sports'
               ]
len(column_list)
```

<pre>
21
</pre>



```python
filtered1.columns = column_list
```


```python
final_df = filtered1
```


```python
final_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tried_golf</th>
      <th>whom_with</th>
      <th>time_per</th>
      <th>cost_per</th>
      <th>research</th>
      <th>spent_last_year</th>
      <th>amount_appropriate</th>
      <th>club_participate</th>
      <th>cur_happiness</th>
      <th>family</th>
      <th>...</th>
      <th>avg_wk_workhr</th>
      <th>sex</th>
      <th>age</th>
      <th>edu</th>
      <th>marrital</th>
      <th>province</th>
      <th>disabled</th>
      <th>exp_golf</th>
      <th>want_golf</th>
      <th>watch_sports</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>10000.0</td>
      <td>4</td>
      <td>100000.0</td>
      <td>50000.0</td>
      <td>1</td>
      <td>6</td>
      <td>3.0</td>
      <td>...</td>
      <td>60.0</td>
      <td>2</td>
      <td>65.0</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>50000.0</td>
      <td>100000.0</td>
      <td>0</td>
      <td>7</td>
      <td>4.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>2</td>
      <td>35.0</td>
      <td>4</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>30000.0</td>
      <td>0</td>
      <td>7</td>
      <td>3.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>2</td>
      <td>35.0</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>700000.0</td>
      <td>200000.0</td>
      <td>0</td>
      <td>8</td>
      <td>2.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>2</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>3.0</td>
      <td>15000.0</td>
      <td>6</td>
      <td>100000.0</td>
      <td>100000.0</td>
      <td>0</td>
      <td>7</td>
      <td>2.0</td>
      <td>...</td>
      <td>12.0</td>
      <td>2</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10041</th>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>1000.0</td>
      <td>1</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>0</td>
      <td>7</td>
      <td>4.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>2</td>
      <td>55.0</td>
      <td>4</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10042</th>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>500.0</td>
      <td>1</td>
      <td>300000.0</td>
      <td>300000.0</td>
      <td>0</td>
      <td>7</td>
      <td>3.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>2</td>
      <td>45.0</td>
      <td>4</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10043</th>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2000.0</td>
      <td>1</td>
      <td>500000.0</td>
      <td>500000.0</td>
      <td>0</td>
      <td>5</td>
      <td>3.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>2</td>
      <td>45.0</td>
      <td>4</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10044</th>
      <td>0</td>
      <td>4</td>
      <td>1.0</td>
      <td>5000.0</td>
      <td>7</td>
      <td>200000.0</td>
      <td>200000.0</td>
      <td>0</td>
      <td>8</td>
      <td>2.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>2</td>
      <td>55.0</td>
      <td>3</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10045</th>
      <td>0</td>
      <td>4</td>
      <td>0.0</td>
      <td>500.0</td>
      <td>5</td>
      <td>150000.0</td>
      <td>150000.0</td>
      <td>0</td>
      <td>6</td>
      <td>4.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>2</td>
      <td>45.0</td>
      <td>3</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10046 rows × 21 columns</p>

</div>



```python
final_df.to_csv('final_df.csv')
```



#### Preprocessing (Clustering)

```python
import pandas as pd 
import numpy as np
df = pd.read_csv("체력측정 및 운동처방 종합 데이터202304.csv")
```


```python
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MBER_SEQ_NO_VALUE</th>
      <th>MESURE_SEQ_NO</th>
      <th>CNTER_NM</th>
      <th>AGRDE_FLAG_NM</th>
      <th>MESURE_PLACE_FLAG_NM</th>
      <th>MESURE_AGE_CO</th>
      <th>INPT_FLAG_NM</th>
      <th>CRTFC_FLAG_NM</th>
      <th>MESURE_DE</th>
      <th>SEXDSTN_FLAG_CD</th>
      <th>...</th>
      <th>MESURE_IEM_033_VALUE</th>
      <th>MESURE_IEM_034_VALUE</th>
      <th>MESURE_IEM_035_VALUE</th>
      <th>MESURE_IEM_036_VALUE</th>
      <th>MESURE_IEM_037_VALUE</th>
      <th>MESURE_IEM_038_VALUE</th>
      <th>MESURE_IEM_039_VALUE</th>
      <th>MESURE_IEM_040_VALUE</th>
      <th>MESURE_IEM_041_VALUE</th>
      <th>MVM_PRSCRPTN_CN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAGxF0mZtZ/3b+x5zazDbd66</td>
      <td>4</td>
      <td>진천</td>
      <td>성인</td>
      <td>출장</td>
      <td>19</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230401</td>
      <td>M</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.372</td>
      <td>NaN</td>
      <td>본운동:팔벌려뛰기,무릎 높여 제자리 달리기,버피운동,앉았다 일어서면서 점프하기,스텝...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAGgD+o0OQjIcaYN2+OH9KgR</td>
      <td>2</td>
      <td>남구(부산)</td>
      <td>성인</td>
      <td>일반</td>
      <td>25</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230401</td>
      <td>M</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>115.0</td>
      <td>46.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.258</td>
      <td>0.613</td>
      <td>마무리운동:목 스트레칭,등/어깨 뒤쪽 스트레칭,가슴/어깨 앞쪽 스트레칭,아래 팔 스...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAEr1ywKnMVvf+c2brlHRiqJ</td>
      <td>7</td>
      <td>진천</td>
      <td>성인</td>
      <td>출장</td>
      <td>19</td>
      <td>관리자</td>
      <td>2등급</td>
      <td>20230401</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.333</td>
      <td>NaN</td>
      <td>본운동:팔꿈치 굽히기,뒤로 팔굽혀펴기,손목 펴기/굽히기,물병 옆으로 들어올리기</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAH61XKoPuTqoG+dtklfWiGi</td>
      <td>1</td>
      <td>남구(부산)</td>
      <td>성인</td>
      <td>일반</td>
      <td>45</td>
      <td>관리자</td>
      <td>3등급</td>
      <td>20230401</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>103.0</td>
      <td>36.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.383</td>
      <td>0.490</td>
      <td>본운동:옆으로 누워 버티기,누워서 다리 들어올리기,네발기기 자세로 팔 다리 들기,윗...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAH8ELoWRwcO2aw8rnMArnnJ</td>
      <td>9</td>
      <td>진천</td>
      <td>성인</td>
      <td>출장</td>
      <td>19</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230401</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.371</td>
      <td>NaN</td>
      <td>본운동:서서 뒤꿈치 들기,의자에 앉아 다리 뻗어 올리기,앉았다 일어서기,한발 앞으로...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18964</th>
      <td>AAGyIS1HMTUWEj/zUNsBGsZc</td>
      <td>1</td>
      <td>계룡</td>
      <td>청소년</td>
      <td>출장</td>
      <td>15</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230413</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18965</th>
      <td>AAGcsTrVJ7+e9hUD1uIaqM6Z</td>
      <td>1</td>
      <td>강릉</td>
      <td>청소년</td>
      <td>출장</td>
      <td>13</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230413</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>본운동:목 스트레칭,등/어깨 뒤쪽 스트레칭,가슴/어깨 앞쪽 스트레칭,엉덩이 스트레칭...</td>
    </tr>
    <tr>
      <th>18966</th>
      <td>AAFSseKC2lknbdiBNmoXSvpN</td>
      <td>1</td>
      <td>북구(광주)</td>
      <td>청소년</td>
      <td>출장</td>
      <td>15</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230413</td>
      <td>M</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18967</th>
      <td>AAGkpHcVYrIhzlN+me9UQgfS</td>
      <td>1</td>
      <td>시흥</td>
      <td>유소년</td>
      <td>출장</td>
      <td>11</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230413</td>
      <td>M</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>준비운동:넙다리 뒤쪽 스트레칭</td>
    </tr>
    <tr>
      <th>18968</th>
      <td>AAGjAkkQ0XEmBnnoHlVGkSd6</td>
      <td>1</td>
      <td>시흥</td>
      <td>유소년</td>
      <td>출장</td>
      <td>11</td>
      <td>관리자</td>
      <td>참가증</td>
      <td>20230413</td>
      <td>F</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>준비운동:전신 루틴 스트레칭 / 본운동:달리기,줄넘기 운동</td>
    </tr>
  </tbody>
</table>
<p>18969 rows × 51 columns</p>

</div>


* 체력 측정이니 결측치는 평균으로 변경!



```python
df.drop(['MBER_SEQ_NO_VALUE', 'MESURE_SEQ_NO',
         'CNTER_NM', 'AGRDE_FLAG_NM', 'MESURE_PLACE_FLAG_NM',
         'MESURE_AGE_CO', 'INPT_FLAG_NM', 'CRTFC_FLAG_NM', 'MESURE_DE',
          ], axis = 1, inplace=True)
```


```python
df['SEXDSTN_FLAG_CD'] = np.where(df['SEXDSTN_FLAG_CD'] == 'M', 1, 0)
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEXDSTN_FLAG_CD</th>
      <th>MESURE_IEM_001_VALUE</th>
      <th>MESURE_IEM_002_VALUE</th>
      <th>MESURE_IEM_003_VALUE</th>
      <th>MESURE_IEM_004_VALUE</th>
      <th>MESURE_IEM_005_VALUE</th>
      <th>MESURE_IEM_006_VALUE</th>
      <th>MESURE_IEM_007_VALUE</th>
      <th>MESURE_IEM_008_VALUE</th>
      <th>MESURE_IEM_009_VALUE</th>
      <th>...</th>
      <th>MESURE_IEM_033_VALUE</th>
      <th>MESURE_IEM_034_VALUE</th>
      <th>MESURE_IEM_035_VALUE</th>
      <th>MESURE_IEM_036_VALUE</th>
      <th>MESURE_IEM_037_VALUE</th>
      <th>MESURE_IEM_038_VALUE</th>
      <th>MESURE_IEM_039_VALUE</th>
      <th>MESURE_IEM_040_VALUE</th>
      <th>MESURE_IEM_041_VALUE</th>
      <th>MVM_PRSCRPTN_CN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>180.6</td>
      <td>75.1</td>
      <td>20.0</td>
      <td>74.0</td>
      <td>65.0</td>
      <td>111.0</td>
      <td>50.4</td>
      <td>52.8</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.372</td>
      <td>NaN</td>
      <td>본운동:팔벌려뛰기,무릎 높여 제자리 달리기,버피운동,앉았다 일어서면서 점프하기,스텝...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>174.4</td>
      <td>63.8</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>70.0</td>
      <td>127.0</td>
      <td>49.9</td>
      <td>52.3</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>115.0</td>
      <td>46.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.258</td>
      <td>0.613</td>
      <td>마무리운동:목 스트레칭,등/어깨 뒤쪽 스트레칭,가슴/어깨 앞쪽 스트레칭,아래 팔 스...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>156.5</td>
      <td>52.8</td>
      <td>34.8</td>
      <td>65.0</td>
      <td>80.0</td>
      <td>128.0</td>
      <td>21.5</td>
      <td>23.5</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.333</td>
      <td>NaN</td>
      <td>본운동:팔꿈치 굽히기,뒤로 팔굽혀펴기,손목 펴기/굽히기,물병 옆으로 들어올리기</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>162.2</td>
      <td>52.9</td>
      <td>24.5</td>
      <td>72.0</td>
      <td>84.0</td>
      <td>140.0</td>
      <td>33.0</td>
      <td>22.7</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>103.0</td>
      <td>36.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.383</td>
      <td>0.490</td>
      <td>본운동:옆으로 누워 버티기,누워서 다리 들어올리기,네발기기 자세로 팔 다리 들기,윗...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>162.0</td>
      <td>65.3</td>
      <td>38.6</td>
      <td>75.0</td>
      <td>77.0</td>
      <td>122.0</td>
      <td>22.2</td>
      <td>27.4</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.371</td>
      <td>NaN</td>
      <td>본운동:서서 뒤꿈치 들기,의자에 앉아 다리 뻗어 올리기,앉았다 일어서기,한발 앞으로...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>

</div>



```python
df.drop(['MVM_PRSCRPTN_CN'], axis = 1, inplace=True)
```


```python
df.drop(['MESURE_IEM_038_VALUE','MESURE_IEM_039_VALUE'], axis = 1, inplace=True)
```


```python
a = df.columns

for col_name in a: 
    if col_name == 'BMI':
        df[col_name].fillna(22.6, inplace=True)
    else:
        df[col_name].fillna(df[col_name].mean(), inplace= True)
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEXDSTN_FLAG_CD</th>
      <th>MESURE_IEM_001_VALUE</th>
      <th>MESURE_IEM_002_VALUE</th>
      <th>MESURE_IEM_003_VALUE</th>
      <th>MESURE_IEM_004_VALUE</th>
      <th>MESURE_IEM_005_VALUE</th>
      <th>MESURE_IEM_006_VALUE</th>
      <th>MESURE_IEM_007_VALUE</th>
      <th>MESURE_IEM_008_VALUE</th>
      <th>MESURE_IEM_009_VALUE</th>
      <th>...</th>
      <th>MESURE_IEM_030_VALUE</th>
      <th>MESURE_IEM_031_VALUE</th>
      <th>MESURE_IEM_032_VALUE</th>
      <th>MESURE_IEM_033_VALUE</th>
      <th>MESURE_IEM_034_VALUE</th>
      <th>MESURE_IEM_035_VALUE</th>
      <th>MESURE_IEM_036_VALUE</th>
      <th>MESURE_IEM_037_VALUE</th>
      <th>MESURE_IEM_040_VALUE</th>
      <th>MESURE_IEM_041_VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>180.6</td>
      <td>75.1</td>
      <td>20.0</td>
      <td>74.0</td>
      <td>65.0</td>
      <td>111.0</td>
      <td>50.4</td>
      <td>52.8</td>
      <td>30.645104</td>
      <td>...</td>
      <td>-29.167766</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>121.233432</td>
      <td>39.818698</td>
      <td>0.372</td>
      <td>0.460894</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>174.4</td>
      <td>63.8</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>70.0</td>
      <td>127.0</td>
      <td>49.9</td>
      <td>52.3</td>
      <td>30.645104</td>
      <td>...</td>
      <td>-29.167766</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>115.000000</td>
      <td>46.800000</td>
      <td>0.258</td>
      <td>0.613000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>156.5</td>
      <td>52.8</td>
      <td>34.8</td>
      <td>65.0</td>
      <td>80.0</td>
      <td>128.0</td>
      <td>21.5</td>
      <td>23.5</td>
      <td>30.645104</td>
      <td>...</td>
      <td>38.300000</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>121.233432</td>
      <td>39.818698</td>
      <td>0.333</td>
      <td>0.460894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>162.2</td>
      <td>52.9</td>
      <td>24.5</td>
      <td>72.0</td>
      <td>84.0</td>
      <td>140.0</td>
      <td>33.0</td>
      <td>22.7</td>
      <td>30.645104</td>
      <td>...</td>
      <td>-29.167766</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>103.000000</td>
      <td>36.200000</td>
      <td>0.383</td>
      <td>0.490000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>162.0</td>
      <td>65.3</td>
      <td>38.6</td>
      <td>75.0</td>
      <td>77.0</td>
      <td>122.0</td>
      <td>22.2</td>
      <td>27.4</td>
      <td>30.645104</td>
      <td>...</td>
      <td>32.700000</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>121.233432</td>
      <td>39.818698</td>
      <td>0.371</td>
      <td>0.460894</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>

</div>



```python
df.columns
```

<pre>
Index(['SEXDSTN_FLAG_CD', 'MESURE_IEM_001_VALUE', 'MESURE_IEM_002_VALUE',
       'MESURE_IEM_003_VALUE', 'MESURE_IEM_004_VALUE', 'MESURE_IEM_005_VALUE',
       'MESURE_IEM_006_VALUE', 'MESURE_IEM_007_VALUE', 'MESURE_IEM_008_VALUE',
       'MESURE_IEM_009_VALUE', 'MESURE_IEM_010_VALUE', 'MESURE_IEM_012_VALUE',
       'MESURE_IEM_013_VALUE', 'MESURE_IEM_014_VALUE', 'MESURE_IEM_015_VALUE',
       'MESURE_IEM_016_VALUE', 'MESURE_IEM_017_VALUE', 'MESURE_IEM_018_VALUE',
       'MESURE_IEM_019_VALUE', 'MESURE_IEM_020_VALUE', 'MESURE_IEM_021_VALUE',
       'MESURE_IEM_022_VALUE', 'MESURE_IEM_023_VALUE', 'MESURE_IEM_024_VALUE',
       'MESURE_IEM_025_VALUE', 'MESURE_IEM_026_VALUE', 'MESURE_IEM_027_VALUE',
       'MESURE_IEM_028_VALUE', 'MESURE_IEM_029_VALUE', 'MESURE_IEM_030_VALUE',
       'MESURE_IEM_031_VALUE', 'MESURE_IEM_032_VALUE', 'MESURE_IEM_033_VALUE',
       'MESURE_IEM_034_VALUE', 'MESURE_IEM_035_VALUE', 'MESURE_IEM_036_VALUE',
       'MESURE_IEM_037_VALUE', 'MESURE_IEM_040_VALUE', 'MESURE_IEM_041_VALUE'],
      dtype='object')
</pre>



```python
df.drop(['MESURE_IEM_003_VALUE','MESURE_IEM_004_VALUE'], axis= 1, inplace = True)
```


```python
df.drop(['MESURE_IEM_041_VALUE'], axis= 1, inplace = True)
```


```python
df.columns
```

<pre>
Index(['SEXDSTN_FLAG_CD', 'MESURE_IEM_001_VALUE', 'MESURE_IEM_002_VALUE',
       'MESURE_IEM_005_VALUE', 'MESURE_IEM_006_VALUE', 'MESURE_IEM_007_VALUE',
       'MESURE_IEM_008_VALUE', 'MESURE_IEM_009_VALUE', 'MESURE_IEM_010_VALUE',
       'MESURE_IEM_012_VALUE', 'MESURE_IEM_013_VALUE', 'MESURE_IEM_014_VALUE',
       'MESURE_IEM_015_VALUE', 'MESURE_IEM_016_VALUE', 'MESURE_IEM_017_VALUE',
       'MESURE_IEM_018_VALUE', 'MESURE_IEM_019_VALUE', 'MESURE_IEM_020_VALUE',
       'MESURE_IEM_021_VALUE', 'MESURE_IEM_022_VALUE', 'MESURE_IEM_023_VALUE',
       'MESURE_IEM_024_VALUE', 'MESURE_IEM_025_VALUE', 'MESURE_IEM_026_VALUE',
       'MESURE_IEM_027_VALUE', 'MESURE_IEM_028_VALUE', 'MESURE_IEM_029_VALUE',
       'MESURE_IEM_030_VALUE', 'MESURE_IEM_031_VALUE', 'MESURE_IEM_032_VALUE',
       'MESURE_IEM_033_VALUE', 'MESURE_IEM_034_VALUE', 'MESURE_IEM_035_VALUE',
       'MESURE_IEM_036_VALUE', 'MESURE_IEM_037_VALUE', 'MESURE_IEM_040_VALUE'],
      dtype='object')
</pre>



```python
col_names = [
    'gender', 'height(cm)', 'weight(kg)', 
    'BP(min)', 'BP(max)', 'grip(left)', 
    'grip(right)', 'situps', 'standing_jump', 
    'torso_bend', 'illinois', 'air_time', 
     'hand_eye', 'hand_eye_no', 'hand_ee_time', 
     'BMI', 'cross_situp', 'back_forth_run', 
     '10M_back_forth_4times(sec)', 'standing_long_jump',
     'chair_sitting', '6min_walk', '2min_still_walk',
     'sit_3M_target', '8walk(sec)', 'relaive_grip', 
     'belly_stretch', 'endurance_run', 'treadmil_stable',
     'treadmil_3min', 'tradmil_6', 'treadmil_9', 
     'treadmil_output', 'step_bpm','step_output', 
     'cog_reaction'
]

df.columns = col_names
```


```python
df.drop('belly_stretch', axis=1, inplace=True)
```


```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>height(cm)</th>
      <th>weight(kg)</th>
      <th>BP(min)</th>
      <th>BP(max)</th>
      <th>grip(left)</th>
      <th>grip(right)</th>
      <th>situps</th>
      <th>standing_jump</th>
      <th>torso_bend</th>
      <th>...</th>
      <th>relaive_grip</th>
      <th>endurance_run</th>
      <th>treadmil_stable</th>
      <th>treadmil_3min</th>
      <th>tradmil_6</th>
      <th>treadmil_9</th>
      <th>treadmil_output</th>
      <th>step_bpm</th>
      <th>step_output</th>
      <th>cog_reaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>180.6</td>
      <td>75.1</td>
      <td>65.0</td>
      <td>111.0</td>
      <td>50.4</td>
      <td>52.8</td>
      <td>30.645104</td>
      <td>39.2164</td>
      <td>15.3</td>
      <td>...</td>
      <td>70.3</td>
      <td>-29.167766</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>121.233432</td>
      <td>39.818698</td>
      <td>0.372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>174.4</td>
      <td>63.8</td>
      <td>70.0</td>
      <td>127.0</td>
      <td>49.9</td>
      <td>52.3</td>
      <td>30.645104</td>
      <td>39.2164</td>
      <td>-13.8</td>
      <td>...</td>
      <td>82.0</td>
      <td>-29.167766</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>115.000000</td>
      <td>46.800000</td>
      <td>0.258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>156.5</td>
      <td>52.8</td>
      <td>80.0</td>
      <td>128.0</td>
      <td>21.5</td>
      <td>23.5</td>
      <td>30.645104</td>
      <td>39.2164</td>
      <td>22.2</td>
      <td>...</td>
      <td>44.5</td>
      <td>38.300000</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>121.233432</td>
      <td>39.818698</td>
      <td>0.333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>162.2</td>
      <td>52.9</td>
      <td>84.0</td>
      <td>140.0</td>
      <td>33.0</td>
      <td>22.7</td>
      <td>30.645104</td>
      <td>39.2164</td>
      <td>11.9</td>
      <td>...</td>
      <td>62.4</td>
      <td>-29.167766</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>103.000000</td>
      <td>36.200000</td>
      <td>0.383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>162.0</td>
      <td>65.3</td>
      <td>77.0</td>
      <td>122.0</td>
      <td>22.2</td>
      <td>27.4</td>
      <td>30.645104</td>
      <td>39.2164</td>
      <td>13.2</td>
      <td>...</td>
      <td>42.0</td>
      <td>32.700000</td>
      <td>85.982379</td>
      <td>115.92511</td>
      <td>137.704194</td>
      <td>155.070513</td>
      <td>39.262252</td>
      <td>121.233432</td>
      <td>39.818698</td>
      <td>0.371</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>

</div>

![image-20231230101851146](../../../../../images/projects/image-20231230101851146.png)

#### Regression Analysis 



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve 
from sklearn.metrics import matthews_corrcoef

```


```python
df = pd.read_csv('final_df.csv', index_col=0)
df.drop('tried_golf', axis = 1, inplace=True)
df.drop('watch_sports', axis = 1, inplace=True)
df.head()
df['sex'] = np.where(df['sex']== 2, 1, 0)
```

#### Feature scaling



```python
scaler = MinMaxScaler()
spent_last_year = scaler.fit_transform(df['spent_last_year'].values.reshape(-1, 1))
cost_per = scaler.fit_transform(df['time_per'].values.reshape(-1,1))
amount_appropriate = scaler.fit_transform(df['amount_appropriate'].values.reshape(-1, 1))
df['spent_last_year'] = spent_last_year
df['cost_per'] = cost_per
df['amount_appropriate'] = amount_appropriate
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>whom_with</th>
      <th>time_per</th>
      <th>cost_per</th>
      <th>research</th>
      <th>spent_last_year</th>
      <th>amount_appropriate</th>
      <th>club_participate</th>
      <th>cur_happiness</th>
      <th>family</th>
      <th>wk_econ_act</th>
      <th>avg_wk_workhr</th>
      <th>sex</th>
      <th>age</th>
      <th>edu</th>
      <th>marrital</th>
      <th>province</th>
      <th>disabled</th>
      <th>exp_golf</th>
      <th>want_golf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>1.0</td>
      <td>0.004505</td>
      <td>4</td>
      <td>0.050</td>
      <td>0.025</td>
      <td>1</td>
      <td>6</td>
      <td>3.0</td>
      <td>1</td>
      <td>60.0</td>
      <td>1</td>
      <td>65.0</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>2.0</td>
      <td>0.009009</td>
      <td>3</td>
      <td>0.025</td>
      <td>0.050</td>
      <td>0</td>
      <td>7</td>
      <td>4.0</td>
      <td>1</td>
      <td>10.0</td>
      <td>1</td>
      <td>35.0</td>
      <td>4</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2.0</td>
      <td>0.009009</td>
      <td>6</td>
      <td>0.000</td>
      <td>0.015</td>
      <td>0</td>
      <td>7</td>
      <td>3.0</td>
      <td>1</td>
      <td>11.0</td>
      <td>1</td>
      <td>35.0</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2.0</td>
      <td>0.009009</td>
      <td>5</td>
      <td>0.350</td>
      <td>0.100</td>
      <td>0</td>
      <td>8</td>
      <td>2.0</td>
      <td>1</td>
      <td>10.0</td>
      <td>1</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3.0</td>
      <td>0.013514</td>
      <td>6</td>
      <td>0.050</td>
      <td>0.050</td>
      <td>0</td>
      <td>7</td>
      <td>2.0</td>
      <td>1</td>
      <td>12.0</td>
      <td>1</td>
      <td>75.0</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


</div>



```python
len(df.columns)
```

<pre>
19
</pre>


#### 골프 희망과 경험 상관계수 



```python
import scipy.stats as stats
golf_want = df['want_golf'].values
golf_exp = df['exp_golf'].values 
crosstab = pd.crosstab(golf_want, golf_exp)

print(stats.chi2_contingency(observed = crosstab))
matthews_corrcoef(golf_want, golf_exp)
```

<pre>
Chi2ContingencyResult(statistic=3389.1639375575833, pvalue=0.0, dof=1, expected_freq=array([[8543.47800119,  546.52199881],
       [ 898.52199881,   57.47800119]]))
</pre>
<pre>
0.5815443978963075
</pre>


* 희망과 경험 연관성 높아 경험을 유도하면 희망하게 될 것

#### 분석 목적 

* 전반적으로 골프참여 여부에 영향을 미치는 요소 파악

#### EDA

**골프 유경험자 ** 

```python
fig = plt.figure(figsize=(8,6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.countplot(df, x='exp_golf', hue='exp_golf')
```

<pre>
<Axes: xlabel='exp_golf', ylabel='count'>
</pre>


<img src="data:image/png;base64,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5efna8uWLfL19VVlZaXLOXUB6+/vL19fX0m65DF1jxLUt4a7bDZbo84HgP+ER6EAoPEaegPR0k+TCAsLkyT16NHDZft1112nEydOKDw8XHa73WVf3c9hYWHOxyMudUzd2vWtAQAAAHNZGsM///nPdcUVV+gvf/mLy/YvvvhCXbt2VWxsrPLz852PU0jS/v37dcUVVygqKkohISGKjIxUbm6uc391dbUOHTqk2NhYSap3DQAAAJjL0hj29fVVcnKynn/+eb355pv65ptv9MILL+ijjz7ShAkTNGLECHXq1EkzZszQ0aNHtXfvXq1cuVITJ050Pic8ceJEbdmyRbt27dKxY8c0b948lZeX63/+538kqUFrAAAAwEyWPjMsSVOnTpWfn59WrVqloqIide/eXRkZGRoyZIgkaePGjXrqqad0991368orr9R9992nqVOnOs+/++67VVJSotWrV+vs2bPq3bu3tmzZog4dOkj64c1y9a0BAAAAM9kcvGX5suXl5UmS+vTpY9kMe/96RmdLqy27PoCmF+TvqRG9g60eAwB+Ehraa5Z/HTMAAABgFWIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAst2L44MGDunDhwiX3FRcX66233mrUUAAAAEBLcCuG77//fh0/fvyS+/Lz8/X44483aigAAACgJXg29MDHHntMJ0+elCQ5HA4tXLhQAQEBFx331VdfqWPHjk03IQAAANBMGnxn+Oabb5bD4ZDD4XBuq/u57le7du3Ur18/PfPMM80yLAAAANCUGnxnOCEhQQkJCZKk8ePHa+HCherevXuzDQYAAAA0twbH8P/3u9/9rqnnAAAAAFqcWzFcXl6uF154Qe+9957KyspUW1vrst9ms2nv3r1NMiAAAADQXNyK4aefflq///3vNXjwYPXq1Uvt2vFxxQAAAGh73IrhnJwczZw5U5MnT27qeQAAAIAW49Yt3aqqKvXt27epZwEAAABalFsxPGzYMH3wwQdNPQsAAADQotx6TGLUqFF68skndfr0acXExMjP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"/>

**골프 희망자 **

```python
fig = plt.figure(figsize=(8,6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.countplot(df, x='want_golf', hue='want_golf')
```

<pre>
<Axes: xlabel='want_golf', ylabel='count'>
</pre>


<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAsMAAAISCAYAAAAp7sLIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAy8ElEQVR4nO3de3BV5b3/8c/OPQECScytKJICAgECQRJkJJAigqVUi6BnjgYOCEIJGAUiiCJgqRZNFEq4qCMXa0XUEgEvp1AQDmoJJPywQIEqEhWUJERIAuSevX9/MNl2FzRh57KIz/s105lkXZ71bWZK3yzWXrE5HA6HAAAAAAN5WD0AAAAAYBViGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMbysnqAlujAgQNyOBzy9va2ehQAAABcQVVVlWw2m2JjY3/0OGLYDQ6HQ/ziPgAAgGtXfVuNGHZD7R3hXr16WTwJAAAAruTQoUP1Oo5nhgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCzeJgEAAGCBmpoaVVVVWT1Gi+Tt7S1PT89GWYsYBgAAaEYOh0N5eXkqKiqyepQWrV27doqIiJDNZmvQOsQwAABAM6oN4bCwMAUEBDQ45kzjcDhUWlqqgoICSVJkZGSD1iOGAQAAmklNTY0zhENCQqwep8Xy9/eXJBUUFCgsLKxBj0zwAToAAIBmUvuMcEBAgMWTtHy1P8OGPndNDAMAADQzHo1ouMb6GRLDAAAAkMPhsHoESxDDAAAABispKdHs2bOVk5NjyfVPnTqlrl27KjMz07lt3bp1uvXWWxUTE6OVK1c26fWJYQAAAIMdPXpUmzdvlt1ut3oUSdKFCxf07LPPKiYmRqtXr9aoUaOa9Hq8TQIAAADXjOLiYtntdg0dOlRxcXFNfj3uDAMAAFwDRo0apalTp7psGzp0qBITE122JScna+LEiSovL9fzzz+vYcOGqWfPnurbt68mTJigo0ePOo997LHHNH78eG3cuFHDhw9Xz549ddddd2n37t2SpL1792rcuHGSpHHjxmns2LFXNfOuXbt09913KyYmRsOHD9d7772n22+/XRkZGc5jCgoKNHfuXA0ePFgxMTEaM2aMduzYccX1MjMzNWTIEEnS448/rq5du17VPO4ghgEAAK4BgwcP1r59+1RTUyPp0rO0J0+e1OnTp3Xy5ElJl14jtmfPHiUmJmr27NnauHGjJk+erDVr1mju3Ln6/PPPNWvWLJcPwx0+fFirV69WSkqKVqxYIU9PTz300EMqLi5Wjx49NH/+fEnS/PnztWDBgnrPm5WVpeTkZEVGRiojI0P333+/FixYoNOnTzuPKSws1JgxY5STk6MZM2YoIyND7du317Rp07Rly5bL1kxMTNTy5cslSVOnTtWbb7559T/Iq8RjEgAAANeAxMRErVq1SgcPHlRsbKz27Nmjjh07qrCwUNnZ2brhhhu0f/9+lZaWauDAgdq1a5fmzZunESNGSJLi4+N14cIFLV68WIWFhQoNDZUknT9/XpmZmerQoYOkS+/nTUpKUlZWloYPH67OnTtLkjp37uz8uj4yMjLUpUsXLV++3Pmas5CQEM2cOdN5zNq1a3X27Flt3bpV7du3l3Qp+sePH6/nnntOI0eOdFkzODhY3bt3lyR16NBBffr0ceMneXW4MwwAAHANiImJUVBQkP7+979LunTntX///urdu7eys7MlSbt371aXLl0UFRWl1atXa8SIEcrPz1dWVpY2bNignTt3SpIqKyud6wYHBztDWJIiIiIkSWVlZW7PWllZqQMHDmjYsGEu7/u944475OX1/b3Wffv2KTY21hnCte68806dOXNGJ06ccHuGxkIMAwAAXAM8PDw0aNAg7dmzR9L3MRwfH699+/ZJkj766CP94he/cH79y1/+UoMGDVJycrK2bNkiHx8fSa7vDK791cW1auO1IW+PKCoqUk1NzWW/UtrT01Pt2rVzfl9cXOy8Q/3vrrvuOkmXXutmNWK4BTL1pdiACfjfN2C2xMREffrppzp48KAKCwsVHx+v/v3769SpUzpw4IA+++wzJSYm6uuvv9a0adPUvXt3/e1vf9P+/fu1fv16Zyg3tZCQEHl7e6uwsNBlu91uV1FRkfP7tm3b6syZM5edX7stKCioSeesD54ZboFsNpv2flGi82U1Vo8CoBG18fdU/06BVo8BwEIDBw6Uw+HQSy+9pKioKIWGhiooKEgBAQFKS0tTUFCQYmNj9de//lUVFRWaPHmyyyMQH330kaSr+4u1p6fnVc/p6empvn37aseOHZo+fbpz+4cffqjq6mrn93FxcfrTn/6kb775xuVRiS1btig0NFQ33nijvv3226u+fmMihluo82U1KiqtrvtAAADQYgQGBio2Nlbbt2/Xf/3Xf0mSvLy81K9fP+3evVt33XWXPDw81KNHD3l5eSktLU0PPPCAKisrlZmZqV27dkmSSktL633NNm3aSLr0mrS2bduqW7du9TovJSVFY8eOVUpKisaMGaNvv/1Wf/zjHyV9/yjGhAkTtGXLFo0fP17Tp09Xu3bttGnTJmVlZemZZ56Rh4f1DylYPwEAAACcBg8eLEnq37+/c1vt17XvHL7xxhv1/PPPKz8/X1OnTnW+Hu21116TzWa7ql+t3KVLF40cOVKvv/66UlNT631ev379lJGRodzcXCUnJ2vt2rV68sknJUmtWrWSJIWGhuqNN95Qjx499Pvf/14PP/ywTp8+rZUrV2r06NH1vlZTsjl4QO2qHTp0SJLUq1cvy2bYfvgcd4aBn5h2AV4a2tP65+cANJ3y8nLl5uYqKipKfn5+Vo/TIDt27FBERIR69Ojh3Pb5559r5MiRWrlypW677bYmvX5dP8v69hqPSQAAAMDp35/5/SEeHh76+OOP9cEHHyg1NVVRUVHKz8/XqlWr9POf/1wDBw5shkkbBzEMAAAASZd+61197uhOnz5dc+bMkZ+fn1atWqWCggK1a9dOCQkJmjVrlnx9fZth2sZBDAMAAECSFBYWpr/85S/1Os7Pz09z5szRnDlzmmGypkMMAwAAQJLk4+Nj6WeirMDbJAAAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAFoQh8PRoq5rt9u1bNkyJSQkqE+fPnrwwQd18uTJRp7OfbxnGAAAoAWx2Wza+0WJzpfVNNs12/h7qn+nQLfOXblypdavX6/FixcrIiJCaWlpmjRpkt599135+Pg08qRXjxgGAABoYc6X1aiotNrqMepUWVmpNWvWKDU1VYmJiZKkJUuWKCEhQdu2bdPIkSOtHVA8JgEAAIAmcuzYMV28eFEDBgxwbgsMDFR0dLSys7MtnOx7xDAAAACaRF5eniQpMjLSZXtYWJhzn9WIYQAAADSJsrIySbrs2WBfX19VVFRYMdJliGEAAAA0CT8/P0mXnh3+dxUVFfL397dipMsQwwAAAGgStY9HFBQUuGwvKChQeHi4FSNdhhgGAABAk+jWrZtat26tvXv3OreVlJToyJEjiouLs3Cy7/FqNQAAADQJHx8fJSUlKT09XcHBwWrfvr3S0tIUERGhYcOGWT2eJGIYAACgxWnj79lirpeSkqLq6mrNmzdP5eXliouL0+rVq+Xt7d2IE7qPGAYAAGhBHA6H278NrqHXtdlsV32ep6enHn30UT366KNNMFXD8cwwAABAC+JOkLbk6zY1YhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAzeKll17S2LFjrR7DBTEMAADQgjgc9hZ53ddff11Lly5tnGEakZfVAwAAAKD+bDYPXfzHVtkvnmu2a3q0ClKr3sPdOjc/P18LFizQ3r171bFjx8YdrBEQwwAAAC2M/eI51ZScsXqMevnnP/8pb29vbdmyRStWrNA333xj9UguiGEAAAA0mSFDhmjIkCFWj/GDeGYYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADG4m0SAAAALYxHq6Cf9PWak+UxXF1drRUrVmjTpk0qKipSdHS0Hn30UfXp00eSdPToUT399NM6fPiwgoODNX78eI0bN855vt1u1/Lly/X222/r/PnziouL0/z583XDDTc4j6lrDQAAgJbC4bC7/QswGnpdm61hDxUsXry4kaZpPJY/JrFq1Sq9/fbbWrRokTZt2qSoqChNmjRJBQUFOnfunCZMmKAOHTpo48aNmjZtmtLT07Vx40bn+StXrtT69eu1aNEibdiwQXa7XZMmTVJlZaUk1WsNAACAlqKhQdrSrtvULL8zvH37do0cOVIDBw6UJD322GN6++239emnnyo3N1fe3t763e9+Jy8vL3Xq1ElfffWVXn75ZY0ePVqVlZVas2aNUlNTlZiYKElasmSJEhIStG3bNo0cOVJvvfXWj64BAAAAc1me+CEhIdq5c6dOnTqlmpoavfnmm/Lx8VG3bt2Uk5Oj+Ph4eXl93+y33HKLvvzySxUWFurYsWO6ePGiBgwY4NwfGBio6OhoZWdnS1KdawAAAMBclsfwE088IW9vb912223q1auXlixZomXLlqlDhw7Ky8tTRESEy/FhYWGSpNOnTysvL0+SFBkZedkxtfvqWgMAAADmsvwxiePHj6tNmzZasWKFwsPD9fbbbys1NVV//vOfVV5eLh8fH5fjfX19JUkVFRUqKyuTpCseU1xcLEl1ruEuh8Oh0tJSt893l81mk7+/f7NfF0DzKSsrk8PhsHoMAE2goqJCdrtdNTU1qqmpsXqcFq2mpkZ2u11lZWWy2+2X7Xc4HLLZbHWuY2kMnz59WrNmzdK6devUr18/SVKvXr10/PhxZWRkyM/Pz/lBuFq1ARsQECA/Pz9JUmVlpfPr2mNqg7GuNdxVVVWlo0ePun2+u/z9/RUdHd3s1wXQfHJzc51/2Qfw0+Pp6any8nKrx2jxysvLVVVVpRMnTvzgMf95Q/RKLI3hf/zjH6qqqlKvXr1ctvfu3Vu7d+/Wz372MxUUFLjsq/0+PDxc1dXVzm0dOnRwOaZr166SpIiIiB9dw13e3t7q3Lmz2+e7qz5/wwHQskVFRXFnGPiJqqmp0ddffy273e5yIw9Xr7S0VN7e3urUqZM8PT0v23/8+PF6rWNpDNc+y/uvf/1LMTExzu2fffaZOnbsqN69e2vDhg2qqalx/pfMyspSVFSUQkJC1KZNG7Vu3Vp79+51xnBJSYmOHDmipKQkSVJcXNyPruEum83WoDvLAPBDeBQK+GkLDg5WYWGhPDw8FBAQwI2uq1T7qGphYaGCg4PVpk2bKx5X35+rpTEcExOjm2++WXPmzNGCBQsUERGhTZs2ac+ePXrjjTd0/fXX65VXXtETTzyhSZMm6eDBg1q3bp2eeuopSZdufSclJSk9PV3BwcFq37690tLSFBERoWHDhkmSRo8e/aNrAAAANKfam4H/+S/XuDrt2rW77CUJ7rA5LP63uOLiYi1dulS7du1ScXGxbrrpJs2cOVPx8fGSpIMHD+rpp5/WkSNHFBoaqgceeMB511e69M8NL7zwgjIzM1VeXu78DXTXX3+985i61rhahw4dkqTLHu9oTtsPn1NRabVl1wfQ+NoFeGloz5/urzwF4KqmpkZVVVVWj9EieXt7X/HRiH9X316zPIZbImIYQFMghgGg8dS31yx/zzAAAABgFWIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgrGsihjdt2qQRI0aoV69e+tWvfqX//d//de47deqUpkyZor59+2rgwIFaunSpampqXM5//fXXddtttykmJkb33Xefjhw54rK/PmsAAADAPJbH8ObNm/XEE0/o/vvv1/vvv6+RI0dq5syZOnDggKqqqjRx4kRJ0oYNG7Rw4UK98cYbWrFihfP8d955R88995wefvhhZWZm6vrrr9eECRN09uxZSarXGgAAADCTl5UXdzgc+uMf/6hx48bp/vvvlyRNnTpVOTk52rdvn7755ht9++23euutt9S2bVvddNNN+u677/Tcc8/pt7/9rXx8fPTiiy8qKSlJd955pyTpmWee0dChQ/X2229rypQp2rp1a51rAAAAwEyW3hnOzc3VN998o1//+tcu21evXq0pU6YoJydHPXr0UNu2bZ37brnlFl24cEFHjx7Vd999py+//FIDBgxw7vfy8lK/fv2UnZ0tSXWuAQAAAHNZHsOSVFpaqokTJ2rAgAG655579OGHH0qS8vLyFBER4XJOWFiYJOn06dPKy8uTJEVGRl52TO2+utYAAACAuSx9TOLChQuSpDlz5mj69OlKTU3V1q1blZycrLVr16q8vFyBgYEu5/j6+kqSKioqVFZWJkmXPerg6+uriooKSapzDXc5HA6Vlpa6fb67bDab/P39m/26AJpPWVmZHA6H1WMAQIvm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sWOvauZdu3bp7rvvVkxMjIYPH6733ntPt99+uzIyMpzHFBQUaO7cuRo8eLBiYmI0ZswY7dix44rrZWZmasiQIZKkxx9/XF27dr2qeQCgoYhhALDI4MGDtW/fPtXU1Ei69CztyZMndfr0aZ08eVKSVFVVpT179igxMVGzZ8/Wxo0bNXnyZK1Zs0Zz587V559/rlmzZrl8GO7w4cNavXq1UlJStGLFCnl6euqhhx5ScXGxevToofnz50uS5s+frwULFtR73qysLCUnJysyMlIZGRm6//77tWDBAp0+fdp5TGFhocaMGaOcnBzNmDFDGRkZat++vaZNm6YtW7ZctmZiYqKWL18uSZo6darefPPNq/9BAkAD8JgEAFgkMTFRq1at0sGDBxUbG6s9e/aoY8eOKiwsVHZ2tm644Qbt379fpaWlGjhwoHbt2qV58+ZpxIgRkqT4+HhduHBBixcvVmFhoUJDQyVJ58+fV2Zmpjp06CBJCggIUFJSkrKysjR8+HB17txZktS5c2fn1/WRkZGhLl26aPny5bLZbJKkkJAQzZw503nM2rVrdfbsWW3dulXt27eXdCn6x48fr+eee04jR450WTM4OFjdu3eXJHXo0EF9+vRx4ycJAO7jzjAAWCQmJkZBQUH6+9//LunSndf+/furd+/eys7OliTt3r1bXbp0UVRUlFavXq0RI0YoPz9fWVlZ2rBhg3bu3ClJqqysdK4bHBzsDGFJioiIkCSVlZW5PWtlZaUOHDigYcOGOUNYku644w55eX1/X2Xfvn2KjY11hnCtO++8U2fOnNGJEyfcngEAmgIxDAAW8fDw0KBBg7Rnzx5J38dwfHy89u3bJ0n66KOP9Itf/ML59S9/+UsNGjRIycnJ2rJli3x8fCS5vjPY39/f5Tq18dqQt0cUFRWppqZGISEhLts9PT3Vrl075/fFxcXOO9T/7rrrrpN06bVuAHAtIYYBwEKJiYn69NNPdfDgQRUWFio+Pl79+/fXqVOndODAAX322WdKTEzU119/rWnTpql79+7629/+pv3792v9+vXOUG5qISEh8vb2VmFhoct2u92uoqIi5/dt27bVmTNnLju/dltQUFCTzgkAV4sYBgALDRw4UA6HQy+99JKioqIUGhqqXr16KSAgQGlpaQoKClJsbKwOHz6siooKTZ48WR06dHDe7f3oo48kXd1vk/P09LzqOT09PdW3b9/L3grx4Ycfqrq62vl9XFycDhw4oG+++cbluC1btig0NFQ33njjVV8bAJoSMQwAFgoMDFRsbKy2b9+u+Ph4SZKXl5f69eun/fv3a9CgQfLw8FCPHj3k5eWltLQ0ffLJJ9q5c6ceeugh7dq1S5JUWlpa72u2adNG0qXXpB07dqze56WkpOjYsWNKSUnR7t27tWHDBj355JOSvn8UY8KECWrXrp3Gjx+vzZs36//+7/80Y8YMZWVlacaMGfLw4P92AFxb+FMJACw2ePBgSVL//v2d22q/rn3n8I033qjnn39e+fn5mjp1qvP1aK+99ppsNttV/WrlLl26aOTIkXr99deVmppa7/P69eunjIwM5ebmKjk5WWvXrnXGcKtWrSRJoaGheuONN9SjRw/9/ve/18MPP6zTp09r5cqVGj16dL2vBQDNxea4mn9bAwAYa8eOHYqIiFCPHj2c2z7//HONHDlSK1eu1G233WbhdADgHt4zDACG+/dnfn+Ih4eHPv74Y33wwQdKTU1VVFSU8vPztWrVKv385z/XwIEDm2FSAGh8xDAAGOzUqVP1uqM7ffp0zZkzR35+flq1apUKCgrUrl07JSQkaNasWfL19W2GaQGg8fGYBAAYrLKyUv/617/qPC4sLEzh4eHNMBEANC9iGAAAAMbibRIAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAY/1/g+qaEJie0z0AAAAASUVORK5CYII="/>

**장애인 등록여부 **

```python
fig = plt.figure(figsize=(8,6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.countplot(df, x='disabled', hue='want_golf')
```

<pre>
<Axes: xlabel='disabled', ylabel='count'>
</pre>


<img src="data:image/png;base64,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"/>

* oversampling 기법 고려 (SMOTE)


* 청소년 인구수 확인



```python
fig = plt.figure(figsize=(8,6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.countplot(df, x='age', hue='exp_golf')
```

<pre>
<Axes: xlabel='age', ylabel='count'>
</pre>


<img src="data:image/png;base64,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AAAAVyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyypXGD5w4ICuXLlS6lhmZqZ27NhRoaIAAACAqlCuMPzII4/o+PHjpY4lJSXp+eefr1BRAAAAQFXwKevESZMm6fTp05IkwzA0bdo01axZs8S8kydPqkGDBpVXIQAAAOAiZb4yfNddd8kwDBmG4dhW/Lj4y8vLS23bttWrr77qkmIBAACAylTmK8OxsbGKjY2VJA0ePFjTpk1Ts2bNXFYYAAAA4GplDsM/t27dusquAwAAAKhy5QrDOTk5euONN/TJJ58oOztbRUVFTuM2m027d++ulAIBAAAAVylXGJ45c6b+9re/qWPHjmrZsqW8vLhdMQAAADxPucLwrl279Oyzz2rEiBGVXQ8AAABQZcp1STc/P1+RkZGVXQsAAABQpcoVhm+//XZ99tlnlV0LAAAAUKXKtUyid+/emjp1qs6fP6+oqCgFBASUmHPvvfdWtDYAAADApcoVhseOHStJ2r59u7Zv315i3GazEYYBAABgeuUKw//85z8ruw4AAACgypUrDN9www2VXQcAAABQ5coVhl9//fXrznnqqafKc2gAAACgylR6GK5Zs6YaNWpEGAYAAIDplSsMHz16tMS2rKwsJSYmatq0aZoyZUqFCwMAAABcrdI+RzkwMFBdu3bV6NGj9dprr1XWYQEAAACXqbQwXOx3v/udjh8/XtmHBQAAACpduZZJlMYwDKWmpmrlypXcbQIAAAAeoVxhODw8XDabrdQxwzBYJgEAAACPUK4wPHr06FLDcM2aNdWtWzc1bdq0onUBAAAALleuMPz0009Xdh0AAABAlSv3muHz589r9erV2r9/vzIzM1WvXj116NBBjz32mIKCgiqzRgAAAMAlynU3idTUVPXt21dvvfWW/Pz8FBERIR8fH61Zs0b33nuv0tLSKrtOAAAAoNKV68rwnDlz5OPjow8//FA33XSTY/v333+voUOHav78+Zo9e3alFQkAAAC4QrmuDO/Zs0djxoxxCsKSdNNNN2n06NH67LPPKqU4AAAAwJXKFYYLCwtVr169Usfq16+vy5cvV6goAAAAoCqUKwy3aNFCH3zwQalj7733nsLCwipUFAAAAFAVyrVmeNSoURo2bJguXryo3r17q2HDhjpz5ox2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ifrggw9KHff39y/xRrjc3FxJUmBgoPz9/SVdXTtc/OfiORV9pmW329W8efNrjpflmUhVCw0N9YhncmVhxv5K9NjV6K/r0WPXqk79leixq5mxv5Ln9Dg5OblM80wdhrds2aJz586pW7duTtunTp2qDz/8UCEhIUpPT3caK34cHBysgoICx7YmTZo4zWnRokWFarPZbAoMDKzQMaqaGV9qqW7osWvRX9ejx65Ff12PHruep/S4rE8mTB2G4+PjlZOT47StZ8+eGjNmjO655x6999572rhxowoLC+Xt7S1J2rdvn0JDQxUUFKRatWqpZs2aSkhIcIThzMxMJSUladCgQVV+PgAAADAXU4fh4ODgUrcHBQUpODhY/fr108qVK/Xiiy9q+PDh+vrrr7V27VpNnz5d0tV1IoMGDVJ8fLzq16+vG264QXPmzFFISIh69uxZlacCAAAAEzJ1GL6eoKAgrVy5UjNnzlTfvn3VsGFDTZw4UX379nXMGTNmjAoKChQXF6ecnBxFR0dr1apVstvtbqwcAAAAZuBxYfibb75xehwZGalNmzZdc763t7cmTJigCRMmuLo0AAAAeBiP+NANAAAAwBUIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLIIwwAAALAswjAAAAAsizAMAAAAyyIMAwAAwLI8IgxfuHBBL730krp27ar27dvr4YcfVmJiomN87969uu+++xQVFaVevXppx44dTvvn5uZq+vTpiomJUbt27fTcc8/p/PnzVX0aAAAAMBmPCMPjxo3TV199pXnz5mnLli1q2bKlhg0bphMnTuj48eMaOXKkunTpoq1bt6p///6aOHGi9u7d69h/2rRp2rNnjxYvXqy33npLJ06c0JgxY9x4RgAAADADH3cXcD2nTp3Sv//9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T0932p6enq7g4GB3lAQAAAATsEQYDg8PV82aNZWQkODYlpmZqaSkJEVHR7uxMgAAALiTJdYM+/r6atCgQYqPj1f9+vV1ww03aM6cOQoJCVHPnj3dXR4AAADcxBJhWJLGjBmjgoICxcXFKScnR9HR0Vq1atV1b48GAACA6ssS9xkGAAAASmOJNcMAAABAaQjDAAAAsCzCMAAAACyLMAwAAADLIgwDAADAsgjDAAAAsCzCsIdbvny5Bg8e7Hg8ePBgtWjRotSv7du3X/M4Q4YMKTH/58e1kgsXLuill15S165d1b59ez388MNKTEx0jJenV+vXr9cdd9yhyMhIDRgwQElJSa4+DVM7d+6cJkyYoM6dO6tdu3YaMWKEjh8/7hiPi4sr0ePY2NhfPeY//vEP9e7dW5GRkbr33nu1d+9eV5+GR0hJSVG7du20detWxzb6W3FpaWml/p4t7jM9rhzbt29X79691aZNG9199936xz/+4Rh74403Sv07+DV79+7Vfffdp6ioKPXq1Us7duxw9SmYUkJCwjWzwh133CHJYv014LHeeecdIzw83Bg0aJBjW0ZGhpGenu74SktLMwYMGGDcfffdxuXLl695rJiYGGPDhg1O+2ZkZFTBWZjPkCFDjD59+hgHDhwwTpw4YUyfPt2IjIw0jh8/bhjGb+/V1q1bjcjISOO9994zvv32W2PChAlGx44djXPnzlXRGZnPgw8+aPTv3984ePCgkZycbDz99NPG7bffbmRlZRmGYRj333+/MW/ePKce/1q/9u7da7Rq1cp46623jOTkZGP27NlG69atjeTk5Ko6JVPKy8sz7rvvPiMsLMzYsmWLYzv9rbhPP/3UaNOmjZGWlubUx+zsbMMw6HFl2L59uxEREWG88847xqlTp4ylS5ca4eHhxpdffmkYhmE888wzxoQJE5x6nJ6efs3jJScnG23atDHmzZtnJCcnGytXrjQiIiKM//znP1V1SqaRm5tbom+7du0yWrRoYfztb38zDMNa/SUMe6DU1FRj5MiRRtu2bY1evXo5heFfWrdundG6dWtHkCvN2bNnjbCwMOPw4cOuKNejnDx50ggLCzMSExMd24qKiowePXoYCxYsKFevevbsabz22muOx/n5+cYf//hHY9myZZVau6e4cOGCMW7cOOObb75xbDty5IgRFhZmHDx40CgqKjLa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6SK73a65c+c6jvPWW2/pwoUL+utf/6obbrhBktS1a1f17t1bCxcu1KJFi6r+5ACgEnFlGACqIV9fX61atUq9e/dWWlqa9u3bp40bN+qTTz6RJKWkpOinn35Sr169nPa7++67nR7v3btXLVu2VHBwsAoKClRQUCAvLy917dpV//nPf6rsfADAVbgyDADV1Oeff65Zs2bpxIkTqlGjhsLDwxUYGChJstvtkqSgoCCnfRo0aOD0+MKFCzp16pRatWpV6vfIzs5WQECAC6oHgKpBGAaAaui7777T6NGj1aNHDy1fvlw33XSTbDab1q9fr88//9yxLvjcuXNO+/3yca1atdSxY0dNnDix1O/j6+vrmhMAgCrCMgkAqIb+93//V7m5uRoxYoSaNGkim80m6erVYklq1KiRmjRpoo8//thpv127djk97tixo1JSUhQaGqo2bdo4vt577z397W9/k7e3d9WcEAC4CGEYAKqhVq1aycfHR3PmzNG///1vffLJJ3r66af16aefSrq6vGHMmDHavXu3pk6dqj179mjlypVauHChJMnL6+o/D4899piKior02GOP6cMPP9TevXs1ZcoUrVu3TqGhoe46PQCoNDbDMAx3FwEAqHwfffSRXn/9dX333XeqU6eO2rZtq0ceeUSDBw/WlClTNHDgQG3atEmrVq3STz/9pFtuuUUDBw7Uiy++qMWLF6tnz56Sri65mDt3rvbu3avc3Fw1bdpUgwcP1v333+/mMwSAiiMMA4BF/f3vf1dERIR+//vfO7Z9+umnGjlypN577z2Fh4e7sToAqBqEYQCwqBEjRuj48eMaO3asGjdurFOnTmnRokVq0qSJ1q1b5+7yAKBKEIYBwKIyMjI0d+5cffbZZzp//rwaNGigu+66S2PGjFGNGjXcXR4AVAnCMAAAACyLu0kAAADAsgjDAAAAsCzCMAAAACyLMAwAAADLIgwDAADAsgjDAAAAsCzCMAAAACyLMAwAAADL+v8AW3r0h4HUP24AAAAASUVORK5CYII="/>

#### 전반적으로 골프참여/희망에 영향을 미치는 요소 파악



```python
X = df.drop(['exp_golf', 'want_golf'], axis=1)
y_exp, y_want = df['exp_golf'], df['want_golf']
```

#### Random Forest

```python
X_train, X_test, y_train, y_test = train_test_split(X, y_exp, 
                                                    test_size=0.3,
                                                    random_state = 13,
                                                    stratify= y_exp)
```


```python
params = {
    'max_depth' : [6, 8, 10, 12, 20],
    'n_estimators' : [50, 100, 150, 200],
    'min_samples_leaf' : [8, 12],
    'min_samples_split' : [8, 12]
    }

rf_clf = RandomForestClassifier(random_state=13, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=4, n_jobs=-1)
grid_cv.fit(X_train, y_train)
```

<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, estimator=RandomForestClassifier(n_jobs=-1, random_state=13),

             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [6, 8, 10, 12, 20],
                         &#x27;min_samples_leaf&#x27;: [8, 12],
                         &#x27;min_samples_split&#x27;: [8, 12],
                         &#x27;n_estimators&#x27;: [50, 100, 150, 200]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4, estimator=RandomForestClassifier(n_jobs=-1, random_state=13),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [6, 8, 10, 12, 20],
                         &#x27;min_samples_leaf&#x27;: [8, 12],
                         &#x27;min_samples_split&#x27;: [8, 12],
                         &#x27;n_estimators&#x27;: [50, 100, 150, 200]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1, random_state=13)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1, random_state=13)</pre></div></div></div></div></div></div></div></div></div></div>



```python
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
target_col = ['rank_test_score', 'mean_test_score','param_n_estimators','param_max_depth' ]
cv_results_df[target_col].sort_values('rank_test_score')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank_test_score</th>
      <th>mean_test_score</th>
      <th>param_n_estimators</th>
      <th>param_max_depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0.944255</td>
      <td>200</td>
      <td>8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0.944255</td>
      <td>200</td>
      <td>8</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>0.943828</td>
      <td>50</td>
      <td>12</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3</td>
      <td>0.943828</td>
      <td>150</td>
      <td>8</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3</td>
      <td>0.943828</td>
      <td>150</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>73</td>
      <td>0.942406</td>
      <td>200</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>77</td>
      <td>0.942122</td>
      <td>100</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>77</td>
      <td>0.942122</td>
      <td>100</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>79</td>
      <td>0.941980</td>
      <td>50</td>
      <td>6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>79</td>
      <td>0.941980</td>
      <td>50</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 4 columns</p>

</div>



```python
rf_clf_best = grid_cv.best_estimator_
rf_clf_best.fit(X_train, y_train)
```

<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(max_depth=8, min_samples_leaf=8, min_samples_split=8,

                       n_estimators=200, n_jobs=-1, random_state=13)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=8, min_samples_leaf=8, min_samples_split=8,
                       n_estimators=200, n_jobs=-1, random_state=13)</pre></div></div></div></div></div>



```python
accuracy_score(y_test, rf_clf_best.predict(X_test))
```

<pre>
0.9442601194426012
</pre>


* CV_accuracy, test_accuracy 둘다 높다

* 하지만 골프 비경험자 대비 경험자가 현저히 적어 recall 확인필요



```python
def get_clf_eval(y_test, pred): 
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    re = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)

    return acc, pre, re, f1, auc

def print_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    acc, pre, re, f1, auc = get_clf_eval(y_test, pred)

    print('==> Confusion matrix')
    print(confusion)
    print('===================')

    print('Accuracy : {0:.4f}, precision : {1:.4f}'.format(acc, pre))
    print('Recall : {0:.4f}, F1 : {1:.4f}, AUC: {2:.4f}'.format(re, f1, auc))
```


```python
rf_clf_pred = rf_clf_best.predict(X_test)
print_clf_eval(y_test, rf_clf_pred)
```

<pre>
==> Confusion matrix
[[2825    8]
 [ 160   21]]
===================
Accuracy : 0.9443, precision : 0.7241
Recall : 0.1160, F1 : 0.2000, AUC: 0.5566
</pre>


Accuracy는 높지만 Recall이 현져히 낮음, 골프 경험자 분류에 실패, LightGBM은 어떨까?



```python
params = {
    'n_estimators' : [100, 200, 300],
    'learning_rate' : [0.01, 0.05],
    'num_leaves' : [20, 50, 80, 100, 150],
    'boosting_type' : ['dart'],
    }

lgbm_clf = LGBMClassifier(random_state=13, n_jobs=-1, boost_from_average=False)
lgbm_cv = GridSearchCV(lgbm_clf, param_grid=params, cv=4, n_jobs=-1)
lgbm_cv.fit(X_train, y_train)
```

<pre>
[LightGBM] [Info] Number of positive: 423, number of negative: 6609
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000438 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 245
[LightGBM] [Info] Number of data points in the train set: 7032, number of used features: 16
</pre>
<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4,

             estimator=LGBMClassifier(boost_from_average=False, n_jobs=-1,
                                      random_state=13),
             n_jobs=-1,
             param_grid={&#x27;boosting_type&#x27;: [&#x27;dart&#x27;],
                         &#x27;learning_rate&#x27;: [0.01, 0.05],
                         &#x27;n_estimators&#x27;: [100, 200, 300],
                         &#x27;num_leaves&#x27;: [20, 50, 80, 100, 150]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4,
             estimator=LGBMClassifier(boost_from_average=False, n_jobs=-1,
                                      random_state=13),
             n_jobs=-1,
             param_grid={&#x27;boosting_type&#x27;: [&#x27;dart&#x27;],
                         &#x27;learning_rate&#x27;: [0.01, 0.05],
                         &#x27;n_estimators&#x27;: [100, 200, 300],
                         &#x27;num_leaves&#x27;: [20, 50, 80, 100, 150]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(boost_from_average=False, n_jobs=-1, random_state=13)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(boost_from_average=False, n_jobs=-1, random_state=13)</pre></div></div></div></div></div></div></div></div></div></div>



```python
accuracy_score(y_test, lgbm_cv.best_estimator_.predict(X_test))
```

<pre>
0.9432647644326476
</pre>



```python
lgbm_clf_best = lgbm_cv.best_estimator_
lgbm_clf_best.fit(X_train, y_train)
```

<pre>
[LightGBM] [Info] Number of positive: 423, number of negative: 6609
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000238 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 245
[LightGBM] [Info] Number of data points in the train set: 7032, number of used features: 16
</pre>
<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LGBMClassifier(boost_from_average=False, boosting_type=&#x27;dart&#x27;,

               learning_rate=0.05, n_estimators=300, n_jobs=-1, num_leaves=20,
               random_state=13)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(boost_from_average=False, boosting_type=&#x27;dart&#x27;,
               learning_rate=0.05, n_estimators=300, n_jobs=-1, num_leaves=20,
               random_state=13)</pre></div></div></div></div></div>



```python
lgbm_clf_pred = lgbm_clf_best.predict(X_test)
print_clf_eval(y_test, lgbm_clf_pred)
```

<pre>
==> Confusion matrix
[[2814   19]
 [ 152   29]]
===================
Accuracy : 0.9433, precision : 0.6042
Recall : 0.1602, F1 : 0.2533, AUC: 0.5768
</pre>


두경우 다 recall이 낮으니 SMOTE로 oversampling 시도



```python
from imblearn.over_sampling import SMOTE 

smote = SMOTE(random_state=13)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
```


```python
y_train_over
```

<pre>
0        0
1        0
2        0
3        1
4        0
        ..
13213    1
13214    1
13215    1
13216    1
13217    1
Name: exp_golf, Length: 13218, dtype: int64
</pre>



```python
fig = plt.figure(figsize=(8,6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.countplot(x = y_train_over)
```

<pre>
<Axes: xlabel='exp_golf', ylabel='count'>
</pre>


<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAsMAAAIRCAYAAACverBmAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAvuUlEQVR4nO3de1SVdaL/8c9W7iFeUIEskyBFRhRNVDrh4TDKNMXMifHUTImVeSG1w1Ezs8STlzQn8RJOpOZ9kjFnvDSNMyPpadaMLUVwbLIBjmHkaAmkoqjcZf/+aLHPb48UtEUe5Pt+reVa8jzf57u/slbf9ebp2Rub3W63CwAAADBQB6sXAAAAAFiFGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABjLzeoF3IqOHTsmu90ud3d3q5cCAACARtTW1spms2nw4MHfOo4YdoHdbhe/qwQAAKDtam6rEcMuaLgjHBERYfFKAAAA0Jjjx483axzPDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQzfgux2u9VLAHCTmPrft6n/bsAEbf2/bzerF4DvzmazKftkuS5XXrN6KQBaUCfvjhoe4mf1MizBvga0T7fCvkYM36IuV17TxYo6q5cBAC2GfQ2AFXhMAgAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGahMxvGfPHj344IOKiIjQQw89pD/84Q+Oc2fOnFFycrKGDBmi+++/X6tWrdK1a9ecrt+2bZu+//3va+DAgXr88ceVl5fndL45cwAAAMA8lsfwu+++q7lz52rs2LHau3evEhISNHPmTB07dky1tbWaMGGCJGn79u2aP3++fvWrX+mNN95wXL9792699tpr+q//+i/t2rVLd9xxh8aPH68LFy5IUrPmAAAAgJncrHxxu92u119/XU888YTGjh0rSZoyZYpyc3N15MgRffHFF/ryyy+1Y8cOde7cWX379tX58+f12muv6ZlnnpGHh4fWrFmjpKQk/fjHP5YkLVmyRKNGjdKvf/1rJScna9++fU3OAQAAADNZeme4qKhIX3zxhX70ox85Hd+wYYOSk5OVm5ur733ve+rcubPj3IgRI3TlyhXl5+fr/Pnz+vzzzxUdHe047+bmpqFDhyonJ0eSmpwDAAAA5rL0znBRUZEkqaKiQhMmTFBeXp7uuOMOTZkyRXFxcSouLlZgYKDTNT179pQknT17Vm5uXy8/KCjoujEFBQWS1OQcgwYNcmntdrtdFRUVLl17I2w2m7y9vVv9dQG0nsrKStntdquX0WrY14D2z4p9zW63y2azNTnO0hi+cuWKJOmFF17Qs88+q1mzZmnfvn2aOnWqNm3apKqqKvn5+Tld4+npKUmqrq5WZWWlJF33qIOnp6eqq6slqck5XFVbW2vJnWVvb2+Fh4e3+usCaD1FRUWO/c0E7GtA+2fVvtacx2EtjWF3d3dJ0oQJE5SYmChJ6t+/v/Ly8rRp0yZ5eXmppqbG6ZqGgPXx8ZGXl5ckNTqm4S5DU3PcyNpDQ0Ndvt5VzfkJB8CtLTg42Lg7wwDaNyv2tcLCwmaNszSGAwICJEl9+/Z1Oh4aGqo//elPGjZsmE6cOOF0rrS01HFtw+MRpaWlCgkJcRrTMHdgYOC3zuEqm812QzENAN+ERwYAtDdW7GvN/UHb0jfQfe9739Ntt92mv/3tb07HT5w4od69eysqKkp5eXmOxykk6fDhw7rtttsUFhYmf39/BQcHKzs723G+rq5Oubm5ioqKkqQm5wAAAIC5LI1hLy8vTZw4UW+88YZ+97vf6R//+IfefPNNffjhhxo/frxGjRqlHj16aPr06SooKND+/fu1YsUKPf30045nQJ5++mlt2rRJu3fvVmFhoV566SVVVVXpP/7jPySpWXMAAADATJY+JiFJU6dOlbe3t1auXKmSkhKFhIRo9erVGj58uCRp/fr1WrBggR599FF17txZjz/+uKZOneq4/tFHH9Xly5e1atUqXbx4UQMGDNCmTZvUrVs3SV+/Wa6pOQAAAGAmm92kd2m0kOPHj0uSIiIiLFvD/k/KdLGizrLXB9Dyuvi4adSArlYvwzLsa0D7Y+W+1txes/zXMQMAAABWIYYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGIoYBAABgLGIYAAAAxiKGAQAAYCxiGAAAAMYihgEAAGAsYhgAAADGsjyGS0pK1K9fv+v+7Nq1S5KUn5+vpKQkRUZGKi4uTlu3bnW6vr6+Xunp6YqJiVFkZKQmTZqk06dPO41pag4AAACYyc3qBRQUFMjT01P79++XzWZzHO/UqZPKyso0fvx4xcXFacGCBfroo4+0YMEC3XbbbRozZowkKSMjQ5mZmVq6dKkCAwO1bNkyTZw4Ue+99548PDyaNQcAAADMZHkMnzhxQn369FHPnj2vO7dlyxa5u7tr4cKFcnNzU0hIiE6dOqV169ZpzJgxqqmp0caNGzVr1izFxsZKklauXKmYmBhlZWUpISFBO3bs+NY5AAAAYC7LY/h///d/FRIS0ui53NxcDRs2TG5u/7fMESNGaO3atTp37py+/PJLXb16VdHR0Y7zfn5+Cg8PV05OjhISEpqco3v37i6t2263q6KiwqVrb4TNZpO3t3ervy6A1lNZWSm73W71MloN+xrQ/lmxr9ntdqenDr6J5TF84sQJde3aVWPHjlVRUZHuuusuTZkyRSNHjlRxcbH69u3rNL7hDvLZs2dVXFwsSQoKCrpuTMO5puZwNYZra2uVn5/v0rU3wtvbW+Hh4a3+ugBaT1FRkSorK61eRqthXwPaP6v2NQ8PjybHWBrDdXV1+uyzzxQaGqo5c+bI19dXe/fu1eTJk7Vp0yZVVVVd94/w9PSUJFVXVzu+qY2NuXTpkiQ1OYer3N3dFRoa6vL1rmrOTzgAbm3BwcHG3RkG0L5Zsa8VFhY2a5ylMezm5qbs7Gx17NhRXl5ekqQBAwbo008/1YYNG+Tl5aWamhqnaxoC1sfHx3FNTU2N4+8NYxr+l1tTc7jKZrPd0PUA8E14ZABAe2PFvtbcH7Qt/2i12267zSlkJemee+5RSUmJAgMDVVpa6nSu4euAgADH4xGNjQkICJCkJucAAACAuSyN4U8//VRDhgxRdna20/FPPvlEoaGhioqK0tGjR3Xt2jXHucOHDys4OFj+/v4KCwuTr6+v0/Xl5eXKy8tTVFSUJDU5BwAAAMxlaQyHhITo7rvv1sKFC5Wbm6uTJ0/q1Vdf1UcffaQpU6ZozJgxunLliubOnavCwkLt2rVLmzdvVnJysqSvnxVOSkpSWlqaDhw4oIKCAs2YMUOBgYGKj4+XpCbnAAAAgLksfWa4Q4cOWrNmjZYvX67p06ervLxc4eHh2rRpk+MTINavX6/FixcrMTFRPXr00OzZs5WYmOiYIyUlRXV1dUpNTVVVVZWioqK0YcMGubu7S5L8/f2bnAMAAABmstlNestyCzl+/LgkKSIiwrI17P+kTBcr6ix7fQAtr4uPm0YN6Gr1MizDvga0P1bua83tNcvfQAcAAABYhRgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABiLGAYAAICx2lQMFxUVafDgwdq1a5fjWH5+vpKSkhQZGam4uDht3brV6Zr6+nqlp6crJiZGkZGRmjRpkk6fPu00pqk5AAAAYKY2E8O1tbWaNWuWKioqHMfKyso0fvx49e7dWzt37tS0adOUlpamnTt3OsZkZGQoMzNTixYt0vbt21VfX6+JEyeqpqam2XMAAADATG5WL6DB6tWr5evr63Rsx44dcnd318KFC+Xm5qaQkBCdOnVK69at05gxY1RTU6ONGzdq1qxZio2NlSStXLlSMTExysrKUkJCQpNzAAAAwFxt4s5wTk6O3nnnHS1dutTpeG5uroYNGyY3t/9r9hEjRujzzz/XuXPnVFBQoKtXryo6Otpx3s/PT+Hh4crJyWnWHAAAADCX5XeGy8vLNXv2bKWmpiooKMjpXHFxsfr27et0rGfPnpKks2fPqri4WJKuu65nz56Oc03N0b17d5fWbbfbnR7paC02m03e3t6t/roAWk9lZaXsdrvVy2g17GtA+2fFvma322Wz2ZocZ3kMz58/X4MHD9aPfvSj685VVVXJw8PD6Zinp6ckqbq6WpWVlZLU6JhLly41aw5X1dbWKj8/3+XrXeXt7a3w8PBWf10AraeoqMixv5mAfQ1o/6za1/65ARtjaQzv2bNHubm5eu+99xo97+Xl5XgjXIOGgPXx8ZGXl5ckqaamxvH3hjENdxmamsNV7u7uCg0Ndfl6VzXnJxwAt7bg4GDj7gwDaN+s2NcKCwubNc7SGN65c6fOnz/vePNbg5dfflm///3vFRgYqNLSUqdzDV8HBASorq7Ocax3795OY/r16ydJTc7hKpvNdkMxDQDfhEcGALQ3Vuxrzf1B29IYTktLU1VVldOx+Ph4paSk6Mc//rHeffddbd++XdeuXVPHjh0lSYcPH1ZwcLD8/f3VqVMn+fr6Kjs72xHD5eXlysvLU1JSkiQpKirqW+cAAACAuSz9NImAgADdddddTn8kyd/fXwEBARozZoyuXLmiuXPnqrCwULt27dLmzZuVnJws6evnQJKSkpSWlqYDBw6ooKBAM2bMUGBgoOLj4yWpyTkAAABgLsvfQPdt/P39tX79ei1evFiJiYnq0aOHZs+ercTERMeYlJQU1dXVKTU1VVVVVYqKitKGDRvk7u7e7DkAAABgJpvdpHdptJDjx49LkiIiIixbw/5PynSxos6y1wfQ8rr4uGnUgK5WL8My7GtA+2PlvtbcXmsTv3QDAAAAsAIxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjuRTDOTk5unr1aqPnysvLtXfv3htaFAAAANAaXIrhJ554QidPnmz0XF5enl588cUbWhQAAADQGtyaO/CFF17Q2bNnJUl2u13z58+Xr6/vdeM+//xzde/eveVWCAAAANwkzb4z/IMf/EB2u112u91xrOHrhj8dOnRQZGSkXn311ZuyWAAAAKAlNfvOcFxcnOLi4iRJ48aN0/z58xUSEnLTFgYAAADcbM2O4f/fL3/5y5ZeBwAAANDqXIrhqqoqvfnmm/rggw9UWVmp+vp6p/M2m0379+9vkQUCAAAAN4tLMbx48WL95je/0bBhw9S/f3916MDHFQMAAODW41IMZ2VlacaMGZo8eXJLrwcAAABoNS7d0q2trdXAgQNbei0AAABAq3Iphu+//379+c9/bum1AAAAAK3KpcckHnzwQb388su6cOGCBg0aJG9v7+vGPPzwwze6NgAAAOCmcimGp0+fLknas2eP9uzZc915m81GDAMAAKDNcymGDxw40NLrAAAAAFqdSzHcq1evll4HAAAA0OpciuFf/OIXTY559tlnXZkaAAAAaDUtHsO+vr7q2bMnMQwAAIA2z6UYLigouO5YRUWFcnNzNX/+fM2bN++GFwYAAADcbC32e5R9fHw0cuRITZs2Ta+99lpLTQsAAADcNC0Www1uv/12nTx5sqWnBQAAAFqcS49JNMZut6u4uFjr16/n0yYAAABwS3AphsPCwmSz2Ro9Z7fbeUwCAAAAtwSXYnjatGmNxrCvr69iY2PVp0+fG10XAAAAcNO5FMP/+Z//2dLrAAAAAFqdy88MX7hwQRs3btSRI0dUXl6url27aujQoXrqqafk7+/fkmsEAAAAbgqXPk2iuLhYiYmJ2rJlizw9PRUeHi43Nzdt2rRJDz/8sEpKSlp6nQAAAECLc+nO8LJly+Tm5qbf//73uvPOOx3HT58+raefflorV67U0qVLW2yRAAAAwM3g0p3hgwcPKiUlxSmEJenOO+/UtGnT9Oc//7lFFgcAAADcTC7F8LVr19S1a9dGz3Xr1k1Xrly5oUUBAAAArcGlGO7Xr5/ee++9Rs+9++676tu37w0tCgAAAGgNLj0zPHXqVE2YMEGXLl3Sgw8+qB49euirr77S3r17dfDgQaWnp7f0OgEAAIAW51IM/8u//IuWLl2qtLQ0p+eDe/TooVdffVWjR49usQUCAAAAN4vLnzNcWlqq8PBwvfDCC7p06ZIKCgq0evVqnhcGAADALcOlGN64caNWrVqlpKQkhYSESJKCgoL02WefaenSpfL09NQjjzzSogsFAAAAWppLMbx9+3ZNnz5dkydPdhwLCgpSamqqunfvrs2bNxPDAAAAaPNc+jSJkpISRURENHpu0KBBOnPmzA0tCgAAAGgNLsVwr169dOjQoUbP5eTkKDAw8IYWBQAAALQGlx6TePTRR7Vs2TLV1tZq1KhR8vf314ULF/TBBx9o06ZNeu6551p6nQAAAECLcymGn3rqKZWUlOiXv/ylNm/e7DjesWNHPfnkkxo/fnxLrQ8AAAC4aVz+aLUXXnhBU6dO1UcffaSLFy/Kz89PAwcO/MZf0wwAAAC0NS7HsCR16tRJMTExLbUWAAAAoFW59AY6AAAAoD0ghgEAAGAsYhgAAADGIoYBAABgLMtj+Pz583r++ec1YsQIDR48WJMnT9bJkycd5/Pz85WUlKTIyEjFxcVp69atTtfX19crPT1dMTExioyM1KRJk3T69GmnMU3NAQAAADNZHsPTpk3TqVOntG7dOv3mN7+Rl5eXnnrqKVVWVqqsrEzjx49X7969tXPnTk2bNk1paWnauXOn4/qMjAxlZmZq0aJF2r59u+rr6zVx4kTV1NRIUrPmAAAAgJlu6KPVbtSlS5fUq1cvJScnq2/fvpKkqVOn6t///d/16aef6tChQ3J3d9fChQvl5uamkJAQRziPGTNGNTU12rhxo2bNmqXY2FhJ0sqVKxUTE6OsrCwlJCRox44d3zoHAAAAzGVpDHfu3FnLly93fH3hwgVt3rxZgYGBCg0N1erVqzVs2DC5uf3fMkeMGKG1a9fq3Llz+vLLL3X16lVFR0c7zvv5+Sk8PFw5OTlKSEhQbm7ut87RvXt3l9Zut9tVUVHh0rU3wmazydvbu9VfF0DrqayslN1ut3oZrYZ9DWj/rNjX7Ha7bDZbk+MsjeH/37x587Rjxw55eHjozTfflI+Pj4qLix13jBv07NlTknT27FkVFxdLkoKCgq4b03CuqTlcjeHa2lrl5+e7dO2N8Pb2Vnh4eKu/LoDWU1RUpMrKSquX0WrY14D2z6p9zcPDo8kxbSaGn3zySf30pz/Vtm3bNG3aNGVmZqqqquq6f4Snp6ckqbq62vFNbWzMpUuXJKnJOVzl7u6u0NBQl693VXN+wgFwawsODjbuzjCA9s2Kfa2wsLBZ49pMDDeE5eLFi/W3v/1Nb7/9try8vBxvhGvQELA+Pj7y8vKSJNXU1Dj+3jCm4X+5NTWHq2w22w1dDwDfhEcGALQ3Vuxrzf1B29JPk7hw4YL27t2ruro6x7EOHTooNDRUpaWlCgwMVGlpqdM1DV8HBAQ4Ho9obExAQIAkNTkHAAAAzGVpDJ87d04zZ87UoUOHHMdqa2uVl5enkJAQRUVF6ejRo7p27Zrj/OHDhxUcHCx/f3+FhYXJ19dX2dnZjvPl5eXKy8tTVFSUJDU5BwAAAMxlaQz37dtXI0eO1CuvvKKcnBydOHFCc+bMUXl5uZ566imNGTNGV65c0dy5c1VYWKhdu3Zp8+bNSk5OlvT1s8JJSUlKS0vTgQMHVFBQoBkzZigwMFDx8fGS1OQcAAAAMJflzwyvWLFCy5cv14wZM3T58mUNHTpU27Zt0+233y5JWr9+vRYvXqzExET16NFDs2fPVmJiouP6lJQU1dXVKTU1VVVVVYqKitKGDRvk7u4uSfL3929yDgAAAJjJZjfpLcst5Pjx45KkiIgIy9aw/5MyXayoa3oggFtGFx83jRrQ1eplWIZ9DWh/rNzXmttrlv86ZgAAAMAqxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxleQxfvHhR//3f/62RI0dqyJAheuyxx5Sbm+s4f+jQIf3kJz/RoEGD9MADD2jv3r1O11dXV2vBggWKjo7W4MGD9dxzz+nChQtOY5qaAwAAAGayPIZnzpypY8eOacWKFdq5c6f69++vCRMm6LPPPtPJkyeVnJysmJgY7dq1S4888ohmz56tQ4cOOa6fP3++Dh48qNWrV2vLli367LPPlJKS4jjfnDkAAABgJjcrX/zUqVP68MMPlZmZqXvvvVeSNG/ePP3lL3/Re++9p/Pnz6tfv36aMWOGJCkkJER5eXlav369oqOjVVJSoj179mjNmjUaOnSoJGnFihV64IEHdOzYMQ0ePFhbtmz51jkAAABgLkvvDHft2lXr1q1TRESE45jNZpPNZlN5eblyc3OvC9YRI0bo6NGjstvtOnr0qONYg+DgYAUEBCgnJ0eSmpwDAAAA5rL0zrCfn5/+9V//1enYvn37dOrUKb300kvavXu3AgMDnc737NlTlZWVKisrU0lJibp27SpPT8/rxhQXF0uSiouLv3WObt26ubR2u92uiooKl669ETabTd7e3q3+ugBaT2VlpVE/rLOvAe2fFfua3W6XzWZrcpylMfzP/vrXv+rFF19UfHy8YmNjVVVVJQ8PD6cxDV/X1NSosrLyuvOS5Onpqerqaklqcg5X1dbWKj8/3+XrXeXt7a3w8PBWf10AraeoqEiVlZVWL6PVsK8B7Z9V+1pjnfjP2kwM79+/X7NmzdKQIUOUlpYm6euo/edgbfja29tbXl5ejQZtdXW14y5DU3O4yt3dXaGhoS5f76rm/IQD4NYWHBxs3J1hAO2bFftaYWFhs8a1iRh+++23tXjxYj3wwAP6+c9/7qj4oKAglZaWOo0tLS2Vj4+POnXqpMDAQF28eFE1NTVO5V9aWqqAgIBmzeEqm80mHx8fl68HgG/CIwMA2hsr9rXm/qBt+UerZWZmatGiRRo7dqxWrFjhFLVDhw7VkSNHnMYfPnxYQ4YMUYcOHXTvvfeqvr7e8UY66evb8CUlJYqKimrWHAAAADCXpTVYVFSkJUuWaPTo0UpOTta5c+f01Vdf6auvvtLly5c1btw4ffzxx0pLS9PJkye1ceNG/fGPf9TEiRMlSQEBAXrooYeUmpqq7Oxsffzxx5o5c6aGDRumyMhISWpyDgAAAJjL0sck9u3bp9raWr3//vt6//33nc4lJiZq6dKlysjI0LJly7RlyxbdcccdWrZsmdNHpS1atEhLlizRs88+K0kaOXKkUlNTHefvueeeJucAAACAmWx2k96l0UKOHz8uSU6fj9za9n9SposVdZa9PoCW18XHTaMGdLV6GZZhXwPaHyv3teb2Gg/NAgAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFhtKobXrl2rcePGOR3Lz89XUlKSIiMjFRcXp61btzqdr6+vV3p6umJiYhQZGalJkybp9OnT32kOAAAAmKnNxPC2bdu0atUqp2NlZWUaP368evfurZ07d2ratGlKS0vTzp07HWMyMjKUmZmpRYsWafv27aqvr9fEiRNVU1PT7DkAAABgJjerF1BSUqKXX35Z2dnZ6tOnj9O5HTt2yN3dXQsXLpSbm5tCQkJ06tQprVu3TmPGjFFNTY02btyoWbNmKTY2VpK0cuVKxcTEKCsrSwkJCU3OAQAAAHNZHsN///vf5e7urt/+9rd644039MUXXzjO5ebmatiwYXJz+79ljhgxQmvXrtW5c+f05Zdf6urVq4qOjnac9/PzU3h4uHJycpSQkNDkHN27d3dp3Xa7XRUVFS5deyNsNpu8vb1b/XUBtJ7KykrZ7Xarl9Fq2NeA9s+Kfc1ut8tmszU5zvIYjouLU1xcXKPniouL1bdvX6djPXv2lCSdPXtWxcXFkqSgoKDrxjSca2oOV2O4trZW+fn5Ll17I7y9vRUeHt7qrwug9RQVFamystLqZbQa9jWg/bNqX/Pw8GhyjOUx/G2qqqqu+0d4enpKkqqrqx3f1MbGXLp0qVlzuMrd3V2hoaEuX++q5vyEA+DWFhwcbNydYQDtmxX7WmFhYbPGtekY9vLycrwRrkFDwPr4+MjLy0uSVFNT4/h7w5iG/+XW1ByustlsN3Q9AHwTHhkA0N5Ysa819wftNvNpEo0JDAxUaWmp07GGrwMCAhyPRzQ2JiAgoFlzAAAAwFxtOoajoqJ09OhRXbt2zXHs8OHDCg4Olr+/v8LCwuTr66vs7GzH+fLycuXl5SkqKqpZcwAAAMBcbTqGx4wZoytXrmju3LkqLCzUrl27tHnzZiUnJ0v6+lnhpKQkpaWl6cCBAyooKNCMGTMUGBio+Pj4Zs0BAAAAc7XpZ4b9/f21fv16LV68WImJierRo4dmz56txMREx5iUlBTV1dUpNTVVVVVVioqK0oYNG+Tu7t7sOQAAAGAmm92ktyy3kOPHj0uSIiIiLFvD/k/KdLGizrLXB9Dyuvi4adSArlYvwzLsa0D7Y+W+1txea9OPSQAAAAA3EzEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwFjEMAAAAYxHDAAAAMBYxDAAAAGMRwwAAADAWMQwAAABjEcMAAAAwljExXF9fr/T0dMXExCgyMlKTJk3S6dOnrV4WAAAALGRMDGdkZCgzM1OLFi3S9u3bVV9fr4kTJ6qmpsbqpQEAAMAiRsRwTU2NNm7cqJSUFMXGxiosLEwrV65UcXGxsrKyrF4eAAAALOJm9QJaQ0FBga5evaro6GjHMT8/P4WHhysnJ0cJCQnfab7a2lrZ7XZ9/PHHLb3UZrHZbOpSVy+/jpa8PICbpEOtdPz4GdntdquX0urY14D2ycp9rba2VjabrclxRsRwcXGxJCkoKMjpeM+ePR3nvouGb2xzvsE3i6ebETf1ASNZubdYiX0NaL+s2NdsNhsx3KCyslKS5OHh4XTc09NTly5d+s7zDR48uEXWBQAAAGsZ8WO4l5eXJF33Zrnq6mp5e3tbsSQAAAC0AUbEcMPjEaWlpU7HS0tLFRAQYMWSAAAA0AYYEcNhYWHy9fVVdna241h5ebny8vIUFRVl4coAAABgJSOeGfbw8FBSUpLS0tLUrVs39erVS8uWLVNgYKDi4+OtXh4AAAAsYkQMS1JKSorq6uqUmpqqqqoqRUVFacOGDXJ3d7d6aQAAALCIzW7iB1oCAAAAMuSZYQAAAKAxxDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDANtUH19vdLT0xUTE6PIyEhNmjRJp0+ftnpZANAi1q5dq3Hjxlm9DEASMQy0SRkZGcrMzNSiRYu0fft21dfXa+LEiaqpqbF6aQBwQ7Zt26ZVq1ZZvQzAgRgG2piamhpt3LhRKSkpio2NVVhYmFauXKni4mJlZWVZvTwAcElJSYmeeeYZpaWlqU+fPlYvB3AghoE2pqCgQFevXlV0dLTjmJ+fn8LDw5WTk2PhygDAdX//+9/l7u6u3/72txo0aJDVywEc3KxeAABnxcXFkqSgoCCn4z179nScA4BbTVxcnOLi4qxeBnAd7gwDbUxlZaUkycPDw+m4p6enqqurrVgSAADtFjEMtDFeXl6SdN2b5aqrq+Xt7W3FkgAAaLeIYaCNaXg8orS01Ol4aWmpAgICrFgSAADtFjEMtDFhYWHy9fVVdna241h5ebny8vIUFRVl4coAAGh/eAMd0MZ4eHgoKSlJaWlp6tatm3r16qVly5YpMDBQ8fHxVi8PAIB2hRgG2qCUlBTV1dUpNTVVVVVVioqK0oYNG+Tu7m710gAAaFdsdrvdbvUiAAAAACvwzDAAAACMRQwDAADAWMQwAAAAjEUMAwAAwFjEMAAAAIxFDAMAAMBYxDAAAACMRQwDAADAWMQwAOCm27Vrl/r166czZ85Ikurq6jRnzhwNHjxYQ4YM0eHDhy1eIQBT8euYAQCt7i9/+Yt2796tqVOn6r777lN4eLjVSwJgKGIYANDqLl68KEn6yU9+ojvvvNPaxQAwGo9JAEAb9Otf/1oPPfSQBgwYoNjYWK1evVrXrl3T2bNnde+992rcuHGOsdXV1XrwwQf10EMPqbq6WtnZ2erXr58OHjyosWPHauDAgYqPj1dmZqZLa9mwYYO+//3va+DAgfrZz36m//mf/1G/fv2UnZ3tGHP8+HFNmDBBw4cP15AhQ/TMM8/o008/bXS+OXPmaM6cOZKkUaNGOf1bAKC1EcMA0MasXbtW8+bNU3R0tNasWaOxY8fqrbfe0rx58xQUFKQ5c+boyJEj2rlzpyRp+fLl+sc//qHly5fL09PTMc+MGTMUHh6uN954Q/fdd58WLFjwnYP4F7/4hdLS0vTDH/5QGRkZGjRokKZPn+405vDhw3rsscckSUuWLNErr7yis2fP6mc/+5lOnjx53ZxTp07VlClTHPO//PLL32lNANCSeEwCANqQy5cvKyMjQz/96U+VmpoqSbr//vvVpUsXpaamavz48XrkkUeUlZWl1157TV26dNHWrVv1/PPPKywszGmu0aNHa+7cuZKkmJgYlZaWKiMjQ4899phsNluTa6moqNBbb72lsWPHatasWY61VFZW6p133nGMW758ue666y6tW7dOHTt2dIwbPXq00tPT9frrrzvN27t3b/Xu3VuS1L9/f91xxx0ufrcA4MZxZxgA2pBjx46pqqpKcXFxqqurc/yJi4uTJH344YeSpFdeeUX19fV69tlnNWzYMD399NPXzZWYmOj0dXx8vL766isVFRU1ay0fffSRqqqq9MADDzgdT0hIcPy9oqJCx48f1w9/+ENHCEuSn5+f/u3f/k1Hjhxp3j8cACzCnWEAaEMa3lg2efLkRs+XlpZKkgICAhQdHa19+/YpNja20Tu9AQEBTl/7+/tLki5dutSstVy4cEGS1K1bt0bnkb6+k22329W9e/frru/evbsuX77crNcCAKsQwwDQhvj5+UmS0tLS1KdPn+vON0TnwYMHtW/fPvXv31+rV6/W6NGjr/tUhrKyMsfjCJJ0/vx5Sc4x+20CAwMd1919992O4w2RLEmdOnWSzWbTuXPnrrv+q6++UpcuXZr1WgBgFR6TAIA2ZNCgQXJ3d1dJSYkiIiIcf9zc3LRixQqdOXNGly9fVmpqqu677z69/fbb8vPz00svvSS73e401/79+52+/uMf/6hevXo5BfK3CQsLU6dOnfT+++87Hc/KynL83cfHRwMGDNAf/vAHXbt2zXH88uXL+tOf/qR77733u34LAKBVcWcYANqQrl27auLEiXr99dd15coVDR8+XCUlJXr99ddls9kUFhamJUuWqKysTFu3bpWvr6/mzZunadOm6e2333b6mLJNmzbJ09NTkZGRysrK0gcffKDly5c3ey2+vr6aOHGi0tPT5e3trWHDhunIkSP61a9+JUnq0OHr+ynPPfecJkyYoMmTJ+vxxx9XbW2t1q1bp5qaGk2bNq1lv0EA0MKIYQBoY6ZPn64ePXooMzNT69evV+fOnRUdHa2ZM2fqr3/9q3bt2qXnn3/ecYd31KhRio+P1/LlyzVy5EjHPC+99JJ2796ttWvX6u6771Z6erp+8IMffKe1JCcny26365133tGGDRs0aNAgzZo1S6+++qp8fHwkSdHR0dq0aZPS09M1c+ZMeXh4aOjQofr5z3+ue+65p+W+MQBwE9js//z/1QAAt7Ts7Gw98cQT2rp1q4YPH+7yPHV1dfrd736n4cOHKygoyHF827ZteuWVV5Sdne14xhkAblXcGQYAw9jtdqfne79Jx44d9dZbb2nLli2aMmWKunbtqhMnTmjVqlV6+OGHCWEA7QIxDACG2b17t1588cUmx23dulVr1qzRihUrNH/+fJWXl+v222/Xk08+qeTk5FZYKQDcfDwmAQCGKSsr05kzZ5ocFxwcLF9f31ZYEQBYhxgGAACAsficYQAAABiLGAYAAICxiGEAAAAYixgGAACAsYhhAAAAGIsYBgAAgLGIYQAAABjr/wETRWCCRU/yPgAAAABJRU5ErkJggg=="/>

#### Smote - randomforest  (Selected Model)

```python
params = {
    'max_depth' : [8, 10, 15, 20],
    'n_estimators' : [50, 100, 150],
    'min_samples_leaf' : [8, 12],
    'min_samples_split' : [8, 12]
    }

rf_clf = RandomForestClassifier(random_state=13, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=4, n_jobs=-1)
grid_cv.fit(X_train_over, y_train_over)
```

<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, estimator=RandomForestClassifier(n_jobs=-1, random_state=13),

             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [8, 10, 15, 20],
                         &#x27;min_samples_leaf&#x27;: [8, 12],
                         &#x27;min_samples_split&#x27;: [8, 12],
                         &#x27;n_estimators&#x27;: [50, 100, 150]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4, estimator=RandomForestClassifier(n_jobs=-1, random_state=13),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [8, 10, 15, 20],
                         &#x27;min_samples_leaf&#x27;: [8, 12],
                         &#x27;min_samples_split&#x27;: [8, 12],
                         &#x27;n_estimators&#x27;: [50, 100, 150]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1, random_state=13)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1, random_state=13)</pre></div></div></div></div></div></div></div></div></div></div>



```python
rf_clf_best = grid_cv.best_estimator_
rf_clf_best.fit(X_train_over, y_train_over)
rf_clf_best_pred = rf_clf_best.predict(X_test)
grid_cv.best_params_
```

<pre>
{'max_depth': 20,
 'min_samples_leaf': 8,
 'min_samples_split': 8,
 'n_estimators': 100}
</pre>



```python
accuracy_score(y_test, rf_clf_best_pred)

print_clf_eval(y_test, rf_clf_best_pred)
```

<pre>
==> Confusion matrix
[[2690  143]
 [ 120   61]]
===================
Accuracy : 0.9127, precision : 0.2990
Recall : 0.3370, F1 : 0.3169, AUC: 0.6433
</pre>



```python
best_cols_values = rf_clf_best.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train.columns)
best_cols = best_cols.sort_values(ascending=False)
best_cols
```

<pre>
amount_appropriate    0.289619
spent_last_year       0.274021
family                0.076470
cost_per              0.065363
time_per              0.061987
research              0.060863
edu                   0.033564
province              0.031339
age                   0.031330
avg_wk_workhr         0.029673
whom_with             0.016928
cur_happiness         0.012499
wk_econ_act           0.008191
marrital              0.006454
club_participate      0.001662
disabled              0.000035
sex                   0.000000
dtype: float64
</pre>



```python
sns.set(style='whitegrid', color_codes = True)
plt.figure(figsize=(8,8))
sns.barplot(x=best_cols, y=best_cols.index, hue=best_cols.index)
```

<pre>
<Axes: xlabel='None', ylabel='None'>
</pre>


<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAzAAAAKrCAYAAAAu3mgvAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAACH/UlEQVR4nOzdeXxN1/7/8feRCAkRYq4hiIogJIbGWAQl5lmlpASlLYq2KuahiKGmoCXiElQNjSkdjMWlRFW1IQgxBa1rLhUJyfn94ed8pYZGmjg5J6/n49HHzdl77bU+ey9/nPfda+9jMBqNRgEAAACABchm7gIAAAAAILUIMAAAAAAsBgEGAAAAgMUgwAAAAACwGAQYAAAAABaDAAMAAADAYhBgAAAAAFgMW3MXAKSXX375RUajUdmzZzd3KQAAAHiK+/fvy2AwyMvLK819cAcGVsNoNJr+Q+ZhNBqVmJjIvGQyzEvmxLxkTsxL5sS8ZE7/NC/p8V2NOzCwGtmzZ1diYqLKli0rBwcHc5eD/+/u3bs6duwY85LJMC+ZE/OSOTEvmRPzkjn907xERUX96zG4AwMAAADAYhBgYHUMBoO5S8BjDAaD7O3tmZdMhnnJnJiXzIl5yZyYl6yLJWSwKnZ2drK3tzd3GXiMvb29KlSoYO4y8DfMS+bEvGROzEvmxLykD2NysgzZLOueBgEGVudG+BY9uHrD3GUAAABkarYF8ilf+zfMXcYLI8DA6jy4ekMP/rhi7jIAAACQASzrfhEAAACALI0AAwAAAMBiEGAAAAAAWAwCDAAAAACLQYABAAAAYDEIMAAAAAAsBgEGAAAAgMUgwKTRDz/8oFOnTpm7DLNyc3NTeHh4qtufPHlSO3fuzLiCAAAAYPUIMGlw8eJF9evXT9euXTN3KWa1Z88eNW/ePNXt+/btq6ioqAysCAAAANbO1twFWCKj0WjuEjKFggULmrsEAAAAZDFmvwMTExOjvn37qkaNGqpUqZIaNWqkxYsXS5KCg4PVo0cPzZ07V7Vr15aXl5dGjx6t33//XX379lWVKlXUpEmTFMuS7t27p1mzZqlRo0by8PBQmzZttHnzZtP+8PBwubm5pajh79t8fHwUGhqqAQMGyMvLS97e3vr000/14MEDXbhwQY0aNZIk+fv7Kzg4OFXneevWLY0cOVL16tVTxYoVVatWLY0cOVLx8fGSpMjISLm5uWnLli1q3LixPD091aNHD8XGxpr66N69uyZOnKghQ4aoSpUqev3117Vw4UJToIqMjFSFChW0cOFCeXt7q3379kpOTtbvv/+ujz76SHXq1JGnp6d69eql48ePm/odNmyYhgwZovHjx6tq1aqqVauWgoKClJiYKEm6cOGC3NzctGDBAtWpU0eNGjXSnTt3UiwhS0xM1JQpU+Tj46NKlSrptdde0wcffKDr16+brunFixc1d+5cde/eXZJ0+/ZtjRo1SjVr1lS1atXk7+/PHRoAAAA8l1kDTHx8vAICApQ3b1599dVXioiIULNmzTRlyhQdO3ZMknTw4EGdOXNGK1as0MiRI7Vq1Sp17NhRvr6+Cg8Pl6urq4YNG2b6Ej9kyBCtX79eo0aN0saNG9W4cWN98MEH2rZt2wvVNnv2bNWoUUMbN27U0KFDtXz5ckVERKho0aJas2aNpIcBKyAgIFX9DRs2TNHR0Zo7d642b96swMBArV+/XqtWrUrRLigoSKNGjdKqVatka2srf39/3b5927R/5cqVcnR0VHh4uAYPHqx58+YpJCTEtD8pKUm7du3SqlWrNHHiRN29e1ddu3bV5cuX9fnnn+urr75Szpw51a1bN128eNF03JYtW/S///1PX331lT799FOtX79eEydOTFHbunXrtHTpUs2aNUu5c+dOsW/q1KnasmWLgoKCtHnzZgUFBWn//v36/PPPJUlr165VkSJFFBAQoODgYBmNRvXp00dxcXFasGCBVq9eLU9PT3Xt2lXR0dGpuqYAAADIesy6hCw+Pl7+/v566623lCtXLknSwIEDtWjRIp04cUKSlJycrHHjxil37twqXbq0pk2bppo1a6pt27aSpK5du+qHH37QlStXdPv2bW3fvl1ffPGFGjRoIEkaMGCAjh8/ri+++EKNGzdOdW1169aVv7+/JKlEiRJatmyZDh06pLZt28rZ2VmS5OTkZKr7n9SpU0c1atQw3ekpXry4li9frpiYmBTtPvnkE9WvX1+SNH36dDVo0EDffPON3nzzTUlS6dKlNXbsWBkMBrm6uio2NlZhYWHq06ePqY+AgACVKlVKkvTll1/qxo0bCg8PN9X92WefqXHjxlqxYoWGDh0qScqTJ4+mTZsme3t7lStXTv/73/80ceJEffzxx6Z+/fz8VLZs2aeen4eHh5o1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okWLdOLECa1du1bNmjVTr169JEmlS5fW2bNntWTJEknS/v37dfjwYe3fv1958+aVJA0ZMkSHDh1SWFiYgoKCzHFaAAAAyOTSFGCKFCmiIkWKpHctsCDOzs7y8/NTRESEoqOjdf78eR0/flySlJycrKNHj+qNN95IcUyNGjVMAebo0aMyGo1q2LBhijaJiYlKSEh4KecAAAAAy5OmADN58uT0rgMW5sqVK+rSpYucnZ3l4+OjunXrysPDQ/Xr15ck2draKjk5+ZnHJycnK3fu3AoPD39in52dXYbVDQAAAMuW5h+ylKTdu3frwIED+vPPP5UvXz5Vr15d9erVS6/akIlFRETo5s2b2rx5s7Jnzy5JOnHihKSHz7eUL19ev/76a4pjfvnlF9Pf5cqV0507d3T//n2VLVvWtH3kyJEqX768unXr9hLOAgAAAJYmTQEmMTFR7733nvbs2SMbGxvly5dPN27c0MKFC1WzZk0tWLCA/xfdyhUpUkTx8fH6/vvvVa1aNZ0+fdp0Zy4xMVF9+vRR3759VblyZTVs2FA///yzli9fbjq+Xr16cnd31+DBgzVixAgVLVpUX375pcLDwxUaGmqu0wIAAEAml6YAExwcrJ9//llTp05VixYtZGNjowcPHigiIkLjxo3T559/rg8++CC9a0Um0qxZMx09elRBQUG6c+eOihUrpk6dOmn79u2KiopS165dNX78eC1YsECfffaZKlWqpK5du5pCjI2NjRYvXqxp06Zp0KBBio+Pl6urq+bOnatatWqZ+ewAAACQWaUpwERERKh///5q3br1/3Vka6u2bdvq2rVrWrlyJQHGyhkMBn300Uf66KOPUmzv2bOnJOnAgQOqVq1airfRffHFFyle/uDs7MzzVAAAAHghafodmOvXr6tChQpP3VehQgVdvnz5XxUFy7dnzx716tVL+/fv16VLl7R9+3YtXbpUbdq0MXdpAAAAsGBpugNTsmRJ/fzzz09d6vPTTz+paNGi/7owWLb+/fvr7t27Gjp0qK5fv66iRYuqR48e6t27t7lLAwAAgAVLU4B58803FRQUpJw5c6pFixYqUKCArl69qoiICIWEhKh///7pXScsjJ2dnUaOHKmRI0eauxQAAABYkTQFmK5duyo6OlrTp0/XZ599ZtpuNBrVrl07vfPOO+lWIAAAAAA8kqYAky1bNk2cOFEBAQE6cOCAbt26JScnJ7322mtydXVN7xoBAAAAQNILBJjAwMB/bPPbb79JeviGqkmTJqW9KgAAAAB4ilQHmMjIyH9sc+PGDcXHxxNgAAAAAGSIVAeYHTt2PHPfgwcPNH/+fC1cuFAFChTQ2LFj06M2IE1K5nU2dwlmkVXPGwAAZC1pegbmcceOHVNgYKBOnDihFi1aaNSoUXJyckqP2oA0CXy9mblLMJuk5GTZZEvTzzsBAABYhDQHmAcPHmjevHkKCQlR3rx5NXfuXDVq1Cg9awNeWGJiouLj42Vvb2/uUsyC8AIAAKxdmgJMdHS06a5L69atNXLkSOXJkye9awPSxGg0mrsEAAAAZJAXCjAPHjzQ3LlztWjRIuXLl0+ff/65GjZsmFG1AQAAAEAKqQ4wR48e1bBhw3Tq1Cm1bdtWw4cPl6OjY0bWBgAAAAAppDrAdO7cWcnJyXJ0dNTFixf1/vvvP7OtwWDQ0qVL06VAAAAAAHgk1QGmatWqpr//6RkDnkGAORkMBnOXAAAAgAyS6gCzbNmyjKwDSBd2dnZZ7g1kvDoZAABkJf/6d2CAzCZo5y6dv3XL3GW8FCWdnDSsQX1zlwEAAPDSEGBgdc7fuqVT166ZuwwAAABkANadAAAAALAYBBgAAAAAFoMAAwAAAMBiEGAAAAAAWAwCDAAAAACLQYABAAAAYDEIMAAAAAAsBgHGAvj4+Cg4ONjcZTzThQsX5ObmpsjIyKfuDw4Olo+Pz0uuCgAAANaIAAMAAADAYhBgAAAAAFiMLBtgYmJi1LdvX9WoUUOVKlVSo0aNtHjxYsXFxal8+fLatWtXivaBgYHq2rWrJCk+Pl5jxoyRt7e3qlatqhEjRujDDz/UsGHDUjX2gAED1K9fP9Pn48ePy83NTaGhoaZty5YtU5MmTZ449q+//lLXrl3VunVrXb9+/R/HCgoKUqtWrUyfb926JXd3d40fP960bceOHfLy8lJCQoKSkpK0ZMkSNW3aVB4eHmratKlWrlxpahsZGakKFSpo4cKF8vb2Vvv27WU0GlOMGRsbqzp16mjo0KFKSkoybV+4cKFef/11Va5cWd27d9fZs2dN+9zc3DRnzhw1bNhQdevWTbEPAAAAeCRLBpj4+HgFBAQob968+uqrrxQREaFmzZppypQpunPnjmrUqKGIiAhT+4SEBG3ZskXt27eXJH3yySfau3evZs6cqa+++kq3b9/WN998k+rxGzZsqAMHDujBgweSpL1798pgMKR4hmTnzp1q1KjRE3X369dP9+7dU1hYmJydnVM1VkxMjK5cuSJJ2rdvn4xG4xNj1a1bVzly5FBQUJDmz5+v/v37a9OmTXrrrbc0ceJELVmyxNQ+KSlJu3bt0qpVqzRx4kQZDAbTvnPnzqlHjx56/fXXFRQUJBsbG0nSxYsXdejQIS1cuFDLly/XlStXNGLEiBS1fvnll5ozZ47mzp2rUqVKpe5iAgAAIEvJsgHG399fo0ePlqurq0qVKqWBAwdKkk6cOKH27dtr27Ztio+Pl/TwDkVSUpJ8fX0VFxenzZs3a8yYMapdu7bKlSunadOmqUCBAqkev0GDBoqPj9fhw4clST/++KMaNWqkgwcP6sGDB7p7964OHDiQIsAkJCTo3Xff1V9//aUlS5Yob968qRqrWrVqcnJy0t69e1OMderUKV29elWStHv3bjVq1Eh37tzRypUrNXDgQLVq1UqlSpWSv7+//Pz8tHDhwhR3WgICAlSqVCm5u7ubtl24cEH+/v6qX7++Jk2apGzZ/u+fV/bs2TV9+nSVL19elStX1ptvvqkjR46kqLVNmzby8PCQp6dnqq8lAAAAspYsGWCcnZ3l5+eniIgIjRkzRj179lSDBg0kScnJyWratKkkafv27ZKkjRs3qnHjxsqdO7eio6MlSV5eXqb+cuTIocqVK7/Q+FWqVNHevXuVmJiogwcPqm/fvkpISNCRI0e0b98+OTg4qGrVqqZjli5dqv379ytPnjxycnJK9Vi2traqV6+efvzxR0kP7/Z07txZBQsWVGRkpI4fP67//e9/ql+/vk6fPq379++rWrVqKfp47bXXdO3aNV27ds207Wl3SMaOHavLly+raNGiKe7KSFL+/PmVO3du0+c8efLo3r17Kdq4uLik+rwAAACQNWXJAHPlyhW1bt1aa9asUeHCheXn56d169aZ9js4OKhZs2batGmTbt68qf/+97+m5WOPlkQlJyf/qxp8fHy0d+9e/fzzz8qTJ48qV64sDw8PRUZGateuXWrYsKFpLEkqV66cwsLC9NNPP2nVqlUvNFajRo30448/6vz587p8+bJq1Kghb29v01jVqlVTvnz5nniW5ZFH52pra2valiNHjifatWvXTiNHjtTnn3+umJiYFPseP5dnyZkz54ucFgAAALKgLBlgIiIidPPmTa1cuVLvvfeemjRpolu3bkmS6Ut8hw4dtHfvXq1fv14FChRQzZo1JT182NxgMJiWf0lSYmKijh49+kI1+Pj46MiRI9q6datq1aolSapdu7b279//1OdfGjRooNdee009e/bU1KlT9fvvv6d6rHr16unmzZsKCwtTlSpV5ODgYBrrhx9+MI3l6uqq7Nmz6+eff05x/MGDB1WwYMF/vPPTokUL+fn5qVKlSgoMDEzxAD8AAACQHrJkgClSpIji4+P1/fff69KlS9qzZ4+GDBki6WEYkaTq1auraNGimjNnjtq0aWN6nqNEiRLy9fXVhAkTtG/fPp06dUojRozQH3/88cSyqecpW7asihUrpjVr1pgCTK1atbR//37dvHlTderUeepx/fv3l7Ozs0aOHJnqsRwdHVW9enWtWrUqxVjnzp3Tr7/+agowuXPnVpcuXTRnzhxFRETo3LlzWrFihb788ksFBASk6vyyZcumCRMm6MSJE1q0aFGqawQAAABSI0sGmGbNmqlXr14KCgqSr6+vJk2apI4dO6pGjRqKiooytWvXrp3++usv0/KxRyZMmKBq1appwIAB6tKli3LlyiUvLy9lz579hepo2LChEhMTTXd3PD09lTNnTtWuXVsODg5PPSZnzpwaP3689uzZozVr1qR5rFdeeUWlSpVS2bJlVaJECVO7wMBA+fv7a/r06WrRooVWrlyp0aNHKyAgINVjvfrqq+rTp4/mzp2rU6dOpfo4AAAA4J8YjM968AFPlZCQoP/+97+qWbNmiofSmzZtqtatW+v99983Y3VZ26Pw+fnpMzr12AsHrFnZ/Pk1v01rc5fxXHfv3tWxY8fk7u7+zGCOl495yZyYl8yJecmcmJfM6Z/m5dH3NQ8PjzSPYfvPTfA4Ozs7jRs3Tq+99pree+892djYaO3atbp06ZKaNWtm7vIAAAAAq0aAeUEGg0ELFy7UtGnT1KVLFyUlJalChQpavHixXF1dNX78+BRvNHuaefPmqXbt2v+6lpCQEM2fP/+5bYYPH65OnTr967EAAACAzIAAkwbu7u5avHjxU/f1799fb7/99nOPL1SoULrU0blzZ73xxhvPbZM/f/50GQsAAADIDAgw6czZ2VnOzs4vZSwnJ6cX+lFLAAAAwNJlybeQAQAAALBMBBgAAAAAFoMlZLA6JbPQsrqsdK4AAAASAQZWaFiD+uYu4aVKSk6WTTZupgIAgKyBbz2wKomJiYqPjzd3GS8V4QUAAGQlfPOB1TEajeYuAQAAABmEAAMAAADAYhBgAAAAAFgMAgwAAAAAi0GAgdUxGAzmLgEAAAAZhAADq2JnZyd7e3tzlyFJSk7mZQIAAADpjd+BgdWZuuug4m7dNmsNJZwcNbR+dbPWAAAAYI0IMLA6cbduK/baLXOXAQAAgAzAEjIAAAAAFoMAAwAAAMBiEGAAAAAAWAwCDAAAAACLQYABAAAAYDEIMAAAAAAsBgEGAAAAgMUgwGRSbm5uCg8PN3cZ/4qPj4+Cg4MlSUajUevWrdO1a9ckSeHh4XJzczNneQAAALBA/JAlMszatWuVI0cOSdJPP/2kYcOGafv27WauCgAAAJaMAIMM4+zsbPrbaDSasRIAAABYC5aQvUTt27fXp59+avq8bds2ubm56fvvvzdtCwoKUo8ePSRJZ86cUY8ePeTh4aF69eppwYIFKfrbuXOnOnfuLC8vL9WtW1eTJ0/WvXv3TPvd3Ny0atUq+fn5ycPDQ76+vjp06JBWrVqlBg0aqGrVqho0aFCKY55nwIAB6tevn+nz8ePH5ebmptDQUNO2ZcuWqUmTJpL+bwlZZGSk/P39JUmNGjVKsTQuPDxcjRs3loeHh9q3b69ff/01VbUAAAAgayLAvEQNGzbU3r17TZ9//PFHGQwGRUZGmrbt3LlTjRo1kiQtX75cbdu21bfffquuXbtqxowZ2rdvnyRp69atevfdd9WgQQOFh4dr3Lhx+vbbbzVkyJAUY86cOVO9e/fWhg0b5OjoqH79+mnz5s1auHChJk+erG3btmnNmjWprv/AgQN68OCBJGnv3r3Prf8RLy8v07Mwa9asUfPmzU37Vq9erRkzZujrr7+WnZ2dBg0alKpaAAAAkDURYF4iHx8fnT59Wr///rukhwGgUaNGpgBw/vx5nTlzRj4+PpIkPz8/tW3bViVKlNB7770nR0dHHTlyRJK0cOFCNWnSRO+9955Kly6tRo0aacyYMdq+fbtOnTplGrNDhw7y8fFRmTJl1KZNG926dUujR49WuXLl1LRpU7m7u+vkyZOpqr9BgwaKj4/X4cOHJT0MYI0aNdLBgwf14MED3b17VwcOHHgiwNjZ2cnJyUnSw2VlOXPmNO2bOHGiKleurHLlyqlXr166dOmS6UF/AAAA4O8IMC9RxYoVVbhwYe3du1eXLl3ShQsX1LdvX8XGxurKlSvauXOn3N3dVaxYMUlSqVKlUhyfJ08eJSQkSJJiYmJUtWrVFPtfe+01075HXFxcTH/b29tLkkqWLGnaljNnTiUmJqaqfmdnZ1WpUkV79+5VYmKiDh48qL59+yohIUFHjhzRvn375ODg8ERdz/P4OebJk0eSUr2kDQAAAFkPD/G/ZI8vI/Pw8FDlypVVuHBhRUZGateuXSnuXtjY2Dxx/KOH4Z/2UHxycrIkydb2/6b18b8fyZYt7bnVx8dH27Zt02uvvaY8efKocuXK8vDwUGRkpC5evKiGDRs+te5ned45AgAAAH/HHZiXzMfHR/v27dO+fftUq1YtSVKtWrW0Y8cORUZGPrH86lnc3Nx06NChFNsOHjwoSXJ1dU3foh/j4+OjI0eOaOvWrab6a9eurf379z/1+ZdHDAZDhtUEAACArIMA85LVqlVLCQkJ2rJlS4oA891336lgwYKqUKFCqvrp3bu3tmzZovnz5+vMmTP64YcfNGHCBDVs2DBDA0zZsmVVrFgxrVmzJkX9+/fv182bN1WnTp2nHufg4CDp4ZvL/vrrrwyrDwAAANaNAPOS2dnZqXbt2sqWLZs8PT0lPQwAycnJpof3U6Np06aaMWOGvvvuO7Vq1UpjxoxRixYtNGvWrIwp/DENGzZUYmKiatasKUny9PRUzpw5Vbt2bVNQ+bty5cqpfv36GjRokFatWpXhNQIAAMA6GYw8cAArERUVJUlaeOaqYq/dMmstrvmdFNy6oVlryCzu3r2rY8eOyd3d/ZkBFy8f85I5MS+ZE/OSOTEvmdM/zcuj72seHh5pHoM7MAAAAAAsBm8hgyRp/PjxWrdu3XPbzJs3T7Vr135JFQEAAABPIsBAktS/f3+9/fbbz21TqFChl1QNAAAA8HQEGEh6+COVzs7O5i4DAAAAeC6egQEAAABgMQgwAAAAACwGS8hgdUo4OZq7hExRAwAAgDUiwMDqDK1f3dwlSJKSk43Kls1g7jIAAACsCkvIYFUSExMVHx9v7jIkifACAACQAQgwsDpGo9HcJQAAACCDEGAAAAAAWAwCDAAAAACLQYABAAAAYDEIMLA6BgMPzwMAAFgrAgysip2dnezt7TN8nORkXhQAAABgDvwODKzOjN2xiruZca9SLpHXXkNed82w/gEAAPBsBBhYnbib8Tp9/a65ywAAAEAGYAkZAAAAAItBgAEAAABgMQgwAAAAACwGAQYAAACAxSDAAAAAALAYBBgAAAAAFoMAAwAAAMBiEGAygcjISLm5uenChQvUAQAAADwHP2QJEy8vL+3Zs0fOzs7mLgUAAAB4KgIMTOzs7FSwYEFzlwEAAAA8E0vInuKvv/7ShAkTVLduXXl5ealbt246cuSIwsPD5ebmlqLt37f5+PhoypQpat68uby9vXXgwIFUj7tr1y61bNlSlSpVUosWLbRz507Tvlu3bmnkyJGqV6+eKlasqFq1amnkyJGKj4+X9H/Lv7Zs2aLGjRvL09NTPXr0UGxsrKmP7t27a+LEiRoyZIiqVKmi119/XQsXLpTRaEzRx6MlZD4+PgoNDdWAAQPk5eUlb29vffrpp3rw4IGpz0OHDumtt95S5cqV1aBBA40bN0537twx7f/tt9/k5+cnLy8v1ahRQwMGDNClS5dM+9evX68WLVrIw8ND9erV08SJE5WYmJjqawYAAICshQDzFIMGDdLu3bs1efJkrV+/XiVKlFBAQID+/PPPVB2/fPlyjRw5UosWLZKnp2eqxw0LC9OoUaO0adMmlSpVSoMGDdJff/0lSRo2bJiio6M1d+5cbd68WYGBgVq/fr1WrVqVoo+goCCNGjVKq1atkq2trfz9/XX79m3T/pUrV8rR0VHh4eEaPHiw5s2bp5CQkGfWNHv2bNWoUUMbN27U0KFDtXz5ckVEREiSjh8/rp49e6pevXrauHGjpk+frqNHjyogIEBGo1FJSUnq27ev6fglS5bo0qVLGj58uOn4kSNHasCAAdq8ebMmTZqkDRs2aNGiRam+ZgAAAMhaWEL2N6dPn9bu3bsVGhqqunXrSpLGjh2rPHnyyMHBIVV91K9fX7Vr137hsYcPHy5vb29J0vvvv69t27YpNjZWlStXVp06dVSjRg3T3Z7ixYtr+fLliomJSdHHJ598ovr160uSpk+frgYNGuibb77Rm2++KUkqXbq0xo4dK4PBIFdXV8XGxiosLEx9+vR5ak1169aVv7+/JKlEiRJatmyZDh06pLZt2yo0NFR16tRRv379JEmlSpXSZ599psaNG+vAgQMqX768bty4oUKFCqlYsWIqUaKEZs2apWvXrkmSLly4IIPBoGLFiumVV17RK6+8otDQUOXOnfuFrx0AAACyBgLM3zwKBI/fOcmRI4cCAwMVHh6eqj5cXFzSNHbp0qVNf+fJk0eSdO/ePUmSn5+fduzYoXXr1uns2bM6deqULly4oDJlyqTo41EAkqS8efOqdOnSKUKOt7e3DAaD6bOXl5dCQkJ048aNp9bk6uqa4rOjo6Pu378vSYqOjta5c+fk5eX1xHGxsbHy9vZW7969NWHCBM2ZM0c1a9ZU/fr15evrK0mqV6+evLy81LFjRxUvXlx16tRRo0aNVKlSpX++WAAAAMiSCDB/Y2v7YpckKSnpiW05c+ZM09jZsj25os9oNCo5OVl9+/bVyZMn1bJlSzVv3lwVK1bUqFGjnmj/9/qTkpJS9Pv3/cnJyZIkGxubp9ZkZ2f31JoeHduqVSvTHZjHPXqT2UcffSQ/Pz/t2rVL+/bt04QJE7Ro0SKtX79eOXLkUFhYmKKjo7Vnzx7t2bNH/fr1U9u2bTV58uSn1gMAAICsjWdg/ubRHYeoqCjTtgcPHsjHx8cUBB5/SP3s2bMZXtOxY8e0e/duzZ49Wx999JFat26tkiVL6vz586Yw8cjjdV+/fl3nzp1TxYoVn7pfevgQfvHixeXk5PTCdb366qs6deqUXFxcTP89ePBAkydP1u+//67Tp09rzJgxyp8/v7p27ao5c+Zo0aJFio2N1fHjx7Vr1y7NnTtXFSpU0DvvvKOwsDANHDhQ33777QvXAgAAgKyBAPM3pUuX1htvvKFx48Zp//79OnPmjEaNGqWEhASVKVNGBoNBwcHBunDhgr777jutW7cuw2sqUKCAbG1t9d133ykuLk5RUVEaNGiQrly58sQbu8aNG6effvpJx48f14cffqiCBQuqWbNmpv0HDx7UnDlzdPbsWa1du1YrVqxQ796901RXQECAoqOjNW7cOMXGxuqXX37Rhx9+qLNnz6pUqVLKly+fvvnmG40ePVqxsbE6c+aM1q1bJycnJ5UpU0bZs2fXvHnztGTJEsXFxenIkSPauXPnU5ekAQAAABIB5qkmTZqkGjVq6IMPPlD79u31+++/KzQ0VJUrV9a4ceO0detW+fr6atWqVRo6dGiG11O4cGEFBQVpx44dat68uT744AMVLlxYPXr00JEjR1K07dKli4YOHaquXbsqZ86cCgsLk729vWl/o0aNFBsbq9atW+uLL75QYGCgunbtmqa6PD09tWjRIh07dkzt2rXTu+++q9KlS2vJkiWys7NTvnz5FBISoosXL6pz585q166dLly4oP/85z/KnTu3ateurYkTJ2rt2rVq2bKlevXqJRcXF82YMeNfXS8AAABYL4Px72uQYJEiIyPl7++v7du3q3jx4k9t0717dxUrVkxBQUEvubqX49HyuMVnDDp9/W6GjVPG2UEzW/OigdS6e/eujh07Jnd391S/yQ8Zj3nJnJiXzIl5yZyYl8zpn+bl0fc1Dw+PNI/BHRgAAAAAFoO3kGWw6tWrP/VNZY/kz59f27Zte4kVAQAAAJaLAJPBwsPDn3hT2OOe9friF+Xt7a0TJ048t82yZcvSZSwAAADAXAgwGaxkyZLmLgEAAACwGjwDAwAAAMBiEGAAAAAAWAyWkMHqlMhr/8+NMnH/AAAAeDYCDKzOkNddM3yM5GSjsmUzZPg4AAAASIklZLAqiYmJio+Pz/BxCC8AAADmQYCB1Xnea6sBAABg2QgwAAAAACwGAQYAAACAxSDAAAAAALAYBBgAAAAAFoMAA6tjMPCGMAAAAGtFgIFVsbOzk739v/uhyeRk3mIGAACQWfFDlrA6G3ff0NVbD9J0bAEnW7V+PV86VwQAAID0QoCB1bl664EuX79v7jIAAACQAVhCBgAAAMBiEGAAAAAAWAwCDAAAAACLQYABAAAAYDEIMAAAAAAsBgEGAAAAgMUgwAAAAACwGASYl8DNzU3h4eHmLiPTO3nypHbu3GnuMgAAAJCJEWCQafTt21dRUVHmLgMAAACZGAEGAAAAgMUgwKRR+/bt9emnn5o+b9u2TW5ubvr+++9N24KCgtSjR48Ux125ckXNmjVTz549de/evVSNlZiYqGnTpqlevXry8vJS586dtWfPnhRtfvvtN/Xo0UNeXl6qXbu2xowZo/j4eElSUlKSlixZoqZNm8rDw0NNmzbVypUrTcdGRkaqQoUK2rVrl1q2bKlKlSqpWbNm2rZt2wtdk23btqlTp07y9PSUh4eH2rdvr//+97+m/UajUUuXLlXTpk1VuXJltWjRQhEREZIkHx8fXbx4UXPnzlX37t1faFwAAABkHQSYNGrYsKH27t1r+vzjjz/KYDAoMjLStG3nzp1q1KiR6fP169fVo0cPFStWTF988YVy5syZqrECAwO1d+9eTZ8+XevWrZOvr6/69etnel4kLi5Ob7/9tgoVKqRVq1YpODhYe/fu1bhx4yQ9DFLz589X//79tWnTJr311luaOHGilixZYhojKSlJ06ZN04gRIxQREaFy5crpk08+0V9//ZWqGo8cOaIBAwaoRYsW2rRpk1avXi1nZ2cNHTpUiYmJkqRFixZp5syZ6t27tyIiIvTmm29q6NCh2r9/v9auXasiRYooICBAwcHBqRoTAAAAWQ8BJo18fHx0+vRp/f7775KkvXv3qlGjRqYAc/78eZ05c0Y+Pj6SpJs3b6pHjx565ZVX9PnnnytHjhypGufcuXOKiIjQ5MmT5e3trVKlSqlnz55q0aKFQkNDJUmrV69W3rx5NWnSJJUrV07VqlXTp59+KhcXF925c0crV67UwIED1apVK5UqVUr+/v7y8/PTwoULZTQaTWMNGjRItWrVUqlSpfTee+/pzp07iomJSVWdNjY2GjVqlHr06KESJUrI3d1d/v7+un79uq5du2a6++Lv769OnTqpZMmS6t69uwYPHqwHDx7I2dlZNjY2cnBwUN68eVM7DQAAAMhibM1dgKWqWLGiChcurL1796p27dq6cOGCpk2bpk6dOunKlSvauXOn3N3dVaxYMUnSzJkzdf/+fVWqVEl2dnapHic6OlqS5Ofnl2L7/fv3lSdPHklSTEyMKlasKFvb/5vOmjVrqmbNmvrtt990//59VatWLcXxr732mpYuXapr166ZtpUpU8b0d+7cuU3jpIa7u7ucnJy0cOFCnT59WufOndPx48clPby7c+PGDV25ckVVqlRJcVyfPn1S1T8AAAAgEWD+lceXkXl4eKhy5coqXLiwIiMjtWvXrhTLx2rXrq0OHTpowIABat68uerWrZuqMR7dIVmxYoVy5cqVYl+2bA9voD0eXJ51/N8lJyc/cezTgtWzjv+7AwcOqFevXmrQoIGqVaumVq1aKT4+Xu+//74kKXv27KnqBwAAAHgelpD9Cz4+Ptq3b5/27dunWrVqSZJq1aqlHTt2KDIyMkWAadq0qd544w01b95co0aN0p07d1I1xquvvirp4cP/Li4upv/Cw8NNvy1TtmxZRUdHKykpyXTc1q1b5ePjI1dXV2XPnl0///xzin4PHjyoggULysnJ6V9dg0cWL14sb29vBQcHq0ePHqpTp45p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IejmsaL68+qDVB2Tp4Ct6nUukJFlZXl3797VsWPH5O7uLgcHB3OXg/+PecmcmJfMiXnJnJiXzOmf5uXR9zUPD480j8EdGAAAAAAWgwADAAAAwGIQYAAAAABYDAIMAAAAAItBgAEAAABgMQgwAAAAACwGP2QJq5M7X+r/Wb9IWwAAAJgf395gdbya5H2h9sZkowzZDBlTDAAAANIVS8hgVRITExUfH/9CxxBeAAAALAcBBlbHaDSauwQAAABkEAIMAAAAAItBgAEAAABgMQgwAAAAACwGAQZWx2DgoXwAAABrRYCBVbGzs5O9vb3pszGZB/oBAACsCb8DA6tzYe01JVx9oBwFbFW8Y35zlwMAAIB0RICB1Um4+kD3fr9v7jIAAACQAVhCBgAAAMBiEGAAAAAAWAwCDAAAAACLQYABAAAAYDEIMAAAAAAsBgEGAAAAgMUgwAAAAACwGASYvwkPD5ebm1uGtX/Zfv75Zx08eFCSdOHCBbm5uSkyMvIfj3uRtql19+5drVixIt36AwAAQNZDgLFyfn5+On/+vCSpaNGi2rNnj7y8vP7xuBdpm1qLFy9WaGhouvUHAACArMfW3AXg5bGxsVHBggXTvW1qGY3GdO0PAAAAWU+WvQPz119/acKECapbt668vLzUrVs3HTly5Il2Pj4+Cg4O/sdtq1evVr169VSlShX169dPFy9eTHUt4eHhev3117V69WpTPe+//74uX75sanPp0iUNHjxYtWrVUsWKFfX6669r2rRpSk5ONvXRpEkTffrpp6pWrZree+8909K2wMBADRs27IllYUajUUuXLlXTpk1VuXJltWjRQhEREZKeXELWvXt3TZw4UUOGDFGVKlX0+uuva+HChSlCybZt29SpUyd5enrKw8ND7du313//+19JUnBwsObOnauLFy/Kzc1NFy5ckCR9/fXX8vX1VeXKleXr66ulS5eazgkAAAD4uywbYAYNGqTdu3dr8uTJWr9+vUqUKKGAgAD9+eefaepv2bJlmj17tlasWKEbN27o/ffff6E7DtevX9fSpUs1a9YsLV26VL///rt69+6tBw8eSJLeffdd3b59W//5z3/0/fffKyAgQIsWLdKOHTtMfZw/f17/+9//tH79eg0ePFh79uyRJA0fPlwjRox4YsxFixZp5syZ6t27tyIiIvTmm29q6NCh2r9//1Nr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"/>

#### Smote - LightGBM

```python
params = {
    'n_estimators' : [100, 200, 300, 400],
    'learning_rate' : [0.01, 0.05],
    'num_leaves' : [20, 50, 80, 100, 150, 200],
    }

lgbm_clf = LGBMClassifier(random_state=13, n_jobs=-1, boost_from_average=False)
lgbm_cv = GridSearchCV(lgbm_clf, param_grid=params, cv=4, n_jobs=-1)
lgbm_cv.fit(X_train_over, y_train_over)
```

<pre>
[LightGBM] [Info] Number of positive: 6609, number of negative: 6609
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000808 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 1840
[LightGBM] [Info] Number of data points in the train set: 13218, number of used features: 16
</pre>
<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4,

             estimator=LGBMClassifier(boost_from_average=False, n_jobs=-1,
                                      random_state=13),
             n_jobs=-1,
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.05],
                         &#x27;n_estimators&#x27;: [100, 200, 300, 400],
                         &#x27;num_leaves&#x27;: [20, 50, 80, 100, 150, 200]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4,
             estimator=LGBMClassifier(boost_from_average=False, n_jobs=-1,
                                      random_state=13),
             n_jobs=-1,
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.05],
                         &#x27;n_estimators&#x27;: [100, 200, 300, 400],
                         &#x27;num_leaves&#x27;: [20, 50, 80, 100, 150, 200]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(boost_from_average=False, n_jobs=-1, random_state=13)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(boost_from_average=False, n_jobs=-1, random_state=13)</pre></div></div></div></div></div></div></div></div></div></div>



```python
lgbm_best = lgbm_cv.best_estimator_
lgbm_best.fit(X_train_over, y_train_over)
pred = lgbm_best.predict(X_test)
```

<pre>
[LightGBM] [Info] Number of positive: 6609, number of negative: 6609
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000919 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 1840
[LightGBM] [Info] Number of data points in the train set: 13218, number of used features: 16
</pre>



```python
accuracy_score(y_test, pred)
print_clf_eval(y_test, pred)
```

<pre>
==> Confusion matrix
[[2782   51]
 [ 144   37]]
===================
Accuracy : 0.9353, precision : 0.4205
Recall : 0.2044, F1 : 0.2751, AUC: 0.5932
</pre>



```python
best_cols_values = lgbm_best.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train.columns)
best_cols = best_cols.sort_values(ascending=False)
best_cols
```

<pre>
province              8258
spent_last_year       6983
avg_wk_workhr         6700
amount_appropriate    6390
time_per              5558
family                5446
age                   4436
research              4025
cur_happiness         4008
whom_with             2960
edu                   2156
club_participate      1022
wk_econ_act            862
marrital               849
disabled                47
sex                      0
cost_per                 0
dtype: int32
</pre>



```python
sns.set(style='whitegrid', color_codes = True)
plt.figure(figsize=(8,8))
sns.barplot(x=best_cols, y=best_cols.index, hue=best_cols.index)
```

<pre>
<Axes: xlabel='None', ylabel='None'>
</pre>


<img src="data:image/png;base64,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kSO1ceNGBQYGasyYMUnuOUlISNDUqVM1fPhwbdq0SWXLltXHH3+sv//+W0FBQQoKClKhQoW0Z88eFS5c+Knfw5QpU7RlyxaFhIRo8+bNCgkJ0f79+/XZZ59Jko4fP64RI0aof//+2rx5syZOnKj169dr0aJFKly4sFavXi3pYWgKCgp64nx169aVq6ur1q9fb9525swZHT58WO3atdPdu3fVtWtXSdLy5cu1bNky5cmTRx07dtSVK1fSVPMja9eu1dKlSzVz5kw5OTk9dW8AAABgG7JauoCX7cqVK4qPj9crr7yiIkWKqEiRIvr888+VkJCgW7duSZI+/vhj1alTR5I0bdo01a1bV998841atmypJUuWaPr06apbt64kqXjx4rp06ZLCwsL01ltvmecZOHCgqlevLkl69913tXnzZsXExMjHx0eOjo6ys7NT/vz50/UevLy81KRJE1WpUkWSVKRIEdWoUUMxMTGSHl6dMBgMKlKkiF555RW98sorCgsLk5OTk+zs7OTq6ipJcnFxUc6cOZ84X9asWdWqVSutX79e7777rqSHV2y8vLxUpkwZrV69Wn/99ZemTp2qrFkf/paaMGGCIiMjtWrVKvXv3/+JNT8SEBCgMmXKpKsvAAAAsB2ZLqh4eHioefPm6tu3r/Lnz6+aNWuqbt26atiwoX7++WdJkq+vr3l87ty5VapUKcXExOjUqVOKi4vTBx98oCxZ/u9i1IMHDxQfH6979+6Zt5UuXdr886MrA/fv338u76FVq1b68ccfNW3aNJ09e1anT5/WmTNnzCGgdu3a8vHxUfv27VW0aFHzUqry5cune8527dpp8eLF+vXXX1WhQgVt2LBBvXr1kiRFR0fr1q1bqlq1apJj4uLizMvmnlTzIyVKlEh3jQAAALAdmS6oSNKnn36q9957T7t379aPP/6ojz76SJUrVzZfLXh0VeCRhIQEZcmSxXwT+syZM5MEkUfs7e2T/fmR53UT+6hRo7R582a1bt1afn5+eu+99xQWFmZeZpU9e3aFh4crOjpae/bs0Z49e9S3b1+1bt1akyZNStecZcqUUcWKFbVhwwbdu3dP165dU/PmzSVJiYmJKlWq1GPLuCTJ0dExTTU/kiNHjnTVBwAAANuS6YLKr7/+qm+++UbDhg1T6dKl1a1bN23YsEEfffSROnXqJEmKiooyL9u6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R8uqfv755yT/Un/w4EHzkqVH89y5c8d8pebs2bNPNc8/r2ikxbFjx7R7926tWrVKFStWlCTdv39f58+fV7FixZKMjYqKUvXq1SVJN27c0Llz59S9e/ck+//p0KFDKlq0qFxcXJKd293dXevWrdP169eVN29eSVJcXJyOHDliXj4nPbzCkZCQYA4mv/zyixwcHFSqVKknPob55s2bWrlypWbMmJHkccWnT5+Wo6NjsseULVtWd+7c0f3795MsJxsxYoTKlSunLl26pDonAAAAbJtFb6YvVKiQjEajvv/+e12+fFl79uzR4MGDJcn8xKmn4ebmpnr16mns2LHauXOnzpw5ozlz5mj79u3mm969vb1lMBgUGhqqixcv6rvvvtPatWufap5HH75jYmKS3Oieknz58ilr1qz67rvvdOHCBUVFRWngwIG6evXqY+9z7Nix+umnn3T8+HF98MEHyp8/v5o0aWLef/DgQc2ePVtnz57VmjVrtGLFCvXs2TPFuVu0aKHcuXNr4MCB+u2333T8+HF9+OGHunv3rjp16mQed+nSJY0dO1axsbHasmWLZs+erS5duiRZDpYSJycnOTs7a/v27Tp37pxOnDihkSNH6ujRo0neX86cOXX27Fldu3ZNtWvXloeHhwYNGqT9+/fr3LlzmjRpkvkBCQAAAMjcLHpFpUmTJjp69KhCQkJ0584dFSlSRB06dND27dsVFRWlwoULP/U5p0+frunTp2v48OH666+/VLZsWYWGhqphw4aSHt4YP3bsWM2fP19ffPGFKleurCFDhujjjz9O8xx58uRRu3btNGXKFJ07d04jRoxIdXzBggUVEhKi0NBQrVixQvnz51fdunXVrVs37dixI8nYTp06aciQIfrzzz9VrVo1hYeHJwkL9evXV2xsrFq2bKkCBQooODhYnTt3TnFuZ2dnLV++XCEhIerWrZukh/fwfPnll0mu5nh7eytLlixq3769nJ2dFRgYqHfeeSdN/ciWLZtmzZqlkJAQtWjRQi4uLvL19dXgwYM1f/58GY1GOTg4qGvXrpo8ebJOnjypDRs2aPHixZo6daoGDhwoo9EoNzc3zZkzx3xFCQAAAJmXwcS3F1qFyMhIBQYGavv27SpatGiyY7p27aoiRYo89ydiDR06VJcuXdKyZcue63lftkfL4i5HzdXNqyctXA0AANYjT/5X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zsZqz549EhwcLB4eHjJt2jT59ttvlb7S0lKZP3++GI1GGT16tGzbtk0sFovq+bt375axY8eKl5eXPP/885Kfn6/qt2UOrVq4cKG89tprTfbpPZvKykqJi4uT8ePHi8lkktmzZ0t2drbSz3NLxCCi4Q//JyIiIiIiXdLuRdVERERERKRbLFSIiIiIiEhzWKgQEREREZHmsFAhIiIiIiLNYaFCRERERESaw0KFiIiIiIg0h4UKERERERFpDgsVIiIiIiLSHBYqREREbeDFF1+Eu7s7fv/99yb7g4ODERMT086rIiLqPFioEBERtRGLxYLY2FjU1tZ29FKIiDodFipERERtxN7eHsXFxdi5c2dHL4WIqNNhoUJERNRG3NzcMH36dOzevRunT59udpzFYsG+ffswdepUeHl5Yfz48di8eTNqamqUMTExMZg3bx7S0tIwceJEeHp64qmnnsKxY8dUc/3111+IioqCv78/hg8fjrlz5yI/P7/N9khE1FZYqBAREbWhN998Ew4ODi1eArZ27Vps2LABISEh+OCDDxAeHo6UlBQsXboUIqKMO336NJKTk7F8+XLs3LkTXbt2RWRkJK5fvw4AqKiowHPPPYczZ85gzZo1SEhIQENDA8LDw3H27Nl22S8R0b3CQoWIiKgN9enTB2+99RaKioqavATMbDYjNTUVy5cvx6uvvorAwEAsWrQI69evxy+//KJ6x+TGjRtISkrClClTMG7cOMTGxuL27dvIysoCAHzyySeorKzEnj17MHXqVISEhCA5ORn9+vXD9u3b223PRET3AgsVIiKiNhYcHIxp06Zh9+7dOHPmjKrvxIkTAIApU6ao2qdMmYKuXbsiOztbaXv44YcxaNAg5WcnJycAQHV1NQAgMzMTbm5ucHR0RH19Perr69GlSxeMHTsWx48fb5O9ERG1FbuOXgAREZEerF69GpmZmYiNjUVaWprSbr1s69FHH1WNt7Ozg4ODA27cuKG09ezZUzXGYDAAABoaGgAAlZWVOH/+PDw8PJpcQ3V1daM5iIi0ioUKERFRO+jTpw/i4uKwbNkyvP/++6p2ALhy5QoGDhyotNfV1eHvv/+Gg4ODzf+Hvb09/P398cYbbzTZ37179//n6omI2h8v/SIiImonISEhCAsLw65du1BRUQEA8Pf3BwBkZGSoxmZkZMBisWDEiBE2z+/v74+SkhIMGTIERqNReRw+fBipqano2rXrvdsMEVEb4zsqRERE7WjNmjXIysrC1atXAQDOzs6YMWMGduzYgerqavj5+aGgoADvvfceRo4ciTFjxtg897x583D48GHMmzcPERERcHBwwNGjR/H5558jNja2rbZERNQmWKgQERG1o759+yIuLg6vvPKK0hYfH4/BgwcjLS0NH330Efr374+XXnoJS5cuRZcutl/84OjoiAMHDiAhIQFxcXGoqanB448/jvj4eDzzzDNtsR0iojZjkLs/oJ2IiIiIiEgDeI8KERERERFpDgsVIiIiIiLSHBYqRERERESkOSxUiIiIiIhIc1ioEBERERGR5rBQISIiIiIizWGhQkREREREmsNChYiIiIiINIeFChERERERaQ4LFSIiIiIi0hwWKkREREREpDn/A31HbkAeZj+fAAAAAElFTkSuQmCC"/>

Let's try dropping some columns



```python
df = pd.read_csv('final_df.csv', index_col=0)
a = df.drop(['tried_golf'], axis=1)
X = a.drop(['exp_golf', 'want_golf'], axis=1)
y = a['exp_golf']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=13, 
                                                    stratify=y)
smote = SMOTE(random_state=13)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
```


```python
def get_result(model, X_train_over, y_train_over, X_test, y_test):
    model.fit(X_train_over, y_train_over)
    pred = model.predict(X_test)

    return get_clf_eval(y_test, pred)
```


```python
def get_result_pd(models, model_names, X_train, y_train, X_test, y_test):
    col_names = ['Accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    tmp = []

    for model in models:
        tmp.append(get_result(model, X_train, y_train, X_test, y_test))
    
    return pd.DataFrame(tmp, columns=col_names, index=model_names)
```


```python
rf_clf = RandomForestClassifier(random_state=13, n_jobs=-1, n_estimators=100)
lgbm_clf = LGBMClassifier(random_state=13, n_jobs=-1, n_estimators=1000,
                          num_leave=54, boost_from_average=False)
lr_clf = LogisticRegression(random_state = 13, solver='liblinear')
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=13)
```


```python
models = [lr_clf, rf_clf, lgbm_clf, dt_clf]
model_names = ['logistic regression', 'RandomForest', 'LightGBM', 'Decision Tree']

results = get_result_pd(models, model_names, X_train_over, y_train_over, X_test, y_test)
```

<pre>
[LightGBM] [Warning] Unknown parameter: num_leave
[LightGBM] [Warning] Unknown parameter: num_leave
[LightGBM] [Info] Number of positive: 7081, number of negative: 7081
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000980 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 1843
[LightGBM] [Info] Number of data points in the train set: 14162, number of used features: 17
[LightGBM] [Warning] Unknown parameter: num_leave
</pre>



```python
results
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }



    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

#### Model Evaluation

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>roc_auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>logistic regression</th>
      <td>0.626990</td>
      <td>0.110119</td>
      <td>0.735099</td>
      <td>0.191544</td>
      <td>0.677588</td>
    </tr>
    <tr>
      <th>RandomForest</th>
      <td>0.931927</td>
      <td>0.391304</td>
      <td>0.238411</td>
      <td>0.296296</td>
      <td>0.607346</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>0.940685</td>
      <td>0.515625</td>
      <td>0.218543</td>
      <td>0.306977</td>
      <td>0.602707</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.873806</td>
      <td>0.237342</td>
      <td>0.496689</td>
      <td>0.321199</td>
      <td>0.697307</td>
    </tr>
  </tbody>
</table>


</div>



```python
params = {
    'max_depth' : [5, 10, 15, 20]
    }

dt_cv = GridSearchCV(dt_clf, param_grid=params, cv=4, n_jobs=-1)
dt_cv.fit(X_train_over, y_train_over)
```

<style>#sk-container-id-7 {color: black;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4,

             estimator=DecisionTreeClassifier(max_depth=5, random_state=13),
             n_jobs=-1, param_grid={&#x27;max_depth&#x27;: [5, 10, 15, 20]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4,
             estimator=DecisionTreeClassifier(max_depth=5, random_state=13),
             n_jobs=-1, param_grid={&#x27;max_depth&#x27;: [5, 10, 15, 20]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=5, random_state=13)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=5, random_state=13)</pre></div></div></div></div></div></div></div></div></div></div>



```python
dt_best  = dt_cv.best_estimator_
dt_best.fit(X_train_over, y_train_over)
predict_dt = dt_best.predict(X_test)
accuracy_score(y_test, predict_dt)
print_clf_eval(y_test, predict_dt)
```

<pre>
==> Confusion matrix
[[2172  189]
 [ 106   45]]
===================
Accuracy : 0.8826, precision : 0.1923
Recall : 0.2980, F1 : 0.2338, AUC: 0.6090
</pre>


#### Tensorflow 사용 

```python
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
```

<pre>
WARNING:tensorflow:From c:\Users\USER\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.



</pre>

```python
model = Sequential([
    Dense(units =50, activation='relu'),
    Dense(units = 50, activation='relu'), 
    Dense(units=1, activation='sigmoid')
])
adam = keras.optimizers.Adam(learning_rate=0.001)
```

<pre>
WARNING:tensorflow:From c:\Users\USER\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.



</pre>

```python
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
```


```python
model.fit(X_train_over, y_train_over, epochs=100)
</pre>
```

<pre>
<keras.src.callbacks.History at 0x1a10079d9d0>
</pre>



```python
predicted = model.predict(X_test)
predicted
```

<pre>
79/79 [==============================] - 0s 1ms/step
</pre>
<pre>
array([[0.4906654],
       [0.4906654],
       [0.4906654],
       ...,
       [0.4906654],
       [0.4906654],
       [0.4906654]], dtype=float32)
</pre>



```python
predicted = tf.squeeze(predicted)
predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
actual = np.array(y_test)
conf_mat = confusion_matrix(actual, predicted)
displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
displ.plot()
```

<pre>
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1a101d03050>
</pre>


<img src="data:image/png;base64,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Bg1y2jZhwgQlJiY6vVJTUx37bTab5s+fr5SUFLVv315Dhw7V8ePHnc6xd+9epaWlqX379kpNTdWKFSuqXBtBAQBgSna7ey9PWbVqlebOnVtp+/79+zV8+HB9/PHHjtcf//hHx/6FCxdq9erVmjJlitasWSObzaYhQ4bIarVKks6dO6fBgwerSZMmWrdunUaNGqXMzEytW7euSvUxegAAmM7lyyO9+/TIvLw8TZw4UdnZ2WratKnzue12ffPNN3r00UcVHR1d6b1Wq1XLli3T+PHj1bNnT0nSnDlzlJKSok2bNqlPnz56++235e/vr8mTJ8vPz08JCQk6evSolixZov79+xuuk44CAMCU7HaLWy93ff311/L399e7776rdu3aOe07duyYiouL1bx586u+d9++fSoqKlK3bt0c28LDw9W6dWtt375dkrRjxw517txZfn7/7Ql07dpVR44c0enTpw3XSUcBAIBrlJOTo9GjR7vcv2XLFpf7UlNTndYcfN+BAwckSStXrtS///1v+fj4qHv37hozZozCwsKUm5srSYqPj3d6X0xMjGNfbm6uWrZsWWn/lbqjoqJ++Jv7PwQFAIAp1eSrHg4cOCAfHx/FxMRo0aJFOnbsmGbMmKGDBw9q+fLlKikpkSQFBAQ4vS8wMFAFBQWSpNLS0qvul6SysjLDtRAUAACm5IkFifHx8T/YNbhWI0aM0IMPPqj69etLklq2bKno6Gj95je/0VdffaWgoCBJl9cqXPm1dDkABAcHS5KCgoIcCxu/v1+SQkJCDNfCGgUAAGoYHx8fR0i4okWLFpIujxSujBzy8/OdjsnPz1dsbKwkKS4u7qr7JTmOMVRL1UoHAKB28PZixh/y5JNP6re//a3Ttq+++kqSdOONNyopKUmhoaHKzs527C8sLNSePXuUnJwsSUpOTtbOnTtVUVHhOObTTz9Vs2bNFBkZabgWggIAwHzcDAl2u0WqxrDwi1/8Qtu2bdMrr7yiY8eO6V//+peeffZZ9enTRwkJCQoICFBaWpoyMzO1ZcsW7du3T2PGjFFcXJx69+4tSerfv78uXryo5557Tt98843Wr1+vN954Q8OGDatSLaxRAACghrn99ts1d+5cLVmyRK+99prCwsJ09913O11hkZ6ervLyck2YMEGlpaVKTk5WVlaW/P39JUmRkZFaunSpXnrpJfXr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npo0aZIKCgqq7WuvX79eiYmJ+u677yRJL7/8shITEw2/Pzc3V48++qhOnDjhdi3fffedEhMTtX79epfHDBo0SIMGDarSea/lPVfzvz8rADULd2ZErda6dWtNnDjR8ftLly7p66+/1uzZs7V371699dZbslgs1V7Hfffdp5SUFMPHb926Vf/617+qsSIAMIaggFotNDRU7du3d9qWnJysoqIizZ8/X19++WWl/dUhLi5OcXFx1f51AMDTGD3AlNq0aSNJOnnypKTLbfTx48crPT1d7du31+DBgyVJZWVlmjFjhnr06KE2bdro7rvv1nvvved0LpvNpoULF6pnz55q166dRo4cWWmscbXRw4YNG9SvXz+1a9dOPXv21KxZs2S1WrV+/Xo988wzkqTbb79dTz/9tOM977zzju666y7HCOXll19WRUWF03k3bdqkvn37qm3bturXr5/27dtX5Z/P2bNnNWnSJN12221q06aNOnfurFGjRl11PLBgwQLdcsst6tChg0aOHKnjx4877T9w4ICGDRumjh07qmPHjho1alSlYwDUXHQUYEqHDx+WJDVu3Nix7f3331ffvn316quvymazyW63a9SoUdq1a5fS09OVkJCgDz74QGPGjJHVatWvf/1rSdLMmTO1YsUKjRgxQu3atdP777+vWbNm/eDXX7VqlSZPnqz77rtPY8eO1fHjxzVjxgwVFBRo9OjRGjFihF599VW98sorjoCxePFizZkzR2lpaXrmmWe0d+9evfzyy8rJydHUqVMlSR9++KHS09N19913KyMjQ3v37lVGRkaVfjZ2u13Dhg1TQUGBxo8fr6ioKO3fv19z587VxIkTlZWV5Th2586dOnPmjJ5//nlVVFRo1qxZeuihh7Rx40aFhobq8OHDeuCBB9S8eXP94Q9/UHl5uV599VUNGDBAf/7znxUZGVml2gBcfwQF1Gp2u13l5eWO3xcUFOizzz7Tq6++qg4dOjg6C5Lk7++vSZMmKSAgQJL0ySef6KOPPtKcOXP0q1/9SpKUkpKikpISZWZmqk+fPiouLtbKlSs1ePBgPfbYY45j8vPz9dFHH121JpvNpgULFqhXr1568cUXHdtLSkr017/+VWFhYWrSpIkkqVWrVmrUqJEuXLighQsX6v7779eECRMkSbfeeqvq1aunCRMmaPDgwWrRooUWLFigtm3baubMmY5aJP1ocPm+/Px8BQcH66mnnlKnTp0kSV26dNGxY8e0du1ap2N9fX21bNkyx1ilefPm+vWvf60NGzYoLS1Nr7zyioKDg/XGG28oNDRUktStWzf16tVLS5cu1VNPPWW4LgDeQVBArbZ9+3bddNNNTtt8fHx0yy23aPLkyU4LGZs3b+4ICZK0bds2WSwW9ejRwylspKam6t1339XBgwd16tQpXbp0SbfddpvT17jzzjtdBoXDhw/rzJkzuuOOO5y2P/LII3rkkUeu+p7PP/9cpaWlSk1NrVSLdDnUNG7cWF9//bWeeOKJSrVUJSjExsZqxYoVstvt+u6773T06FF9++232rVrl6xWq9OxHTt2dFp70apVKzVu3Fjbt29XWlqaPv30U3Xu3FlBQUGOukNDQ9WpUydt3brVcE0AvIeggFrtpptu0qRJkyRJFotFgYGBio+Pd/zr9vvq1Knj9Pvz58/LbrerY8eOVz13fn6+CgsLJUn169d32hcdHe2ypvPnz0tSldruV97z6KOPuqyloKBAdru9Ui0xMTGGv84V7777rmbPnq2cnBzVq1dPrVq1UlBQUKXjoqKiKm2LjIx0/FzOnz+v9957r9K6DkmKiIiocl0Arj+CAmq1OnXq6Oabb76m94aFhSkkJEQrVqy46v4bbrhBu3fvliSdOXNGzZs3d+y78sF+NeHh4ZIuLxj8vnPnzmnPnj3q0KGDy/dkZmaqadOmlfZHRUWpXr168vHx0enTp532/VAtV7Njxw499dRTGjRokB555BHFxsZKkmbMmKGdO3c6HXu1e1GcOnXK8T2EhYXplltucSwO/T4/P/76AX4KuOoBcKFz584qLi6W3W7XzTff7HgdOHBACxYsUHl5uTp06KCgoCD97W9/c3rvP/7xD5fnbd68uerXr1/pmD//+c969NFHdenSJfn4OP9fs127dvL391deXp5TLX5+fpo9e7a+++47BQYGqkOHDtq0aZPTHR0//PDDKn3fn3/+uWw2mx5//HFHSKioqHCMCmw2m+PYnTt36sKFC47ff/nllzpx4oS6du0q6fLP8JtvvlGrVq0cNbdp00ZvvPGGPvjggyrVBcA7iPSACz169FBycrJGjhypkSNHKiEhQbt379b8+fOVkpLiaJ2PHDlSc+fOVXBwsLp27ap//etfPxgUfH199fjjj2vy5MmKjIxUamqqDh8+rPnz52vgwIGqW7euo4PwwQcfqHv37kpISNCQIUM0b948Xbx4UV26dFFeXp7mzZsni8WipKQkSdLYsWP18MMP67HHHtP999+vw4cPa9GiRVX6vtu2bStJmjx5svr376+CggKtWrXKcZllcXGxY3Rjs9n06KOPavjw4Tp37pxmzZqlli1bqm/fvo6fzQMPPKBhw4ZpwIABCgwM1Nq1a7V582bNnz+/SnUB8A6CAuCCj4+PlixZonnz5mnx4sU6c+aMYmNjNXjwYI0aNcpx3LBhwxQSEqLly5dr+fLl6tChg5566im98MILLs89cOBAhYSEKCsrS2vXrlVcXJyGDh2qoUOHSrp8lcEtt9yiWbNmadu2bVqyZIlGjx6t6OhorV69WkuXLlXdunXVrVs3jR07VmFhYZKkTp066bXXXtPs2bP12GOPqVGjRpo6daqGDx9u+Pvu0qWLnn/+eb3++uv629/+pqioKHXp0kWvvPKKRo0apZ07d6pHjx6SpF69eqlBgwbKyMhQeXm5brvtNj333HMKDAyUJCUlJWnVqlWaM2eOnnzySdntdrVs2VILFizQ7bffXtX/JAC8wGLnqTMAAMAF1igAAACXCAoAAMAlggIAAHCJoAAAAFwiKAAAAJcICgAAwCWCAgAAcImgAAAAXCIoAAAAlwgKAADAJYICAABwiaAAAABc+v9fF4B26zNg2wAAAABJRU5ErkJggg=="/>


```python
score = model.evaluate(X_test, y_test, batch_size=128)
score
```

<pre>
20/20 [==============================] - 0s 1ms/step - loss: 0.6800 - accuracy: 0.9395
</pre>
<pre>
[0.6800302267074585, 0.9394904375076294]
</pre>


#### 골프를 희망하는 사람들 분석 (나이, 가족 수 가장 중요)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y_want, 
                                                    test_size=0.3,
                                                    random_state = 13,
                                                    stratify= y_exp)
```

Smote



```python
smote = SMOTE(random_state=13)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
```


```python
params = {
    'max_depth' : [8, 10, 15, 20],
    'n_estimators' : [50, 100, 150],
    'min_samples_leaf' : [8, 12],
    'min_samples_split' : [8, 12]
    }

rf_clf = RandomForestClassifier(random_state=13, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=4, n_jobs=-1)
grid_cv.fit(X_train_over, y_train_over)
```

<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=4, estimator=RandomForestClassifier(n_jobs=-1, random_state=13),

             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [8, 10, 15, 20],
                         &#x27;min_samples_leaf&#x27;: [8, 12],
                         &#x27;min_samples_split&#x27;: [8, 12],
                         &#x27;n_estimators&#x27;: [50, 100, 150]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=4, estimator=RandomForestClassifier(n_jobs=-1, random_state=13),
             n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [8, 10, 15, 20],
                         &#x27;min_samples_leaf&#x27;: [8, 12],
                         &#x27;min_samples_split&#x27;: [8, 12],
                         &#x27;n_estimators&#x27;: [50, 100, 150]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1, random_state=13)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1, random_state=13)</pre></div></div></div></div></div></div></div></div></div></div>



```python
rf_clf_best = grid_cv.best_estimator_
rf_clf_best.fit(X_train_over, y_train_over)
rf_clf_best_pred = rf_clf_best.predict(X_test)
grid_cv.best_params_
```

<pre>
{'max_depth': 15,
 'min_samples_leaf': 8,
 'min_samples_split': 8,
 'n_estimators': 150}
</pre>



```python
accuracy_score(y_test, rf_clf_best_pred)

print_clf_eval(y_test, rf_clf_best_pred)
```

<pre>
==> Confusion matrix
[[2605  121]
 [ 220   68]]
===================
Accuracy : 0.8869, precision : 0.3598
Recall : 0.2361, F1 : 0.2851, AUC: 0.5959
</pre>



```python
best_cols_values = rf_clf_best.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train.columns)
best_cols = best_cols.sort_values(ascending=False)
best_cols
```

<pre>
age                   0.186219
family                0.150802
time_per              0.101217
amount_appropriate    0.098041
avg_wk_workhr         0.091939
research              0.081548
spent_last_year       0.074917
province              0.043174
whom_with             0.043113
edu                   0.035754
cost_per              0.027068
wk_econ_act           0.019960
cur_happiness         0.016269
marrital              0.013599
watch_sports          0.010537
club_participate      0.005683
disabled              0.000161
sex                   0.000000
dtype: float64
</pre>



```python
sns.set(style='whitegrid', color_codes = True)
plt.figure(figsize=(8,8))
sns.barplot(x=best_cols, y=best_cols.index, hue=best_cols.index)
```

<pre>
<Axes: xlabel='None', ylabel='None'>
</pre>


<img src="data:image/png;base64,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EhISMtRTAAAA2C+rLv0yGo0KCwtTp06dlCtXLklS3759NW/ePB0+fFiSlJaWppEjR8rd3V2lS5fWxIkTVatWLbVp00aS1LFjR33//fe6cOGC/vrrL23ZskWfffaZGjRoIEnq06ePDh06pM8++0yNGjXKcG1169ZVWFiYJKl48eJavHix9u7dqzZt2sjLy0uS5Onpaa77UerUqaOaNWuar9wUK1ZMS5YsUWJiYrpxH374oerXry9JmjRpkho0aKCvv/5ar732miSpdOnSGjFihAwGg7y9vZWUlKRFixapR48e5nOEh4erVKlSkqTPP/9cV65cUUxMjLnuTz75RI0aNdLSpUs1cOBASVLu3Lk1ceJEubq6qly5cvrPf/6jMWPG6IMPPjCfNzQ0VGXLlr3v+/Pz81PTpk1Vo0YNSVLRokUVGBhofn9eXl5ycHCQm5ub8uTJo507d2rfvn3atWuX8uTJI+lOyNy7d68WLVqkyMjIDPUVAAAA9smqQcXLy0uhoaFav369EhISdOrUKfOSpLS0NElSvnz50v3rvZubm0qUKGF+7eLiIunO0qO74aZ69erp5qlZs6YmT578WLV5e3une+3h4aFbt2491jn+KTQ0VFu3btWqVat04sQJHT16VGfOnFGZMmXSjQsICDD/nCdPHpUuXTpdmAkICJDBYDC/9vf319y5c3XlyhXztrshRbqztK5UqVLmkCLd6VnlypXTnbdy5cpydXVNd95bt27p+PHjyps3rySpZMmSD3x/rVu31k8//aRJkybpxIkTOnbsmI4fP24OLv924MABmUwmNWzYMN32lJQUJScnP3AeAAAAZA9WDSoXLlxQhw4d5OXlpaCgINWtW1d+fn7mKwqS5OTkdM9xOXI83oo1k8kkR8cHv9XU1NR7tjk7O9/3PJZIS0tTz549deTIEbVo0ULNmjVTxYoVNXTo0HvG/rvO1NTUdO/33/vvBjoHBwfztpw5cz6y5rS0tHTn+nef73feu6HwfoYNG6YNGzaoTZs2CgoK0jvvvKPo6GidP3/+gfO7u7vf9/HG9+s9AAAAsherBpX169fr6tWr2rBhg/mD8t2rIpaEgrvLqn755Zd0/1K/Z88e85Klu/Ncv37dfKXmxIkTjzXPP69oZMTBgwf1ww8/aPny5apSpYok6datWzp16pSKFy+ebmx8fLxq164tSbp8+bJOnjyprl27ptv/T3v37lWxYsXk6el537l9fHy0evVqXbp0Sfny5ZMkJScna//+/eblc9KdKxypqanmYPLrr7/K1dVVpUuXfuRjmK9cuaJly5ZpypQp6R5XfOzYMbm5ud33mHLlyun69eu6detWuuVkQ4YMUfny5fX6668/dE4AAADYN6veTF+4cGEZjUZ99913OnfunGJjYzVgwABJMj9x6nF4e3urYcOGGjlypLZt26bjx49rxowZ2rJli/mm96pVq8pgMCgqKkpnzpzRt99+q1WrVj3WPHc/fCcmJqa70f1B8ufPL0dHR3377bc6ffq04uPj1a9fP124cOGe9zly5Ej9/PPPOnTokN577z0VKFBATZs2Ne/fs2ePpk+frhMnTmjlypVaunSpunfv/sC5W7ZsqTx58qhfv376/fffdejQIb3//vu6ceOGOnToYB539uxZjRw5UklJSdq4caOmT5+u119/Pd1ysAdxd3eXh4eHtmzZopMnT+rw4cMaOnSoDhw4kO795cqVSydOnNDFixdVr149+fr6qn///tq1a5dOnjypcePGmR+QAAAAgOzNqldUmjZtqgMHDigyMlLXr19X0aJF1a5dO23ZskXx8fEqUqTIY59z8uTJmjx5sgYPHqz//ve/KleunKKiovTSSy9JunNj/MiRIzV79mx9/vnnql69ugYOHKgPP/www3PkzZtXr7zyiiZMmKCTJ09qyJAhDx1fqFAhRUZGKioqSkuXLlWBAgXUoEEDdenSRVu3bk03tkOHDho4cKCuXr2qWrVqadGiRenCQnBwsJKSktSqVSsVLFhQERER6tix4wPn9vDw0JIlSxQZGakuXbpIunMPzxdffJHuak7VqlWVI0cOvfrqq/Lw8FBYWJjeeuutDPXDyclJ06ZNU2RkpFq2bClPT08FBARowIABmj17toxGo1xdXdW5c2eNHz9eR44c0dq1azV//nxNnDhR/fr1k9FolLe3t2bMmGG+ogQAAIDsy2Di2wttQlxcnMLCwrRlyxYVK1bsvmM6d+6sokWLZvoTsQYNGqSzZ89q8eLFmXreZ+3usrg/f5+pqxeOWrkaAMgceQqU1UudPrtn+40bN3Tw4EH5+vo+cJktHoz+WY7eWY7e/e/zmp+f3yPHWv0LHwEAAADg36y69MsezJ07V7NmzXromI8++kjt2rV7RhUB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/nr99de1f//+e8YFBQUpKirqkduWL1+uevXqqUqVKurVq5fOnj2b4VpiYmL04osvavny5eZ63nnnHZ0/f9485ty5c+rfv79q166tihUr6sUXX9TEiROVlpZmPsdLL72kjz/+WNWrV9fbb79tXpIWERGhQYMG3bOcy2QyaeHChWrSpIkqV66s5s2ba/369ZLuXfrVuXNnjRkzRgMGDFCVKlX04osvas6cOenCx+bNm9WuXTtVrVpVfn5+atu2rX788UdJUlRUlGbMmKGzZ8/Kx8dHZ86ckSR99dVXCgkJUeXKlRUSEqKFCxea3xMAAACyr2wbVPr166cffvhB48aN0+rVq1W8eHGFh4frv//9r0XnW7x4saZNm6alS5fqypUreueddx7rCsLly5e1cOFCTZ06VQsXLtQff/yh7t276/bt25Kkt956S3/99Zf+7//+T999953Cw8M1b948bd261XyOU6dO6T//+Y9Wr16t/v37KzY2VpL00UcfafDgwffMOW/ePE2ZMkXdu3fX+vXr9dprr2ngwIHatWvXfWv84osv5OHhoZiYGPXv318zZ87U3LlzJUn79+9Xnz591Lx5c61bt07Lly+Xl5eXBg4cqJSUFIWHhys8PFyFCxdWbGysihQpomXLlmnChAnq3bu3vv76a/Xr109z587VpEmTMtw3AAAA2KdsufTr2LFj+uGHHxQdHa26detKkkaMGKHcuXPLzc3NonNOnDhR5cuXlySNHz9eTZo00c6dOxUYGJih42/duqXx48erUqVK5vM1a9ZMO3fuVM2aNdW6dWuFhISoSJEikqQuXbpo7ty5Onz4sBo1amQ+z9tvv63ixYunO7eHh4c8PDx07do187a7V1PCwsLUrl07SXeumty8edMcjv6tdOnSGjFihAwGg7y9vZWUlKRFixapR48ecnBw0NChQxUaGmoeHxYWph49eujSpUsqUqSI3Nzc0i0pmzVrlt566y01b95cklS8eHFdv35dI0eO1LvvvqucOXNmqHcAAACwP9kyqCQmJkqSqlatat6WM2dORUREKCYm5rHPlytXLnNIkaRSpUrJ09NTiYmJGQ4quXLlMocUSfL29jafo169enr99df13Xff6ffff9fJkyd1+PBhXbx48Z5lUqVKlcrQfFeuXNGFCxdUpUqVdNt79OghSealWf8UEBAgg8Fgfu3v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vZ07dyokJETBwcEO7W3bttXhw4fVqFEjNWzYUJLUtWtXHT9+XL/97W+1YMGCGq+rNlQ3dg9DznM+dvcj51UfOc81e6++5zxncUUFlnDvMmphYaFDe2FhoVq3bl2pf5s2bSr1LS0t1fXr19WqVSs1a9ZMfn5+/3O8h43hKVwdu3uysrIUFxenRo0aadu2bXr66acdnnP/C7Yke9wfdPnbitwRO19f30r3IQcGBqqkpEQ3btyoN/tOct/eq6io0O7duzVs2LAqz9ukSRP7C7b0za07HTp00KVLl5xeS22rbuwehpznfOzuIec5FztyXs333uOQ85xFoQJL6Ny5s5588kmHzwQvKirS8ePHFRERUal/RESELl68qHPnztnbDh06JEnq3r27vLy81K1bN3vbPQcPHrTf+/mwMTyFq2MnSf/+97/16quvqmPHjkpLS6uUgN977z0NGjRIxhh723/+8x9du3ZNzz33nEvX506ujp0xRv3799eqVascnpeTk6OWLVvK39+/3uw7yT17T/rm/QDXrl1Tr169Ko2xd+9ehYeHO/zfgrKyMuXm5tbrvfcw5DznYyeR85yNHTmv5ntPejxyntPq9kPHgP+3bNkyExkZaf72t785fDZ5aWmpKSsrM4WFheb27dvGGGMqKipMXFycGTFihDly5IjZv3+/iY6ONomJifbxvvjiCxMUFGQ2bdpk8vPzzeLFi01ISIj9IxIfZQxP4crY3b171wwYMMD069fPnD9/3uEz2//73/8aY4zJyckxwcHBZu7cuebMmTPm0KFDJjY21sTFxZmKioo6i4MzXL3vUlJSTFhYmMP/YwgJCTHbt29/5DE8iavjZ4wxO3fuNMHBwaa8vLzS+YqLi010dLQZO3asycnJMbm5uebtt982ERER5vLly7WyZlepTuy+LTo6utLHnJLznIsdOa9m+46cV7P4GfP45DxnUKjAMsrKysySJUtMz549TVhYmElISDAFBQXGGGMKCgpMYGCgSU9Pt/e/cuWKmTJligkLCzNRUVFm3rx55s6dOw5j7ty50wwYMMB07drVjBgxwuzbt8/h+KOM4QlcGbusrCwTGBhY5SM6Oto+xr59+8yYMWNMWFiYiYyMNElJSQ/8HH0rc/W+u3v3rlm1apXp16+fCQ4ONoMGDbK/YD/qGJ7EHT+3qampplevXg8857lz58yUKVNMZGSkCQ0NNfHx8ebkyZPuWaAbVTd29/tfv/CQ86oXO3JezfYdOa/mP7ePS85zhpcx913HBAAAAAAL4D0qAAAAACyHQgUAAACA5VCoAAAAALAcChUAAAAAlkOhAgAAAMByKFQAAAAAWA6FCgAAAADLoVABAAAAYDkUKgAAuMFPfvITdenSRTk5OVUef+mll5SYmFjLswIAz0GhAgCAm5SXlyspKUmlpaV1PRUA8DgUKgAAuEnjxo2Vl5en1atX1/VUAMDjUKgAAOAmQUFBio2N1YYNG3T06NEH9isvL1daWpqGDRumkJAQ9e3bV7/61a9ks9nsfRITEzVx4kSlp6dr0KBBev755/WjH/1Ie/fudRjrwoULevvttxUZGanQ0FD99Kc/1fHjx922RgBwFwoVAADcaNasWfL39/+ft4DNnTtX7777rvr376+1a9dq3Lhx2rp1qyZPnixjjL3f0aNHtXHjRk2dOlWrV6+Wj4+PpkyZohs3bkiSrl69qri4OB07dkxz5szR0qVLVVFRoXHjxun06dO1sl4AcBUKFQAA3Khp06b6xS9+oVOnTlV5C1h+fr527NihqVOnatq0aXrxxReVkJCgBQsW6Msvv3S4YlJcXKx169YpJiZGP/jBD5SUlKQ7d+7owIEDkqQPPvhA169f16ZNmzRs2DD1799fGzdu1FNPPaXly5fX2poBwBUoVAAAcLOXXnpJw4cP14YNG3Ts2DGHY4cOHZIkxcTEOLTHxMTIx8dHBw8etLc1b95czzzzjP37Nm3aSJJu374tSdq/f7+CgoLUunVrlZWVqaysTN7e3urTp4/27dvnlrUBgLs0qOsJAADwOJg9e7b279+vpKQkpaen29vv3bbVsmVLh/4NGjSQv7+/iouL7W2NGjVy6OPl5SVJqqiokCRdv35d586dU3BwcJVzuH37dqUxAMCqKFQAAKgFTZs21fz58/Xmm29qzZo1Du2SdPnyZbVr187efvfuXV27dk3+/v6PfI7GjRsrMjJSM2fOrPK4r6+vk7MHgNrHrV8AANSS/v37a+jQoUpNTdXVq1clSZGRkZKkjIwMh74ZGRkqLy9X9+7dH3n8yMhIffXVV/re976nrl272h+ffPKJduzYIR8fH9ctBgDcjCsqAADUojlz5ujAgQO6cuWKJOm5557TiBEjtGLFCt2+fVsRERE6ceKEVq1apaioKPXu3fuRx544caI++eQTTZw4UfHx8fL399euXbv0+9//XklJSe5aEgC4BYUKAAC1qFmzZpo/f77eeuste1tycrICAgKUnp6u9evXq1WrVpowYYImT54sb+9Hv/mhdevW2rZtm5YuXar58+fLZrPp2WefVXJyskaPHu2O5QCA23iZ+z+gHQAAAAAsgPeoAAAAALAcChUAAAAAlkOhAgAAAMByKFQAAAAAWA6FCgAAAADLoVABAAAAYDkUKgAAAAAsh0IFAAAAgOVQqAAAAACwHAoVAAAAAJZDoQIAAADAcv4P3Tu8ywAeAcgAAAAASUVORK5CYII="/>

1) 저렴하게 

2) 가족단위 골프 추천!



### Reflection 

