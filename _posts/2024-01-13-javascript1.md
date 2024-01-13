---

layout: single
title: 'Javascript Basics'
categories: Javascript
tag: [javascript, blog, webdev]
toc: true
author_profile: false
sidebar:
    nav: 'docs'
search: false
sidebar:
    nav: "counts"
use_math: true
---

<div class ="notice--success">
Let's start exploring web development. 
</div>

Operators are quite similar to those in python. 

In javascript, the '==' operatore does not compare data type. In order to compare data types we must use the '===' operator. 

```javascript
> '3' == 3
< true

> '3' === 3
< false
```



```javascript
> 10 > 5 && 6 < 8; 
< true

> 10 < 5 || 6 < 8; 
< true
```

The ! operator returns the opposite logical operator. 

```javascript
> !true; 
< false 

> !false;
< true
```

We can use !! 

```javascript
> !!'ab'
< true

> !!false 
< false 
```

When we run the !! operator on **false, ' '(empty string), 0, Nan,** the boolean result is false. Later on we'll also learn that **undefined & null** also return false. 

