---

layout: single
title: '[1] Javascript Syntax'
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
## Boolean Operators

Operators are quite similar to those in python. 

In javascript, the '==' operator does not compare data types. In order to compare data types we must use the '===' operator. 

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
> !!'ab';
< true

> !!false 
< false 
```

When we run the !! operator on **false, ' '(empty string), 0, Nan,** the boolean result is false. Later on we'll also learn that **undefined & null** also return false. 

**Question One ** 

```javascript
> 5 + 4 * 3 === 27;
```

Make the above statement true. 

```javascript
> (5 + 4) * 3 === 27;
< true

> !(5 + 4 * 3 === 27);
< true
```



## Undefined and Null Data Types 

If there is nothing to return, javascript returns undefined.

```javascript
> undefined == null; 
< true

> undefined === null;
< false 
```



## Variables 

We can announce variables and save data. When we declare variables javascript returns undefined. 

```javascript 
> let total = 5000 + 8000 + 10000 + 9000 ;
< undefined

> total;
< 32000
```

If we don't assign a value to a variable, javascript will return undefined when the variable is called. 

```javascript
> let empty; 
< undefined 

> empty;
< undefined
```

In javascript we can use symbols like **$, _ ,** etc., as variable names 



### Changing Variables 

```javascript
> let change = '바꿔 봐'; 
< undefined 
> change = '바꿨다';
< '바꿨다'
```

javascript returns the assigned change as an output when a variable is changed.



## Constants 

Like the name suggests constants are immutable. 

```javascript
> const value1 = '12345'
< undefined
> value1 = 'chocolate sundae'
< Uncaught TypeError: Assignement to constant variable....
```

In the past variable was announced using **var not let**

**Question Two **

a and b are variables that contain certain values. Change the values of the two variables. 

```javascript
> let a = '123';
< undefined 

> let b = '456';
< undefined 

> let c = a; 
< undefined 

> a = b
< '456'

> b = c
< '123'
```



## Conditional Statements 

### if statements

> if (a logged in user);
>
> show information; 

```javascript
if (true) {
    console.log('Hello, if!');
    console.log('Hello, again!');
}
```

```javascript
if (0) {
    console.log('Hello, if!');
}
```

{} is not a must but it is a nice way to organize your codes

### else, else  if, switch 

> if (condition) {
>
> activation;
>
> } else if{ 
>
> activation;
>
> }
>
> else if { 
>
> activation; 
>
> }

```javascript
const score = 90; 
	if (score >= 90) { 
    	console.log('A+')
    } else if (score < 90 && score >= 80) {
        console.log('A')
    } else if (score < 80 && score >= 70) {
        console.log('B+')
    } else if (score < 70 && score >= 70) {
        console.log('B')
    } else {
        console.log('F')
    }
```

**javascript  does not support 80 < score < 90**

We must also prevent callback hell by indenting properly 

``` javascript
let first = true; 
let second = false;
if (first && second) { 
	console.log('the first condition has been satisfied!');
    console.log('the second condition has been satifised!');
} else if (first) {
    console.log('the first condition has been satisfied!');
    console.log('the second condition has not been satisfied!');
} else { 
	console.log('the first condition has not been satisfied')
}
```

A switch statement also runs if the conditions are satisfied but can be different from an if statement. 

``` javascript
let value = 'A'; 
switch (value) { 
	case 'A';
        console.log('A');
    case 'B': 
        console.log('B');
    case 'C':
        console.log('C'); 
}
> A
> B 
> C
```

If the condition == case the activation runs. The difference is that unlike else if statements all the cases beneath the one that has been satisfied runs. 

**USE {} to prevent errors!!! **

```javascript
let value = 'A'; 
switch (value) { 
	case 'A';
        console.log('A');
        break
    case 'B': 
        console.log('B');
        break
    case 'C':
        console.log('C');
        break
    default:
        console.log('do nothing!');
}
> A
```

## Conditional Operators 

```javascript
> 5 > 0 ? '참입니다' : '거짓입니다'
< "참입니다"
```

We use conditional operators to assign variables based on conditions.

```javascript
let condition = true; 
let value = condition ?'참' : '거짓'; 
console.log(value)

if (condition) {
    value = '참'
} else { 
	value = '거짓';
}
console.log(value);
```

```javascript 
> let condition1 = true; 
  let condition2 = false; 
  let value = condition1 ? (condition2 ? '둘 다 참' : 'condition1만 참') : 'condition1이 거짓'; 
  console.log(value)
< condition1만 참
```

``` javascript 
> let condition1 = false;
  let condition2 = true; 
  let value = condition1 ? 'condition1이 참' : conditions2 ? 'condition2 가 참' : '둘 다 거짓';
  console.log(value);
```

## While Loop

> while (condition); 
>
>   activation; 

```javascript 
> let i = 0; 
  while (i < 100) {
      console.log('Hello, while!');
      i++; 
  }
```

i +=1 can be expressed as i++. They can be different! 

## For Loop 

> for (start; condition; termination) {
>
> ​	activation;
>
> }

``` javascript 
> for (let i = 0; i < 100; i++) {
    console.log('Hello, for!'); 
}
```

**Question** 

Write a code that prints from 1 to 100 using a for loop and a while loop

```javascript
> for (let i = 0; i < 100; i++) { 
	console.log(i + 1);
}
```

```javascript
> let i = 0; 
  while (i < 100) {
      console.log(i + 1)
      i++;
  }
```

## Break and Continue 

```javascript 
> let i = 0; 
  while (true) {
      if (i === 5) {break};
      i++;
  }
console.log(1); 
```

``` javascript
> let i = 0; 
  while (i < 10) {
      i += 1;
      if (i % 2 === 0) {
          continue;
      }
      console.log(i);
  }
```

continue skips the current loop then goes to the next

``` javascript
> for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) {
        console.log(i, j);
    }
}
```

**Question**

Print the multiplication table up to 9 without an even numbers 

```javascript
for (let i = 1; i < 10; i ++ ) { 
	for (let j = 1; i < 10; i++) {

		if (i % 2 === 0 || j % 2 === 0) {continue;}
		console.log(i, '*', j, '=', i * j)
 }
}
```

``` javascript
for (let i = 0; i < 5; i++) {
    console.log('*'.repeat(i + 1))
}

< *
< **
< ***
< ****
< *****
```

```javascript
for (let i = 5; i >= 1; i -=1) {
    console.log('*'.repeat(i))
}

< *****
< ****
< ***
< **
< *
```

## Objects

### Arrays 

```javascript
const apple = '사과';
const orange = '오렌지';
const pear = '배';
const strawberry = '딸기';

const fruits = ['사과', '오렌지', '배', '딸기']

console.log(fruits[0]);
console.log(fruits[1]); //etc
```

```javascript
const arrayOfArrays = [[1, 2, 3], [4, 5]];
arrayOfArrays[0]; // [1, 2, 3]
const a = 10;
const b = 20; 
const variableArray = [a, b, 30];
variableArray[1]; //20
```

arrays allow redundancy

```javascript
const everything = ['사과', 1, undefined, true, '배열', null]; 
console.log(everything.length)
```

we can utilize the length function to find the last element of an array 

```javascript 
const findLastElement = ['a', 'b', 'c', 'd', 'e'];
console.log(findLastElement[findLastElement.length -1]);
```

**adding a value to the array from the left ** 

```javascript 
> const target = ['b', 'c', 'd', 'e', 'f']; 
  target.unshift('a');
  console.log(target);

< ['a', 'b', 'c', 'd', 'e', 'f']
```

**we can modify elements of the array but we CANNOT modify the array itself** 

* push : adds an element from the right then returns the length of the changed array 
* pop : removes the last element of the array then returns that last element 
* unshift : adds a new element from the left then returns the length of the changed array 
* shift : removes the first element then returns that element 

```javascript
var arr = [1, 2, 3];

arr.pop();  // 3

arr.push("new");  // 3
console.log(arr);  //-> [ 1, 2, 'new' ]

arr.shift();  // 1

arr.unshift("new");  // 3
console.log(arr);  //-> [ 'new', 2, 'new' ]
```

```javascript
const target = ['가', '나', '다', '라', '마']; 
target.splice(1, 3, '타', '파'); //removes '나', '다', '라', then adds '타', '파'
```

**we can use splice to append a new element into a specific index position ** 

```javascript 
const arr = ['가', '나', '다', '라', '마'];
arr.split(1, 0, '바')
```

**Checking whether an array contains a value ** 

```javascript 
> const target = ['가', '나', '다', '라', '마'];
  const result = target.includes('다'); //returns true
  const result2 = target.includes('카'); //returns false
```

``` javascript
> target.indexOf('가')
< 0
```

**printing all elements in an array ** 

```javascript
const target = ['가', '나', '다', '라','마']; 
let i = 0; 
while (i < target.length) { 
	console.log(target[i]); 
    i++; 
}
```

This method also works for strings! Just like python. 

```javascript 
for (let i = 0; i < arr.length; i++) {
    console.log(arr[i]);
}
```

removing a redundant element from an array 

```javascript 
const arr = ['가','라','나','다','라','마', '라'];

while (arr.includes('라')) {          //or while (arr.indexOf('라') != -1)
    let i = arr.indexOf('라');
    arr.splice(i,1);
}
```

**0 in an if condition is always considered false!!! ** 

```javascript
> const arr = [1, 2, 3, 4, 5]; 
> arr.indexOf(1)
< 0 

if (arr.indexOf(1)) {
    console.log('1 has been found');
} else { 
  console.log('1 has not been found');
}
< 1 has not been found
```

instead we have to do the following 

```javascript
if (arr.indexOf(1) > -1) {
    console.log('1 has been found');
} else { 
  console.log('1 has not been found');
}
< 1 has been found 
```



## Function 

> function a() {}
>
> const b = function() {};
>
> const c = () => {}; 

```javascript
> function a() {
    console.log('Hello');
    console.log('Function');
}
> a();
< Hello
< Function
```

```javascript 
> function b() {
    return '반환값';
}

> b()
< '반환값'
```

```javascript
> function c() {
    return 'hello';
    console.log('hi');
}
> c()
< 'hello'
```

return ends the function 

**returning multiple values ** is done with arrays

## Parameters and Arguments

```javascript
> function a(parameter) {
    console.log(parameter);
}
a('argument'); 
```

within a function we can call arguments which returns an array consisting of all the arguments 

```javascript
> function a(w, x, y, z) {
    console.log(w, x, y, z);
    console.log(arguments);
}
a('Hello', 'Parameter', 'Argument');

< Hello Parameter Argument undefined
< Arguments(3) ['Hello', 'Parameter', 'Argument']
```

**Question **

make an arrow function that returns the multiplication of the parameters 

```javascrip
const f = (x, y, z) => {
	return x * y * z 
}
```

## Object Literals

we use object literals to group variables together. This allows us to reuse variable names. 

![image-20240114205030828](/images/2024-01-13-javascript1/image-20240114205030828.png)

![image-20240114205044883](/images/2024-01-13-javascript1/image-20240114205044883.png)

``` javascript
const sengmin = {
    name: '윤성민', 
    year: 1994, 
    month: 8, 
    date: 12, 
    gender: 'M'
};
console.log(sengmin.name);
console.log(sengmin['name']);
console.log(sengmin.date);
console.log(sengmin['date']);
```

``` javascript
> const obj = {
    bc: 'hello',
    '2ca': 'hello1', 
    'c a': 'hello2',
    'c-a': 'hello3',
};

obj.bc; //hello
obj.2ca //error
obj['2ca'];
```

**if we wrap a key in apostrophes, we must call them using []**

we can add or modify properties by doing the following 

```javascript
> sengmin.gender = 'F';
```

```javascript
> delete sengmin.gender;
```

### Why Arrays and Functions are Objects

The reason arrays and functions are objects is because both arrays and functions have all the qualities of an object. We can add attributes to or remove attributes from functions and arrays. 

``` javascript
> function hello() {}
  hello.a = 'really?';
  const array = [];
  array.b = 'wow';
  console.log(hello.a);
  console.log(array.b);
```

``` javascript 
const debug = {
    log: function(value) {
        console.log(value);
    },
};
debug.log('Hello, Method');
```

### Comparing objects 

> {} === {}

returns false 

numbers, strings, bool, null, undefined which are not objects return true when compared. 

```javascript 
'str' === 'str';
123 === 123;
false === false; 
null === null; 
```

In order to compare objects we need to assign them to a variable 

``` javascript
const a = {name: 'sengmin'};
const array = [1, 2, a];
console.log(a === array[2]);
```

```javascript 
> const a = {name, 'sengmin'};
  const array = [1, 2, a]; 
  array === [1, 2, a]; 
< false
```

The arrays are not the same as a new object was created in the above process 

![image-20240114212152638](/images/2024-01-13-javascript1/image-20240114212152638.png)

**for non objects ** 

![image-20240114212351525](/images/2024-01-13-javascript1/image-20240114212351525.png)

```javascript 
const sengmin = {
    name: {
        first: '성민', 
        last: '윤',
    },
    gender: 'm',
};

sengmin.name.last; 
sengmin['name']['last'];
```

it might be convenient to think that there are separate memory storages for objects and non objects
