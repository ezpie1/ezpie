---
layout: ../../../layout/CourseLayout.astro
title: "PyTorch And Numpy"
permalink: /courses/machine-learning
---

So now it's time to add up some maths... no not really, but instead we will look at one of the most important python library for working with machine learning, that is numpy.

NumPy stands for "Numerical Python," and it's a popular library in the Python programming language. It provides a set of tools and functions that make it easier to work with numerical data in an efficient and convenient way.

With NumPy, you can create and manipulate arrays, which are like containers that hold numbers. These arrays can have one dimension (like a vector), two dimensions (like a matrix), or even more dimensions. Arrays in NumPy are particularly useful because they allow you to perform operations on multiple numbers at once, saving you time and effort.

For example, let's say you have a dataset containing the heights of a group of people. You can use NumPy to create an array that holds all those heights. Then, you can easily perform operations on that array, like calculating the average height, finding the tallest person, or even doing more complex mathematical computations.

And because of this PyTorch has features just to work with numpy arrays

`starting point as numpy array` -> `needed as pytorch tensor` - use `torch.from_numpy(ndarray)`

When I started learning machine learning, I learned numpy first then pytorch, and I would also recommend you that spend sometime learning numpy, it's just really helpful, cause when we start with our data we don't start with tensors, we start with numpy arrays, then we convert those data into pytorch tensors.

## Converting numpy array to tensor

First let's import torch and numpy

```python
import torch
import numpy as np
```

Did you notice that we are importing numpy as np? Why we are doing this? That's cause that's what the python community prefers calling numpy. Just a fancy, short name for better referencing the library's functions and objects throughout the codebase.

Now let's create a numpy array and convert it into a tensor

```python
arr = np.arange(1., 10.) # points for making float data type
tensor = torch.from_numpy(arr)

arr, tensor
```

The given output looks like this:

```
(array([1., 2., 3., 4., 5., 6., 7., 8., 9.]),

tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=torch.float64))
```

If you see the tensor is of float64 data type, which in fact is the default data type of numpy, but it doesn't means it's the default for pytorch, pytorch's default data type is float32, or basically anything with 32-bit, so now we have a problem, how do we convert the data type of this tensor from float64 to float32?

For this pytorch comes with a useful function `type()`, and it just simplifies the process, and if that doesn't makes you want to learn pytorch, I don't know what will.

```python
arr = np.arange(1., 10.) # points for making float data type
tensor = torch.from_numpy(arr).type(torch.float32) # default type is float32

arr, tensor
```

And there you go, a numpy array converted into a tensor of type float32 rather then the default numpy type float64.

## Converting from tensor into numpy array

Another thing we could do with pytorch is convert a tensor into numpy array, and why do you want to do that? Well let's just say, for fun! I mean why not? Just kidding, there are several reasons why you might want to convert a pytorch tensor into a numpy array, and here are some:

1. **Compatibility with other libraries:** numpy is a widely used library in the Python ecosystem, and many other libraries for data manipulation, visualization, and analysis are built on top of numpy. By converting a pytorch tensor into a numpy array, you can seamlessly integrate it with other libraries and leverage their functionalities.

2. **Visualization and plotting:** Matplotlib provides a rich set of tools for visualizing and plotting data, but it's build on top of numpy, so in order to work with matplotlib you must Convert a pytorch tensor to a numpy array first allowing you to take advantage of matplotlib's plotting capabilities to visualize your data, inspect model predictions, or analyze the results of your machine learning models.

3. **CPU operations:** pytorch tensors are primarily designed to work efficiently on GPUs, which are optimized for parallel computations. However, there might be situations where you need to perform operations on the CPU or use libraries that only support numpy arrays. In such cases, converting the pytorch tensor to a numpy array allows you to perform computations on the CPU or interface with other CPU-based libraries.

Now let's try converting a tensor into a numpy array.

```python
tensor = torch.arange(1., 10.)
arr = tensor.numpy()

tensor, arr
```

And there you go, a simple tensor converted into a numpy array!

# Tasks

You tasks for today are as following:

1. Make a tensor data type and convert it into a numpy array, use matplotlib to plot the numpy array.

2. Make a numpy array and convert it into a tensor, change the values of the numpy array and see if it changes the value of the tensor.

<p class="text-white bg-blue-400 p-2 rounded-md">
<b>NOTE:</b> Make sure your tensors are of float32 data type, not float64
</p>
