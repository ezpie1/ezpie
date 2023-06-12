---
layout: ../../../layout/CourseLayout.astro
title: "Creating a dataset for binary classification"
permalink: /courses/machine-learning
---

So now we are getting in to the real stuff.

We can create a model that can predict the future so now let's create a model that is more on complex then predicting the future.

Let's teach a computer which dot belong's to which category.

## Creating the dataset

For this chapter let's just start simple, let's create a dataset and study our dataset for once.

First let's import our dependencies

```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

torch.__version__ # Just to check
```

So for just checking we are displaying what version we are using. And for today(and for this year, let's hope not) we are using pytorch version 2.0.1

Nothing like we haven't covered yet right, let's continue.

No in order to create a dataset of 2 different values or more in ML style, classes, we will use the `sklearn.datasets` module and use it's `make_blobs` method.

```python
#@title Create data of two classes(two different values)
X, y = make_blobs(n_samples=1000, n_features=2,
                            centers=2, 
                            cluster_std=1.5,
                            random_state=42)
```

As you can see we are just creating a data set with 1000 samples, now why that many? Well cause for learning we should prefer taking a big dataset rather then a small one, cause small is small amount of data and no one likes small things.

Next we are defining the number of features our dataset should have and we have them as 2. That's cause we are creating a 2D dataset, meaning that it will have an `X` axis and a `y` axis.

But big datasets can also have a number of features from 20 to even 60 features.

Next all we are doing is telling the `make_blobs` method to add a little randomness to our dataset so that it's not to closed and not to easy to understand.

Now again the same old `random_state`, don't even need to explain what that is.

OK so we have the data let's convert it to torch datatype and let's split them into training and testing datasets.

First in order to convert them let's use the `torch.from_numpy` method which I mentioned in 
<a href="pytorch-and-numpy" class="text-blue-600 hover:text-blue-400 hover:underline">
the PyTorch and Numpy chapter</a>

```python
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
```

Nothing new here so let's go ahead and split our data into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

So again nothing new let's continue with it.

Now maybe it's time to look into our data.

Let's plot out how our data set looks like

```python
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], # first feature - x1
            X[:, 1], # second feature - x2
            c=y, cmap=plt.cm.RdYlBu) # color the output
```

And there we have it, a simple dataset on which we will create a pytorch model and train it to predict weather a given value or dot in this graph is red or blue.

![a simple dataset of two different options](https://github.com/EzpieCo/ezpie/assets/104765117/6c79b26a-a65e-46b1-8fbd-015a192b36a0)

Now let's finish this chapter by setting up device agnostic and put and end to this chapter.

```python
device = "cuda" if torch.cuda.is_available() else 'cpu'

device
```

There we have it our data is created and now all we have to do is create a model that can learn patterns in this data.