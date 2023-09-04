---
layout: ../../../layout/CourseLayout.astro
title: "Understanding non-linearity"
permalink: /courses/machine-learning
---

From this chapter onward we will focus(maybe again) on binary classification, but with a twist.

To be exact we will in fact twist the data set and the model, but the training loop will remain the same. Why? That's cause training is almost same for every kind of data, just that, the model has to be perfect for that dataset.

In this chapter we will create the dataset for the use. Which will look like this.

<!-- Add image -->

![Over view of the dataset](https://user-images.githubusercontent.com/104765117/265205331-abb58ff6-7354-49d8-8eef-40055d67a614.png)

As you can see this image has 2 circles with 2 deferent labels, that is blue and red.

No judging, those the color I find good.

But what's more interesting is that how can you decide wether a dot is blue or red?

That's a question I won't answer. Like any good story, let's keep going through the process and find it out.

## Making the dataset

For this chapter let's just make the dataset and be done with the boring part.

Let's import the required libraries

```python
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

As you can see we are just importing the normal things, but with something new, that's the `from sklearn.datasets import make_circles`. This method will help us make the two circles you show above.

The next thing we need is data, cause no data no predictions.

For the data we will make 2 different circles with the **make_circle** method provided by sklearn.

```python

X, y = make_circles(1000, # simple of dots
                    noise=0.05, # some randomness
                    random_state=23) # the same thing - the dataset should be same even after multiple runs
```

But being humans we love visuls, let's see what we have here.

```python
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
```

For checking the data a bit more like what makes a dot red or blue, let's use pandas **DataFrame** method.

```python
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})


![visulizing the circle dataset]
circles.head() # first five(5) values
```

And like you can see the output:

```
     X1	          X2	       label
0	   0.768261	    0.157294	   1
1	   -0.426287	  -0.642347	   1
2	   -0.889509	  0.482045	   0
3	   0.765902	    0.283268	   1
4	   0.843128	    -0.194731	   1
```

This doesn't makes much since for a human but a lot for a machine.

But being humans we love visuals, let's see what we have here.

```python
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
```

![visualizing the circle dataset](https://user-images.githubusercontent.com/104765117/265205331-abb58ff6-7354-49d8-8eef-40055d67a614.png)

As you can see this is just a crazy dataset with all the values in circular pattern.

And as usual we will convert our data into pytorch data type and split them into training and testing data.

```python
# Convert from numpy to pytorch floating data type
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
```

There's nothing new here, and will never be too. Cause this is just how you convert and split dataset.

And to end this chapter let's just do the simple and I would say an important thing...

Check which device it is.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

So that's all for this chapter, in the next one will will create a model for this dataset and also training it.
