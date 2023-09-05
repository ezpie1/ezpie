---
layout: ../../../layout/CourseLayout.astro
title: "Creating dataset for multi-classification"
permalink: /courses/machine-learning
---

Now that we have **mastered** binary classification.

Now it's time to put our skills... no not to the test, but learn about something a lot more well known.

That is multi-classification.

But before creating any data, let's understand some things in multi classification and itself.

## Understanding multi-classification

Multi-classification involves more then just 2 labels, for our case we will use 5.

In multi-classification we will have a replacement for our old friend sigmoid.

Who will replace him? The answer is softmax.

Let me explain what softmax is.

Like sigmoid for binary we have softmax for multi-classification problems. When we use sigmoid, the values are squeezed between 0 and 1. Meaning that the output, after round off, would be either a 1 or a 0, but in multi-classification, remember that multi means more then 2, our labels are of 5 for examples, so that means we would need to assign specific values to each label or model predicts and show that.

In more simple terms, if our model predicts value like these - `[1.5703, -19.9038, 15.7842, 8.6973, 1.7559]`, you can see they are 5, case that's the number of labels and if we pass this into a softmax function we would get something like this

`[6.7085e-07, 3.1664e-16, 9.9916e-01, 8.3534e-04, 8.0771e-07]`

Now this is just a probability and we won't be rounding them off, we would do something magical.

Just saying add a `.argmax(dim=1)` to the probabilities variable and you get `tensor([2])`.

And as you can see something like this long array would give an output of 2.

Now let's just stop talking and try implementing it in code!

First let's get some raw logits

```python
with torch.inference_mode():
  y_logits = model1(X_test)

y_logits[:1]
```

Then pass those values into a softmax function

```python
y_probs = torch.softmax(y_logits, dim=1)

y_probs[:1]
```

And now just apply the `.argmax` to it

```python
y_preds = y_probs.argmax(dim=1)

y_preds[:5]
```

What does `dim=1` means? Well in simple words it means, **dim=1** as an argument in softmax means that the softmax operation is applied independently to each row in the y_logits tensor, that is this value `[1.5703, -19.9038, 15.7842, 8.6973, 1.7559]`. In `argmax` it means to compute all the values so that the sum of each row equals **1**.

Why 1? That's cause it's maths and maths love's 1. That's how I remembered it and for simplicity, remember that dividing 1 into groups is simple, cause it's 1. **Don't ask why** that's just how it is.

## Making the dataset

OK so we are done with the understanding part, let's make some data and play with it!

First let's import all the packages

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
```

Now let's use the **make_blobs** method from the sklearn library to create that same red blue dataset from the first binary classification chapter, just with a twist.

Now it will have 5 labels!

```python
X, y = make_blobs(n_samples=1000,
                            n_features=2,
                            centers=5,
                            cluster_std=1.5,
                            random_state=23)

print(f'X First 5: {X[:5]} \n y first 5: {y[:5]}')
```

Let's visualize convert to torch data type and split into training and testing

```python
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
```

You may have notice that new **LongTensor data type** up there. You see later on this course we will run into an error which wants the labels to be of this type, so to prevent it from happening let's just add it before hand, but yes do let your curious mind see what that error is and try understanding it. I won't do because errors aren't fun.

Now let's visualize.

```python
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
```

Now I won't show the image so you have to run this code yourself and stop reading this course if you're not coding along.

Also try setting up device agnostic by yourself.

That's it for this chapter in the next one we will make the model and the training loop. But we will come up with a question while making the model, so stay ready.
