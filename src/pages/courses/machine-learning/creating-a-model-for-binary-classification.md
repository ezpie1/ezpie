---
layout: ../../../layout/CourseLayout.astro
title: "Introduction of PyTorch Machine Learning Crash Course"
permalink: /courses/machine-learning
---

Now that we have some data let's make a model that can learn patterns and predict in this data.

Before we do that we need to cover one small thing which is visualizing with an online tool known as <a href="https://playground.tensorflow.org" class="text-blue-600 hover:text-blue-400 hover:underline">TensorFlow Playground</a>

## What's tensorflow playground?

Tensorflow playground is a web version of tensorflow, where you can train a model to fit the data.

The fun part about it is it can be used to understand what will be better to use in creating a model.

So head over to tensorflow and let's try training a model that can fit the simple dataset we have.

<img src="https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/70a8f3e8-b33c-4a25-b82c-52fa75b73cf5" loading="lazy">

As you can see this is the closest representation of our dataset.

Here we haven't really covered a few things, but few we already know, such as the **epoch**, **Learning rate**, **Problem type**.

So let's remove all the hidden layers so that we can have a good experimentation.

<img src="https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/934537b0-eb77-4cdd-9819-c51b00aa5fa9" loading="lazy">

Since this is a really simple and easy dataset, cause all we have to do is draw a line between them, the model will learn it in less this 10 - 15 epochs.

Later on the course we will be trying something complex so to demonstrate that we will be using this thing.

## Creating the model

Now that we know about a tool we can use to understand what our model may need, let's create the model.

Now since you saw that the model learned pattern in just 10 - 15 epochs with no hidden layers and just the 2 features, let's try doing that shall we.

```python
from torch import nn

class BlobClassification(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(2, 2)

  def forward(self, x):
    return self.layer1(x)

model1 = BlobClassification().to(device)
model1
```

In the next chapter we will see if this works or not.

For now let's go with this, but there's just one last surprise for you developer...

Let's recreate the model, but with a lot simpler way!

```python
model2 = nn.Sequential(
    nn.Linear(2, 2)
).to(device)

model2
```

If you see we can create a model with the `nn.Sequential` module.

I know you may be saying, "Ezpie! Why didn't you told this before!". You see this is not always that good, cause in creating a class it gives us the chance to update the forward pass the way we like and upgrade and flip-flop it any way we like.