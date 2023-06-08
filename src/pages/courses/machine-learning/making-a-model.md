---
layout: ../../../layout/CourseLayout.astro
title: "Making a PyTorch model"
permalink: /courses/machine-learning
---

In the last chapter, we created a really simple straight line, saying mathematically, a linear line with the help of a linear regression formula.

Now we will see how we can make a PyTorch model that can learn patterns in that simple line and predict the values we will provide it.

## Creating a Model. The important points

In Creating a model there are a few major points you must know.

1. To create a model we must subclass the `nn.Module` class into our model.
2. We must know how many hidden layers we will need. This is done with experiments.
3. We must also know how many neurons will be required. ALso with experiments.
4. We must define what the forward pass should be.

These are just the simple points you must know and just to tell you these is not even what all pytorch models need, this is just the basic thing ever pytorch model has in it. These points are more all suitable for our straight line aka linear line.

## Creating the Model with code.

Now that we know what all our model needs, let's write it down shall we.

First of all we need to sub class the `nn.Module` class into our main model class. Now nn.Module is the main source from where we get all the important parts our model requires, such as the forward pass.

```python
# using nn class from pytorch
from torch import nn # already imported in the last chapter

# One way of making a neural network, simplest define the parameters yourself
class LinearModel(nn.Module): # Base class of building neural network(nn gives the name)
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.rand(0,  dtype=float, requires_grad=True))
    self.bais = nn.Parameter(torch.rand(0, dtype=float, requires_grad=True))

  def forward(self, x):
    return self.bais + (self.weight * x)
```

Now let's discuss what this code is doing.

First we are importing the nn class from torch, which <a href="preparing-data" class="text-blue-600 hover:text-blue-400 hover:underline">we already imported in the last chapter</a>. Next we have the main class which is subclassed from nn.Module, covering the first point as mentioned.

Next you can see we are creating the 2 values which we used to create the linear line, the weight and bais. Now if you see we are using something called `nn.Parameter` here in order to create the 2 values.

In a neural network these 2 are the most important thing, <a href="preparing-data" class="text-blue-600 hover:text-blue-400 hover:underline">we have already covered them in the last chapter</a>

Next if you notice in the forward method, we are doing something similar... Hmmm, can you recall? Yes, we are performing a linear regression formula in the forward method, so from this we can say that our model just adjust the weight and bais in order to determine which adjustment makes the regression formula same as the values in the dataset! Magic right?

Now you may have notice that the comment on top of the class say's one way of making a neural network. You maybe thinking that why? Is there another way of doing this? Well yes there is any in fact that way is the most common way and the most of the time used.

```python
# Other most common way to make a model
class LinearModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(1, 1)

  def forward(self, x):
    return self.layer1(x)
```

Now this code is even smaller and also more precise of what it's doing, like you can just make out that the layer1 variable holds a simple layer that takes an input and gives an output. Yeah let me explain what this is.

In neural network there can be more then just one neuron or even more then one hidden layer, what's a hidden layer? Well it's just an extra layer that can hold more neurons within it, we can add say 2 hidden layers with 5 neurons each or we can add 1 hidden layer with 10 neurons, the output may differ some times with some problems.

So like I said, some neural networks can have more then one neuron, and since each neuron has 1 weight and 1 bais, I don't even know how many weights and biases can 168 neurons have! And no joke writing 336 lines of code just to tell the model how many weights and biases it needs, is more then unnecessary!

So to simple it out we use the Linear method from nn, now you may be thinking what are those 1 values in passed to the Linear method? You see the number of features we have in our data is just 1 that is the x axis values, so that's what the first 1 is for, the second 1 is for the output, that means the y label which is just 1, so this model just has 1 input layer and 1 output layer, with no hidden layer in between, that's the reason why we have 1 input and 1 output, but later on the course we will create neural networks with more then just 0 hidden layers with mor then 1 input and output, so just hold on tight.

## Creating the instance

Now that we have a model, let's get used to it as well, just like our data, let's create an instance of the model, but in this part I will focus more on the `LinearModelV2`, cause it's fun... no reason at all, but you can do that same with the `LinearModel` class also.

```python
torch.manual_seed(32)

# Creating the instance
model2 = LinearModelV2()
print(model2)

# Checking out the parameters
list(model2.parameters())
```

Now if you see I have also setup a `manual_seed` for the model, now what's manual seed? Well it's the same as the `random_state` <a href="preparing-data" class="text-blue-600 hover:text-blue-400 hover:underline">which we talked about in the last chapter.</a> So yeah, just trying to make sure that the random parameters aka our weight and bais remain the same as long as the manual seed is the same.

Now let's see what the output is given to us as:

```
LinearModelV2(
  (layer1): Linear(in_features=1, out_features=1, bias=True)
)
[Parameter containing:
 tensor([[0.7513]], requires_grad=True),
 Parameter containing:
 tensor([-0.4559], requires_grad=True)]
```

Just as I said, the in features is 1, the x axis/feature, the out features is 1, the y axis/label.

Now if you also tried the `LinearModel` class and tried printing this, you may have seen what the tensors where, nothing, that is because we initialized our parameters as 0, so no random value was selected for them, this doesn't really effects the model, cause it just starts with a random value, so weather you start with a 0 or a random value, it's the same.

## What we have covered.

So till now what have we covered.

![most common pytorch workflow](/images/courses/ml/pytorch-workflow.svg)

So till now we have covered, preparing and loading our data, and creating a model, which is not complete as you can see we haven't pick a loss function yet, which we will cover in the next chapter.

## Task

Your task for today is yet a bit simple.

Create two neural network, one with the `nn.Linear` method and one with `nn.Parameter`

1. Try creating a neural network using the `nn.Parameter` method with 2 neurons.
2. And One with the `nn.Linear` with 2 hidden layer each with 2 neurons.

Don't worry if you can't solve the second one as we haven't covered creating that long neural network, but I will give you the answers tomorrow, and you can also <a href="https://github.com/ezpieco/pytorch-crash-course" class="text-blue-600 hover:text-blue-400 hover:underline">visit the github repository and star it</a> so that you stay updated with the latest stuff.
