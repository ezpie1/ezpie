---
layout: ../../../layout/CourseLayout.astro
title: "Training the model"
permalink: /courses/machine-learning
---

In this chapter we will cover how to train our model.

In training our model will be able to learn patterns in the dataset.

But first let's solve the tasks I gave you in the last chapter.

The tasks where:

1. Try creating a neural network using the `nn.Parameter` method with 2 neurons.
2. And One with the `nn.Linear` with 2 hidden layer each with 2 neurons.

Let's solve each one of them.

## Task 1 answer

The first task is to create a neural network with `nn.Parameters` and give 2 neurons.

Now as I explained that one neuron contains 1 weight and 1 bais so we will be creating up to 4 parameters... right?

Here's the code

```python
import torch
from torch import nn

class LinearModel(nn.Module): 
  def __init__(self):
    super().__init__()
    self.weight1 = nn.Parameter(torch.rand(1,  dtype=float, requires_grad=True))
    self.bais = nn.Parameter(torch.rand(1, dtype=float, requires_grad=True))
    self.weight2 = nn.Parameter(torch.rand(1,  dtype=float, requires_grad=True))

  def forward(self, x):
    return self.bais + self.weight1 * x + self.weight2 * x
```

Now you maybe seeing that our model uses only 1 bais even though I said each neuron has 1 weight and 1 bais right? Well you see what I meant was(I'm sorry for not explaining it properly) that, each neuron receives weights multiplied by the features and for just additional adjustment we have the bais, so each neuron has 1 bais added which **receives** weights multiplied by features.

![example neural network](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Neural_network_example.svg/1200px-Neural_network_example.svg.png)

As you can see this neural network has 2 inputs(green dots), 1 hidden layer with 4 neurons(blue dots) and 1 output(purple).

In this example each neuron has it's own weight which get's multiplied by the features, then each hidden neuron receiving the value adds that value with the bais and this process repeats tell it reach's the output neuron.

You can also see that each line has a different thickness, that's cause that's the amount of weight each neuron is putting in that neuron.

## Task 2

Task 2 say's that we have to create a neural network with 2 hidden layers each with 2 neurons, now we don't have a particular input layer nor output layer defined, so let's take each input and output layer to have 1 neuron.

Here's the code for that:

```python
class Task2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x))
```

Now in this code what we are doing is really simple.

You see `layer1` takes in 1 input aka the input layer, then it gives an output as 2 which is the first hidden layer with 2 neurons.

The second one, which is the second hidden layer, takes in 2 neurons, which is the number of neurons in the last hidden layer, then it gives an output as 1 aka the output layer.

So I want you to imagine that cause, if I tell you the answer then what are you even trying to learn?

So there are your answers to the tasks in the last chapter, now let's get started with training the model.

## Training a model

So we have our model ready, but if we try predicting with it, it will fail to provide with accurate results.

First let's try checking what exactly or how exactly is our model preforming.

Cause it's a good idea to see how it's doing from scratch.

```python
with torch.inference_mode():
  y_preds = model2(X_test)

y_preds[:5], y_test[:5] # Checking the first 5 values
```

In PyTorch when we are doing predictions there are a few options which we need to disable, cause those options are used by pytorch for the training loop, to do that we can use the `inference_mode()` method.

if you see that the numerical values are also different, but it's always a nice idea, and in fact it's always a good idea, to plot the predictions as it helps us know how well our model is preforming, though we know that for now it has random values so it won't do well, but let's just checkout.

```python
# Ploting the values to check how far are the values
plot(predictions=y_preds)
```

![Random predictions of our model](https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/e0aed162-fff7-44e7-92eb-90a9f9aa50cb)

Ah, looks bad right? But yeah, it's looking good, cause now we know that our model is at least performing something.

Now let's checkout how can we improve our model's performance.

## Improving our model's performance
Right now our model is like a new born child who as no idea what even is this.

For a machine learning model to learn it has to do a little training, just like how a human may do in order to learn something, now you may say that our model has only numerical values to learn from, how is it going to do that? For this our model performs a bunch of mathematical equations, which I will explain one by one.

First let's see what all steps we will tell our model to do.

1. Perform forward pass
2. perform the loss function
3. perform the gradient descent
4. perform backpropagation
5. Update the model parameters

If you see we have all ready seen what forward pass is, now let's discuss the mathematical equations, loss function, gradient descent and backpropagation.

### Loss function

Now in machine learning, a loss function is like a judgmental critic that tells us how well our model is performing. It's our way of measuring the "cost" or "error" of our predictions compared to the actual, ground truth values.

Now, let's put our mathematical glasses and explore the formal definition. A loss function, denoted by L, is a mathematical function that takes in our model's predictions (let's call them y_pred) and the true values (let's call them y_train) and spits out a single number that represents the "loss" or "cost" of our predictions.

Mathematically, a loss function can be denoted as L(y_pred, y_train), where y_pred represents the predicted output generated by our model and y_train represents the true output that we expect. The loss function takes these two values as inputs and produces a single scalar value that represents the "cost" or "error" associated with the predictions.

The specific form of the loss function depends on the problem at hand. For our example a regression tasks where we aim to predict continuous values aka the future! A commonly used loss function is the mean squared error (MSE) which we will use too. It calculates the average of the squared differences between each predicted value and the corresponding true value.

But let's not get lost in the cold, abstract realm of mathematics. Think of a loss function as a referee in a game, where the game is training our model to make accurate predictions. The referee watches our model play, carefully scrutinizing each move, and at the end of each game, blows the whistle and assigns a score based on how well our model performed.

In simpler terms, a loss function tells us how wrong our predictions are. It's like a teacher grading your work—when you make a mistake, your score goes down, but when you get things right, your score goes up. The goal is to find the best set of model parameters that minimizes this loss, or in other words, maximizes our accuracy or performance.

So all a loss function does is it takes the difference of all the predicted values and subtracts them from the true onces and finds the mean/average of them.

### gradient descent

<div class="mt-5 aspect-w-16 aspect-h-9">
    <iframe loading="lazy"
    src="https://www.youtube.com/embed/i62czvwDlsw" frameborder="0" allowfullscreen
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
    </iframe>
</div>

Imagine you're standing at the top of a steep mountain, and your mission is to find the lowest point in the valley below. You can only take small steps, but you're determined to reach the bottom as quickly as possible. How would you navigate this mountainous terrain? Well, that's where gradient descent comes into play!

In machine learning, gradient descent is a popular optimization algorithm used to minimize the value of a given function, typically a loss function. The idea is to iteratively update the parameters of our model in a way that gradually brings us closer to the optimal set of parameters that minimize the loss.

Mathematically speaking, gradient descent exploits the concept of derivatives. Imagine the loss function as a bumpy surface, where each point on the surface represents a specific set of model parameters and the height of the surface represents the value of the loss function. The goal is to find the point on this surface that corresponds to the lowest elevation, indicating the minimum value of the loss.

To do this, we start by randomly initializing the model parameters. Then, we calculate the derivative (gradient) of the loss function with respect to each parameter. This derivative tells us the direction of steepest ascent on the loss surface. However, since we want to go in the opposite direction, we take the negative of the gradient.

With the negative gradient in hand, we update the model parameters by taking a small step in that direction. This step size is known as the learning rate, which determines the size of our steps. We repeat this process iteratively, recalculating the gradient and updating the parameters until we reach a point where the loss function is minimized.

Going back to our mountain analogy, imagine yourself standing on a particular point of the bumpy surface. You check the slope of the terrain around you and take a step in the direction that leads to the steepest descent. Then, you recalculate the slope at your new position and continue taking steps until you reach the lowest point.

The magic of gradient descent lies in its iterative nature. By repeatedly updating the parameters based on the negative gradient, we gradually descend down the loss surface, inching closer to the optimal set of parameters. It's like finding your way down the mountain by repeatedly adjusting your steps based on the steepness of the terrain.

### Backpropagation

<div class="mt-5 aspect-w-16 aspect-h-9">
    <iframe loading="lazy"
    src="https://www.youtube.com/embed/Ilg3gGewQ5U" frameborder="0" allowfullscreen
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture">
    </iframe>
</div>

Imagine you're a detective trying to solve a complex puzzle. You have multiple clues scattered around, and each clue provides a hint towards unraveling the solution. But here's the twist: you need to figure out not only the solution but also how much each clue contributed to it. That's where backpropagation comes in!

In the realm of neural networks, backpropagation is a key algorithm that allows us to efficiently calculate the gradients of the loss function with respect to the parameters of the network. These gradients help us determine the impact of each parameter on the overall error, enabling us to update the parameters and improve the model's performance.

Let's break it down step by step. Imagine you have a neural network with multiple layers, and you're training it to make predictions. During the forward pass, you feed your input data through the network, and it undergoes a series of transformations and computations in each layer, ultimately producing an output prediction.

Now, the difference between this predicted output and the true output gives us a measure of the error. The goal of backpropagation is to distribute this error backward through the network, layer by layer, and attribute a portion of the error to each parameter. This attribution process is known as computing the gradients.

To compute these gradients, backpropagation starts from the final layer and works its way backward. It calculates the partial derivatives of the loss function with respect to the parameters in each layer. By leveraging the chain rule from calculus, these derivatives are computed by multiplying the gradients from the subsequent layers with the local gradients of the current layer.

Think of it as passing the error baton backward. Each layer receives the error signal from the layer above, and then it calculates its own contribution to the overall error and passes it backward to the previous layer. This way, the network "learns" how much each parameter in each layer should be adjusted to minimize the error.

Once we have these gradients, we can use them to update the parameters of the network using an optimization algorithm like gradient descent, as we discussed earlier. The gradients tell us the direction and magnitude of the adjustments needed for each parameter, so we can iteratively refine the model's parameters to improve its performance.

Backpropagation is like being a detective who carefully analyzes each clue, understands its significance in solving the puzzle, and passes on the crucial information to the previous investigators. By efficiently propagating the error signal backward, we can train complex neural networks with many layers, unlocking their full potential in capturing intricate patterns and making accurate predictions.

## Starting with the code

Now that you know what are the main mathematical equations our training loop takes to train the model, let's create the training loop then!

So let's go through each step one by one.

First we have to setup our loss function and gradient descent or optimizer as it's called.

```python
lossFunction = nn.L1Loss()

optimizer = torch.optim.SGD(model2.parameters(), 0.01)
```

As you can see we have set our loss function to L1Loss, now what's L1Loss? Well it's a really weird thing, but it turns out that L1Loss is another way of saying MSE loss/Mean Absolute Error! So yes you have to get use to different naming convection even though they are the same.

Next you can see we have the optimizer setup, and here we are using the SGD optimizer from the optim module.

If you notice we have passed our model's parameters as well, now if you would have read the explanation you may have get it that the optimizer optimizes the model's parameters, which in our case are only 1 weight and 1 bais.

You may be thinking, "then what is 0.01 here?". You see in order to change our model's parameters, since they are numbers, we need a number to change them, know you may say why not go for something like `10` or even `60` rather then `0.01`, but you see the more bigger our changing value, which is known as learning rate, the worst it will be for use cause, imagine you are only .1 away from your destination, but you take 10 at once, well then you will be left out far away from your destination. So all I can say is the perfect value comes from experience.

So let's end the talk and let's really get into the fun part!

Now let's code out the 5 steps we need in order to train our model.

```python
def trainer(epochs=10): # Number of times the model will loop over the training data

  for epoch in range(epochs):
    # Set the model in training mode
    model2.train()

    # 1. preform the forward pass
    y_pred = model2(X_train) # give the model the training features

    # 2. perform the loss function
    train_loss = lossFunction(y_pred, y_train) # give the loss function the predicted values and the actural values

    # 3. Perform the gradent descent
    optimizer.zero_grad()

    # 4. Perform the backpropagation
    train_loss.backward()

    # 5. update the model parameters
    optimizer.step()

    # Testing loop
    # Set the model to testing mode
    model2.eval()

    with torch.inference_mode():
      # 1. perform the forward pass
      test_pred = model2(X_test)

      # 2. perform the loss function
      test_loss = lossFunction(test_pred, y_test)
    
      # Printing data
      if epoch < 30:
        print(f'Epoch: {epoch} | Training loss: {train_loss:.3f} | Testing loss: {test_loss:.3f}')
        
      elif epoch > 30 and epoch < 100:
          if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Training loss: {train_loss:.3f} | Testing loss: {test_loss:.3f}')
          
      elif epoch > 100:
          if epoch % 100 == 0:
            print(f'Epoch: {epoch} | Training loss: {train_loss:.3f} | Testing loss: {test_loss:.3f}')


trainer()
```

Now as you can see I have put our training loop in a function, that's cause what we will be doing later will need us to rewrite the same training loop, so yeah functioning it out.

Now if you see epochs is just another way of saying number of times our model should loop over the dataset to learn patterns.

If you see we have first set our model into training mode with `model2.train()`, this sets our model into training mode and makes sure that it performs, gradient descent, backpropagation and upgrade's the optimizer.

Next we are just doing the usual forward pass.

Then we are calculating the loss aka the MSE.

Afterwards we are performing the gradient descent, nothing like we haven't covered yet.

Then we are performing the backpropagation and finally updating the model parameters.

And for the last part we are testing our model in the testing data.

So in a nutshell, our model at first learns patterns in the data using the training dataset, and then gets tested using the testing dataset.

If you see the output, it's kind of interesting.

```
Epoch: 0 | Training loss: 0.524 | Testing loss: 0.557
Epoch: 1 | Training loss: 0.511 | Testing loss: 0.545
Epoch: 2 | Training loss: 0.498 | Testing loss: 0.533
Epoch: 3 | Training loss: 0.485 | Testing loss: 0.522
Epoch: 4 | Training loss: 0.473 | Testing loss: 0.510
Epoch: 5 | Training loss: 0.460 | Testing loss: 0.498
Epoch: 6 | Training loss: 0.447 | Testing loss: 0.486
Epoch: 7 | Training loss: 0.434 | Testing loss: 0.474
Epoch: 8 | Training loss: 0.422 | Testing loss: 0.463
Epoch: 9 | Training loss: 0.409 | Testing loss: 0.451
```

At the beginning our model is kind of not what we want, cause if you see our loss is kind of hight! Yes, it maybe looking like that it's less then 1, but our output will always be between 1 and 0, so we can say that our model is performing bad, cause the values must be around 0.0**, so how can we do that?

## Improving our model

Now that we know our model is doing good, and why even though the loss if bad? That's cause the loss is also going down with **more** training, so what can we do in order to improve the model? Well you may have guessed, we just **more** the epochs.

For this reason I created the training loop as a function, not as a code cell.

But first let's check has some improvement proven that the training is working or not?

Let's plot our data using our plot function

```python
with torch.inference_mode():
  new_preds = model2(X_test)

plot(predictions=y_preds) # our old predictions
plot(predictions=new_preds) # our new predictions
```

As you can see for comparing I'm plotting both the new and old predictions. Let's see the output

**Old Predictions**

![old predictions of our model](https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/ee3aabbb-4fd3-4c5f-a86a-4073ba4cbecb)

**New predictions**

![New predictions of our model after training](https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/c291ee1c-417f-4667-8ef4-bfca7812e220)

So yes we can say that our model has learned patterns or not at least some patterns in the data, now since our learning rate was also a bit low it didn't reach the actual points, so yeah even changing the learning rate can improve the model predictions.

Now let's try increasing the number of epochs so that our model get's more time to learn from the dataset.

But since we will be redoing it in a new code block you may see that instead of start from where we left, the last time we did a loop.

```python
trainer(1000)
```

You may say that we are looping for a 1000 times, but since we have all ready looped for 10 times, it's actually 1010 times!

```
Epoch: 0 | Training loss: 0.396 | Testing loss: 0.439
Epoch: 1 | Training loss: 0.383 | Testing loss: 0.427
Epoch: 2 | Training loss: 0.370 | Testing loss: 0.415
Epoch: 3 | Training loss: 0.358 | Testing loss: 0.403
Epoch: 4 | Training loss: 0.345 | Testing loss: 0.392
Epoch: 5 | Training loss: 0.332 | Testing loss: 0.380
Epoch: 6 | Training loss: 0.319 | Testing loss: 0.368
Epoch: 7 | Training loss: 0.307 | Testing loss: 0.356
Epoch: 8 | Training loss: 0.294 | Testing loss: 0.344
Epoch: 9 | Training loss: 0.281 | Testing loss: 0.333
....
Epoch: 200 | Training loss: 0.034 | Testing loss: 0.044
Epoch: 300 | Training loss: 0.009 | Testing loss: 0.004
Epoch: 400 | Training loss: 0.009 | Testing loss: 0.004
Epoch: 500 | Training loss: 0.009 | Testing loss: 0.004
Epoch: 600 | Training loss: 0.009 | Testing loss: 0.004
Epoch: 700 | Training loss: 0.009 | Testing loss: 0.004
Epoch: 800 | Training loss: 0.009 | Testing loss: 0.004
Epoch: 900 | Training loss: 0.009 | Testing loss: 0.004
```

So as you can see we have reached our goal that is making the loss reach 0.0**!

Now let's plot our predictions and see how it is.

```python
with torch.inference_mode():
  new_preds = model2(X_test)

plot(predictions=y_preds) # our old predictions
plot(predictions=new_preds) # our new predictions
```

And here we have it!

**Old Predictions**

![old predictions of our model](https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/ee3aabbb-4fd3-4c5f-a86a-4073ba4cbecb)

**New Predictions**

![new predictions of our model after training for 1000 times plus 10](https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/fb8cded4-2054-4e9f-83cf-387ca1775c91)

And look at it! From totally random to almost accurate! Isn't that fun?

And now that our model is so good with this type of data, if we try testing it with totally newer one we can easily do it, and that exactly what we will do in the next chapter.

## Task
Your task for this chapter are

1. Try improving the model with the 2 ways we discussed.

2. Try to make a totally new dataset that continues with this data and check does the model performs just the same way as it did which this one.