---
layout: ../../../layout/CourseLayout.astro
title: "Creating a training loop for binary classification"
permalink: /courses/machine-learning
---

Now that we have a dataset and a model, we can learn patterns in the dataset with it.

For this we will learn about few things in classification problems:

- Sigmoid activation
- Logits
- And a little bit of maths(hope you like this)

## Predicting how wrong the models is

First thing, we need to see how much wrong our model is by predicting some values.

For this let's try doing what we did in the <a href="/courses/machine-learning/training-the-model" class="text-blue-600 hover:text-blue-500">linear regression chapter</a>.

```python
# Put things into device <- real important

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
```

```python
with torch.inference_mode():
  preds = model1(X_test)

print(f'predictions: {preds[:5]} \n labels: {y_test[:5]}')
```

You may be thinking that this will work, but what if I tell you it won't?

Yes it won't work, here's the output:

```
predictions: tensor([[11.6267],
        [-6.0835],
        [ 4.4050],
        [11.9127],
        [-8.0395]])
labels: tensor([1., 0., 1., 1., 0.])
```

Not only is the predictions a matrix but also they aren't even 1s and 0s!

What could have gone wrong?

You may be able to find the answer to the matrix one with... Yes squeeze() method! Let's try that

```python
with torch.inference_mode():
  preds = model1(X_test).squeeze()

print(f'predictions: {preds[:5]} \n labels: {y_test[:5]}')
```

And yes just a vector now

```
predictions: tensor([11.6267, -6.0835,  4.4050, 11.9127, -8.0395])
 labels: tensor([1., 0., 1., 1., 0.])
```

But still what about the 1s and 0s?

For this we need to discuss about some maths!

### The maths of binary classification

So in binary classification there's on thing called the sigmoid activation function. What exactly is it?

In Machine learning, as per me, the sigmoid activation function is really important, only in case of binary classification. The sigmoid activation function is something called a non-linear activation function. By non-linear I mean that sigmoid turns a value into other value and if you put that in a graph it will look like this:

<div align="center">
  <img src="https://pytorch.org/docs/stable/_images/Sigmoid.png" alt="image of sigmoid activation function in a graphical point">
  <p class="italic">From pytorch docs</p>
</div>

As you can make out, if you give the sigmoid function a value of 0, it will turn it into 1 or something like that.

At its core, the sigmoid activation function uses a simple yet powerful mathematical expression.

<div align="center">
<img src="https://149695847.v2.pressablecdn.com/wp-content/uploads/2018/01/sigmoid-equation.png" alt="Sigmoid activation formula">
<p class="italic">From analyticsindiamag.com</p>
</div>

Let's brake it down shall we:

- **x** is the number predicted, let's take the value _11.6267_ from our model's prediction.
- If you have studied calculus you may recall that **e** is known as Euler's number, which is around 2.71828

Rest is simple, you take the value x(11.6267) into the function.

So the formula says, 1 plus **e** raise to the power **-x**, and divide 1 from the sum.

Let's try this out with python. First you give it a try and then look at the answer given below.

Done? Ok here's the answer

```python
sigmoid = lambda x: 1 / (1 + 2.71828 ** -x)

sigmoid(11.6267)
```

And guess what the output was - _0.9999910754183369_

OK! It's between 0 and 1 and it's really big, but it's between 0 and 1, just what we needed.

In case you still didn't get what's happening and why do it, here's a one case scenario.

Imagine a scenario where a neural network is employed to determine whether an email is spam or not(classic).
The network's final output is a raw score aka the raw logit that isn't 0 or 1. This is where the sigmoid activation
function enters the picture. By applying the sigmoid function to the raw logit, it undergoes a transformation that
converts it into a probability, like the one we calculated above. This probability signifies the network's confidence
that the email falls into the positive class (spam) category.

The rest is decided by something called round off, you did studied it in you 3 class right? So just take the output value and round it off and we will get 1 or 0 as the final output.

### Words to know before going forward

OK, I know you maybe thinking what's a logit, cause I used it a lot. Logits are just the raw output of the model's predictions, in the above scenario _11.6267_ is a logit, _0.9999910754183369_ is a prediction probability and _1_ is the prediction label.

You should know this cause I will use them a lot and in case you forget what they mean, just remember these steps:
_logits -> propabilities -> labels_

## Predicting how wrong the model is(the right way)

OK we know what a logit is we know what a sigmoid is we know how to convert raw logits into labels of 0 and 1.

Now let's try seeing what exactly did our model mean by it's predictions.

Let's follow the step I mentioned above.

First, let's get the raw logits

```python
with torch.inference_mode():
  y_labels = model1(X_test).squeeze()

y_labels[:5] # Print to see the outputs change
```

Now that we got the logits let's convert them into probabilities.

Lucky for us we don't need to know the sigmoid activation formula, cause pytorch has a method called **torch.sigmoid** for this.

```python
y_probs = torch.sigmoid(y_labels)

y_probs[:5]
```

Nothing new here just converting the logits into probabilities. And for the final blow, let's see what our model is saying to us by rounding off.

```python
y_preds = torch.round(y_probs)

print(f"Predictions: {y_preds[:5]} \n Lables: {y_test[:5]}")
```

In fact pytorch got a round method to do the rounding off for us.

And now we know what our model means by those weird numbers that weren't 0 or 1.

But as you can see the predictions are just not good.

```
Predictions: tensor([0., 1., 0., 0., 1.])
Labels: tensor([1., 0., 1., 1., 0.])
```

This get's us into the training loop, the main topic of this chapter.

## Picking the loss function and optimizer

OK before the training loop let's get ourself a cost(other name of loss) function and optimizer.

As for the optimizer it can be anything, like anything. Pick any optimizer you want the way to use it is the same. For the sake of knowledge we will still ues the SGD.

For the loss function we have two options which you can read at the <a href="https://pytorch.org/docs/stable/nn.html#loss-functions" class="text-blue-600 hover:text-blue-500">PyTorch Doc</a>.

If you see we have two loss functions with the same name, at the starting to be exact.

They are - **BCELoss** and **BCEWithLogitLoss**

Now both are the same, but as you may recall that loss function calculates how wrong the model is.

You may say, "SO what does it has to do with the loss functions?".

You see the BCELoss doesn't comes with a sigmoid activation function, that means if we try passing _y_logits_ as the input, we would be calculating the difference between a value more then 1 or less then 0 and the label. And we know that the labels and the y predictions should be same to calculate the wrongness of our model.

So we will need to calculate and store the sigmoid output and then pass it into the **BCELoss**.

For a more consistent or something something, we use the **BCEWithLogitLoss**, this loss function allows use to pass the raw logits into the function and the function, having a sigmoid activation, calculates the loss without any error.

Thus, we will have **BCEWithLogitLoss** as the loss function.

And here's the setup code for this:

```python
# Setup loss function
loss_fn = nn.BCEWithLogitsLoss() # Has the sigmoid activation function

optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.1)
```

OK now that we have everything, let's make the training loop, in fact, you all give it a try first. Try to scratch your head and find the setup.

Just remember everything is like the linear regression training loop, just that calculate the prediction labels.

OK, if you're done with the training loop or not, here's the final training loop code:

```python
#@title Building the training loop

# Manual seed setting
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put data to device(cuda)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

def trainer(epochs=100):
  for epoch in range(epochs):
    model1.train()

    # Forward pass
    y_logits = model1(X_train).squeeze()
    # y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labels

    # Calculate accuracy
    loss = loss_fn(y_logits, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Gradient descent
    optimizer.step()

    model1.eval()
    with torch.inference_mode():
      # Forward pass
      test_logits = model1(X_test).squeeze()
      # test_pred = torch.round(torch.sigmoid(test_logits))

      # Calculate loss
      test_loss = loss_fn(test_logits, y_test)

    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Train Loss: {loss:.5f} | Test Loss: {test_loss:.5f}")

trainer()
```

IF you see the entire code is the exact same as the linear regression problem, just that you may have seen I have
commented out the **y_pred** and **test_pred**, cause you see we don't need them, unless we are using **BCELoss** not
**BCEWithLogitLoss**.

Finally let's see the loss go down.

```
Epoch: 0 | Train Loss: 0.72748 | Test Loss: 0.26738
Epoch: 10 | Train Loss: 0.06119 | Test Loss: 0.04717
Epoch: 20 | Train Loss: 0.03718 | Test Loss: 0.02668
Epoch: 30 | Train Loss: 0.02840 | Test Loss: 0.01886
Epoch: 40 | Train Loss: 0.02378 | Test Loss: 0.01470
Epoch: 50 | Train Loss: 0.02090 | Test Loss: 0.01210
Epoch: 60 | Train Loss: 0.01892 | Test Loss: 0.01032
Epoch: 70 | Train Loss: 0.01746 | Test Loss: 0.00902
Epoch: 80 | Train Loss: 0.01633 | Test Loss: 0.00803
Epoch: 90 | Train Loss: 0.01543 | Test Loss: 0.00725
```

Looks good right? I mean, if you have notice how we went from 0.2 to 0.007 in 100 loops, real good.

Sure, I know this is just numbers on a blank page, but on looking at the loss go down, it makes sense that the model must be doing real good now.

To see our **human prediction**, let's try the same thing we did in the beginning, but with the sigmoid activation.

```python
model1.eval()
with torch.inference_mode():
  y_logits = model1(X_test).squeeze()
  y_preds = torch.round(torch.sigmoid(y_logits))

print(f'Predictions: {y_preds[:5]}\n Labels: {y_test[:5]}')
```

And the output was as good as the loss

```
Predictions: tensor([1., 0., 1., 1., 0.])
Labels: tensor([1., 0., 1., 1., 0.])
```

This might have given you the idea that our model is doing way better then it was doing in the beginning.

Yes yes I know! I know you're thinking that, "Ezpie! Just visualize this thing!". But sadly even I don't know how to, but
don't lose hope. Try finding out how to visualize this on your own, cause that's what makes a programmer a programmer, he
knows how to find answers better then an average human.

## Tasks

OK now that you understand what binary classification is, it's time for some fun exercises!

Yes! Now the tables have turned! You wont get an easy task but a bit tricky one.

Don't worry not too tricky!

Here's what you have to do.

- Spend some days creating a binary data with 10,000 samples, also make the day with a lot of noise, like a lot, but not to much or you will find out about our next topic(non-linearity!).

- Create a model for that data

- Make a training loop to train the model

- Make a github repository for it(useful for me to understand how you made it.)

- Save the model and upload it to our <a href="https://discord.gg/jR7fjqSCDk" class="text-blue-600 hover:text-blue-500">discord server</a>.

And I will see how well the model is! May the best ML engineer win!
