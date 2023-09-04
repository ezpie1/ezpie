---
layout: ../../../layout/CourseLayout.astro
title: "Making the model"
permalink: /courses/machine-learning
---

We have the data and now it's time to make a model for that data.

And also we will make the training loop for it.

**Spoiler:** Things won't work as we expect them to! The model won't learn patterns!

## Making the model

OK, we have the data, but we don't have a model for the data, so let's create a model that can learn patterns in our data.

I want you guys to do this part and come back to see if your self made non-copied model will work or not.

Done? No? Then here's the answer.

```python
class CircleModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Linear(2, 10)
    self.layer2 = nn.Linear(10, 10)
    self.layer3 = nn.Linear(10, 1)

  def forward(self, x):
    return self.layer3(self.layer2(self.layer1(x))) # x -> layer1 -> layer2 -> layer3 -> output

model1 = CircleModel().to(device)
model1
```

That's all for making the model, and you may have notice we have a third layer.

Why? Well that's cause I played around a bit and found out that 3 layers where good.

In case you want to visualize this, you're in luck.

Here's the visualized version of the neural network

![neural network visual view](/images/courses/ml/neural-network.svg)

Now we will make the training loop.

## Training loop

This is simple so just make it yourself and then run it.

First let's get the loss function and optimizer

```python
# Setup loss function
loss_fn = nn.BCEWithLogitsLoss() # Has the sigmoid activation function

optimizer = torch.optim.SGD(params=model1.parameters(),
                            lr=0.1)
```

Now let's put everything into the target device.

```python
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
```

Also I think this is a good time to introduce you to another measurement of finding out about our models performance. That is accuracy.

Just as the word say we will calculate how accurate our model is.

For this try implementing it yourself once, if not able to do then look below for the answer.

This will teach you how to think programmatically.

```python
def accuracy(y_pred, y_true):
  correct = torch.eq(y_true, y_pred).sum().item()
  accuracy = (correct/len(y_pred)) * 100
  return accuracy
```

With this function we will be able to calculate how **good** not how bad our model is doing.

Now let's make the training loop

```python
def trainer(epochs=100):

  for epoch in range(epochs):
    model1.train()

    # Predict
    train_logit = model1(X_train).squeeze()
    train_pred = torch.round(torch.sigmoid(train_logit)) # logits -> probabilities -> labels

    # calculate loss and accuracy
    train_loss = loss_fn(train_logit, y_train)
    train_acc = accuracy(train_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # backpropagation
    train_loss.backward()

    # Gradient decent
    optimizer.step()

    model1.eval()
    with torch.no_grad():
      test_logit = model1(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logit))

      # calculate loss and accuracy
      test_loss = loss_fn(test_logit, y_test)
      test_acc = accuracy(test_pred, y_test)

    if epoch % 20 == 0:
      print(f'Epoch: {epoch} | Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%')

trainer()
```

But you will find some funny outputs

```
Epoch: 0 | Train loss: 0.69729 | Train accuracy: 50.38% | Test loss: 0.70532 | Test accuracy: 48.50%
Epoch: 20 | Train loss: 0.69326 | Train accuracy: 49.50% | Test loss: 0.69888 | Test accuracy: 43.50%
Epoch: 40 | Train loss: 0.69299 | Train accuracy: 49.38% | Test loss: 0.69773 | Test accuracy: 47.50%
Epoch: 60 | Train loss: 0.69291 | Train accuracy: 49.75% | Test loss: 0.69724 | Test accuracy: 47.00%
Epoch: 80 | Train loss: 0.69286 | Train accuracy: 49.75% | Test loss: 0.69693 | Test accuracy: 48.00%
```

Hmmm, why is the loss not going down and why is the accuracy the same?

Maybe trying to train a little longer may help...

But! It wont! Yes it just wont.

Why? That's cause... Let's just visualize it first.

I searched the internet and found this nice function to display a decision boundary, that is making a boundary for the blue dots and red dots.

```python
def plot_decision_boundary(model, X, y):
  # Numpy and matplot work well with CPU
  model.to("cpu")
  X, y = X.to("cpu"), y.to("cpu")

  # Setup prediction boundaries and grid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

  # Make features
  X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

  # Make predictions
  model.eval()
  with torch.inference_mode():
    y_logits = model(X_to_pred_on)

  # Logits -> Probabilities -> Prediction Labels
  y_pred = torch.round(torch.sigmoid(y_logits))

  # Reshape preds and plot
  y_pred = y_pred.reshape(xx.shape).detach().numpy()
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
```

Now let's see what's happening

```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model1, X_test, y_test)
```

![Decision boundary of our model's predictions](https://user-images.githubusercontent.com/104765117/265269662-3d3c1b85-adcb-423d-abdf-41b18b7e2e58.png)

As you can see this proves why our accuracy is around 50%.

Can you see something here?

You may have made out that our data is not a straight line it has curves and our model is trained for straight line right?

Why you ask? Remember linear regression?

The model we used then was created the same way as this model is and if you see thi model will preform good with straight line data, cause that's what our model is doing, using linear regression formula to calculate a straight line!

In a more mathematical style this is what I mean.

![neural network visual view](/images/courses/ml/neural-network.svg)

Take the neuron at the top of the first hidden layer and say that the two input neurons have a value of about 0.46 and 0.37 each.

Now take the weight as say something like 0.05 for 0.46 and 0.9 for 0.37 and the bias about 0.0352

Now calculate the value of the top neuron.

The formula is like this: y = wx<sub>1</sub> + wx<sub>2</sub> + wx<sub>n</sub> + b

This with the given values will be:

y = 0.05 _ 0.46 + 0.9 _ 0.37 + 0.0352 = 0.3912

And if you were to repeat this process for every neuron, which I will not do and you shouldn't either, cause will the weight and bias was just made up. But if you were to find these wights and biases and plot a graph you will find that the graph is just another straight line, whereas we have a dataset which is, well not straight!

So for this we need to make changes to our hidden layers outputs, let's say we give them a spacial layer after each hidden layer which will convert the outputs of the previous hidden layer into something that may cause the line to turn a bit up or down rather then being straight.

This spacial layer has a name, called the activation function.

Sounds familiar? Yes the sigmoid activation function is also a function which we can use for this, but it's not really useful for these kinds of things.

Rather we have a popular activation function called the **ReLU activation function** that will suit this case a lot.

Go to the <a href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html" class="text-blue-600 hover:text-blue-500">pytorch documentation for it</a>.

You will see that the ReLU activation function has a bit different formula, which does kind of looks weird.

But no need to worry about the formula, let's just see the graph

![ReLU activation function graph](https://pytorch.org/docs/stable/_images/ReLU.png)

As you can see from this graph, the value 0 turned into around 6, and imagine doing this for about every single output of the hidden layer we may get something pretty cool to see.

OK enough of talking and now let's try adding this ReLU function in our model and see what happens.

## Adding the missing ingredient: Non-linearity

So now let's add the ReLU activation function into our model, train it and see how it affects the outputs.

```python
class CircleModelV2(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(2, 10)
    self.layer2 = nn.Linear(10, 10)
    self.layer3 = nn.Linear(10, 1)
    self.relu = nn.ReLU() # Non-linear activaion function

  def forward(self, x):
    return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model2 = CircleModelV2().to(device)
model2
```

And yes I totally made a new class, cause why not?

And now let's recreate the training loop, cause the previous one has _model1_ which doesn't has non-linearity.

```python
#@title Preparing the training loop

torch.manual_seed(32)
torch.cuda.manual_seed(32)

def trainer2(epochs=1000):

  for epoch in range(epochs):
    model2.train()

    # Predict
    train_logit = model2(X_train).squeeze()
    train_pred = torch.round(torch.sigmoid(train_logit)) # logits -> probabilities -> labels

    # calculate loss and accuracy
    train_loss = loss_fn(train_logit, y_train)
    train_acc = accuracy(train_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # backpropagation
    train_loss.backward()

    # Gradient decent
    optimizer.step()

    model2.eval()
    with torch.inference_mode():
      test_logit = model2(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logit))

      # calculate loss and accuracy
      test_loss = loss_fn(test_logit, y_test)
      test_acc = accuracy(test_pred, y_test)

    if epoch % 10 == 0:
      print(f'Epoch: {epoch} | Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%')

trainer2()
```

And look at the loss go down and the accuracy go up!

```python
Epoch: 0 | Train loss: 0.69544 | Train accuracy: 43.38% | Test loss: 0.69415 | Test accuracy: 45.00%
Epoch: 10 | Train loss: 0.69510 | Train accuracy: 43.12% | Test loss: 0.69392 | Test accuracy: 45.00%
Epoch: 20 | Train loss: 0.69480 | Train accuracy: 43.25% | Test loss: 0.69374 | Test accuracy: 45.00%
Epoch: 30 | Train loss: 0.69452 | Train accuracy: 44.75% | Test loss: 0.69361 | Test accuracy: 46.50%
Epoch: 40 | Train loss: 0.69425 | Train accuracy: 47.88% | Test loss: 0.69351 | Test accuracy: 47.50%
Epoch: 50 | Train loss: 0.69399 | Train accuracy: 50.25% | Test loss: 0.69343 | Test accuracy: 48.50%
Epoch: 60 | Train loss: 0.69374 | Train accuracy: 50.38% | Test loss: 0.69336 | Test accuracy: 48.50%
Epoch: 70 | Train loss: 0.69350 | Train accuracy: 50.38% | Test loss: 0.69330 | Test accuracy: 48.50%
Epoch: 80 | Train loss: 0.69327 | Train accuracy: 50.38% | Test loss: 0.69324 | Test accuracy: 48.50%
Epoch: 90 | Train loss: 0.69305 | Train accuracy: 50.38% | Test loss: 0.69318 | Test accuracy: 48.50%

...

Epoch: 900 | Train loss: 0.64068 | Train accuracy: 66.12% | Test loss: 0.65741 | Test accuracy: 60.50%
Epoch: 910 | Train loss: 0.63807 | Train accuracy: 68.50% | Test loss: 0.65495 | Test accuracy: 62.50%
Epoch: 920 | Train loss: 0.63532 | Train accuracy: 70.75% | Test loss: 0.65232 | Test accuracy: 63.00%
Epoch: 930 | Train loss: 0.63244 | Train accuracy: 72.88% | Test loss: 0.64959 | Test accuracy: 66.50%
Epoch: 940 | Train loss: 0.62939 | Train accuracy: 74.00% | Test loss: 0.64674 | Test accuracy: 67.00%
Epoch: 950 | Train loss: 0.62615 | Train accuracy: 74.62% | Test loss: 0.64375 | Test accuracy: 68.50%
Epoch: 960 | Train loss: 0.62275 | Train accuracy: 75.38% | Test loss: 0.64062 | Test accuracy: 70.50%
Epoch: 970 | Train loss: 0.61917 | Train accuracy: 76.12% | Test loss: 0.63733 | Test accuracy: 72.50%
Epoch: 980 | Train loss: 0.61539 | Train accuracy: 77.25% | Test loss: 0.63391 | Test accuracy: 72.50%
Epoch: 990 | Train loss: 0.61143 | Train accuracy: 79.25% | Test loss: 0.63026 | Test accuracy: 73.50%
```

Yes I did trained it much longer then previously, cause I played about and saw that it started to do the magic a bit late. So yeah, it works!

And let's plot out the decision boundary, also I converted into a function, cause why not.

```python
def ploter():
  plt.figure(figsize=(12, 6))

  plt.subplot(1, 2, 1)
  plt.title("Train")
  plot_decision_boundary(model2, X_train, y_train)

  plt.subplot(1, 2, 2)
  plt.title("Test")
  plot_decision_boundary(model2, X_test, y_test)

ploter()
```

And here we have it.

![Decision boundary after adding non-linearity to our model](https://user-images.githubusercontent.com/104765117/265270870-ba402401-0d77-4f9a-b024-a3ecaab6169d.png)

As you can see the boundary is a lot more better, but it's not perfect.

We can just do it right now and make this a real long chapter, but let's keep that for the other chapter and why not you all try to improve the model.

## Task

So you're task for this chapter is simple, improve the model, bring it to perfection, you can do this by following these steps:

- Try increasing the number of hidden layers
- Try increasing the number of neurons per hidden layer
- Try to train for a longer time

That all, and see you in the next chapter.
