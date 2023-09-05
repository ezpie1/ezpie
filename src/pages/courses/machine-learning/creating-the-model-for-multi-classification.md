---
layout: ../../../layout/CourseLayout.astro
title: "Creating The model for multi-classification"
permalink: /courses/machine-learning
---

Now that we have the dataset which looks like this

![visualizing the multi-classification dataset](https://user-images.githubusercontent.com/104765117/265690700-fa61080b-0f64-421b-9fd0-c3b10811f652.png)

As you can see, the dataset has 5 values - red, yellow, blue, lightblue and orange.

We already know how to convert our model's raw logits into labels, but what we don't know is how is our model going to be structured?

You see I didn't say how we will make the model, cause we already know that, but what I did said is that how is it going to be structured? That means, how many neurons, how many hidden layers, and the most important is it going to be non-linear or linear?

To answer this we need to experiment.

Let's go with something simple, that is 10 neurons per hidden layer, 2 hidden layers and...

Well even I don't know about the last one, you see our dataset consist of some mixed up values too, but if you try making lines you will see they may fit up.

To answer this last question, let's experiment and find out, starting with the linear option

## Making a linear model

Let's make a model with linearity first and see if it can learn patterns in the dataset, if not then we will go for non-linearity.

```python
# Create multi-class model
class ModelV1(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Linear(2, 10)
    self.layer2 = nn.Linear(10, 10)
    self.layer3 = nn.Linear(10, 5)

  def forward(self, x):
    return self.layer3(self.layer2(self.layer1(x)))

# Create instance
model1 = ModelV1().to(device)
model1
```

Just to differentiate I have named this one with a **V1** to indicate it's version 1 of the model.

Now let's create another model with non-linearity.

```python
# Non-linear model
class ModelV2(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Linear(2, 8)
    self.layer2 = nn.Linear(8, 16)
    self.layer3 = nn.Linear(16, 5)

    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

# Create instance
model2 = ModelV2().to(device)
model2
```

And now let's setup a loss function and an optimizer for the models, but we have another problem, that is we don't know which loss function to pick. We can't just pick a loss function made for binary classification, cause it's binary classification and this is multi-classification, just two different problems!

Try going through the pytorch documentation and find a loss function for yourself.

Found? No? OK the answer is the **CrossEntropyLoss**. It even makes sense to put a binary before it and we will get **B**inary **C**ross **E**ntropy **L**oss or BCE loss.

Let's set them up and make the training loops.

```python
# Loss function
lossFn = nn.CrossEntropyLoss()

# For model1
optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.1)

# For model2
optimizer2 = torch.optim.SGD(params=model2.parameters(), lr=0.1)
```

As you can see we have to setup 2 different optimizers and just one loss function, cause the optimizer updates the model's weights and biases whereas the loss function calculates how wrong the model is.

And finally for the training loops

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put train test into device agnostic
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Trainer for Linear model
def trainer(epochs=1000):
  # Train loop
  for epoch in range(epochs):
    model1.train()

    # Forward pass
    y_logit = model1(X_train)

    # Convert logits -> pred probs -> pred labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

    # Calculate loss
    loss = lossFn(y_logit, y_train)
    acc = accuracy(y_pred, y_train)

    # gradient descend
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # optimizer step
    optimizer.step()

    # Test loop
    model1.eval()
    with torch.inference_mode():
      # Forward pass
      test_logit = model1(X_test)

      # Convert logits -> pred probs -> pred labels
      test_pred = torch.softmax(test_logit, dim=1).argmax(dim=1)

      # Calculate loss
      test_loss = lossFn(test_logit, y_test)
      test_acc = accuracy(test_pred, y_test)

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Train accuracy: {acc:.2f} | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")

# Trainer for non-linear model
def trainer2(epochs=1000):
  # Train loop
  for epoch in range(epochs):
    model2.train()

    # Forward pass
    y_logit = model2(X_train)

    # Convert logits -> pred probs -> pred labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

    # Calculate loss
    loss = lossFn(y_logit, y_train)
    acc = accuracy(y_pred, y_train)

    # gradient descend
    optimizer2.zero_grad()

    # Backpropagation
    loss.backward()

    # optimizer step
    optimizer2.step()

    # Test loop
    model2.eval()
    with torch.inference_mode():
      # Forward pass
      test_logit = model2(X_test)

      # Convert logits -> pred probs -> pred labels
      test_pred = torch.softmax(test_logit, dim=1).argmax(dim=1)

      # Calculate loss
      test_loss = lossFn(test_logit, y_test)
      test_acc = accuracy(test_pred, y_test)

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Train accuracy: {acc:.2f}% | Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
```

As you can see I defined 2 different functions for this, though yes you can create just one function with parameters to pass the model class, but that's for you to solve, **I just like working hard not smart**!

Also I setup the accuracy function for this and wound recommend you too do it.

When you will run each of them you will see something weird.

For model1 training I got these outputs

```
Epoch: 0 | Train loss: 2.31888 | Train accuracy: 0.62 | Test loss: 1.34170 | Test accuracy: 45.50
Epoch: 100 | Train loss: 0.22591 | Train accuracy: 92.50 | Test loss: 0.17964 | Test accuracy: 94.00
Epoch: 200 | Train loss: 0.18862 | Train accuracy: 92.25 | Test loss: 0.14965 | Test accuracy: 93.50
Epoch: 300 | Train loss: 0.16862 | Train accuracy: 93.50 | Test loss: 0.13524 | Test accuracy: 94.00
Epoch: 400 | Train loss: 0.15820 | Train accuracy: 93.62 | Test loss: 0.12809 | Test accuracy: 94.50
Epoch: 500 | Train loss: 0.15224 | Train accuracy: 93.88 | Test loss: 0.12374 | Test accuracy: 94.50
Epoch: 600 | Train loss: 0.14857 | Train accuracy: 94.25 | Test loss: 0.12079 | Test accuracy: 95.00
Epoch: 700 | Train loss: 0.14614 | Train accuracy: 94.25 | Test loss: 0.11862 | Test accuracy: 95.00
Epoch: 800 | Train loss: 0.14443 | Train accuracy: 94.50 | Test loss: 0.11694 | Test accuracy: 95.00
Epoch: 900 | Train loss: 0.14318 | Train accuracy: 94.50 | Test loss: 0.11559 | Test accuracy: 95.00
```

And for model2 it wasn't what I expected

```
Epoch: 0 | Train loss: 3.41792 | Train accuracy: 0.00% | Test loss: 3.10749 | Test accuracy: 0.00%
Epoch: 100 | Train loss: 1.31619 | Train accuracy: 60.38% | Test loss: 1.31112 | Test accuracy: 59.00%
Epoch: 200 | Train loss: 1.28730 | Train accuracy: 60.50% | Test loss: 1.28615 | Test accuracy: 59.00%
Epoch: 300 | Train loss: 0.79143 | Train accuracy: 73.88% | Test loss: 0.88678 | Test accuracy: 70.50%
Epoch: 400 | Train loss: 0.73142 | Train accuracy: 75.62% | Test loss: 0.82119 | Test accuracy: 72.50%
Epoch: 500 | Train loss: 0.70449 | Train accuracy: 76.12% | Test loss: 0.79714 | Test accuracy: 72.00%
Epoch: 600 | Train loss: 0.68900 | Train accuracy: 76.38% | Test loss: 0.78510 | Test accuracy: 72.00%
Epoch: 700 | Train loss: 0.67934 | Train accuracy: 76.62% | Test loss: 0.77843 | Test accuracy: 71.50%
Epoch: 800 | Train loss: 0.67300 | Train accuracy: 76.75% | Test loss: 0.77433 | Test accuracy: 71.50%
Epoch: 900 | Train loss: 0.66867 | Train accuracy: 76.88% | Test loss: 0.77181 | Test accuracy: 72.00%
```

At the end it's proven that linear model is better for such a dataset, but this won't be true in all case, imagine having a real missed up dataset with values all around, at that time our non-linear model might have won.

And just for visualizing let's output their decision boundaries

for model1

```python
import numpy as np

def plot_decision_boundary(model, X, y):
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

    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
```

The output was just like what the accuracy said.

**Code**

```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model1, X_test, y_test)
```

![decision boundary of model1's training for multi-classification](https://user-images.githubusercontent.com/104765117/265704461-2a3213fd-6413-4af9-be02-0ab1075032a9.png)

And for model2 it was almost the same as model1

**Code**

```python
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model2, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model2, X_test, y_test)
```

![decision boundary of model2's training for multi-classification](https://user-images.githubusercontent.com/104765117/265706059-399684cc-d8de-426f-bd29-03f6e864a517.png)

As you can see we have these two totally different outputs for different model structure for our current problem.

From this we can conclude that for different problems we need to have different model structure and also experimenting is the most important thing while making models.

Experiment, Experiment, Experiment, and Experiment.

Also training for longer or adjusting the number of neurons or hidden layer didn't work so don't try doing it.

## Task

OK so we are done with multi-classification part of the course.

Now it's time to really put your learned skills to the test.

I want you to make a dataset of about 8 labels, the model structure is totally up to you.

After you have made the dataset and the model is trained save the model and share it with me in our <a href="https://discord.gg/jR7fjqSCDk" class="text-blue-600 hover:text-blue-500">discord server</a>.
