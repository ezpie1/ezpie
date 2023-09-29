---
layout: ../../../layout/CourseLayout.astro
title: "Creating model for computer vision with non-linearity"
permalink: /courses/machine-learning
---

OK we got our dataset and we have created a dataloader.

Now let's create a model that can learn patterns in this data.

Our first model will be of non-linearity, but without CNN, so simply saying, our first model isn't made for image classification, but will still perform well enough.

## Understanding what a Flatten layer is

This is an important layer in image classification or computer vision in general.

What is a flatten layer?

A flatten layer just apply's multiplication on our height and width and convert them into one.

What do I mean by that?

Take our image dimensions **[1, 28, 28]**.

If you pass this into a flatten layer it would convert it into [1, 28*28] that would be **[1, 784]**.

See what happened? Yes, we now have a dimension of 1 with 784 pixels! So just saying in simple words, the flatten layer converts the image from a square to a straight line of pixels, and not 28 pixels by 28 pixels.

To see this in code let's do this.

```python

flatten = nn.Flatten()

# get first sample
x = train_features[0]

output = flatten(x) # performs forward pass on it's on

print(f'Shape before flattening: {x.shape}')
print(f'Shape after flattening: {output.shape}') # HINT: Multiple the last 2 dimensions
```

Output:

```
Shape before flattening: torch.Size([1, 28, 28])
Shape after flattening: torch.Size([1, 784])
```

And yeah, the flatten layer converted the 28 by 28 grid into one single line. Great!

## Making Model1

Now let's make a model that can learn patterns and predict values shall we.

Let's setup a device agnostic first, something which I keep forgetting to mention!

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```

Now let's really create the model

```python
class FashionMNISTModelV1(nn.Module):
  def __init__(self, input_shape, hidden_units, output_shape):
    super().__init__()

    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_shape, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, output_shape)
    )

  def forward(self, x):
    return self.layers(x)
```

Did you notice something? Yeah, to make this easy on me I have added `nn.Sequential()` to make the neural network. Simple and easy!

Also we have added the flatten layer before the linear layer, cause our...

Well if you saw that video when I was explaining <a href="courses/machine-learning/what-is-a-neural-network" class="text-blue-500 hover:text-blue-600">what neural networks are</a>. You may have seen that in <a href="https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw" class="text-blue-500 hover:text-blue-600">3blues1brown</a>'s video, the starting of the neural network has 784 input neurons.

Looks like he was using the same dataset?

The input will of course be 784, because each value of each of the pixels is important, and depending on those values we are able to predict what the image is displaying.

Back in the model, you can see that we have two linear layers that are taking input and affecting the output.

Also we just have 1 non-linear layer applying non-linear activation ReLU at each on the hidden neurons.

Now let's setup our first model

```python
model1 = FashionMNISTModelV1(784, 20, len(label_names)).to(device)
model1
```

Now I want to setup 3 things... well 4 things to be exact.

These are very common and the last one is a bit spacial.

They are, the accuracy function, loss function, optimizer and an evaluation function to store the results of each model.

An evaluation function can be used to store, say the accuracy, loss and the name of the model so that we know which model performs better and what architecture is used to build it.

```python
def accuracy_fn(y_true, y_pred):

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model1.parameters(), 0.1)
```

Nothing new over here, now let's make that eval function

```python
def eval(model, data_loader, loss_fn, accuracy, device = device):

  loss, acc = 0, 0

  model.eval()
  with torch.inference_mode():

    for X, y in data_loader:

      X, y = X.to(device), y.to(device)

      y_pred = model(X)

      loss += loss_fn(y_pred, y)
      acc += accuracy(y, y_pred.argmax(dim=1))

    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"Model name": model.__class__.__name__,
          "Model loss": loss.item(),
          "Model accuracy": acc}
```

As you can see, we are just doing a testing loop kind of and calculating the loss, accuracy and then finally returning a dictionary of model name, loss and accuracy.

You can in fact also add a model training time to see how fast one model trains to get to the same loss as the other, and this helps a lot in making decisions for the perfect model. But sadly we won't cover that in this course.

That's all, let's wrap it up and move to the next chapter which will be focused on making a training loop.
