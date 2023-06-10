---
layout: ../../../layout/CourseLayout.astro
title: "Saving and Loading the model"
permalink: /courses/machine-learning
---

OK, so we have created a dataset, we have created a model that can learn patterns in that dataset, we also created a training loop to train the model with that dataset and we finally tested the model and all went OK.

Now it's time we try saving the model for later on use.

In order to do that we will use pytorch's `torch.save()` and `torch.load()` methods to save and load the model.

## Why do we need to save our model?

First Let's just answer why we may want to do that.

Now in our case we won't be publishing our model, but in case of OpenAI, <a href="" class="text-blue-600 hover:text-blue-400 hover:underline">as I mentioned they use pytorch</a>, they need to publish their model like ChatGPT.

Now you may say, "But why not just the code? Why save and load the model? Why not let it learn as it helps others?". Well that would be interesting, but you see in case their server went down or crashed, the entire hard work, training and testing all that would go down as well.

And you don't want that as a tech company.

So the simple answer is, make the model, train and  test the model, save the model, recreate the **model class**, and finally load the model's data and you have a good working AI that in case crashed, will not loss all it's learned data.

## Saving the model
So now that we know why we should save our model it's time to do that.

PyTorch provides a method called `torch.save()` for this very purpose.

Now just the thing is, what should be saved?

When we are saving our model we have the option to save the model itself or the model's parameters and use them by recreate the model class.

Now what I want you to do is <a href="" class="text-blue-600 hover:text-blue-400 hover:underline">read this page from pytorch documentation</a> and figure out which one is better.

For this course let's go with the state_dict.

What we need to do is just save our model's state, meaning it's parameters, into a pickle file.

What's pickle? Pickle's just a python module that allow's us to serialize and deserialize the data.

Why do this rather then simply writing it to a file? Well that even I don't know.

Now let's just stop with the talk and save our model.

```python
torch.save(model2.state_dict(), 'linear.pt')
print("Model saved at current folder as linear.pt")
```

Now there's nothing to complicated, rather we are just saving the state_dict aka the parameters of our model in a file called linear.pt, `.pt` is the pytorch default extension for saving any pytorch model.

## Loading our model and predicting with it

Now let's try loading our model and predicting with it.

But first we will need to recreate the model class which we created earlier.

```python
class CopyLinear(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(1, 1)

  def forward(self, x):
    return self.layer1(x)

# Creating the instence
modelCopy = CopyLinear()
modelCopy
```

Nothing new here just the same old stuff. 

Now let's load the model's state and assign it to this model.

```python
modelCopy.load_state_dict(torch.load('linear.pt'))
print('model state loaded')
```

Here we are just using the `load_state_dict` method to assign the state to the model, and for loading the model's state dict that we saved, we are using `torch.load()`.

Now let's try to test weather the loaded model works or not

```python
modelCopy.eval()
with torch.inference_mode():
  newer_preds = modelCopy(X_test)

plot(predictions=newer_preds)
```

So yeah nothing like we haven't covered yet, just setting the model in testing mode, performing forward pass and plotting the values.

![loaded model's predictions](https://github.com/EzpieCo/PyTorch-Crash-Course/assets/104765117/fb8cded4-2054-4e9f-83cf-387ca1775c91)

And yeah, all the same.

## Task: Predicting future values

OK, now enough of this, let's see if our model can predict future values.

ALso if you have done this part as I already told you in the last chapter this was you task, you can skip this part if you like.

Let's create future dataset that continues the values after 1, which was our last data in the dataset.

```python
X = torch.arange(1, 2, 0.02).unsqueeze(1) # continuing from the last point till 2
y = bais + (weight * X)
```

Really simple linear regression formula being performed, nothing unusual here.

```python
def new_plot(test_data=X, test_labels=y, predictions=None):
  
  plt.figure(figsize=(10, 7))

  plt.scatter(test_data, test_labels, c="r", s=4, label="Test data")

  if predictions != None:
    plt.scatter(test_data, predictions, c="b", s=4, label="Predictions")

  plt.legend(prop={"size": 14})

new_plot()
```

Just for clarity let's just define a new plotting function.

![new dataset that continues the previous one](https://github.com/EzpieCo/ezpie/assets/104765117/12bf0410-3f94-449a-9ce8-59d800a15d86)

And finally let's predict how well this model can perform.

And just for fun let's use the loaded model instead.

```python
with torch.inference_mode():
  newer_y_preds = modelCopy(X)

new_plot(predictions=newer_y_preds)
```

![Predicted future values using our model that didn't know about this dataset](https://github.com/EzpieCo/ezpie/assets/104765117/4d8b699f-618f-4a63-88b8-a8b4664afaef)

And Voila!

There you have it! A PyTorch model that can predict values that it has never seen! Not random nor perfect, but close!

If this doesn't conveys you to learn machine learning with PyTorch, then I don't know what well.

## Task

You task for today is simple, just try to improve the model, try training longer, try increasing the number of data, try increasing the number of neurons.

See if the values line up properly or not.

<p class="bg-red-500 text-white p-2 rounded-lg">
    <b>WARNING:</b>
    This is a spoiler alert!
</p>

From the next chapter onwards we will create and train a model to learn patterns in data and classify weather the dot belongs to red or blue category

![a simple dataset of two different options](https://github.com/EzpieCo/ezpie/assets/104765117/6c79b26a-a65e-46b1-8fbd-015a192b36a0)