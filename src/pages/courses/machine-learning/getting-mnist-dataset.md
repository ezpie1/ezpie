---
layout: ../../../layout/CourseLayout.astro
title: "Getting MNIST dataset"
permalink: /courses/machine-learning
---

Now let's get into doing some computer vision.

In this chapter we will get the MNIST dataset using a pytorch module called `datasets`.

Using `datasets` we can get any dataset we want, more precisely the datasets that are available in pytorch.

## Getting the dataset

First let's get our training and testing datasets using `datasets`.

Let's import the necessary packages.

```python
# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# torchvision for image classification
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# Matplotlib for plotting
import matplotlib.pyplot as plt
```

Here we are importing torch and matplotlib as usual, but we also have imported a new package, that is **torchvision**.

**torchvision** is used for computer vision related problems, or more englishry image related.

Next let's use the `torchvision.datasets` to get the MNIST dataset.

```python
train_data = datasets.MNIST(
  root = "data", # Where to put the data
  train = True, # Wether get the training data or testing
  download = True, # Should the data be downloaded
  transform = ToTensor(), # How to transform the data
  target_transform = None # How to transform the labels - y
)

test_data = datasets.MNIST(
  root = "data",
  train = False,
  download = True,
  transform = ToTensor(),
  target_transform = None
)
```

As you can see the parameters are self explanatory, but let me still explain what is happening.

The `root` attribute is used for storing the dataset, for our case we are storing the training dataset in a directory called data.

the `train` attribute is used for telling **datasets** if we want the training dataset or the testing, and since our first dataset is for training we have put it to true.

the `download` attribute tell's wether or not the data should be downloaded or not, if the data doesn't exist it will download, if exist then it won't download.

the `transform` attribute is by far the most important one, without it we would have to convert the images into tensors by hand, like assign each pixel a value in a tensor, and since each image is 25 by 25 pixels that would mean 625 pixels to assign! To help us out we use the **ToTensor()** method.

Finally we have the `target_transform` this is used for how to transform the labels, we have set it to None cause we don't need it, our labels will be between 0 and 1 so we don't have to.

You don't really need to understand each line of code here, just fix it that this is how you can do it.

## Getting to know about the data

Since our data isn't made by us, we have 0 knowledge about what it is.

SO now let's try to get to know about our dataset here.

First, what should be know about our data?

First let's see how the tensor data is, that is how each image tensor is related to a label.

```python
# See first training sample
image, label = train_data[0]
image, label
```

This gave a large output so I won't display what it was, but there were a lot of numbers in that tensor!

Next let's check what all labels we have, cause we don't know what dataset it is... Well technically I know, but say we don't know anything for now.

```python

label_names = train_data.classes
label_names
```

This gave the output as:

```
['0 - zero',
 '1 - one',
 '2 - two',
 '3 - three',
 '4 - four',
 '5 - five',
 '6 - six',
 '7 - seven',
 '8 - eight',
 '9 - nine']
```

As you can see we have a list of 10 different labels, that are 0 to 9! Just like I said.

Now let's check the important thing, the shape of the image and label.

```python
print(f'Image shape: {image.shape} -> [color channel, height, width]')
print(f'Label shape: {label.shape}')
```

Output:

```
Image shape: torch.Size([1, 28, 28]) -> [color channel, height, width]
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-5-ecda01b07575> in <cell line: 2>()
      1 print(f'Image shape: {image.shape} -> [color channel, height, width]')
----> 2 print(f'Image label: {label.shape}')

AttributeError: 'int' object has no attribute 'shape'
```

Ahh! Did you show that? Our labels are of type int and not tensor.

I did mentioned that we won't be converting our labels into tensors and look, here's what happened.

And anyways we don't need to convert them into tensors, they are good as int.

But did you notice? I printed a small text after the image shape?

Remember? Yeah, pytorch uses the color channel first law, that's what is being displayed. Let's just remove the error and see what we get.

```python
print(f'Image shape: {image.shape} -> [color channel, height, width]')
```

Output:

```
Image shape: torch.Size([1, 28, 28]) -> [color channel, height, width]
```

As you can see we have a tensor of type? Yes, vector.

The first value tell's us that the image consists of only black and white colors, the second and last on tell you that the image is of what height and width.

And also that I used the wrong number!?! Turns out it's 28 by 28 and not 25 by 25.

Ops!

But you get the main point right? Our image is of black and white with height and width of 28. Meaning total of 284 pixels!

That's a lot of numbers, let's see a visual representation of these numbers.

Lucky enough, matplotlib can plot images too.

```python

image, label = train_data[0]

print(f'Image shape: {image.shape}')

plt.imshow(image.squeeze(), cmap="gray")
plt.title(label_names[label])
plt.axis(False)
```

As you can see we are getting the first image sample and plotting it by hiding the axis and setting the title as the label of the current image.

![First image sample of the training data. PyTorch Computer vision](https://user-images.githubusercontent.com/104765117/270110804-897b7cb8-d435-4a4e-b23c-fb8eb055da45.png)

As you can see it's just an image of the number 5, but a lot pixelated and handwritten(that's what MNIST consists of).

You can also see some more random images by just going index by index or just picking a number.

But I guess that's all for getting to know our data, now let's create a really useful thing called dataloader.

## Creating a Dataloader

First, what's a dataloader?

Well a dataloader is typically a module which we use to manage and load training or testing data efficiently. Dataloaders
are essential when working with models, as they handle various data-related tasks, such as loading data, batching data into small chunks, and shuffling data.

The 3 main reasons why one should use dataloaders are:

**Data Loading:** Dataloaders are responsible for loading data from our sources, such as files, databases, APIs or the way we did using pytorch's `datasets` module. They abstract away the data loading process, making it easier for us to work with datasets.

**Batching:** Dataloaders group the data into batches, which are just smaller chunks of the dataset. Batch processing is a common practice in machine learning because it allows models to update their parameters more frequently, which can speed up training, the most common batch size, and the one we will use, is 32 samples per batch.

**Shuffling:** Dataloaders can also shuffle the data before each epoch, this prevent the model from learning the order of the data and helps train the model learn randomly. Shuffling helps ensure that the model generalizes well to unseen data.

These are the 3 reasons why I believe that dataloaders are important, but for others it can be a lot more reasons, but these are the common ones.

Now that we know what a dataloader is let's create 2 for our training and testing datasets.

```python
BATCH_SIZE = 32

train_dataloader = DataLoader(
    dataset = train_data, # Which dataset to use
    batch_size = BATCH_SIZE, # Number of samples per batch
    shuffle = True # Should the samples be randomize?
)

test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = False
)

train_dataloader, test_dataloader
```

As you can see we are creating a dataloader for each of the dataset, the dataset attribute as you can guess tell's which dataset is it suppose create a dataloader on.

The batch number as you can guess tell's how many samples each batch is suppose to have.

The shuffle attribute is to tell wether or not the dataset should be shuffled. For the training set we have set it to true, cause we want the model to see random samples when training, whereas we can allow the model to see same patterns of samples, cause it won't update the model's parameters, get what I mean?

Now let's check how many batches of 32 samples do we have.

```python
print(f'Length of training data: {len(train_dataloader)} batches of {BATCH_SIZE}')
print(f'Length of training data: {len(test_dataloader)} batches of {BATCH_SIZE}')
```

Output:

```
Length of training data: 1875 batches of 32
Length of training data: 313 batches of 32
```

As you can see we have 1875 batches of 32 samples each. Do the maths, you will see that if you divide the number of training sample to 32 you will get almost 1875, cause when I checked it's not exact, it just gets round off, but don't worry the model will still perform to it's best.

Now let's check the dimensions of these things

```python
train_features, train_labels = next(iter(train_dataloader))
train_features.shape, train_labels.shape
```

Output:

```
(torch.Size([32, 1, 28, 28]), torch.Size([32]))
```

As you can see from this our samples have 32 as their new dimension, don't ask why, that's just the number of samples for each batch and that's just how it is.

Also you may have notice that our labels now have a shape of 32. Any guesses?

Yes, that's the number of sample each batch has, so basic saying, the dimension here are of the dataloader and not of each label.

That's all for now, in the next chapter we will create 2 model's, that is one with non-linearity and the other build using CNN.

This would be done because we just want to experiment what will happen with our other options.
