---
layout: ../../../layout/CourseLayout.astro
title: "Preparing the data and loading it"
permalink: /courses/machine-learning
---

In this chapter we will be creating the data we need to really make a pytorch model, cause without data there is no model.

## What is data

Data refers to the information that we use to train, validate, and test our machine learning models. It serves as the foundation for making predictions, gaining insights, and solving real-world problems using machine learning techniques.

Data can come in various forms, such as structured, unstructured, or semi-structured. Structured data is organized and follows a predefined format, typically stored in databases or tabular formats like CSV files. Unstructured data, on the other hand, lacks a specific structure and can include text, images, audio, video, or free-form documents. Semi-structured data lies in between, having some organization but not conforming strictly to a predefined schema.

Hence data an be anything

- Audio
- Video
- Text
- Image

Machine learning is all about:

- Getting data and turning it into numerical values
- Building a model that can learn patterns in that data

For this we will create a <a href="/blogs/Understand-Linear-Regression-with-Real-World-Concept" class="text-blue-600 hover:text-blue-400 hover:underline">linear regression</a> model aka model that will predict values in a straight line.

First let's import the necessary dependencies

```python
# importing dependencies
import torch
from torch import nn # nn, short for neural network, is the main building block of neural networks.
import matplotlib.pyplot as plt

# Checking pytorch version

torch.__version__ # version 2.0.1 above will work fine
```

Now let's create the values.

```python
# Creating weight and bais
weight = 0.5 # Can be anything, but line will differ
bais = 0.2 # same with the bais

# Creating the features and labels
X = torch.arange(0, 1, 0.02).unsqueeze(1) # remove all the 1 dimensions

# linear regression formula: y = b + wx ... wx*n* (number of times)
y = bais + (weight * X)
```

So what is weight and bais? In machine learning weight can be defined as the amount of strength each neuron must put in order to make it the most important node, in a more basic context of understanding follow along this experiment.

Say you work for a company **human intelligence** and they asked you to find out how can we predict weather a human is good in english or not, now think about the useful features aka the measurement.

You may come up with features like this, number of books read, which country they belong and so on, we can say that if the person is from country like US, England, Australia, etc they will have a better chance of being good in english, and yeah more the books read the better, but we need a way to put some stress on these values, like if the person has read over 10 books you may think he is good in english, but if the person is from a country where english is not a major language you may think his's not good in english, yet since he has a read a good amount of books we want to put stress aka weight in saying that no matter of country, if he has read enough of books he will have higher changes.

Bais on the other hand is just an additional parameter in a neural network that helps adjust the output of a neuron.

With both weights and biases the neural network performs just the way it should and predicts by transforming the input through a series of mathematical operations. So in recap, weight determines the strength and influence the connections between neurons, while biases help introduce flexibility and control the activations of neurons.

The next thing we have in this code is the `X` and `y` variables, here `X` is in capital, cause that's the prefer way the machine learning community wants, so `X` is the feature and `y` is the label or what we want to predict with the help of `X`.

## Getting to know our data

Now let's try to understand our data, cause in machine learning it's important to know what and how your data is.

```python
# getting the first 10 values
print(f"first ten features: {X[:10]}")
print(f"first ten labels: {y[:10]}")

# checking lenght of features and labels
print(f"number of features: {len(X)}")
print(f"number of labels: {len(y)}")

# Checking the shape of our data
print(f"shape of features: {X.shape}")
print(f"shape of labels: {y.shape}")
```

So yeah really basic stuff going on here, nothing like we have not covered yet.

**Output**

```
first ten features: tensor([[0.0000],
        [0.0200],
        [0.0400],
        [0.0600],
        [0.0800],
        [0.1000],
        [0.1200],
        [0.1400],
        [0.1600],
        [0.1800]])
first ten labels: tensor([[0.2000],
        [0.2100],
        [0.2200],
        [0.2300],
        [0.2400],
        [0.2500],
        [0.2600],
        [0.2700],
        [0.2800],
        [0.2900]])
number of features: 50
number of labels: 50
shape of features: torch.Size([50, 1])
shape of labels: torch.Size([50, 1])
```

If we see our data is kind of straight and simple, some data in form of tensor, so can you guess what our tensor type is? How to find it is simple look at the number of dimensions, here we have a size with 2 values, 50 and 1, therefore our number of dimensions must be 2, and which tensor has 2 dimensions? Matrix! If you guessed it right, good job, you are getting it in the right order!

## Splitting the data into random training and testing sets

In machine learning we need to train our model just the same way we would do to train ourselves, think for example you are a student learning in a school, in the beginning the school teaches us with material they provide, then they take tests, which you can take as a training set, then you have the final exams which checks weather you have learned from all that you were thought or not, this can be taken as a testing set.

Now in order to split our data in training and testing set, which we can do manual, but is not a good practice cause in that case we may skip some data which are a bit different then all the data we had in the training set, say for example in our training set we have data linear, but in our testing set we have data in non-linear, this will cause issues for our model, as it only has knowledge of linear data not non-linear. Hence as machine learning engineers we must split our data randomly.

Now in order to split our data randomly we will use another python package know as sklearn short for scikit-learn, sklearn provides with a lot of useful methods that can help us make datasets, split our data intro training and testing sets, which we are doing, and a lot more.

Now let's split our data into training and testing sets.

```python
# import train_test_split from sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 0.2/20% of data will be for testing and 80% will be for training
                                                    random_state=46) # changes the testing data, but stays same if value same
```

As you can see, we have split our data into 2 parts, yes these are 4 variables, but if you notice the little detail, the first are the **Features** split into training and testing features, and the other 2 are for the labels, hence they are 2 parts not 4.

If you see we have defined that the test size should be 0.2 or more mathematically 0.2 means 20%, so that means 20% of our data will be for testing and 80% will be for training, which is a good split and is also the most recommended split you will hear from any machine learning engineer. The next thing you may wander is what is a random state? Now the if I would have explained it in the comment, it would have been to long, so yeah, the comment is not right, but what random state means is, that the randomly chosen values will stay the same as long as you don't change the random_state value, so go ahead and try changing it and see what happens, also rerun it with the same value.

Now if we check what the values will be, which I want you to do, you will see that the values are just boring numbers and I know you all hate maths, sadly you have to get used to it. So let's use the number 1 rule of machine learning engineer, that is, **Visualize the data!**

Let's create a simple function that plots a graph for us with the training and testing data.

```python
# Ploting function
def plot(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=None):

  plt.figure(figsize=(10, 7))

  # training data in green
  plt.scatter(train_data, train_labels, c="g", s=4, label="Training data")

  # test data in red
  plt.scatter(test_data, test_labels, c="r", s=4, label="Test data")

  if predictions != None:
    # Plot the predictions in blue
    plt.scatter(test_data, predictions, c="b", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
```

Nothing to complicated just a simple function using matplotlib to create a graph with the training and testing values. Go ahead call the function and see the magic happen.

![training and testing values in visualization](https://github.com/EzpieCo/ezpie/assets/104765117/3b268314-1f09-4e17-9990-cefc6ea07c92)

And look at that! Now we have a training and testing data visualized! As you can see the testing data is randomly picked, therefore when we will train our model and test it, it will be better then what we could have done in a more easier way, that is to pick all the last 20% of the data.

In the next chapter we will build a model that can adjust it's weight and bais, or more exactly the y axis with only knowing the x axis values.
