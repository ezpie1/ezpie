---
layout: ../../../layout/CourseLayout.astro
title: "Understanding Classification problems"
permalink: /courses/machine-learning
---

In this chapter we will understand what a classification problem is, what it provides us, and how to solve it.

First of all let's see what a classification problem even is.

In this chapter we won't be writing any code so just seat back and relax.

## What's a classification problem

So let's bring the picture from the <a href="intro" class="text-blue-600 hover:text-blue-400 hover:underline">intro chapter</a>

![Example of a binary classification problem](https://github.com/EzpieCo/ezpie/assets/104765117/6c79b26a-a65e-46b1-8fbd-015a192b36a0)

Now if you remember we also added a line to separate the two categories and added a dot like this. 

<img src="https://github.com/EzpieCo/ezpie/assets/104765117/2f23adeb-37c0-4908-ab4b-d8928d4eba38" alt="a random dot in the dataset" loading="lazy">

As you can see with the help of this line we can say that the green dot belongs to the blue category.

So from this we can say that a classification problem is just basically differentiating or answering to which **class**/category a dot may belong to.

## What is binary classification

We know what classification is so can you guess what binary classification would be?

Just like the name says, binary means 1s and 0s and classification means which category a dot belongs, hence binary classification means which of the **two category** the dot may belong to.

## Multi-class classification

So binary classification means which of the two category so I guess I don't even need to explain what multi-class classification might be.

As the name says it involves classifying weather the dot, or what in real world would be called - a thing, belongs to which category.

Later on this course we will apply this sort of classification for classifying images!

## Diving a bit deep

Now we now the different types of classification let's dive a little deeper and see what can we get.

Let's take an example from our real world and see where can we use binary classification.

1. Depicting weather an email is spam or not spam
2. Fraud Detection in Financial Transactions
3. Product Recommendation

These are just examples you can search for even more and the list will never end!

Depicting weather an email is spam or not spam is a classic example, but for this google doesn't use pytorch rather something they only created that is TensorFlow.

Fraud detection is also a great example as one can't just keep an eye in a screen and watch a bunch of numbers turn green to red, cause there can be millions or not at least billions of transactions a second, and something like a computer comes to the rescue, with the capability to read at 1K words a millisecond why not just put a computer trained to do so.

Product recommendation is more on something which we all have seen, cause literally while searching on amazon we get all what we need in the top is because of this only.

So now we know some stuff about classification, from the next chapter onward we will focus on binary classification, and why not just try using the example we have since the intro chapter and remove the curiosity of how can we teach a computer to classify this.