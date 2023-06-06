---
layout: ../../../layout/CourseLayout.astro
title: "A PyTorch Model's journey"
permalink: /courses/machine-learning
---

Every machine learning engineer or deep learning engineer has one essence, that is to take data, build
a model(neural network) to understand patterns in that data and predict what the value will be of an unseen data.

From the next chapter onward we will focus more on the fun part, that is making data, building a model to understand that data and predict with data.

To Start of **straight** and simple we will learn how to make a model that can predict values in a straight line, a simple straight line.

We will build a pytorch model that will learn the pattern in that straight line and make predictions.

# The simple workflow

We will cover a really, yet a very common workflow, with the help of which we will create our model.

![most common pytorch workflow](/images/courses/ml/pytorch-workflow.svg)

This is a straight and simple workflow, with the help of this workflow we will be able to build a simple pytorch model out of the box!

<table class="table-auto border-collapse border border-slate-400">
  <thead>
    <tr>
        <th class="border border-slate-300 p-2">
            Sections
        </th>
        <th class="border border-slate-300 p-2">
            What they are
        </th>
    </tr>
  </thead>
  <tbody>
    <tr class="hover:bg-slate-200 duration-300">
        <td class="border border-slate-300 p-2">
            Get some data
        </td>
        <td class="border border-slate-300 p-2">
            The most important thing to do, without data there will be no perfect model.
            But for now we will work with a straight line.
        </td>
    </tr>
    <tr class="hover:bg-slate-200 duration-300">
        <td class="border border-slate-300 p-2">
            Build the model
        </td>
        <td class="border border-slate-300 p-2">
            This is the stage where we will build the model, pick a loss function and optimizer for our model to work with.
        </td>
    </tr>
    <tr class="hover:bg-slate-200 duration-300">
        <td class="border border-slate-300 p-2">
            Fit the model with the data
        </td>
        <td class="border border-slate-300 p-2">
            This is the training loop, where we will adjust our model to fix the data. The testing loop also gets involved here.
        </td>
    </tr>
    <tr class="hover:bg-slate-200 duration-300">
        <td class="border border-slate-300 p-2">
            Predict with the model
        </td>
        <td class="border border-slate-300 p-2">
            Same as the test loop, yet a bit different, as we will use more data.
        </td>
    </tr>
    <tr class="hover:bg-slate-200 duration-300">
        <td class="border border-slate-300 p-2">
            Improve mode with Ideas and Experiments
        </td>
        <td class="border border-slate-300 p-2">
            We will use the try and error method to find the perfect setup for our model so that it will improve better.
        </td>
    </tr>
  </tbody>
</table>

We will be covering each and ever single step so that it's in the tip of your tongue. Like this you will be able to make your own pytorch model without the help of this workflow.
