---
layout: ../../../layout/CourseLayout.astro
title: "Improving the model"
permalink: /courses/machine-learning
---

We have the data, we have the model and the model has learned patterns in the data.

Just the thing is that the model is still not perfect.

And sadly no one likes incomplete perfection.

So the main discussion of this chapter will be how to improve a model.

## Improving the model

In fact we can do improvement by 2 where easy methods... well one is a bit of luck one.

The two methods we can use are:

- Improve the dataset
- Improve the model itself

By improve the dataset, what I mean is that we can just add more data points in the dataset, like give more data.

This is a lot common if you will work for big tech companies. They have a lot of data and you can train a model with that tons of data.

But we don't always get this advantage, somethings we may have a billon of data(like GPT-4), but yet the model may not perform good.

At this stage we have to go for the alternative, improve the model itself.

## Improving the model

To improve the model we can either do any of these things or do weird combos of them.

- Add more hidden layers
- Add more neurons per hidden layer
- Train for a bit longer

And since I already now the answer, let's just do that. That is train for a bit more longer.

Let's do a 1000 more epochs by running our good function - `trainer2()`. See how useful it is to have functioned things?

And here's the output:

```
Epoch: 0 | Train loss: 0.60726 | Train accuracy: 80.12% | Test loss: 0.62644 | Test accuracy: 74.00%
Epoch: 10 | Train loss: 0.60288 | Train accuracy: 80.25% | Test loss: 0.62246 | Test accuracy: 76.00%
Epoch: 20 | Train loss: 0.59825 | Train accuracy: 82.38% | Test loss: 0.61831 | Test accuracy: 77.00%
Epoch: 30 | Train loss: 0.59339 | Train accuracy: 82.88% | Test loss: 0.61394 | Test accuracy: 77.50%
Epoch: 40 | Train loss: 0.58828 | Train accuracy: 82.38% | Test loss: 0.60941 | Test accuracy: 78.00%
Epoch: 50 | Train loss: 0.58293 | Train accuracy: 82.75% | Test loss: 0.60465 | Test accuracy: 78.00%
Epoch: 60 | Train loss: 0.57734 | Train accuracy: 82.75% | Test loss: 0.59962 | Test accuracy: 78.00%
Epoch: 70 | Train loss: 0.57149 | Train accuracy: 82.62% | Test loss: 0.59433 | Test accuracy: 78.00%
Epoch: 80 | Train loss: 0.56539 | Train accuracy: 83.00% | Test loss: 0.58876 | Test accuracy: 79.50%
Epoch: 90 | Train loss: 0.55902 | Train accuracy: 83.62% | Test loss: 0.58298 | Test accuracy: 80.00%
Epoch: 100 | Train loss: 0.55239 | Train accuracy: 84.38% | Test loss: 0.57698 | Test accuracy: 80.00%

...

Epoch: 900 | Train loss: 0.12588 | Train accuracy: 96.50% | Test loss: 0.14579 | Test accuracy: 94.50%
Epoch: 910 | Train loss: 0.12570 | Train accuracy: 96.00% | Test loss: 0.14555 | Test accuracy: 94.50%
Epoch: 920 | Train loss: 0.15099 | Train accuracy: 95.12% | Test loss: 0.17568 | Test accuracy: 91.50%
Epoch: 930 | Train loss: 1.07601 | Train accuracy: 58.63% | Test loss: 1.06462 | Test accuracy: 61.00%
Epoch: 940 | Train loss: 0.19751 | Train accuracy: 92.25% | Test loss: 0.19443 | Test accuracy: 91.00%
Epoch: 950 | Train loss: 0.13850 | Train accuracy: 96.75% | Test loss: 0.16344 | Test accuracy: 94.50%
Epoch: 960 | Train loss: 0.13368 | Train accuracy: 97.12% | Test loss: 0.15692 | Test accuracy: 94.50%
Epoch: 970 | Train loss: 0.13009 | Train accuracy: 97.00% | Test loss: 0.15215 | Test accuracy: 95.00%
Epoch: 980 | Train loss: 0.12714 | Train accuracy: 96.62% | Test loss: 0.14852 | Test accuracy: 95.00%
Epoch: 990 | Train loss: 0.12435 | Train accuracy: 96.62% | Test loss: 0.14538 | Test accuracy: 95.00%
```

As you can see we have reached a stage we all were waiting for, that is our model is 95% accuracy, not a 100, but close, and anyway it would have still been the same after you run the `ploter()` function.

Functions, aren't they just too useful?

![decision boundary of our model after training longer](https://user-images.githubusercontent.com/104765117/265467490-e5bafd52-6b82-49bf-9f04-26f9213a9d8c.png)

With this we have came to an end to binary-classification, and from the next chapter we will create data for multi-classification, make a model and train that model.

Also we will come across a new loss function and a replacement for **sigmoid**, for multi-classification and not binary-classification.
