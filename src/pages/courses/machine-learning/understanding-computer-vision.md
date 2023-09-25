---
layout: ../../../layout/CourseLayout.astro
title: "Understanding Computer Vision"
permalink: /courses/machine-learning
---

Now we are getting into the cool stuff.

From this chapter onwards we will be training a model to learn patterns in an image and predict!

But first let's understand a bit about computer vision problems so that when we will code you can understand most of it.

## Computer vision Flow

First let's understand what the computer vision flow is.

Say you have an image like this.

![a 25 by 25 image of five](https://user-images.githubusercontent.com/104765117/270110804-897b7cb8-d435-4a4e-b23c-fb8eb055da45.png)

were pixel image yeah?

As you can see this is an image of a number five(hint, this is what we will work on).

But for a computer what can we think about this? How can we think about this? And how we will learn patterns in this?

Let's answer these questions one by one.

Now This image is of 25 by 25 and the colors in it are black and white, so can you think of a pytorch dimension for this image?

No you are wrong, well unless you are right, the answer is `[1, 25, 25]`.

Let we explain this tensor, the tensor consists of the color channel, width and height of the image respectively. Since PyTorch follows color channel first that's why we have the **1** before and not at the end, other library's may take color channel last.

Think about it, the image consists of only black or white right? So that means either the value of one pixel will be 1 or 0 or between them, that's where the 1 comes from.

In real world problems you may get images consisting of 3 color channels that are _RGB_ or **R**ed **G**reen **B**lue.

So yeah that's just how the dimensions of our image will be shown as a tensor.

Now we need to process this image, which of course the neural network will do and not us.

OK we got our image as a tensor we have done some maths now we have to predict if it's one thing or another.

Now here's the problem, our last layer will of course have linear options, meaning one thing or another in values ranging till the number of labels our model is trained for.

If you think about it our output size should be a vector, cause if you remember our multi-classification model's predictions after turning into prediction labels were a vector of 5 values, all adding up to 1.

In that same manner we need our second last layer's output to be a vector so that we can perform some linear regression formula and get a vector shape output with all values adding up to 1.

In case that sounds intimidating, don't worry, we will write it in code later on. For now let's see this as a visual representation.

![Process an image goes from for computer vision](https://user-images.githubusercontent.com/104765117/269628199-d2864068-4ad8-4cb4-b9c0-3273bc9a03f8.png)

Now I know that the image is not the same image but just replace it in your mind for now.

Now say that the image intend of a dog is of the number 5 in black and white with height and width as 25.

What will the dimensions be? Yes [1, 25, 25].

Now intend of processing the image as Red Green Blue we will process it as black and white. Where ever the pixel is black we will give that pixel a value of 0, or in case the pixel is white we will give 1 and so on from the top left pixel till the bottom right one.

This is absolutely be done by our model and we won't need to do much.

Now let's go step by step.

First the processed image is converted into tensor data.

Then the tensor is passed into the model. After wards the model uses it's first layer, convolutional layer, and passes it into the ReLU activation function, which then passes the tensors into the pooling layer.

**NOTE:** This step can repeat again and again in case of large networks.

After wards the tensor data is passed into a linear layer which apply's linear regression formula and we get our output us a 5.

So I guess this visual representation of what I said moments ago explains what exactly is happening with a 25 by 25 black and white image and how the network can find patterns in the image so that it could guess that the image is showing 5 and not just any random thing.

## Convolutional Neural Network(CNN)

So you heard we saying convolutional layer right?

So the question for now is what is a convolutional layer or convolutional neural network more exactly?

Now let's go a bit in the english, convolution means a thing that is difficult to follow right? Even google says so.

So A Convolutional neural network can most probably mean a neural network built to solve difficult to follow problems right?

Well kind of yes.

But more exactly it means that a convolutional neural network is best of image recognition and can be best explain using this simple step by step process for solving a puzzle.

So just imagine you have a puzzle made up of small pieces, say 625, just like our image 25 by 25, and you want to figure out what the big picture is. A CNN works a bit like this:

First, look at the tiny pieces of the image, like small squares of 3 by 3, meaning 9 pixels at a time and examines them one by one. Similar to how you might start with one piece of a puzzle.

Next the CNN isn't just randomly looking at these pieces, it's looking for specific patterns, like edges, colors, or shapes. It's like trying to find pieces with matching edges in your puzzle.

After checking lots of these small pieces and finding patterns in them, the CNN gradually starts to build up a bigger understanding of the whole image. It's like how you slowly complete your puzzle by putting pieces together.

And once the CNN has looked at all the pieces and found patterns, it makes a guess about what the whole picture is. It's as if you finally see the complete image in your puzzle.

This is probably the best way to understand what CNN models really do, look at chunks of pixels, understand patterns, and final make a guess.

Don't worry we will right it in code and also I have a spacial website to display all the things in a CNN model, to show you later on.

For now just stick with the puzzle solving idea.

## Pooling layer

This is the last layer in our example image shown above, but what is a pooling layer? Does it consists of a pool?

Well of course not, but what is does is kind of sample of what our convolution layer does.

Say you take 3 by 3 square from the top left corner of the 25 by 25 image showing 5.

Now our convolution layer learned a bit of patterns in that area right? So it guessed a little that some pixels have a bigger value. Like say you took the 3 by 3 square and show that a pixel had a large value in that 3 by 3 square because our convolution layer thinks it's important.

Now that our convolution layer is the main player so we remove all the other pixels but keep the one with the lager value.

Kind of like finding the maximum value among 9 different values.

Didn't understood what I said? Just hang on we are getting to a more visual part.

For visualizing what is happening let's pay a visit to a really nice site that uses CNN to examine few sample images and guesses what they are.

Let's head over to <a href="https://poloclub.github.io/cnn-explainer/" class="text-blue-500 hover:text-blue-600">CNN Explainer</a>

This is my personal favorite site when I want to create or understand about CNNs.

## The CNN Explainer Site

Now let's see what is it this CNN explainer site.

First let's try understanding what happens if you click on a neuron on the first convolution layer.

As you can see(I won't display so you have to visit the site) the input layers and the neuron are highlighted.

Nothing much right? Now let's click on the first input and see what's happening behind the scenes.

You may see we have a small panel that shows the input image and how a small 3 by 3 square is moving and adding all pixel values and displaying it on the right. You may have notice that the image turned from 64 by 64 to 62 by 62.

That's what happens in a convolution layer, kind of like a pooling layer right? Yes a convolution layer is just a pooling layer that understands patterns in the input tensor and then learns from it and passes it to the next layer, that is the ReLU activation function.

Now the values at the end are again passed into another convolution layer which learns even more patterns in it, and since I told you that a convolution layer is kind of like a pooling layer, the dimensions mentioned right above each layer also goes from 62 to 60, 60 to 30 and so on till the end of the network.

So why is it happening? Well thats cause **the model wants to learn the most important details in the image**.

Yes! The model wants only important details, unnecessary details are a waste of resources and time.

So from the start the dimensions are [3, 62, 62], which after being passed to the convolution layer gets converted into [3, 60, 60] and goes on till it reaches the end.

Like this the model makes the image so small that at the end only important parts of the image are left, if you click the last neuron at the second last layer(the layer before the linear layer), you will see that the image is a lot more pixelated.

So know you probably understood why our image is being so pixelated at the end compare to what it is at the beginning.

## Terms in a convolution layer

Let's end this chapter with the final few terms or hyperparameter or things which we can adjust as per our wants at make the neural network.

First scroll up till you reach a headline saying Understanding Hyperparameter

### Input size

This is usual the size of the image, in the case of the site, 62 by 62.

### Padding

Try to adjust this, you will see that we get and extra amount of pixels that the outline of the square.

Padding simply conserves data at the borders of the activation maps or the image, which leads to better performance, and it can help preserve the input's spatial size, which allows an us ML engineers to build deeper, higher performing CNNs.

### Kernel

See that small 3 by 3 square just hanging out moving from top left to bottom right?

That's the kernel, the job of a kernel is to extract information, just like what I said for convolution layer, they extract some really important information from the image and deduct the image size with more important information.

If you pick a small kernel size say like 2 by 2, for an input size of 7 by 7, you can see in the display(in CNN explainer) that the output size is 6 by 6. This leads to better performance, as the model learns a lot of information and we can even build a larger neural network having many convolution layers in it.

If you were to pick a larger kernel size, say 5 by 5, for an input size of 7 by 7, you can see that the output size is a lot more smaller and we can't add another convolution layer, cause of course the input size then will be 4 by 4 and the kernel 5 by 5, doesn't makes sense. Also this is bad for performance as the model won't learn much information about the image and will most of the time make bad decisions.

### Stride

The stride is just a value by which the kernel should move. Say you take an input size 7, kernel size 3 and a stride of say 2, you will see that the kernel now moves 2 pixels at a time, and even the output size gets smaller, cause from every 9 pixel one most valuable pixel is being picked at some are even being skipped because of the stride.

OK this is good enough for now.

In the next chapter we will create some data, or to be more exact we will use a dataset called MNIST which has a collection of handwritten numbers from 0 to 9.

Sounds cool, see you there.
