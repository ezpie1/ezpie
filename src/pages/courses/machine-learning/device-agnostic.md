---
layout: ../../../layout/CourseLayout.astro
title: "Why use device agnostic and setting up"
permalink: /courses/machine-learning
---

So in this chapter we won't do much of the actual code, but rather we will discuss how can we increase the speed and computational power.

Though this may not sound of any importance for now, cause what we are doing doesn't requires a GPU, but later on this course you will find it time consuming to wait for your model to train as it will take up a lot of CPU, resulting in a lot of time consumption.

# What is device agnostic and Why setup

Device agnostic refers to the capability of a software or system to work seamlessly across different devices or platforms without requiring specific modifications or adaptations for each device.

In the context of machine learning, device agnostic refers to algorithms or models that can run efficiently and effectively on various hardware devices, such as CPUs, GPUs, etc. A device-agnostic model is not tightly coupled to a specific hardware device and can adapt to different devices without major changes to its implementation.

## Setting up device agnostic

The first thing to do is to setup a device agnostic, that means check if GPU is available or not. This can be done in a simple one liner code like this:

```python
torch.cuda.is_available()
```

Now cuda is a parallel computing platform and API model created by NVIDIA. Cuda allows machine learning engineers like you and me to use GPU for computational purpose. It's the most use GPU and therefore is also pytorch's default GPU.

The output of the code block above will tell us weather we have a GPU or not, if not just change your runtime to GPU and rerun all the code blocks.

Now that we know we have a GPU we can setup a device agnostic with this one liner:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

One line of code, but a really big helpful one. Now let's try to make a tensor and put it into the device agnostic.

```python

# By default the tensor will be in CPU
tensor = torch.arange(1., 10.) # float32 just for fun! You can use any type

tensor, tensor.device # This will not be in GPU
```

Now that we have a tensor and we know that it's in a CPU, let's put that tensor into a GPU for faster computing. For this we can add just one simple word to the tensor, `to(device)`. This method helps us to put our tensors into the device agnostic which is available.

```python
tensor = torch.arange(1., 10.).to(device) # Puts the tensor into the device

tensor # This will result in the tensor and the device it's in
```

# Getting into error and fixing them

We can get into errors where if any one of the tensors are on another device while the other is/are on another, then performing any kind of action with the tensors with different devices can result in errors.

In order to fix these errors we need the tensors to be on the same device, GPU or a CPU.

First let's create 2 tensors in different devices.

```python
# This one will be on a CPU
CPU_tensor = torch.arange(1., 10.)

# This one will be on a GPU
GPU_tensor = torch.arange(2., 11.).to(device)

CPU_tensor, GPU_tensor
```

As you can see the CPU tensor is in the CPU and the GPU tensor is in the GPU(cuda). Now if we try to perform any kind of action, such as adding the 2 tensors, we will get a device error.

```python
ErrorSum = CPU_tensor + GPU_tensor # This will give an error

ErrorSum
```

This will give us this fancy looking error:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

From this we can say that in order to perform any action with our tensors, they must be on the same device.

# Tasks

So your tasks for today are as following:

1. Try finding a solution to the problem we had when adding the 2 tensors, on finding the solution head over to the discord server and go to the answers channel and be the first one to answer this question.

2. Make observation, make two tensors of the same type value and everything, but put one tensor in GPU and one on CPU, try to find out how much time it takes to perform any kind of action on them like adding 2 GPU tensors, adding 2 CPU tensors. Find the answer and show me the results and the answer channel at the discord server.
