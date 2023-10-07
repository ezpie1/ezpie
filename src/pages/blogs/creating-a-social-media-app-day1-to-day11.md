---
layout: ../../layout/PostLayout.astro
title: "Creating a social media app: Day 1 - 11"
description: This is part 1 of the series where I would be posting about the progress of our new project called lambda, which is a social media app. In this part I will show you what we did from day 1 till day 11, from naming, logo designing and designing the over all app using figma and then final today went into starting with code, setting up the backend and frontend.
published: 10/7/2023
author: Ezpie
permalink: /blogs/creating-a-social-media-app-day1-to-day11
image: /images/thumbnail/making-a-social-media-app.png
---

Hi everyone! It's me Ezpie, and today I have a special gift for you all! That is... we are creating a social media app!

This is a big project on which We are working on and this is a series of blog post made for that.

In this series we will show you guys how we start a project, from designing to selecting the backend and frontend stuff and all other things as well.

## Day 1: Selecting name and logo

Day one was simple, I picked a name... And what name you ask?

Lambda!

As you may know, if you have used python, lambda means an anonymous function without a name. But Lambda is also a greek letter that means, in science, wavelength.

So what does lambda the social media app mean?

Well it's none of the above!

Lambda is just the name of the bird of our logo, I know it's like stealing twitter's logo, but hey, it's X now and there is no such thing as twitter any more!

So lambda is just a bird looking towards, where it's beck is pointing, to find something to eat, just like the users looking for something to read.

## Day 2 - 10: Design

Just a thing, we made this post after we were done with the design,  so saying, today is the day for the code, but let me explain what all we did in the design stage.

We already have a tool for making design, that is figma.com, which is a really cool app to use for making designs.

### Selecting color palette

First let's select the color palette, I usual use the coolors.co for this, this tool allows me to pick some colors I have in mind and then press the space bar again and again.

Just visit the site and you will get what I mean.

### Creating the designs

So for the 10 days time I got into designing the website, like creating the sign up page, login page, homepage, post page, and finally the profile view page. That's a lot of pages to work on!

![Designs for lambda](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/34pg3gopl1akxkyb0ar6.png)
_Image of the designs_

As you can see it took 10 days for all this!

I'm not much of a designer and for me this is a bit difficult, but if anyone of you guys are interested... well to start a social media app you can join us, cause sadly Ngdream and Elliot, both have got work to do, for which they aren't involved in this project much.

Lest to say, the design is good enough from my POV.

## Day 11: Selecting the architecture and starting with code

Now we are in the part which I can really understand!

For the architecture, I went simple, and by simple I mean really simple, the backend is basically using Supabase for auth, database, and storage.

I know some of you may think this is a bad idea, but hey I'm no expert, I'm still learning thing you know?

For the frontend, which I would say I'm more experience in is also simple, just Nextjs, simple and easy. Nothing complicated for me.

### Setting Supabase

Time to setup Supabase, I opened Supabase and creating a new project in it.

I did what every starting point for a project had to do... Read the docs!

I had to laterally read the documentation about each and everything from how to setup supabase with nextjs and other things. I can't remember things for much long you know?

Anyways I setup supabase by creating a bunch of tables, Like user table, which will have information about username list of followers, detail information provided by user. That's all for table users.

Table for blogs, having information like, author/username, content, and number of likes.

For now, that's all I think is impotent to focus on.

Next was to setup the env variables, that is, the supabase url and anon key.

These are the 2 important values without which you cannot have access to your project supabase

I have already created the nextjs project for this so all I had to do was setup the env variables before doing anything... well before that install some supabase related packages also.

## Creating account

Now let's get into the real fun.

To get started all I did was created a middleware using supabase auth helper library.

This middleware will only allow signed in users to view and all log out users need to first sign in or login in order to proceed.

I'm not going to show any code as that would be giving away to much of secret information... Nan just joking.

The code will be available soon once we have this project completed, so just [follow us on GitHub](https://github.com/ezpie1) to stay updated.

### Signup form

OK time to make the sign up form

This is by far the most dumbest part for me, because I spent almost  hours in just removing the black ring showing up when ever you focus on the input!

Did took a lot of time but final got fixed.

Also I ran into many errors which were as following:

- Server-side signup for adding information into user table problems
- Couldn't generate types because of `npx supabase login` not working
- Login server-side errors
- environment variables naming errors

Yeah a lot of dumb ones they were but hey! It finally works! I can now signup for a lambda account, only thing is that there is noting to see in the homepage!

## Ending... for Now

That's all for today I will be posting these blogs everyday to keep you all updated about our work in this big project.

Till then have a happy hacking!

Wait wait wait!

I'm not done yet!

If you want to stay updated about all this, we also have a newsletter which you can join by just scrolling up and filling the small form in the footer of our website.

We will keep all newsletter joiners updated about this so do join if you want to see a new app in the web.