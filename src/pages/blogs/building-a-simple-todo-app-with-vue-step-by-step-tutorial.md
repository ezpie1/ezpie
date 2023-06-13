---
layout: ../../layout/PostLayout.astro
title: "Building a Simple Todo App with Vue.js: A Step-by-Step Tutorial"
description: This comprehensive tutorial is tailored for both beginners and those with some experience in Vue.js. 
published: 6/13/2023
author: Elliot
permalink: /blogs/building-a-simple-todo-app-with-vue-step-by-step-tutorial
image: /favicon.svg
---


Are you tired of running into errors and bugs while coding your Vue.js applications? Finding reliable solutions to these issues can be time-consuming and frustrating, especially when you rely solely on search engine results. You may have missed out on some excellent blogs or articles that offer better solutions to your problems. <br>

<h3>Why you need this tutorial</h3>

That's where this tutorial comes to your rescue! We understand the challenges developers face when it comes to troubleshooting and enhancing their Vue.js projects. In this tutorial, we aim to provide you with practical guidance and a comprehensive walkthrough to create a simple yet powerful todo application using Vue.js. By following our step-by-step instructions, you'll gain valuable insights and overcome the common obstacles encountered while coding.

<h3>What to expect</h3>

At the end of this tutorial, you can expect to have a fully functional and impressive todo app at your fingertips. We will cover essential features such as adding, editing, and deleting todos, along with the option to switch between dark and light modes. By mastering these techniques, you'll not only have a practical solution to managing your tasks but also gain a deeper understanding of Vue.js development.

<h3> Prerequisites</h3>

Before we dive into coding, make sure you have the following prerequisites

1. Basic knowledge of HTML, CSS, and JavaScript
2. Node.js installed on your machine.
3. A text editor of your choice (e.g., Visual Studio Code, Sublime Text).
 
<h3>Step 1: Setting up the project</h3>

To create our todo app, we'll use the Vue CLI (Command Line Interface) to set up a new Vue.js project. Open your terminal and follow these steps

<b>1.Install the Vue CLI globally by running the following command</b>
  
```
npm install -g @vue/cli
```
<b> 2. Create a new Vue project by running the following command </b>

```
vue create todo-app
```
<b> 3. Choose the default preset when prompted and wait for the project to be created</b>
 
<b> 4. Navigate to the project directory</b>

```
cd todo-app
```

<h3> Step 2: Creating the TodoList Component </h3>

In this step, we'll create the main component for our todo app. Open the src/components/TodoList.vue file and replace the existing code with the following:

```javascript
<template>
  <div class="todo-list">
    <!-- Todo list goes here -->
  </div>
</template>

<script>
  export default {
    name: 'TodoList',
    data() {
      return {
        todos: [],
        newTodo: '',
      };
    },
    methods: {
      addTodo() {
        // Add todo logic here
      },
      removeTodo() {
        // Remove todo logic here
      },
    },
  };
</script>

<style scoped>
  .todo-list {
    /* Todo list styles here */
  }
</style>
```
  
<h3>Explanation</h3>

We define the template for the TodoList component, which will contain the list of todos.

We declare the todos array and newTodo variable in the component's data. These will store our todo items and the new todo input, respectively.

We create the addTodo and removeTodo methods, which will handle adding and removing todos.

<h3>Step 3: Adding the TodoItem Component</h3>

Next, let's create a subcomponent called TodoItem. This component will represent each individual todo item. Create a new file called TodoItem.vue inside the src/components directory and add the following code:

```javascript
<template>
  <div class="todo-item">
    <!-- Todo item content here -->
  </div>
</template>

<script>
  export default {
    name: 'TodoItem',
    props: ['todo'],
    methods: {
      editTodo() {
        // Edit todo logic here
      },
      updateTodo() {
        // Update todo logic here
      },
    },
  };
</script>

<style scoped>
  .todo-item {
    /* Todo item styles here */
  }
</style>
```

<h3>Explanation</h3>

We define the template for the TodoItem component, which represents an individual todo item.

We declare a prop called todo, which will be passed from the parent component (TodoList) to represent the current todo item.

We create the editTodo and updateTodo methods, which will handle editing and updating todos.
  
<h3>Step 4: Building the Todo App Interface</h3>

Now, let's integrate the TodoList and TodoItem components into our main App component. Open the src/App.vue file and replace the existing code with the following:
  
```javascript
<template>
  <div class="app">
    <div class="container">
      <h1 class="heading">Todo App</h1>
      <div class="input-container">
        <input
          v-model="newTodo"
          class="input-field"
          placeholder="Enter a new todo"
        />
        <button @click="addTodo" class="add-button">Add</button>
      </div>
      <ul class="todo-list">
        <TodoItem
          v-for="(todo, index) in todos"
          :key="index"
          :todo="todo"
        />
    
    </div>
  </div>
</template>

<script>
  import TodoItem from './components/TodoItem.vue';

  export default {
    name: 'App',
    components: {
      TodoItem,
    },
    data() {
      return {
        todos: [],
        newTodo: '',
      };
    },
    methods: {
      addTodo() {
        // Add todo logic here
      },
    },
  };
</script>

<style>
/* App styles here */
</style>
```
<h3>Explanation</h3>

We import the TodoItem component and register it in the App component's components section.

We define the template for the App component, which represents the main structure of our todo app.

We use v-model to bind the newTodo variable to the input field, allowing two-way data binding.

 We use v-for to iterate over the todos array and render TodoItem components for each todo item.

<h3>Step 5: Implementing Todo Functionality</h3>

Now it's time to add the actual functionality to our todo app. Go back to the TodoList component (src/components/TodoList.vue) and update the code as follows:
 
```javascript
<template>
  <div class="todo-list">
    <div class="todo-item" v-for="(todo, index) in todos" :key="index">
      {{ todo }}
      <button class="delete-button" @click="removeTodo(index)">
        Delete
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TodoList',
  data() {
    return {
      todos: [],
      newTodo: '',
    };
  },
  methods: {
    addTodo() {
      if (this.newTodo.trim() !== '') {
        this.todos.push(this.newTodo);
        this.newTodo = '';
      }
    },
    removeTodo(index) {
      this.todos.splice(index, 1);
    },
  },
};
</script>

<style scoped>
.todo-list {
  /* Todo list styles here */
}

.todo-item {
  /* Todo item styles here */
}

.delete-button {
  /* Delete button styles here */
}
</style>
```

<h3>Explanation</h3>

We update the template in the TodoList component to iterate over the todos array and render todo items dynamically.

We add a Delete button for each todo item and bind the removeTodo method to its click event.
In the addTodo method, we check if the newTodo input is not empty before adding it to the todos array.

In the removeTodo method, we splice the selected todo item from the todos array based on its index.

<h3>Step 6: Styling the Todo App</h3>

To make our app visually appealing, we'll add some basic styles using CSS. Open the src/App.vue file and update the style section as follows:

```css
<style>
.container {
  max-width: 400px;
  margin: 0 auto;
  padding: 1rem;
  background-color: #f5f5f5;
  border-radius: 8px;
}

.heading {
  text-align: center;
  font-size: 24px;
  margin-bottom: 1rem;
}

.input-container {
  display: flex;
  margin-bottom: 1rem;
}

.input-field {
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #dddddd;
  border-radius: 4px;
  outline: none;
}

.add-button {
  padding: 0.5rem 1rem;
  border-radius: 8px;
  border: none;
  outline: none;
  font-size: 16px;
  color: #ffffff;
  background-color: #4CD964;
  cursor: pointer;
}

.todo-list {
  list-style: none;
  padding: 0;
}

.todo-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid #dddddd;
  transition: transform 0.3s;
}

.todo-item:last-child {
  border-bottom: none;
}

.delete-button {
  padding: 0.5rem 1rem;
  border-radius: 8px;
  border: none;
  outline: none;
  font-size: 16px;
  color: #ffffff;
  background-color: #FF3B30;
  cursor: pointer;
}

.delete-button:hover {
  opacity: 0.8;
}
</style>
```
  
<h3> Explanation</h3>

We apply styling to different sections of the app using CSS classes.

We define styles for the container, heading, input field, add button, todo list, todo item, and delete button.
 
<h3>Debugging Process</h3>

During the development process, it's common to encounter errors or unexpected behavior. Here are a few tips to help with debugging:

Console Logging: Use console.log() statements to log variable values and debug information. Check the browser console for any logged messages that might help identify the issue.

Vue Devtools: Install the Vue Devtools browser extension, which allows you to inspect the Vue component hierarchy, state, and props in real-time. It can help identify any component-related issues.

Read Error Messages: Pay attention to error messages displayed in the browser console. They often provide valuable information about the cause of the issue. Read the error message carefully and try to understand what it's telling you.

Review Code: Double-check your code for any syntax errors, missing closing tags, or incorrect function names. Sometimes, a simple typo can cause unexpected behavior.

Isolate the Problem: If you encounter an issue, try to isolate it by removing or commenting out sections of code to identify the specific part causing the problem. This process of elimination can help pinpoint the source of the issue.
  
<h3>Conclusion</h3>

Congratulations! You have successfully built a simple todo app with Vue.js. In this tutorial, we covered the step-by-step process of setting up a Vue project, creating components, implementing todo functionality, and styling the app. We also discussed some debugging techniques to help you troubleshoot any issues you may encounter.

Feel free to further customize the app and explore additional Vue.js features to enhance its functionality. Vue.js offers a rich ecosystem of libraries and tools that can be leveraged to add more advanced features to your app.
  
Download project source code <a href="https://github.com/mcjill" class="text-blue-600 hover:text-blue-400 hover:underline">Here</a>
  
  
  
  
