---
layout: ../../layout/PostLayout.astro
title: Building a Modal Using Angular and TailwindCSS
description: In this blog learn how to create modal forms in angular using tailwindCSS. Modern design principles are at the heart of this approach, ensuring a seamless and engaging user experience. Unravel the code behind creating elegant modals, complete with responsive design, intuitive user controls, and dynamic form fields. 
published: 19/08/2023
author: Ayush Agarwal
permalink: /blogs/build-modal-angular-tailwindCSS
image: /images/thumbnail/build-modal-angular-tailwindCSS.svg
---

Greetings, fellow developers! In our coding journey, we often find ourselves tasked with crafting forms. Initially, our go-to approach might involve fashioning a button or hyperlink that, when triggered, whisks us away to a separate form page. However, this technique, while tried and true, can demand additional exertion from users as they navigate through various sections. Intrigued by a more efficient approach? Allow me to introduce you to the concept of "**Modals**."

Rather than redirecting users to an entirely new page or tab, modals elegantly present the form right within the current page's context. If you're still a tad unclear, don't fret. We'll delve deeper into this concept by taking a sneak peek at the snapshot of the final page we're embarking on building today.

%[https://vimeo.com/855911949?share=copy] 

When the button is clicked, we observe the form comes right on the same instead of going to a new page.

Now let's get into building this simple modal.

### Pre Requirements

* Basic knowledge of Angular components and directives.
    
* Experience in using Tailwind CSS.
    

### Project Setup

Next, we will set up our angular project and integrate tailwind CSS.

Use the below command in your terminal to create an Angular project and navigate into it.

```bash
 ng new angular_tailwind
 cd angular_tailwind
```

With this our initial angular project is ready. Next, we need to set up Tailwind CSS in our project. It is a small effort and I already have a blog published to help you achieve it. Find the link to it below:

[Integrate Tailwind CSS in your angular application](https://blogs.ayushdev.com/how-to-integrate-tailwind-css-in-your-angular-project)

To test that your initial setup is working correctly, use the `ng serve` command to start the application and it should run without any errors.

### Build the Modal

#### Form Button on Home Page:

Firstly we will create a button on the homepage upon click of which the modal will open. We use tailwind CSS to create a simple button on the home page.

```xml
<div class="flex items-center justify-center h-screen bg-teal-500">
  <button class="bg-white text-teal-500 py-2 px-4 rounded shadow">
    Fill Form
  </button>
</div>
```

#### Modal Component:

It is a good practice in Angular applications to create new components in a dedicated folder. Therefore create a folder named **components** inside `src/app`.

Next in the terminal execute the below command to create the modal component to be used in our application.

```bash
ng generate component components/form-modal
```

After this, our modal component is ready. Open the `form-modal.component.html` file. In this, we write the HTML for our form modal.

```xml
<div *ngIf="isModalOpen" class="fixed inset-0 flex items-center justify-center z-50">
    <div class="fixed inset-0 z-40 bg-black opacity-50"></div>
    <div class="bg-white w-1/4 p-6 rounded-lg shadow-lg relative z-50" (mouseleave)="closeModal()">
      <h2 class="text-xl font-semibold mb-4">Details</h2>
      <form>
        <div class="mb-4">
            <label for="firstName" class="block text-gray-700">First Name:</label>
            <input type="text" id="firstName" name="firstName" class="w-full border border-gray-300 px-3 py-2 rounded-md">
          </div>
          <div class="mb-4">
            <label for="lastName" class="block text-gray-700">Last Name:</label>
            <input type="text" id="lastName" name="lastName" class="w-full border border-gray-300 px-3 py-2 rounded-md">
          </div>
        <div class="mb-4">
          <label for="phone" class="block text-gray-700">Phone:</label>
          <input type="tel" id="phone" name="phone" class="w-full border border-gray-300 px-3 py-2 rounded-md">
        </div>
        <div class="text-right">
          <button type="submit" class="bg-teal-500 text-white px-4 py-2 rounded-md">Submit</button>
        </div>
      </form>
      <button class="absolute top-0 right-0 m-2 text-gray-700 hover:text-gray-900" (click)="closeModal()">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
        </svg>
      </button>
    </div>
  </div>
  
```

#### Understanding the Code:

Let's break down the essential components of this code and understand how it enables the creation of engaging and user-friendly form interfaces.

* **Modal Display Logic**: The `*ngIf="isModalOpen"` directive controls the visibility of the modal. When `isModalOpen` is true, the modal content becomes visible, allowing users to interact with the form within it.
    
* **Overlay and Background Dimming**: To bring focus to the modal content, an overlay is created using a `fixed` element. This overlay, with a semi-transparent black background (`bg-black opacity-50`), subtly dims the underlying content.
    
* **Close Button**: A close button in the top-right corner lets users dismiss the modal. It's implemented with an SVG icon that, when clicked, triggers the `closeModal()` method. This method can be customized to toggle the `isModalOpen` flag, closing the modal.
    
* **(mouseleave)="closeModal()"** closes the modal automatically when the mouse cursor leaves the modal making the UI more interactive.
    

Next, we need to implement the logic to close and open the modal in the `form-modal.component.ts` file.

```typescript
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-form-modal',
  templateUrl: './form-modal.component.html',
})
export class FormModalComponent implements OnInit {

  constructor() { }
  ngOnInit(): void {
  }

  isModalOpen = false;

  openModal() {
    this.isModalOpen = true;
  }

  closeModal() {
    this.isModalOpen = false;
  } 
}
```

This code defines the logic behind an Angular component responsible for creating interactive modals.

* **Modal State**: The variable `isModalOpen` is initialized as `false`, signifying that the modal is closed by default.
    
* **Modal Interaction Methods**: The `openModal()` method sets `isModalOpen` to `true`, enabling the modal's visibility. Conversely, the `closeModal()` method sets `isModalOpen` to `false`, hiding the modal.
    

#### Including the modal component

In the final step, we just need to use the modal-form selector in `app.component.html` to render the modal on our page.

```typescript
<div class="flex items-center justify-center h-screen bg-teal-500">
  <button class="bg-white text-teal-500 py-2 px-4 rounded shadow" (click)="formModal.openModal()">Fill Form</button>
</div>

<app-form-modal #formModal></app-form-modal>
```

* **Click Event**: The `(click)="formModal.openModal()"` event handler is attached to the button. When the "Fill Form" button is clicked, this handler triggers the `openModal()` method of the `formModal` instance.
    
* **Modal Component**: Just below the button, the `<app-form-modal #formModal></app-form-modal>` element is used to incorporate the previously defined `FormModalComponent`. This is where the modal's logic resides.
    

### Launch the Application

That's it, we have come to the end of creating a modal in Angular. Execute the `ng serve` command in terminal and voila!!.

In conclusion, interactive models in Angular offer a modern solution to streamline user interactions. By seamlessly presenting forms and content within the existing context, users can engage without the disruption of navigating to new pages. This approach not only enhances user experience but aligns with contemporary design principles. With a combination of well-structured HTML, dynamic component logic, and thoughtful user interface elements, developers can create a user-friendly environment that encourages meaningful engagement.

[Project Github URL](https://github.com/ayushhagarwal/angular_tailwind_modal)

Lastly, Your support keeps me going, and I give my best to these blogs! If you’ve found value, consider fueling the blog with a coffee ☕️ donation at the below link.

[**Buy me a COFFEE!**](https://www.buymeacoffee.com/ayushdev)

Thank you! 🙏