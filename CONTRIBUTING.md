# How to Contribute to Ezpie

Ezpie is not just another free and open-source site for average developers. It's for learning by teaching, it's not easy to code by just learning how to, but when you share your experience with others, you learn more then just how to! 😃

**WARNING:** Before you proceed, please take a minute to read ezpie's [Code of Conduct](https://github.com/EzpieCo/ezpie/blob/master/CODE_OF_CONDUCT.md). We enforce it across our repo so that contributing to ezpie is a fun experience for everyone.

---

## Dependencies and Requirement

Most of Ezpie is written with [astro](https://astro.build). Like this there are not limits for anyone, if you are a react developer you can add react components, vue, svelte etc all are always together!

If you want to help, but don't know react, now you can use vue components instead! So spacial thanks to astro!

---

# Setting up your system for better development

To make development easy you can install the vscode extension [astro](https://marketplace.visualstudio.com/items?itemName=astro-build.astro-vscode)

After that fork this repository and clone it into your system, then run the `setup.sh` script in order to install the node modules and run the dev server

```
git clone <your-fork>

cd script
chmod +x setup.sh
./setup.sh
```

## How to contribute?

To get started you can follow any of these steps:

3. If you have a post for [Pull Request follow these steps](#blog-pr)
4. If have discovered an issue with the style or with any other thing you can [follow these steps to report](#issue-filling)

<div id="blog-pr">
  <h2>How to Submit a PR for Blog?</h2>
  <p>To submit a PR for blog just follow these steps:</p>
  <ol>
    <li>Create a file in the <em>src/blog/*.md</em></li>
    <li>Copy paste this frontmatter in the top:</li>
    <br>
    <pre>
      ---
      layout: ../../layout/PostLayout.astro
      title: Some good title
      description: Give a nice one para description(anything between a line or para)
      published: the date when you wrote it - month/day/year format only
      author: your name(not needed if you want to be unknow)
      permalink: /blogs/file-name
      image: /images/thumbnail/thumbnail(if you have)
      ---
    </pre>
    <li>If you have a nice thumbnail you can add that too in the <em>public/images/thumbnail/*.[png, svg, any]</em></li>
    <li>Once you are done you can submit a PR</li>
  </ol>
</div>

<div id="issue-filling">
  <h2>How to submit an issue?</h2>
  <p>To file an issue just follow these steps</p>
  <ol>
    <li>Go to the <a href="https://github.com/EzpieCo/ezpie/issues">issue tab</a> to make sure there is not same issue</li>
    <li>Open an issue with the required templete</li>
    <li>If your issue is a style or code one then use the <i>code</i> and corresponding language label. eg: in case of a tailwind error use the code and tailwind label</li>
  </ol>
</div>

## That's it on your first Ezpie contribution 😄

That's all it takes to make your first step in contributing to ezpie! Just follow the steps and you are all done!

If you ever need any help you can always jump into making an issue or jump into our [discord server!](https://discord.gg/jR7fjqSCDk)
