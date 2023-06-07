---
layout: ../../layout/PostLayout.astro
title: Understand Linear Regression with Real World Concept
description:  Learn about linear regression, a powerful statistical technique used to predict outcomes. Discover the core concepts, explore real-world applications in real estate and sales forecasting, and see practical code snippets in Python.
published: 5/7/2023
author: Elliot
permalink: /blogs/Understand-linear-regression-with-real-world-concept
image: /images/thumbnail/https://www.canva.com/design/DAFlIDkE_Zk/W3rz0OA3UWparxRdO3SgKw/view?utm_content=DAFlIDkE_Zk&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink
---

In the vast realm of machine learning, where algorithms analyze and make predictions based on data, linear regression stands tall as a fundamental and powerful technique. By leveraging the principles of statistics, linear regression enables us to understand and predict the relationships between variables. In this article, we will demystify the concept of linear regression, breaking down complex terms into simple explanations.

<b> What is Linear Regression? </b> <br>
Linear regression is a statistical technique that helps us understand the relationship between two variables: the independent variable (X) and the dependent variable (Y). It allows us to analyze how changes in the independent variable influence the dependent variable. In simple terms, linear regression aims to draw a straight line that best fits the data points, enabling us to make predictions based on this line.

Let me put it this way, imagine you have a dataset with information about people's ages (X) and their corresponding heights (Y). You want to understand how age affects height. Linear regression helps you find a line that best represents the relationship between age and height. By analyzing this line, you can predict the height of a person based on their age. For example, if the line shows that height generally increases as age increases, you can estimate how tall someone might be at a certain age. Linear regression helps us make sense of data and make predictions using a straight line.

<b> Line of Best Fit</b> <br>
The line of best fit is like having a group of children of different ages and you want to understand if there's a relationship between their age and their height. You measure the age of each child (in years) and their corresponding height (in inches). Now, you can plot these data points on a graph, with age on the x-axis and height on the y-axis.

By using linear regression, you can draw a line that represents the relationship between age and height. This line is called the line of best fit. It's like drawing a straight line that comes closest to all the data points, minimizing the difference between the line and the actual heights of the children.

The line of best fit allows you to see if there's a pattern or trend between age and height. For example, if the line of best fit slopes upward from left to right, it suggests that as children get older, they tend to get taller. On the other hand, if the line of best fit is horizontal, it indicates that there might not be a strong relationship between age and height.

Using this line of best fit, you can make predictions about the height of a child based on their age. For instance, if you have a child whose age is not in your original data, you can estimate their height by finding where their age intersects with the line of best fit on the graph. This prediction gives you an idea of how tall they might be based on the observed relationship between age and height in your data set.

<b> Making Prediction </b> <br>
Linear regression not only helps us understand the relationship between variables but also enables us to make predictions based on this relationship. Let's say we have collected data on the number of study hours (X) and the corresponding exam scores (Y) of a group of students. Using linear regression, we can build a predictive model to estimate a student's exam score based on their study hours. By plugging in the number of study hours into our equation, we can predict the expected exam score.

For instance, consider a scenario where you want to predict a child's height based on their age. You gather data on the ages and heights of several children. By applying linear regression, you can create a model that predicts a child's height based on their age. If you input a child's age into the model, it will provide an estimated height for that age.

<b>Assessing Accuracy</b> <br>
Now that we have our predictive model, we need to assess its accuracy. One common way to evaluate the accuracy of a linear regression model is by calculating the coefficient of determination, also known as R-squared (R²). R-squared measures how well the regression line fits the data points. It ranges from 0 to 1, where 1 indicates a perfect fit. For example, an R-squared value of 0.80 means that 80% of the variability in the dependent variable can be explained by the independent variable(s).

In our height and age example, if the R-squared value is close to 1, it means that the model can accurately predict a child's height based on their age. On the other hand, a lower R-squared value suggests that the relationship between age and height may not be as strong, and the predictions may have less accuracy.

<b> Real-world Applications of Linear Regression </b> <br>

<b>Medical Research:</b> In medical research, linear regression can be used to analyze the relationship between variables and predict outcomes. For example, let's say researchers want to understand the correlation between a person's age and their blood pressure. They can collect data from a group of individuals, including their ages and corresponding blood pressure readings. By applying linear regression, they can determine if there is a linear relationship between age and blood pressure and make predictions about future blood pressure based on age.

<b>Scenario:</b> <br>
Researchers collected data from 100 individuals, recording their ages (independent variable) and blood pressure readings (dependent variable). They can use the following Python code to perform linear regression and make predictions:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Age data
ages = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).reshape(-1, 1)

# Blood pressure data
blood_pressure = np.array([120, 125, 130, 135, 140, 145, 150, 155, 160, 165])

# Create a linear regression model
model = LinearRegression()

# Fit the model using the age and blood pressure data
model.fit(ages, blood_pressure)

# Predict blood pressure for a new age
new_age = np.array([[55]])
predicted_blood_pressure = model.predict(new_age)

print("Predicted blood pressure:", predicted_blood_pressure)
```

<br>
<b>Demand Analysis:</b> Companies often use linear regression to analyze the relationship between product demand and various factors. For instance, a retailer may want to understand how product price, advertising expenditure, and competitor prices affect sales. By collecting data on these variables and applying linear regression, they can estimate the impact of each factor on product demand and optimize their pricing and marketing strategies accordingly.

<b>Scenario:<b> <br>
A retailer collects data on the monthly sales volume (dependent variable) of a specific product and the corresponding average prices of the product (independent variable). They also track the average monthly advertising expenditure for that product. Using this data, the retailer can perform linear regression to determine how changes in price and advertising impact product sales.
<br>

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Product price data
prices = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 32]).reshape(-1, 1)

# Sales volume data
sales = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])

# Advertising expenditure data
advertising = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]).reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Combine the price and advertising data
features = np.concatenate((prices, advertising), axis=1)

# Fit the model using the combined features and sales data
model.fit(features, sales)

# Predict sales for a new price and advertising expenditure
new_price = np.array([[35]])
new_advertising = np.array([[1500]])
predicted_sales = model.predict(np.concatenate((new_price, new_advertising), axis=1))

print("Predicted sales volume:", predicted_sales)
```

<br>
<b> Conclusion </b>

Linear regression is a powerful tool for analyzing and predicting the relationship between two variables. By understanding the core concepts of linear regression, such as the line of best fit, slope, and intercept, we can make accurate predictions in various real-world scenarios. Whether it's estimating house prices or forecasting sales, linear regression enables us to uncover valuable insights and make data-driven decisions. So, the next time you come across a prediction or forecast, remember that behind the scenes, linear regression may be at work,
