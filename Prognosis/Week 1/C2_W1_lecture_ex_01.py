#!/usr/bin/env python
# coding: utf-8

# # Course 2 week 1 lecture notebook 01
# # Create a Linear Model

# ## Linear model using scikit-learn
# 
# We'll practice using a scikit-learn model for linear regression. You will do something similar in this week's assignment (but with a logistic regression model).
# 
# [sklearn.linear_model.LinearRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

# First, import `LinearRegression`, which is a Python 'class'.

# In[9]:


# Import the module 'LinearRegression' from sklearn
from sklearn.linear_model import LinearRegression


# Next, use the class to create an object of type LinearRegression.

# In[10]:


# Create an object of type LinearRegression
model = LinearRegression()
model


# Generate some data by importing a module 'load_data', which is implemented for you.  The features in `X' are: 
# 
# - Age: (years)
# - Systolic_BP: Systolic blood pressure (mmHg)
# - Diastolic_BP: Diastolic blood pressure (mmHg)
# - Cholesterol: (mg/DL)
# 
# The labels in `y` indicate whether the patient has a disease (diabetic retinopathy).
# - y = 1 : patient has retinopathy.
# - y = 0 : patient does not have retinopathy.

# In[11]:


# Import the load_data function from the utils module
from utils import load_data


# In[12]:


# Generate features and labels using the imported function
X, y = load_data(100)


# Explore the data by viewing the features and the labels

# In[13]:


# View the features
X.head()


# In[14]:


# Plot a histogram of the Age feature
X['Age'].hist();


# In[15]:


# Plot a histogram of the systolic blood pressure feature
X['Systolic_BP'].hist();


# In[16]:


# Plot a histogram of the diastolic blood pressure feature
X['Diastolic_BP'].hist();


# In[17]:


# Plot a histogram of the cholesterol feature
X['Cholesterol'].hist();


# Also take a look at the labels

# In[18]:


# View a few values of the labels
y.head()


# In[19]:


# Plot a histogram of the labels
y.hist();


# Fit the LinearRegression using the features in `X` and the labels in `y`.  To "fit" the model is another way of saying that we are training the model on the data.

# In[20]:


# Fit the linear regression model
model.fit(X, y)
model


# - View the coefficients of the trained model.
# - The coefficients are the 'weights' or $\beta$s associated with each feature
# - You'll use the coefficients for making predictions.
# $$\hat{y} = \beta_1x_1 + \beta_2x_2 + ... \beta_N x_N$$

# In[21]:


# View the coefficients of the model
model.coef_


# In the assignment, you will do something similar, but using a logistic regression, so that the output of the prediction will be bounded between 0 and 1.

# ### This is the end of this practice section.
# 
# Please continue on with the lecture videos!
# 
# ---
