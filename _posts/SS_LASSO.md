layout: post
title: "Using GitHub pages"

# Self-study Codio activity: LASSO regression
---

## Overview

In this activity, you will build a LASSO regression model using the LASSO loss function and scipy's `minimize` function.

This activity is designed to build your familiarity and comfort coding in Python while also helping you review key topics from each module. As you progress through the activity, questions will get increasingly more complex. It is important that you adopt a programmer's mindset when completing this activity. Remember to run your code from each cell before submitting your activity, as doing so will give you a chance to fix any errors before submitting.



### Learning outcome addressed

- Apply nonlinear optimisation to regularised least squares regression.



## Index:

- [Question 1](#Question-1)
- [Question 2](#Question-2)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Small Example

Below, a simple data set involving data about dining at a restaurant and the tips given to wait staff is loaded and displayed.  The goal will be to predict the tip using the bill. 


```python
tips = sns.load_dataset('tips')
```


```python
tips.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.scatterplot(data = tips, x = 'total_bill', y = 'tip')
plt.title('Total bill vs. tip')
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.grid();
```


    
![png](SS_LASSO_files/SS_LASSO_6_0.png)
    


### The LASSO model

The loss function for a LASSO regression model is given below as `lasso`.  The function takes in a parameter vector `beta` and returns the squared error plus the penalty term.  This will be used to find a line of best fit for the tips data.  In this example, there is no y-intercept.


```python
def lasso(beta):
    loss = (beta*x - y)**2 + alpha*np.abs(beta)
    return np.mean(loss)
```


```python
x = tips['total_bill']
y = tips['tip']
alpha = 0.1
```


```python
#lasso loss with slope of 1
lasso(1)
```




    346.18162254098354



This is a minimisation problem where your goal is to find the parameter `beta` that minimises the `lasso` function.  Below, the `minimize` function is used to determine the parameters that minimise the `lasso` loss function. The results are displayed.


```python
from scipy.optimize import minimize
```


```python
results = minimize(lasso, x0 = 0)
```


```python
print(f'The parameter that minimizes the lasso loss with alpha: {alpha} is slope: {results.x[0] :.4f}')
```

    The parameter that minimizes the lasso loss with alpha: 0.1 is slope: 0.1436
    

Using more inputs is a simple adjustment to the loss function given below.  This is then implemented using the `total_bill` and `size` columns to demonstrate.  For more features, you would simply need to instantiate the `minimize` function with `x0` to match the number of features.  To add an intercept, you would add one and concatenate a column of ones on `x` before multiplying.

If you have multiple features, you can adjust the objective function accordingly and simply specify initial values for the parameters on each feature.  Below, this is demonstrated with `total_bill` and `size`.


```python
def lasso(beta):
    loss = (x@beta - y)**2 + alpha*np.abs(np.sum(beta))
    return np.mean(loss)
```


```python
x = tips[['total_bill', 'size']]
```


```python
#prediction with total_bill coef = 1 and size coef = 1
lasso(np.array([1, 1]))
```




    448.87121270491804




```python
#find parameters that minimize the loss
results = minimize(lasso, x0 = (0, 0))
```


```python
print(f'The parameters for the new model are total bill: {results.x[0]: .4f} and size: {results.x[1]: .4f}')
```

    The parameters for the new model are total bill:  0.1067 and size:  0.3111
    

### Using `scikit-learn`

In practice you would likely use `sklearn` to build a LASSO regression model.  Below, the estimator is imported and a model with no $y$-intercept is fit on the tips data.  The resulting coefficient is displayed -- note the similarity to the results from the `minimize` function.


```python
from sklearn.linear_model import Lasso
```


```python
#use same alpha and no y-intercept
lasso = Lasso(alpha = 0.1, fit_intercept=False)
```


```python
x = x[['total_bill']]
#fit model on total bill
lasso.fit(x, y)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Lasso(alpha=0.1, fit_intercept=False)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Lasso</label><div class="sk-toggleable__content"><pre>Lasso(alpha=0.1, fit_intercept=False)</pre></div></div></div></div></div>




```python
print(f'The sklearn lasso model is minimized with beta = {lasso.coef_[0]: .4f}')
```

    The sklearn lasso model is minimized with beta =  0.1435
    

###### [Back to top](#Index:)

### Question 1

Use `sklearn` and the `Lasso` estimator with `alpha = 0.01` to build and fit a LASSO regression model on the `credit` data given as `X_train`, `y_train` below.


```python
credit = pd.read_csv('credit.csv', index_col=0)
```


```python
#task is to predict the Balance column
credit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Limit</th>
      <th>Rating</th>
      <th>Cards</th>
      <th>Age</th>
      <th>Education</th>
      <th>Gender</th>
      <th>Student</th>
      <th>Married</th>
      <th>Ethnicity</th>
      <th>Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14.891</td>
      <td>3606</td>
      <td>283</td>
      <td>2</td>
      <td>34</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>106.025</td>
      <td>6645</td>
      <td>483</td>
      <td>3</td>
      <td>82</td>
      <td>15</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Asian</td>
      <td>903</td>
    </tr>
    <tr>
      <th>3</th>
      <td>104.593</td>
      <td>7075</td>
      <td>514</td>
      <td>4</td>
      <td>71</td>
      <td>11</td>
      <td>Male</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>580</td>
    </tr>
    <tr>
      <th>4</th>
      <td>148.924</td>
      <td>9504</td>
      <td>681</td>
      <td>3</td>
      <td>36</td>
      <td>11</td>
      <td>Female</td>
      <td>No</td>
      <td>No</td>
      <td>Asian</td>
      <td>964</td>
    </tr>
    <tr>
      <th>5</th>
      <td>55.882</td>
      <td>4897</td>
      <td>357</td>
      <td>2</td>
      <td>68</td>
      <td>16</td>
      <td>Male</td>
      <td>No</td>
      <td>Yes</td>
      <td>Caucasian</td>
      <td>331</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = credit[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']]
y = credit['Balance']

sscaler = StandardScaler()
#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
#scale the data
X_train = sscaler.fit_transform(X_train)
X_test = sscaler.transform(X_test)
```


```python
model = ''
###BEGIN SOLUTION
model = Lasso(alpha = 0.1).fit(X_train, y_train)
###END SOLUTION
#Answer check
for col, coef in zip(X.columns.tolist(), model.coef_):
    print(f'Feature {col} has coefficient {coef: .4f}')
```

    Feature Income has coefficient -252.1683
    Feature Limit has coefficient  326.5163
    Feature Rating has coefficient  278.5763
    Feature Cards has coefficient  8.8645
    Feature Age has coefficient -17.7623
    Feature Education has coefficient  3.9519
    


```python
### BEGIN HIDDEN TESTS
model_ = Lasso(alpha = 0.1).fit(X_train, y_train)



#
#
#
np.testing.assert_array_almost_equal(model.coef_, model_.coef_)
### END HIDDEN TESTS
```

###### [Back to top](#Index:)

### Question 2

Fit a second LASSO model as `model2` on the training data using `alpha = 100` -- much more of a penalty than that of the first model.


```python
model2 = ''
###BEGIN SOLUTION
model2 = Lasso(alpha = 100).fit(X_train, y_train)
###END SOLUTION
#Answer check
for col, coef in zip(X.columns.tolist(), model2.coef_):
    print(f'Feature {col} has coefficient {coef: .4f}')
```

    Feature Income has coefficient -0.0000
    Feature Limit has coefficient  162.4033
    Feature Rating has coefficient  143.2957
    Feature Cards has coefficient  0.0000
    Feature Age has coefficient -0.0000
    Feature Education has coefficient  0.0000
    


```python
### BEGIN HIDDEN TESTS
model2_ = Lasso(alpha = 100).fit(X_train, y_train)



#
#
#
np.testing.assert_array_almost_equal(model2.coef_, model2_.coef_)
### END HIDDEN TESTS
```

###### [Back to top](#Index:)

### Question 3

Compare the coefficients from the model with `alpha = 0.1` to that of the model with `alpha = 100`.  This is an important feature of the LASSO model -- it eliminates features that might be considered 'less important' to the model.  While it typically underperforms other regression models, the LASSO model can be used to select features.  Which of the features below are the important features based on the `alpha = 100` model?  Assign your answer `a`, `b`, `c` or `d` as a string to `ans3` below.

```
a. Income and age
b. Limit and rating
c. Age and education
d. All the features are equally important
```


```python
ans3 = ''
###BEGIN SOLUTION
ans3 = 'b'
###END SOLUTION
print(type(ans3))
print(ans3)
```

    <class 'str'>
    b
    


```python
### BEGIN HIDDEN TESTS
ans3_ = 'b'



#
#
#
assert ans3 == ans3_
### END HIDDEN TESTS
```

Great job!  The LASSO is another example of a nonlinear optimisation problem; however, because it is a convex loss function, you are guaranteed to find a minimimum value.  


```python

```
