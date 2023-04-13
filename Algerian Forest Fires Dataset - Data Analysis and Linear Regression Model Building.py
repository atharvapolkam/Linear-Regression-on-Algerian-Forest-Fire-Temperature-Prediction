#!/usr/bin/env python
# coding: utf-8

# # Algerian Forest Fires Dataset - Data Analysis and Linear Regression Model Building

# Problem statement
# 
# Here is algerian dataset which has attributes Attribute Information:
# 
# Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)
# Weather data observations
# 
# Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 
# RH : Relative Humidity in %: 21 to 90
# 
# Ws :Wind speed in km/h: 6 to 29
# 
# Rain: total day in mm: 0 to 16.8
# 
# FWI Components
# 
# Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# 
# Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# 
# Drought Code (DC) index from the FWI system: 7 to 220.4
# 
# Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# 
# Buildup Index (BUI) index from the FWI system: 1.1 to 68
# 
# Fire Weather Index (FWI) Index: 0 to 31.1
# 
# Classes: two classes, namely Fire and not Fire
# 
# We need to find the temprature as based on the other conditions.

# # We have to perform :-

# 1.Data Collection
# 
# 2.Exploratory Data Analysis
# 
# 3.Data Cleaning
# 
# 4.Model Building
# 
# 5.Model Predictions
# 

# # Import Data and Required Packages

# In[1]:


#Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings


warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading the data
df = pd.read_csv('D:\data science\LINEAR REGRESSION OF ALGERIAN FOREST\Algerian_forest_fires_dataset_UPDATE.csv',header=1)
df.head()


# In[3]:


# drop rows
df.drop([122,123],inplace=True)
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)


# In[4]:


df.loc[:122, 'region'] = 'bejaia'
df.loc[122:, 'region'] = 'Sidi-Bel Abbes'


# In[5]:


#Stripping the names of the columns
df.columns = [i.strip() for i in df.columns]
df.columns


# In[6]:



#Stripping the Classes Features data

df.Classes = df.Classes.str.strip()
df['Classes'].unique()


# # Understanding the dataset

# In[7]:


#displaying first 5 rows of dataset
df.head()


# In[8]:



#displaying last 5 rows of dataset
df.tail()


# In[9]:



#know the datatypes
df.dtypes


# In[10]:


## Changing The DataTypes of the Columns


# In[11]:


df['day']=df['day'].astype(int)
df['month']=df['month'].astype(int)
df['year']=df['year'].astype(int)
df['Temperature']=df['Temperature'].astype(int)
df['RH']=df['RH'].astype(int)
df['Rain']=df['Rain'].astype(float)
df['FFMC']=df['FFMC'].astype(float)
df['DMC']=df['DMC'].astype(float)
df['BUI']=df['BUI'].astype(float)
df['ISI']=df['ISI'].astype(float)
df['Ws']=df['Ws'].astype(float)



df.info()


# In[12]:


#displaying the shape of dataset : No. of rows and No. of columns
df.shape


# In[13]:


#getting overall statistics about the dataframe
df.describe(include='all')


# In[14]:


# how many columns are in the dataframe
df.columns
#there are spaces after the Rain and Classes columns so i rename to take out spaces


# In[15]:



#renaming column names so as to remove spaces nehind
 
df.rename(columns={'Rain ': 'Rain', 'Classes  ': 'classes'}, inplace=True)


# In[16]:



df.nunique()


# In[17]:


## Applying Label encoding in DC,FWI,region features

from sklearn.preprocessing import LabelEncoder
LabelEncoder=LabelEncoder()


# In[18]:


df['DC']=LabelEncoder.fit_transform(df['DC'])
df['FWI']=LabelEncoder.fit_transform(df['FWI'])
df['region']=LabelEncoder.fit_transform(df['region'])


# In[19]:


df.dtypes


# In[20]:



df.head()


# In[21]:



#checking for null values.
df.isnull().sum()


# In[22]:



#dropped the year and classes attribute as the fires all occured in the same year (2012)
#i dropped classes too as i couldnt compare variables 
df = df.drop(['year', 'Classes'], axis=1)


# In[23]:



#check after droppping year attribute

df.head()


# # Univariate Analysis

# In[24]:


numeric_features= [feature for feature in df.columns if df[feature].dtype !='o']


# In[25]:


numeric_features


# In[26]:


plt.figure(figsize=(15,15))
plt.suptitle('Univariate Analysis of Numerical Features',fontsize=20,fontweight='bold',alpha=0.8,y=1)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=df[numeric_features[i]],shade=True, color='b')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# Observations:
# 
# Temperature : temperature noon (temperature max) in Celsius degrees: 22 to 42
# 
# RH : Relative Humidity in %: 21 to 90
# 
# Ws :Wind speed in km/h: 6 to 29
# 
# Rain: total day in mm: 0 to 16.8
# 
# (FFMC) Fine Fuel Moisture Code index from the FWI system: 28.6 to 92.5
# 
# (DMC) Duff Moisture Code index from the FWI system: 1.1 to 65.9
# 
# (DC) Drought Code index from the FWI system: 7 to 220.4
# 
# (ISI) Initial Spread Index from the FWI system: 0 to 18.5
# 
# (BUI) Buildup Index from the FWI system: 1.1 to 68
# 
# (FWI) Fire Weather Index: 0 to 31.1
# 
# We can also see outliers in most of these features.

# In[27]:


x_axis= numeric_features
y= "Temperature"

for col in x_axis:
    sns.regplot(x=col,y=y,data=df)
    plt.xlabel(col, fontsize = 13)
    plt.ylabel(y,fontsize = 13)
    plt.title("scatter Plot of" +  col + " & " + y , fontsize =15)
    plt.grid()
    plt.show()


# Observations:
# 
# As we are considering Temperature as our dependent feature we are checking relation of each feature with it.
# 
# RH shows negative correlation with Temperature as increase in RH reduces the Temperature.
# 
# WS too shows negative correlation with Temperature but not as strong as RH.
# 
# FFMC shows strong positive correlation as Temperature increases with FFMC which is clearly seen from the plot.
# 
# ISI shows positive correlation with Temperature.

# In[28]:


#HEATMAP

#A heatmap is a graphical representation of data that uses a system of color-coding to represent different values. 
#using a correlation heatmap to view rlationship between variables
sns.heatmap(df.corr(),annot=True,cmap='viridis',linewidths=0.2)


# In[29]:


#HISTOGRAM

#A histogram is basically used to represent data provided in a form of some groups.It is accurate method for the graphical 
#representation of numerical data distribution.It is a type of bar plot where X-axis represents the bin ranges while Y-axis 
#gives information about frequency.
df.hist(figsize=(20,14),color='b')


# In[30]:


#LINEPLOT
#A Line plot can be defined as a graph that displays data as points or check marks above a number line, showing the frequency
#of each value.

sns.lineplot(x='Temperature',y='day',data=df,color='g')


# In[31]:


## Visualization of Target Feature
plt.subplots(figsize=(14,7))
sns.histplot(x=df.Temperature, ec = "black", color='blue', kde=True)
plt.title("Temperature Distribution", weight="bold",fontsize=20, pad=20)
plt.ylabel("Count", weight="bold", fontsize=15)
plt.xlabel("Temperatures", weight="bold", fontsize=12)
plt.show()


# In[32]:


#PAIRPLOT

#A pairplot plot a pairwise relationships in a dataset. The pairplot function creates a grid of Axes such that each variable in
#data will by shared in the y-axis across a single row and in the x-axis across a single column.
sns.pairplot(df)


# In[33]:


#JOINTPLOT

#Seaborn's jointplot displays a relationship between 2 variables (bivariate) as well as 1D profiles (univariate) in the margins.
#This plot is a convenience class that wraps JointGrid.

sns.jointplot(x='month',y='Temperature',data=df,color='r')


# In[34]:


#Barplot

plt.style.use("default")
sns.barplot(x="day", y="Temperature",data=df)
plt.title("Day vs Temperature",fontsize=15)
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.show()


# In[35]:



sns.scatterplot(x='day',y='Temperature',data=df,color='g')


# In[36]:


## Temperature Vs ISI

plt.figure(figsize=(10,10))
sns.regplot(x='Temperature',y='ISI',data=df)


# In[37]:


## Checking the outliers of the target 'Temperature' feature
sns.boxplot(df['Temperature'])


# In[38]:


#Boxplot of Rain Vs Temperature


# In[39]:


sns.boxplot(x ='Temperature', y ='Rain', data = df)


# In[40]:


#Boxplot of 'FFMC' Vs Temperature

sns.boxplot(x ='Temperature', y ='FFMC', data = df)


# In[41]:


# Boxplot of ISI Vs Temperature

sns.boxplot(x ='Temperature', y ='ISI', data = df)


# In[42]:


# Boxplot of region Vs Temperature

sns.boxplot(x ='region', y ='Temperature', data = df)


# In[43]:


# Boxplot of BUI Vs Temperature
sns.boxplot(x ='Temperature', y ='BUI', data = df)


# In[44]:


## Boxplot DMC Vs Temperature

sns.boxplot(x ='Temperature', y ='DMC', data = df)


# # Questions

# In[45]:


#What is the highest temperature in the dataset
df.Temperature.max()


# In[46]:


#What is the lowest temperature in the dataset
df.Temperature.min()


# In[47]:


#When did it rain the most
#the 31st day in the 8th month with 16.8mm of rainfall
highest_rain = df.sort_values(by='Rain', ascending=False)[['Rain', 'day','month']].head(1)
highest_rain


# In[48]:


#What did it rain the least
#the 6th month with 0.0mm of rainfall
lowest_rain = df.sort_values(by='Rain', ascending=True)[['Rain', 'day', 'month']].head(1)
lowest_rain


# In[49]:


#What month is the hottest
#the 8th month
highest_month = df.sort_values(by='Temperature', ascending=False)[['month']].head(1)
highest_month


# In[50]:


#what day has the highest temperature in the dataset?
#The highest Temperature is 42 degrees and it occured on the 17th of  the 8th Month).
highest_temp = df.sort_values(by='Temperature', ascending=False)[['Temperature','day','month']].head(1)
highest_temp


# In[51]:


highest_temp = df.sort_values(by='Temperature', ascending=False)[['Temperature', 'day','month', 'Rain']].head(1)

lowest_temp =  df.sort_values(by='Temperature', ascending=True)[['Temperature', 'day','month', 'Rain']].head(1)

print("Highest Temperatures")
print(highest_temp)

print()

print("Lowest Temperatures")
print(lowest_temp)


# # Creating Dependent and Independent features

# In[52]:


## Independent Features

x=pd.DataFrame(df, columns=['RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI','region'])  

## Dependent Features

y=pd.DataFrame(df,columns=['Temperature'])


# In[53]:


x


# In[54]:


y


# # TrainTest Split

# In[55]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=10)


# In[56]:


x_train.shape


# In[57]:


x_test.shape


# In[58]:


y_train.shape


# In[59]:


y_test.shape


# In[60]:


x_train


# In[61]:


x_test


# In[62]:


y_train


# In[63]:


y_test


# # Standardizing or Feature Scaling

# In[64]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()#intialising


# In[65]:


scaler


# In[66]:


x_train=scaler.fit_transform(x_train)


# In[67]:


x_test=scaler.transform(x_test)


# In[68]:


x_train


# In[69]:


x_test


# # Linear Regression Model Training

# In[70]:


from sklearn.linear_model import LinearRegression


# In[71]:


regression=LinearRegression()
regression
regression.fit(x_train,y_train)


# In[72]:


##print the coefficient
print(regression.coef_)


# In[73]:


#print the intercept
print(regression.intercept_)


# In[74]:


##prediction for the test data
reg_pred = regression.predict(x_test)


# In[75]:


reg_pred


# In[76]:


##assumptions of linear regression

plt.scatter(y_test, reg_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")


# In[77]:


#residuals
residuals= y_test-reg_pred


# In[78]:


sns.displot(residuals, kind='kde')#distrib


# In[79]:


## Scatter plot with predictions and residual
## uniform distribution
plt.scatter(reg_pred, residuals)


# In[80]:


## Performance Metrics


# In[81]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[82]:


## R square and adjusted R square


# In[83]:


## R square
from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)
print(score)


# In[84]:


## Adjusted R Square
#display adjusted R square
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# Ridege Regression Algorithm

# In[86]:


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge
ridge.fit(x_train, y_train)


# In[87]:


## print the coefficients 
print(ridge.coef_)


# In[88]:


## print the intercept
print(ridge.intercept_)


# In[89]:


## prediction for the test data
ridge_pred = ridge.predict(x_test)
ridge_pred


# In[90]:


## Assumptions of Ridge Regression

plt.scatter(y_test, ridge_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")


# In[91]:


## residuals 
residuals = y_test-ridge_pred


# In[92]:


residuals


# In[93]:


sns.displot(residuals, kind = "kde") # Distribution of residuals


# In[94]:



## Scatter plot with predictions and residual
## uniform distribution
plt.scatter(ridge_pred, residuals)


# In[95]:


## residuals 
residuals = y_test-ridge_pred


# In[96]:


residuals


# In[97]:


sns.displot(residuals, kind = "kde") # Distribution of residuals


# In[98]:



## Scatter plot with predictions and residual
## uniform distribution
plt.scatter(ridge_pred, residuals)


# In[99]:


## Performance Metrics

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, ridge_pred))
print(mean_absolute_error(y_test, ridge_pred))
print(np.sqrt(mean_squared_error(y_test, ridge_pred)))


# In[100]:


## R square and adjusted R square


# In[101]:


# R square
from sklearn.metrics import r2_score
score = r2_score(y_test, ridge_pred)
print(score)


# In[102]:


## Adjusted R Square
#display adjusted R square
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# # Lasso Regression Model Training
# 

# In[103]:


from sklearn.linear_model import Lasso
lasso=Lasso()
lasso
lasso.fit(x_train,y_train)


# In[104]:


## print the coefficients
print(lasso.coef_)


# In[105]:


## print the intercept
print(lasso.intercept_)


# In[106]:



## prediction for the test data
lasso_pred = lasso.predict(x_test)
lasso_pred


# In[107]:



## Assumptions of Lasso Regression

plt.scatter(y_test, lasso_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")


# In[108]:


## Performance Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, lasso_pred))
print(mean_absolute_error(y_test, lasso_pred))
print(np.sqrt(mean_squared_error(y_test, lasso_pred)))


# In[109]:



## Rsquare and adjusted R square


# In[110]:


# R square
from sklearn.metrics import r2_score
score = r2_score(y_test, lasso_pred)
print(score)


# In[111]:


## Adjusted R Square
#display adjusted R square
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# 
# # Elastic Net Regression Model Training

# In[112]:


from sklearn.linear_model import ElasticNet
elastic = ElasticNet()
elastic
elastic.fit(x_train, y_train)


# In[113]:


## print the coefficients
print(elastic.coef_)


# In[114]:


## print the intercept
print(elastic.intercept_)


# In[115]:



## prediction for the test data
elastic_pred = elastic.predict(x_test)
elastic_pred


# In[116]:


## Assumption of ElasticNet Regression

plt.scatter(y_test, elastic_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")


# In[117]:


## Performance Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(y_test, elastic_pred))
print(mean_absolute_error(y_test, elastic_pred))
print(np.sqrt(mean_squared_error(y_test, elastic_pred)))


# In[118]:



# R square
from sklearn.metrics import r2_score
score = r2_score(y_test, elastic_pred)
print(score)


# In[119]:


## Adjusted R Square
#display adjusted R square
1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# In[ ]:




