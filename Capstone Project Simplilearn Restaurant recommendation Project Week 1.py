#!/usr/bin/env python
# coding: utf-8

# # Introduction/Business Problem

# A restaurant consolidator is looking to revamp its B-to-C portal using intelligent automation tech. It is in search of different matrix to identify and recommend restaurants. To make sure an effective model can be achieved it is important to understand the behaviour of the data in hand.

# In[3]:


# Import required Libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import json
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
from IPython.display import display, Math, Latex
get_ipython().run_line_magic('matplotlib', 'inline')
    
pd.set_option('display.width', 450)
pd.set_option('display.max_columns',100)
pd.set_option('display.notebook_repr_html', True)


# In[4]:


import pandas as pd

sheet1 = None
with pd.ExcelFile(r"/Users/priyankakesari/Library/Containers/com.microsoft.Excel/Data/Desktop/Capstone Project/1582800386_project1datadictionary/data.xlsx") as reader:
    business_df = pd.read_excel(reader, sheet_name='zomato')


# In[5]:


business_df.head()


# # Preliminary data inspection 

# In[6]:


business_df.shape


# In[7]:


business_df.columns


# In[8]:


business_df.describe(include = 'all')


# In[9]:


# check null values
business_df.isnull().sum()


# from above output we can infer that mostly values are which are null is in one coloumn i.e. Cuisines while as Resturant Name has 1 value which is missing which can be ignored.
# Hence we have to deal with null values of cuisines and make sure it does not affect our sample data

# In[10]:


business_df2 = business_df.replace(np.nan, 0)


# In[11]:


# To check if any null values exist after replacing them with 0
business_df2.isnull().sum()


# In[12]:


# Inspecting Duplicates and Removing the same for more structured data

business_df2[business_df2[['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines', 'Average Cost for two', 'Currency', 'Has Table booking', 'Has Online delivery', 'Price range', 'Aggregate rating', 'Rating color', 'Rating text', 'Votes']].duplicated() == True]

business_df2[business_df2[['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 'Locality', 'Locality Verbose', 'Longitude', 'Latitude', 'Cuisines', 'Average Cost for two', 'Currency', 'Has Table booking', 'Has Online delivery', 'Price range', 'Aggregate rating', 'Rating color', 'Rating text', 'Votes']].duplicated()]


# In[13]:


business_df2.shape


# It can be infer from above that there are no duplicates in the dataset

# # Performing EDA

# In[14]:


# Exploring geographical distribution of the restaurants 
business_df3 = business_df2.groupby(['City'])['City'].count().reset_index(name='counts')
print(business_df3)


# In[15]:


# identify the cities with the maximum and minimum number of restaurants

business_df3.groupby('City')['counts'].max().reset_index().sort_values(['counts'], ascending=False)


#  Maximum no. of restaurants is in New delhi i.e. 5473 and And there are about 46 restaurnts  which are only 1 in a particular city

# In[16]:


# Exploring the franchise with most national presence

import pandas as pd

sheet1 = None
with pd.ExcelFile(r"/Users/priyankakesari/Library/Containers/com.microsoft.Excel/Data/Desktop/Capstone Project/1582800386_project1datadictionary/Country-Code.xlsx") as reader:
    country_code_df = pd.read_excel(reader, sheet_name='Sheet1')


# In[17]:


country_code_df.head()


# In[18]:


# Joining two excel files

merged_data = pd.merge(country_code_df,business_df2)
merged_data


# In[19]:


# the franchise with most national presence

most_national_presence = merged_data.groupby(['Country','Restaurant Name'])['Restaurant Name'].count().reset_index(name='counts of Restaurant Name').sort_values(['counts of Restaurant Name'], ascending=False)

print(most_national_presence)


# In[20]:


# Finding ratio between restaurants that allow table booking vs. those that do not allow table booking

restaurants_df = pd.DataFrame(business_df2, columns=['Restaurant Name', 'Has Table booking'])
restaurants_df


# In[21]:


restaurants_df.shape


# In[22]:


# Inspecting duplicates
restaurants_df[restaurants_df[[ 'Restaurant Name', 'Has Table booking']].duplicated() == True]


# In[23]:


restaurants_new_df = restaurants_df.drop_duplicates( subset = ['Restaurant Name', 'Has Table booking'],
keep = False)


# In[24]:


restaurants_new_df.shape


# In[25]:


count_has_table_booking = restaurants_new_df.groupby(['Has Table booking'])['Restaurant Name'].count().reset_index(name='count')
count_has_table_booking


# In[26]:


Count_of_No = count_has_table_booking.iloc[0][1]
Count_of_No


# In[27]:


Count_of_Yes = count_has_table_booking.iloc[1][1]
Count_of_Yes


# In[28]:


ratio_of_restaurants_in_terms_of_Table_booking = (Count_of_Yes/Count_of_No)
ratio_of_restaurants_in_terms_of_Table_booking


# Hence 0.16 is the ratio between resturants that has table booking and those who do not have table booking

# In[29]:


#Percentage of restaurants providing online delivery

percentage_restaurants = pd.DataFrame(business_df2, columns=['Restaurant Name', 'Has Online delivery'])
percentage_restaurants.shape


# In[30]:


# Inspecting and removing duplicates
percentage_restaurants[percentage_restaurants[[ 'Restaurant Name', 'Has Online delivery']].duplicated() == True]

percentage_restaurants_new = percentage_restaurants.drop_duplicates( subset = ['Restaurant Name', 'Has Online delivery'],
keep = False)

percentage_restaurants_new


# In[31]:


percentage_restaurants_new.shape


# In[32]:


count_has_online_delivery = percentage_restaurants_new.groupby(['Has Online delivery'])['Restaurant Name'].count().reset_index(name='count')
count_has_online_delivery


# In[33]:


Count_of_No = count_has_online_delivery.iloc[0][1]
Count_of_No


# In[34]:


Count_of_Yes = count_has_online_delivery.iloc[1][1]
Count_of_Yes


# In[35]:


# percentage of restaurants providing online delivery

percentage_of_restaurants_in_terms_of_online_delivery = (Count_of_Yes/(Count_of_No + Count_of_Yes)) *100
percentage_of_restaurants_in_terms_of_online_delivery


# Percentage of restuarnts those are providing online delivery  is 20.9%

# In[36]:


# the difference in number of votes for the restaurants that deliver and the restaurants that do not deliver

votes_df = pd.DataFrame(business_df2, columns=['Restaurant Name', 'Has Online delivery','Votes'])
votes_df


# In[37]:


votes_df_new = votes_df.groupby(['Restaurant Name', 'Has Online delivery'])['Votes'].sum().reset_index(name='votes')
votes_df_new


# In[38]:


votes_df_new.shape


# In[39]:


votes_df_new2= votes_df_new.groupby(['Has Online delivery'])['votes'].sum().reset_index(name='votes_new')
votes_df_new2


# In[40]:


Count_of_No = votes_df_new2.iloc[0][1]
Count_of_No


# In[41]:


Count_of_Yes = votes_df_new2.iloc[1][1]
Count_of_Yes


# In[42]:


# Difference between number of votes for the restaurants that deliver and the restaurants that do not deliver

Difference_in_terms_of_delivery = (Count_of_No - Count_of_Yes)
Difference_in_terms_of_delivery # Difference between number of votes for the restaurants that deliver and the restaurants that do not deliver

