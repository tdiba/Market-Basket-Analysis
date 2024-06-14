#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Collection and Preparation

# In[1]:


import pandas as pd


# In[2]:


#Load dataset
df=pd.read_excel(r"C:\Users\USER\Documents\Data Portfolio Projects\Retail\Market Basket Analysis\Jomo_Holdings_Transactions.xlsx")
df.head()


# In[ ]:





# What is the structure of the data?
# 

# In[3]:


# Display data types of each column
df.dtypes


# In[ ]:





# Are there any missing values?

# In[4]:


# Check for missing values
df.isnull().sum()


# In[ ]:





# What preprocessing steps are necessary?

# In[5]:


# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])


# In[6]:


# Display the updated data types
df.dtypes


# In[ ]:





# In[ ]:





# ### 2. Descriptive Analysis

# What are the overall sales trends?

# In[7]:


import matplotlib.pyplot as plt


# In[8]:


# Calculate total sales and number of transactions
total_sales = df['Quantity'] * df['Price']
df['TotalSales'] = total_sales
num_transactions = df['TransactionID'].nunique()
average_transaction_value = total_sales.sum() / num_transactions


# In[9]:


# Print the summary statistics
summary_stats = {
    "Total Sales": total_sales.sum(),
    "Number of Transactions": num_transactions,
    "Average Transaction Value": average_transaction_value
}
summary_stats


# In[10]:


# Plot sales trends over time
sales_trends = df.groupby('Date')['TotalSales'].sum()
sales_trends.plot(kind='line', title='Sales Trends Over Time', xlabel='Date', ylabel='Total Sales')
plt.show()


# In[ ]:





# Who are the customers?

# In[11]:


# Display customer demographics summary
customer_demographics = df[['CustomerID', 'CustomerAge', 'CustomerGender', 'CustomerLocation']].drop_duplicates()
customer_demographics_summary = customer_demographics.describe(include='all')
customer_demographics_summary


# In[12]:


# Visualize customer distribution by gender and location
gender_distribution = customer_demographics['CustomerGender'].value_counts()
location_distribution = customer_demographics['CustomerLocation'].value_counts()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
gender_distribution.plot(kind='bar', ax=axes[0], title='Customer Gender Distribution', xlabel='Gender', ylabel='Count')
location_distribution.plot(kind='bar', ax=axes[1], title='Customer Location Distribution', xlabel='Location', ylabel='Count')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# ### 3. Market Basket Analysis

# What are the frequent itemsets?
# 
# The Apriori algorithm will be used to find frequent itemsets.

# In[13]:


from mlxtend.frequent_patterns import apriori, association_rules


# In[14]:


get_ipython().system('pip install mlxtend')


# In[15]:


from mlxtend.frequent_patterns import apriori, association_rules


# In[16]:


# Prepare data for market basket analysis
basket = df.groupby(['TransactionID', 'ProductName'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)


# In[17]:


# Find frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

frequent_itemsets


# In[ ]:





# In[ ]:




