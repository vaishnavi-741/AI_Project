#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# """A. Sales Dataset (Mandatory)
# Date
# 
# Product ID
# 
# Units Sold
# 
# Price
# 
# Revenue
# 
# B. Inventory Dataset (Mandatory)
# Product ID
# 
# Stock Level
# 
# Restock Date
# 
# Warehouse/Store ID
# 
# C. Competitor Pricing Dataset (Optional but Useful)
# Product ID
# 
# Competitor Price
# 
# Competitor Name
# 
# D. Product Master Data
# Product Name
# 
# Category
# 
# Cost Price"""

# In[80]:


data = pd.read_csv("revenue_lift_9pct_30000.csv")
data


# In[82]:


data.shape


# In[84]:


data.isnull().sum()


# In[86]:


data.duplicated().sum()


# In[88]:


data.dtypes


# In[90]:


data.describe()


# In[92]:


# Negative or invalid values
print("\nInvalid values check:")
print("Units Sold <= 0:", (data["Units Sold"] <= 0).sum())
print("Price <= 0:", (data["Price"] <= 0).sum())
print("Revenue <= 0:", (data["Revenue"] <= 0).sum())
print("Cost Price <= 0:", (data["Cost Price"] <= 0).sum())
print("Stock Level < 0:", (data["Stock Level"] < 0).sum())


# In[94]:


def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        data = data[(data[col] >= lower_limit) & (data[col] <= upper_limit)]
    return data


# In[96]:


data.shape


# In[98]:


#KPI 1: Revenue Lift
data["Expected_Revenue"] = data["Cost Price"] * data["Units Sold"]
revenue_lift = ((data["Revenue"].sum() - data["Expected_Revenue"].sum()) / data["Expected_Revenue"].sum()) * 100
print(f"Revenue Lift: {revenue_lift:.2f}%")


# In[100]:


#KPI 2: Profit Margin
data["Profit"] = data["Revenue"] - (data["Cost Price"] * data["Units Sold"])
profit_margin = (data["Profit"].sum() / data["Revenue"].sum()) * 100
print(f"Profit Margin: {profit_margin:.2f}%")


# In[102]:


#KPI 3: Conversion Rate
conversion_rate = (data["Units Sold"].sum() / data["Stock Level"].sum()) * 100
print(f"Conversion Rate: {conversion_rate:.2f}%")


# In[104]:


#KPI 4: Inventory Turnover
inventory_turnover = data["Units Sold"].sum() / data["Stock Level"].sum()
print(f"Inventory Turnover: {inventory_turnover:.2f}")


# In[106]:


# -------- 1. TIME FEATURES --------
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Weekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)


# In[108]:


def season(month):
    if month in [12,1,2]: return "Winter"
    if month in [3,4,5]: return "Summer"
    if month in [6,7,8]: return "Rainy"
    return "Festive"


# In[110]:


data['Season'] = data['Month'].apply(season)


# In[112]:


data['Holiday'] = data['Month'].apply(lambda x: 1 if x in [10,11,12] else 0)


# In[114]:


data['Lag_Price'] = data.groupby('Product ID')['Price'].shift(1)
data['Price_Change_%'] = ((data['Price'] - data['Lag_Price']) / data['Lag_Price']) * 100
data['Discount_%'] = ((data['Cost Price'] - data['Price']) / data['Cost Price']) * 100


# In[116]:


# --------------------- DEMAND FEATURES ---------------------
data["Lag_Sales_1"] = data.groupby("Product ID")["Units Sold"].shift(1)
data["Lag_Sales_7"] = data.groupby("Product ID")["Units Sold"].shift(7)
data["Lag_Sales_30"] = data.groupby("Product ID")["Units Sold"].shift(30)


# In[118]:


data["Rolling_7"] = data.groupby("Product ID")["Units Sold"].transform(lambda x: x.rolling(7).mean())
data["Rolling_30"] = data.groupby("Product ID")["Units Sold"].transform(lambda x: x.rolling(30).mean())

data["Demand_Volatility"] = data.groupby("Product ID")["Units Sold"].transform(lambda x: x.rolling(7).std())

# --------------------- PRICE ELASTICITY ---------------------
data["Elasticity"] = (data["Price_Change_%"]) / (data["Units Sold"].pct_change())


# In[119]:


def classify(e):
    if e > 1: return "High"
    if e > 0.5: return "Medium"
    return "Low"

data["Elasticity_Class"] = data["Elasticity"].apply(classify)


# In[120]:


# --------------------- COMPETITOR FEATURES ---------------------
data["Competitor_Diff"] = data["Price"] - data["Competitor Price"]
data["Competitor_Ratio"] = data["Price"] / data["Competitor Price"]
data["Competitor_Cheaper"] = data.apply(lambda x: 1 if x["Competitor Price"] < x["Price"] else 0, axis=1)

# --------------------- INVENTORY FEATURES ---------------------
data["Inventory_Ratio"] = data["Stock Level"] / (data["Rolling_7"] + 1)
data["Days_To_Stockout"] = data["Stock Level"] / (data["Units Sold"] + 1)

data["Low_Stock"] = data["Stock Level"].apply(lambda x: 1 if x < 10 else 0)
data["Over_Stock"] = data["Stock Level"].apply(lambda x: 1 if x > 200 else 0)

# --------------------- PROFIT FEATURES ---------------------
data["Profit_Per_Unit"] = data["Price"] - data["Cost Price"]
data["Profit_Margin_%"] = (data["Profit_Per_Unit"] / data["Price"]) * 100

# --------------------- INTERACTION FEATURES ---------------------
data["Weekend_Price"] = data["Weekend"] * data["Price"]
data["Season_Discount"] = data["Discount_%"] * data["Month"]
data["Inventory_Price"] = data["Stock Level"] * data["Price"]

# --------------------- CATEGORICAL ENCODING ---------------------
df = pd.get_dummies(data, columns=["Category", "Season", "Elasticity_Class"], drop_first=True)


# In[121]:


df.columns


# In[122]:


df


# In[128]:


df.isnull().sum()


# In[130]:


df = df.drop(columns=['Lag_Sales_7','Lag_Sales_30','Rolling_7','Rolling_30','Demand_Volatility'])


# In[132]:


df.shape


# In[134]:


# --------------------- FINAL CLEANING ---------------------
df = df.fillna(0)
df = df.drop_duplicates()


# In[136]:


df.isnull().sum()


# In[138]:


df.shape


# In[144]:


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols


# In[146]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=90)
plt.title("Boxplot of Numeric Features (Outlier Detection)")
plt.show()


# In[148]:


# Identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Remove outliers using IQR
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]


# In[150]:


df.shape


# In[ ]:




