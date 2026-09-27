import pandas as pd 
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# open excel file 


df = pd.read_excel(r"D:\Python programs\Dataset for Data Analytics.xlsx")
print("Data set shape ", df.shape)
print("Missing Values \n", df.isnull().sum())
print("Duplicate Values ", df.duplicated().sum())

# Code explanation :  load the data set into pandas data frame and check its dimensions, missing values per column, and duplicate row counts.

# 1. Fill missing CouponCode entries

df['CouponCode'] = df['CouponCode'].fillna('No Coupon')

# 2. Convert Date column to datetime format

df['Date'] = pd.to_datetime(df['Date'])

# 3. Feature Engineering: Extract Year, Month, and Day of Week

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.strftime('%Y-%m')
df['Day of week'] = df['Date'].dt.day_name()

# 4. Data Validation: Verify TotalPrice accuracy

df['Calculated Total_price'] = (df['Quantity'] * df['UnitPrice']).round(2)

mismatch_count = (df['TotalPrice'] != df['Calculated Total_price']).sum()

print ("Price calculation mismach count :  ", mismatch_count)

# Basic statistics for continuous numerical columns

des_stats = df[['Quantity', 'UnitPrice','ItemsInCart','TotalPrice']]

print(des_stats)

# Hypothesis Testing (Two-Sample Independent t-Test)

# Split the Data into Two Comparison Groups

coupon_orders = df[df['CouponCode'] != 'No Coupon']['TotalPrice']
non_coupon_orders = df[df['CouponCode'] == 'No Coupon']['TotalPrice']

# Run the Two-Sample Independent T-Test

t_stat, p_value_coupon = stats.ttest_ind(coupon_orders, non_coupon_orders)

print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value_coupon:.4f}")


# Measure linear relationships between variables

corr_matrix = df[['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']].corr()
print(corr_matrix)

# Chi-Square Test of Independence

# Cross-tabulate Payment Method and Referral Source
contingency_table = pd.crosstab(df['PaymentMethod'], df['ReferralSource'])

# Run Chi-Square test
chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p_val_chi:.4f}")


sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Monthly Revenue Trend Line Chart
monthly_revenue = df.groupby('Month')['TotalPrice'].sum().reset_index()
sns.lineplot(data=monthly_revenue, x='Month', y='TotalPrice', ax=axes[0, 0], marker='o', color='navy')
axes[0, 0].set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Total Revenue ($)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Revenue by Product Category Bar Chart
prod_revenue = df.groupby('Product')['TotalPrice'].sum().sort_values(ascending=False).reset_index()
sns.barplot(data=prod_revenue, x='TotalPrice', y='Product', ax=axes[0, 1], palette='Blues_r')
axes[0, 1].set_title('Total Revenue by Product', fontsize=12, fontweight='bold')

# 3. Order Status Distribution Pie Chart
status_counts = df['OrderStatus'].value_counts()
axes[1, 0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
axes[1, 0].set_title('Order Status Distribution', fontsize=12, fontweight='bold')

# 4. Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()