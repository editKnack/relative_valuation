# -*- coding: utf-8 -*-
"""Relative Valuation with LightGBM

## Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, mean_squared_error, r2_score

"""
# Loading & Cleaning"""

#Read Files
folder_path = '/content/data'

dfs = []

# Traverse through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_path = os.path.join(folder_path, filename)
        # Read the second sheet
        df = pd.read_excel(file_path, sheet_name=1)
        dfs.append(df)

# Concatenate all DataFrames
df = pd.concat(dfs, ignore_index=True)
df.shape

df.head()

df.columns.values[0] = 'yr'

df.columns

# copy year values from first column to new column 'year'
df['year'] = df.iloc[:,0]

# Delete old year column
df = df.drop(df.columns[0], axis=1)

df.columns

# trim whitespaces in columns
df.columns = df.columns.str.strip()

# remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# drop all columns that end in a number
df = df.drop(df.filter(regex='\d+$').columns, axis=1)

#make copy of original DF
cp_df = df.iloc[:,:].copy(deep=True)

#create LSTM df
nn_df = df.copy()

df.head()

#check for null entries
print(df.isnull().any(axis = 1).sum())

#check for duplicate records
print(df.duplicated().sum())

#get records post covid
postcov_df = df[df['year'] >= 2021]
precov_df = df[df['year'] < 2021]

print(postcov_df.shape)
print(precov_df.shape)

# divide datasets by target variables
m2b_df = postcov_df[['ML_lnm2b','proﬁtability_an','rd_sale_wr','industry','roe_wr','roic_an','gprof_wr','roce_wr','aftret_invcapx_wr','CAPMbeta','chbe_an']]
v2a_df = postcov_df[['ML_lnv2a','sales2rec_an','roic_an','industry','rd_sale_wr','rect_turn_wr','roa_wr','chbe_an','CAPMbeta','lt_ppent_wr','Accrual_wr']]
v2s_df = postcov_df[['ML_lnv2s','at_turn_wr','opleverage_an','gpm_wr','ebitda2revenue_an','rd_sale_wr','sales2cash_an','npm_wr','chbe_an','CAPMbeta','sale_ac']]

print(m2b_df.shape)
print(v2a_df.shape)
print(v2s_df.shape)

#change variable name to ASCII characters
m2b_df = m2b_df.rename(columns={'proﬁtability_an': 'profitability_an'})

print(m2b_df.info())
print(v2a_df.info())
print(v2s_df.info())

"""# Exploration"""

m2b_df.describe()

v2a_df.describe()

v2s_df.describe()

print(m2b_df['ML_lnm2b'].describe())
print(v2a_df['ML_lnv2a'].describe())
print(v2s_df['ML_lnv2s'].describe())

print(m2b_df['ML_lnm2b'].skew())
print(v2a_df['ML_lnv2a'].skew())
print(v2s_df['ML_lnv2s'].skew())

print(m2b_df['ML_lnm2b'].kurtosis())
print(v2a_df['ML_lnv2a'].kurtosis())
print(v2s_df['ML_lnv2s'].kurtosis())

sns.histplot(m2b_df['ML_lnm2b'])

sns.histplot(v2a_df['ML_lnv2a'])

sns.histplot(v2s_df['ML_lnv2s'])

boxplot = m2b_df.boxplot(column=['ML_lnm2b'])

boxplot = v2a_df.boxplot(column=['ML_lnv2a'])

boxplot=v2s_df.boxplot(column='ML_lnv2s')

m2b_df.corr()['ML_lnm2b'].abs().sort_values(ascending = False)

m2b_df.plot.scatter(x='profitability_an', y='ML_lnm2b')

v2a_df.corr()['ML_lnv2a'].abs().sort_values(ascending = False)

v2a_df.plot.scatter(x='roa_wr', y='ML_lnv2a')

v2s_df.corr()['ML_lnv2s'].abs().sort_values(ascending = False)

v2s_df.plot.scatter(x='gpm_wr', y='ML_lnv2s')

plt.figure(figsize=(15,8))
sns.heatmap(m2b_df.corr())

plt.figure(figsize=(15,8))
sns.heatmap(v2a_df.corr())

plt.figure(figsize=(15,8))
sns.heatmap(v2s_df.corr())

# Create a dictionary to store the correlated variables and their scores
correlated_variables1 = {}

# Get the correlation matrix
correlation_matrix = m2b_df.corr()

# Loop through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        # Check if the correlation score is greater than or equal to 0.7
        if abs(correlation_matrix.iloc[i, j]) >= 0.7:
            # Add the correlated variables and their scores to the dictionary
            correlated_variables1[(correlation_matrix.columns[i], correlation_matrix.columns[j])] = correlation_matrix.iloc[i, j]

# Print the dictionary
print(correlated_variables1)

# filtered correlation matrix of variables that have a score of 0.7+

m2b_df_filtered = m2b_df[list(set([x[0] for x in correlated_variables1.keys()] + [x[1] for x in correlated_variables1.keys()]))]

# Create a heatmap of the filtered DataFrame
plt.figure(figsize=(12, 8))
sns.heatmap(m2b_df_filtered.corr(), annot=True, cmap="coolwarm")
plt.show()

# Create a dictionary to store the correlated variables and their scores
correlated_variables2 = {}

# Get the correlation matrix
correlation_matrix = v2a_df.corr()

# Loop through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        # Check if the correlation score is greater than or equal to 0.7
        if abs(correlation_matrix.iloc[i, j]) >= 0.7:
            # Add the correlated variables and their scores to the dictionary
            correlated_variables2[(correlation_matrix.columns[i], correlation_matrix.columns[j])] = correlation_matrix.iloc[i, j]

# Print the dictionary
print(correlated_variables2)

# filtered correlation matrix of variables that have a score of 0.7+

v2a_df_filtered = v2a_df[list(set([x[0] for x in correlated_variables2.keys()] + [x[1] for x in correlated_variables2.keys()]))]

# Create a heatmap of the filtered DataFrame
plt.figure(figsize=(12, 8))
sns.heatmap(v2a_df_filtered.corr(), annot=True, cmap="coolwarm")
plt.show()

# Create a dictionary to store the correlated variables and their scores
correlated_variables3 = {}

# Get the correlation matrix
correlation_matrix = v2s_df.corr()

# Loop through the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        # Check if the correlation score is greater than or equal to 0.7
        if abs(correlation_matrix.iloc[i, j]) >= 0.7:
            # Add the correlated variables and their scores to the dictionary
            correlated_variables3[(correlation_matrix.columns[i], correlation_matrix.columns[j])] = correlation_matrix.iloc[i, j]

# Print the dictionary
print(correlated_variables3)

# filtered correlation matrix of variables that have a score of 0.7+

v2s_df_filtered = v2s_df[list(set([x[0] for x in correlated_variables3.keys()] + [x[1] for x in correlated_variables3.keys()]))]

# Create a heatmap of the filtered DataFrame
plt.figure(figsize=(12, 8))
sns.heatmap(v2s_df_filtered.corr(), annot=True, cmap="coolwarm")
plt.show()

# confidence level (ci) of 90%
sns.barplot(x = 'industry', y = 'ML_lnm2b', errorbar=('ci', 95), data = m2b_df)

# confidence level of 90%
sns.barplot(x = 'industry', y = 'ML_lnv2a', errorbar=('ci', 95), estimator=np.max, data = v2a_df)

# binning methods
def sturges_formula(n):
    return int(np.ceil(np.log2(n) + 1))

def sqrt_choice(n):
    return int(np.ceil(np.sqrt(n)))

def freedman_diaconis_rule(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr * len(data) ** (-1/3)
    return int(np.ceil((max(data) - min(data)) / bin_width))

# Number of bins according to different methods
n = len(v2s_df)
bins_sturges = sturges_formula(n)
bins_sqrt = sqrt_choice(n)
bins_fd = freedman_diaconis_rule(v2s_df['gpm_wr'])

print(bins_sturges)
print(bins_sqrt)
print(bins_fd)

# Net profit by Value to Sales
bins = pd.qcut(v2s_df['gpm_wr'], 16)
sns.barplot(x="ML_lnv2s", y=bins, data=v2s_df)
plt.ylabel("gpm_wr Bins")
plt.xlabel("ML_lnv2s")
plt.show()

"""# Pre-Processing

#### Outliers
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

print(m2b_df.shape)
print(v2a_df.shape)
print(v2s_df.shape)

def remove_outliers(df, col, threshold):
    # Calculate the z-score for each data point in the column
    z_scores = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    # Define the threshold for outlier identification
    threshold = threshold

    # Select data points that are not outliers
    return( df[(abs(z_scores) <= threshold)] )

m2b_df = remove_outliers(m2b_df, 'ML_lnm2b',1.8)
v2a_df = remove_outliers(v2a_df, 'ML_lnv2a',1.7)
v2s_df = remove_outliers(v2s_df, 'ML_lnv2s',0.9)

print(m2b_df.shape)
print(v2a_df.shape)
print(v2s_df.shape)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('ML_lnm2b, ML_lnv2a, and ML_lnv2s')
m2b_df.boxplot(column=['ML_lnm2b'], ax=axes[0])
v2a_df.boxplot(column=['ML_lnv2a'], ax=axes[1])
v2s_df.boxplot(column=['ML_lnv2s'], ax=axes[2])
plt.show()

"""#### Scaling"""

m2b_df_cp = m2b_df.copy()

def scale_select_features(df):
  # Exclude target variables, year, and industry column
  features = df.columns[~df.columns.isin(['ML_lnm2b', 'ML_lnv2a', 'ML_lnv2s', 'industry','year'])]

  # Calculate standard deviation of features
  std_devs = df[features].std(ddof=0)

  # Select features with standard deviation greater than 1 and greater than the mean
  selected_features = std_devs[(std_devs > 1) & (std_devs > df[features].mean())].index.tolist()
  print(selected_features)

  # Normalize selected features using MinMaxScaler
  scaler = MinMaxScaler()
  df[selected_features] = scaler.fit_transform(df[selected_features])

scale_select_features(m2b_df)
scale_select_features(v2a_df)
scale_select_features(v2s_df)

m2b_df.describe().loc[['mean', 'std']]

v2a_df.describe().loc[['mean', 'std']]

v2s_df.describe().loc[['mean', 'std']]

"""#### Label Categorical columns"""

m2b_df['industry'] = m2b_df['industry'].astype('category')
v2a_df['industry'] = v2a_df['industry'].astype('category')

"""# Feature Selection

#### Calculate VIF
"""

def calculate_vif(X):
    # Add a constant column to calculate VIF
    X = pd.DataFrame(X)
    X['constant'] = 1

    vif = pd.DataFrame()
    vif['Feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.drop(index=X.shape[1]-1)  # Drop the constant column

# Calculate VIF for training data
m2b_vif = calculate_vif(m2b_df.iloc[:,1:])
print("m2b vif:\n",m2b_vif)

v2a_vif = calculate_vif(v2a_df.iloc[:,1:])
print("v2a vif:\n",v2a_vif)

v2s_vif = calculate_vif(v2s_df.iloc[:,1:])
print("v2s vif:",v2s_vif)

#drop variables with high vif scores
m2b_df = m2b_df.drop(['roic_an', 'roce_wr', 'aftret_invcapx_wr'], axis=1)
v2a_df = v2a_df.drop(['sales2rec_an', 'rect_turn_wr'], axis=1)

#verify columns
print(m2b_df.columns)
print(v2a_df.columns)
print(v2s_df.columns)

# Calculate new VIF for training data. Anything less than 10 is acceptable.
m2b_vif = calculate_vif(m2b_df.iloc[:,1:])
print("m2b vif:\n",m2b_vif)

v2a_vif = calculate_vif(v2a_df.iloc[:,1:])
print("v2a vif:\n",v2a_vif)

v2s_vif = calculate_vif(v2s_df.iloc[:,1:])
print("v2s vif:",v2s_vif)

#plot VIF Scores
import matplotlib.pyplot as plt
plt.hist(m2b_vif['VIF'], label='m2b_vif')
plt.hist(v2a_vif['VIF'], label='v2a_vif')
plt.hist(v2s_vif['VIF'], label='v2s_vif')
plt.title('VIF scores')
plt.ylabel('Frequency')
plt.xlabel('VIF')
plt.legend()
plt.show()

print(m2b_df.shape)
print(v2a_df.shape)
print(v2s_df.shape)

"""### Spliting the Data"""

import random
random.seed(333)  # Set a seed for reproducibility

#randomize indices
def get_indices (df):
  indices = list(range(len(df)))
  random.shuffle(indices)
  return indices

def slice_df(indices):
  #Ratio: 60:20:20
  train_size = int(0.6 * len(indices))
  valid_size = int(0.2 * len(indices))

  train_ind = indices[:train_size]
  valid_ind = indices[train_size:train_size + valid_size]
  test_ind = indices[train_size + valid_size:]

  return (train_ind, valid_ind, test_ind)

def get_slices(df, train_ind, valid_ind, test_ind):
  t_df = df.iloc[train_ind]
  v_df = df.iloc[valid_ind]
  tst_df = df.iloc[test_ind]
  return t_df, v_df, tst_df

#select m2b data.

indices = get_indices(m2b_df)
train_ind, valid_ind, test_ind = slice_df(indices)

# Save the indices to a text file on Google Drive
with open('/content/gdrive/My Drive/EY_data/round3/m2b/m2b_train__indices.txt', 'w') as f:
  for index in train_ind:
    f.write(str(index) + '\n')

with open('/content/gdrive/My Drive/EY_data/round3/m2b/m2b_valid_indices.txt', 'w') as f:
  for index in valid_ind:
    f.write(str(index) + '\n')

with open('/content/gdrive/My Drive/EY_data/round3/m2b/m2b_test_indices.txt', 'w') as f:
  for index in test_ind:
    f.write(str(index) + '\n')

t_m2b_df, v_m2b_df, tst_m2b_df = get_slices(m2b_df, train_ind, valid_ind, test_ind)

#select v2a data.
indices = get_indices(v2a_df)

train_ind, valid_ind, test_ind = slice_df(indices)

# Save the indices to a text file on Google Drive
with open('/content/gdrive/My Drive/EY_data/round3/v2a/v2a_train__indices.txt', 'w') as f:
  for index in train_ind:
    f.write(str(index) + '\n')

with open('/content/gdrive/My Drive/EY_data/round3/v2a/v2a_valid_indices.txt', 'w') as f:
  for index in valid_ind:
    f.write(str(index) + '\n')

with open('/content/gdrive/My Drive/EY_data/round3/v2a/v2a_test_indices.txt', 'w') as f:
  for index in test_ind:
    f.write(str(index) + '\n')

t_v2a_df, v_v2a_df, tst_v2a_df = get_slices(v2a_df, train_ind, valid_ind, test_ind)

#select v2s data.
indices = get_indices(v2s_df)
train_ind, valid_ind, test_ind = slice_df(indices)

# Save the indices to a text file on Google Drive
with open('/content/gdrive/My Drive/EY_data/round3/v2s/v2s_train__indices.txt', 'w') as f:
  for index in train_ind:
    f.write(str(index) + '\n')

with open('/content/gdrive/My Drive/EY_data/round3/v2s/v2s_valid_indices.txt', 'w') as f:
  for index in valid_ind:
    f.write(str(index) + '\n')

with open('/content/gdrive/My Drive/EY_data/round3/v2s/v2s_test_indices.txt', 'w') as f:
  for index in test_ind:
    f.write(str(index) + '\n')

t_v2s_df, v_v2s_df, tst_v2s_df = get_slices(v2s_df, train_ind, valid_ind, test_ind)

print(t_m2b_df.shape)
print(v_m2b_df.shape)
print(tst_m2b_df.shape)

print(t_v2a_df.shape)
print(v_v2a_df.shape)
print(tst_v2a_df.shape)

print(t_v2s_df.shape)
print(v_v2s_df.shape)
print(tst_v2s_df.shape)

# Create X and y variables for training, validation and testing.

#m2b
X1 = t_m2b_df.iloc[:,1:]
y1 = t_m2b_df.iloc[:,0]

v_x1 = v_m2b_df.iloc[:,1:]
v_y1 = v_m2b_df.iloc[:,0]

t_x1 = tst_m2b_df.iloc[:,1:]
t_y1 = tst_m2b_df.iloc[:,0]


#v2a
X2 = t_v2a_df.iloc[:,1:]
y2 = t_v2a_df.iloc[:,0]

v_x2 = v_v2a_df.iloc[:,1:]
v_y2 = v_v2a_df.iloc[:,0]

t_x2 = tst_v2a_df.iloc[:,1:]
t_y2 = tst_v2a_df.iloc[:,0]

#v2s
X3 = t_v2s_df.iloc[:,1:]
y3 = t_v2s_df.iloc[:,0]

v_x3 = v_v2s_df.iloc[:,1:]
v_y3 = v_v2s_df.iloc[:,0]

t_x3 = tst_v2s_df.iloc[:,1:]
t_y3 = tst_v2s_df.iloc[:,0]

"""# Model Metrics"""

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return rmse, pearson_corr, r_squared
    #t_y1,y1_test_pred

def perc_val_err (actual_vals, pred_vals):
  actual = np.array(actual_vals)
  pred = np.array(pred_vals)
  #val_err = (pred - actual)
  val_err = (pred / actual) - 1
  return val_err

"""# Exploring Linear Relations with OLS"""

#make copies for the OLS model
ols_X1 = X1.copy()
ols_v_x1 = v_x1.copy()
ols_t_x1 = t_x1.copy()

ols_X2 = X2.copy()
ols_v_x2 = v_x2.copy()
ols_t_x2 = t_x2.copy()

ols_X3 = X3.copy()
ols_v_x3 = v_x3.copy()
ols_t_x3 = t_x3.copy()

ols_X1.head()

"""#### Initial OLS Models"""

#build ols model
def ols_model(X,y):
  X = sm.add_constant(X)
  model = sm.OLS(y, X).fit()
  return model

#call model
m2b_ols = ols_model(ols_X1, y1)
v2a_ols = ols_model(ols_X2, y2)
v2s_ols = ols_model(ols_X3, y3)

print(m2b_ols.summary())
print(v2a_ols.summary())
print(v2s_ols.summary())

# Fit an OLS model with 'industry' as a categorical variable
m2b_ols = smf.ols('ML_lnm2b ~ C(industry) + ' + '+'.join(ols_X1.columns.drop('industry')), data=pd.concat([ols_X1, y1], axis=1)).fit()

# Print the model summary
print(m2b_ols.summary())

# Fit an OLS model with 'industry' as a categorical variable
v2a_ols = smf.ols('ML_lnv2a ~ C(industry) + ' + '+'.join(ols_X2.columns.drop('industry')), data=pd.concat([ols_X2, y2], axis=1)).fit()

# Print the model summary
print(v2a_ols.summary())

"""#### Set Industry as a Dummy"""

#create dumies for industry column
ols_X1 = pd.get_dummies(ols_X1, columns=['industry'], drop_first=True)
ols_v_x1 = pd.get_dummies(ols_v_x1, columns=['industry'], drop_first=True)
ols_t_x1 = pd.get_dummies(ols_t_x1, columns=['industry'], drop_first=True)
ols_X2 = pd.get_dummies(ols_X2, columns=['industry'], drop_first=True)
ols_v_x2 = pd.get_dummies(ols_v_x2, columns=['industry'], drop_first=True)
ols_t_x2 = pd.get_dummies(ols_t_x2, columns=['industry'], drop_first=True)

# Encoding: Replace True and False with 1 and 0 in all columns
ols_X1 = ols_X1.replace({True: 1, False: 0})
ols_v_x1 = ols_v_x1.replace({True: 1, False: 0})
ols_t_x1 = ols_t_x1.replace({True: 1, False: 0})
ols_X2 = ols_X2.replace({True: 1, False: 0})
ols_v_x2 = ols_v_x2.replace({True: 1, False: 0})
ols_t_x2 = ols_t_x2.replace({True: 1, False: 0})

ols_X1.columns

ols_m2b = ols_model(ols_X1, y1)
ols_v2a = ols_model(ols_X2, y2)


print(ols_m2b.summary())
print(ols_v2a.summary())

"""#### Drop Insignificant Variables"""

def get_insignificant_vars(model):
  tvalues = model.tvalues

  # Identify variables with t-value less than 1.96 for 95% two-tail confidence level
  insignificant_vars = tvalues[abs(tvalues) < 1.96].index

  # Exclude 'const' if present
  insignificant_vars = [var for var in insignificant_vars if var != 'const']
  return insignificant_vars

# Drop insignificant variables from all dfs
ols_X1 = ols_X1.drop(get_insignificant_vars(ols_m2b), axis=1)
ols_v_x1 = ols_v_x1.drop(get_insignificant_vars(ols_m2b), axis=1)
ols_t_x1 = ols_t_x1.drop(get_insignificant_vars(ols_m2b), axis=1)

ols_X2 = ols_X2.drop(get_insignificant_vars(ols_v2a), axis=1)
ols_v_x2 = ols_v_x2.drop(get_insignificant_vars(ols_v2a), axis=1)
ols_t_x2 = ols_t_x2.drop(get_insignificant_vars(ols_v2a), axis=1)

ols_X3 = ols_X3.drop(get_insignificant_vars(v2s_ols), axis=1)
ols_v_x3 = ols_v_x3.drop(get_insignificant_vars(v2s_ols), axis=1)
ols_t_x3 = ols_t_x3.drop(get_insignificant_vars(v2s_ols), axis=1)

#show new variables
print(ols_X1.columns,"\n")
print(ols_X2.columns,"\n")
print(ols_X3.columns)

"""#### m2b_OLS Training & Testing"""

ols_m2b = ols_model(ols_X1, y1)
ols_v2a = ols_model(ols_X2, y2)
ols_v2s = ols_model(ols_X3, y3)

print(ols_m2b.summary())
print(ols_v2a.summary())
print(ols_v2s.summary())

#print("Is ols_m2b fitted?", ols_m2b.fittedvalues is not None)
print("Model parameters:", ols_m2b.params)

#drop the insignificant industry column with a t value of nan
ols_X1 = ols_X1.drop('industry_16', axis=1)
ols_v_x1 = ols_v_x1.drop('industry_16', axis=1)
ols_t_x1 = ols_t_x1.drop('industry_16', axis=1)

ols_X2 = ols_X2.drop(get_insignificant_vars(ols_v2a), axis=1)
ols_v_x2 = ols_v_x2.drop(get_insignificant_vars(ols_v2a), axis=1)
ols_t_x2 = ols_t_x2.drop(get_insignificant_vars(ols_v2a), axis=1)

ols_X2.columns

ols_m2b = ols_model(ols_X1, y1)
ols_v2a = ols_model(ols_X2, y2)

print(ols_m2b.summary())
print(ols_v2a.summary())

#print variable importance
def get_feature_importance(model):
  coefs = model.params
  sorted_coefs = coefs.sort_values(ascending=False)
  return sorted_coefs

# Validate on the validation set
ols_v_x1 = sm.add_constant(ols_v_x1)
y1_pred_val = ols_m2b.predict(ols_v_x1)
rmse_val, pearson_corr_val, r_squared_val = get_metrics(v_y1, y1_pred_val)
err_val = perc_val_err(v_y1, y1_pred_val)
avg_err_val = np.mean(err_val)

# Test on the test set
ols_t_x1 = sm.add_constant(ols_t_x1)
y1_pred_test = ols_m2b.predict(ols_t_x1)
rmse_test, pearson_corr_test, r_squared_test = get_metrics(t_y1, y1_pred_test)
err_test = perc_val_err(t_y1, y1_pred_test)
avg_err_test = np.mean(err_test)

# Print the validation and test metrics
print("Validation Results for m2b:")
print(f"RMSE: {rmse_val}")
print(f"Pearson Correlation: {pearson_corr_val}")
print(f"R-squared: {r_squared_val}")
print(f"Average Percentage Error: {abs(avg_err_val)}")

print("\nTest Results:")
print(f"RMSE: {rmse_test}")
print(f"Pearson Correlation: {pearson_corr_test}")
print(f"R-squared: {r_squared_test}")
print(f"Average Percentage Error: {abs(avg_err_test)}\n")

print("Feature Importance:", get_feature_importance(ols_m2b))

# Validate on the validation set
ols_v_x2 = sm.add_constant(ols_v_x2)
y2_pred_val = ols_v2a.predict(ols_v_x2)
rmse_val, pearson_corr_val, r_squared_val = get_metrics(v_y2, y2_pred_val)
err_val = perc_val_err(v_y2, y2_pred_val)
avg_err_val = np.mean(err_val)

# Test on the test set
ols_t_x2 = sm.add_constant(ols_t_x2)
y2_pred_test = ols_v2a.predict(ols_t_x2)
rmse_test, pearson_corr_test, r_squared_test = get_metrics(t_y2, y2_pred_test)
err_test = perc_val_err(t_y2, y2_pred_test)
avg_err_test = np.mean(err_test)

# Print the validation and test metrics
print("Validation Results for v2a:")
print(f"RMSE: {rmse_val}")
print(f"Pearson Correlation: {pearson_corr_val}")
print(f"R-squared: {r_squared_val}")
print(f"Average Percentage Error: {abs(avg_err_val)}")

print("\nTest Results:")
print(f"RMSE: {rmse_test}")
print(f"Pearson Correlation: {pearson_corr_test}")
print(f"R-squared: {r_squared_test}")
print(f"Average Percentage Error: {abs(avg_err_test)}\n")

print("Feature Importance:", get_feature_importance(ols_v2a))

# Validate on the validation set
ols_v_x3 = sm.add_constant(ols_v_x3)
y3_pred_val = ols_v2s.predict(ols_v_x3)
rmse_val, pearson_corr_val, r_squared_val = get_metrics(v_y3, y3_pred_val)
err_val = perc_val_err(v_y3, y3_pred_val)
avg_err_val = np.mean(err_val)

# Test on the test set
ols_t_x3 = sm.add_constant(ols_t_x3)
y3_pred_test = ols_v2s.predict(ols_t_x3)
rmse_test, pearson_corr_test, r_squared_test = get_metrics(t_y3, y3_pred_test)
err_test = perc_val_err(t_y3, y3_pred_test)
avg_err_test = np.mean(err_test)

# Print the validation and test metrics
print("Validation Results for v2s:")
print(f"RMSE: {rmse_val}")
print(f"Pearson Correlation: {pearson_corr_val}")
print(f"R-squared: {r_squared_val}")
print(f"Average Percentage Error: {abs(avg_err_val)}")

print("\nTest Results:")
print(f"RMSE: {rmse_test}")
print(f"Pearson Correlation: {pearson_corr_test}")
print(f"R-squared: {r_squared_test}")
print(f"Average Percentage Error: {abs(avg_err_test)}\n")

print("Feature Importances:", get_feature_importance(ols_v2s))

"""# Modeling with LightGBM"""

print(X1.shape)
print(X2.shape)
print(X3.shape)

print(X1.columns)
print(X2.columns)
print(X3.columns)

# Commented out IPython magic to ensure Python compatibility.
# %pip install lightgbm

# Commented out IPython magic to ensure Python compatibility.
# %pip install optuna

import lightgbm as lgb
import optuna

#calculate leaf and min_data estimates
print(int(2*np.sqrt(len(X1))))
print(int(2*np.sqrt(len(X2))))
print(int(2*np.sqrt(len(X3))))

print(int(0.05*len(X1)))
print(int(0.05*len(X2)))
print(int(0.05*len(X3)))

"""#### Automate Hyper Parameter Tuning

##### Define Objective and Init Study
"""

#Initialize study
def study_init(X, y, v_x, v_y,categorical_feature):
  study = optuna.create_study(direction='minimize')
  study.optimize(lambda trial: objective(trial, X, y, v_x, v_y,categorical_feature), n_trials=100)
  return study

#Optuna Objective: least RMSE
def objective(trial,X,y,v_x,v_y,categorical_feature):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2,31),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2,20),
        'learning_rate': 0.1,
        'early_stopping_round':trial.suggest_int('early_stopping_round', 10, 100),
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }
    num_boost_round = trial.suggest_int('num_boost_round', 10, 1000)

    if categorical_feature != None:
      params['categorical_feature'] = categorical_feature

    model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_boost_round,
                valid_sets=[lgb.Dataset(v_x, label=v_y)])

    preds = model.predict(v_x, num_iteration=model.best_iteration)

    trial.set_user_attr('num_boost_round', model.best_iteration)

    rmse = mean_squared_error(v_y, preds, squared=False)
    return rmse

"""##### m2b Study"""

#Run study for m2b
study = study_init(X1, y1, v_x1, v_y1, 'industry')
print(f'Best Parameters: {study.best_params}')
print(f'Best RMSE: {study.best_value}')

#31,20,100,1000 (5,57%)
#10,3,10,100 (4.9,51%)
#5,3,10,30 (4.7,55%)

"""##### v2a Study"""

#Run study for v2a
study = study_init(X2, y2, v_x2, v_y2,'industry')
print(f'Best Parameters: {study.best_params}')
print(f'Best RMSE: {study.best_value}')

"""##### v2s Study"""

#Run study for v2s
study = study_init(X3, y3, v_x3, v_y3, None)
print(f'Best Parameters: {study.best_params}')
print(f'Best RMSE: {study.best_value}')

"""#### Train

##### Model Params
"""

def tree_params(leaves,min_data,es):
  params = {
    'num_leaves': leaves,
    'min_data_in_leaf':min_data,
    'learning_rate': 0.1,
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'early_stopping_round':es,
    'verbosity': -1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'metric': ['rmse']
  }
  return params

"""##### m2b"""

categorical_feature = ['industry']

#31,58,80,100
m2b_model = lgb.train(tree_params(31,58,25), lgb.Dataset(X1, label=y1), num_boost_round=0,
                valid_sets=[lgb.Dataset(v_x1, label=v_y1)], categorical_feature=categorical_feature)
print("m2b Trees:",m2b_model.best_iteration)

print("\nModel Parameters:")
for param, value in m2b_model.params.items():
    print(f"{param}: {value}")

    # Print feature importances
print("\nFeature Importances:")
for feature, importance in zip(X1.columns, m2b_model.feature_importance(importance_type='split')):
    print(f"{feature}: {importance}")

"""##### v2a"""

categorical_feature = ['industry']

#6,3,15,10
v2a_model = lgb.train(tree_params(6,3,5), lgb.Dataset(X2, label=y2), num_boost_round=10,
                valid_sets=[lgb.Dataset(v_x2, label=v_y2)], categorical_feature=categorical_feature)
print("v2a Trees:",v2a_model.best_iteration)

print("\nModel Parameters:")
for param, value in v2a_model.params.items():
    print(f"{param}: {value}")

    # Print feature importances
print("\nFeature Importances:")
for feature, importance in zip(X2.columns, v2a_model.feature_importance(importance_type='split')):
    print(f"{feature}: {importance}")

"""##### v2s"""

#6,4,25, 50 (27.2, 67%)
v2s_model = lgb.train(tree_params(6,4,25), lgb.Dataset(X3, label=y3), num_boost_round=50,
                valid_sets=[lgb.Dataset(v_x3, label=v_y3)])
print("v2s Trees:",v2s_model.best_iteration)
#6, 194 - 211 trees, 0.31 oos err
#3, 102 trees,

print("\nModel Parameters:")
for param, value in v2s_model.params.items():
    print(f"{param}: {value}")

    # Print feature importances
print("\nFeature Importances:")
for feature, importance in zip(X3.columns, v2s_model.feature_importance(importance_type='split')):
    print(f"{feature}: {importance}")

"""#### Validate

##### m2b
"""

# Predict on the m2b validation set
y1_pred = m2b_model.predict(v_x1, num_iteration=m2b_model.best_iteration)
rmse, pearson_corr, r_squared = get_metrics(v_y1, y1_pred)
print(f'm2b RMSE: {rmse}')
print(f"m2b R-squared: {r_squared:.2f}")
print(f'm2b Pearson Corr: {pearson_corr}')

"""##### v2a"""

#Predict on the v2a validation set
y2_pred = v2a_model.predict(v_x2, num_iteration=v2a_model.best_iteration)
rmse, pearson_corr, r_squared = get_metrics(v_y2, y2_pred)
print(f'v2a RMSE: {rmse}')
print(f"v2a R-squared: {r_squared:.2f}")
print(f'v2a Pearson Corr: {pearson_corr}')

"""##### v2s"""

#Predict on the v2a validation set
y3_pred = v2s_model.predict(v_x3, num_iteration=v2s_model.best_iteration)
rmse, pearson_corr, r_squared = get_metrics(v_y3, y3_pred)
print(f'v2s RMSE: {rmse}')
print(f"v2s R-squared: {r_squared:.2f}")
print(f'v2s Pearson Corr: {pearson_corr}')

"""#### Out of Sample Test

##### m2b
"""

#Testing m2b Model using oos data
y1_test_pred = m2b_model.predict(t_x1, num_iteration=m2b_model.best_iteration)
rmse, pearson_corr, r_squared = get_metrics(t_y1, y1_test_pred)

err1 = perc_val_err (t_y1, y1_test_pred)
avg_err1 = np.mean(err1)

print ("oos m2b error ratio:",abs(avg_err1))
print(f'm2b RMSE: {rmse}')
print(f"m2b R-squared: {r_squared:.2f}")
print("pearson_corr:", pearson_corr)

"""##### v2a"""

#Testing v2a Model using oos data
y2_test_pred = v2a_model.predict(t_x2, num_iteration=v2a_model.best_iteration)
rmse, pearson_corr, r_squared = get_metrics(t_y2, y2_test_pred)

#print(f'v2a Pearson Corr: {pearson_corr}')

err2 = perc_val_err (t_y2, y2_test_pred)
avg_err2 = np.mean(err2)

print ("oos v2a error ratio:",abs(avg_err2))
print(f'v2a RMSE: {rmse}')
print(f"v2a R-squared: {r_squared:.2f}")
print("pearson_corr:", pearson_corr)

"""##### v2s"""

#Testing v2s Model using oos data
y3_test_pred = v2s_model.predict(t_x3, num_iteration=v2s_model.best_iteration)
rmse, pearson_corr, r_squared = get_metrics(t_y3, y3_test_pred)

#print(f'v2s Pearson Corr: {pearson_corr}')

err3 = perc_val_err (t_y3, y3_test_pred)
avg_err3 = np.mean(err3)

print ("oos v2s error ratio:",abs(avg_err3))
print(f'v2s RMSE: {rmse}')
print(f"v2s R-squared: {r_squared:.2f}")
print("pearson_corr:", pearson_corr)

"""### Show Model Trees"""

lgb.plot_tree(m2b_model, tree_index=0, figsize=(10, 5), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.title('m2b Tree 0')
plt.show()

lgb.plot_tree(v2a_model, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.title('v2a Tree 0')
plt.show()

lgb.plot_tree(v2s_model, tree_index=0, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.title('v2s Tree 0')
plt.show()

#print(m2b_model.model_to_string())
#print(v2a_model.model_to_string())
print(v2s_model.model_to_string())

"""### SHapley Additive exPlanations (SHAP) Evaluation"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install shap

import shap

# SHAP values calculation
def get_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

# Plot SHAP feature importance
shap.summary_plot(get_shap_values(m2b_model, X1), X1, plot_type="bar", feature_names=X1.columns)
shap.summary_plot(get_shap_values(v2a_model, X2), X2, plot_type="bar", feature_names=X2.columns)
shap.summary_plot(get_shap_values(v2s_model, X3), X3, plot_type="bar", feature_names=X3.columns)

# Plot SHAP summary plot
shap.summary_plot(get_shap_values(m2b_model, X1), X1, feature_names=X1.columns)
shap.summary_plot(get_shap_values(v2a_model, X2), X2, feature_names=X2.columns)
shap.summary_plot(get_shap_values(v2s_model, X3), X3, feature_names=X3.columns)

"""##### Identify Best Features

"""

# prompt: return shap values of all variables in descending order for all 3 models m2b_model, v2a_model, and v2s_model

import numpy as np
def get_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def get_top_features(model, X):

  # Calculate mean absolute SHAP values for each feature
  mean_shap_values = np.abs(get_shap_values(model, X)).mean(axis=0)

  # Get top 5 features based on mean absolute SHAP values
  top_features = np.argsort(mean_shap_values)[-5:][::-1]
  top_features_names = X.columns[top_features]

  return top_features_names

# Get shap values for all models
m2b_shap_values = get_shap_values(m2b_model, X1)
v2a_shap_values = get_shap_values(v2a_model, X2)
v2s_shap_values = get_shap_values(v2s_model, X3)

# Get top features for each model
m2b_top_features = get_top_features(m2b_model, X1)
v2a_top_features = get_top_features(v2a_model, X2)
v2s_top_features = get_top_features(v2s_model, X3)

# Print shap values in descending order for each model
print("m2b SHAP Values:")
for i in range(len(m2b_shap_values[0])):
    feature_name = X1.columns[i]
    shap_value = m2b_shap_values[0][i]
    print(f"{feature_name}: {shap_value}")

print("\nv2a SHAP Values:")
for i in range(len(v2a_shap_values[0])):
    feature_name = X2.columns[i]
    shap_value = v2a_shap_values[0][i]
    print(f"{feature_name}: {shap_value}")

print("\nv2s SHAP Values:")
for i in range(len(v2s_shap_values[0])):
    feature_name = X3.columns[i]
    shap_value = v2s_shap_values[0][i]
    print(f"{feature_name}: {shap_value}")

def get_top_features(model, X):

  # Calculate mean absolute SHAP values for each feature
  mean_shap_values = np.abs(get_shap_values(model, X)).mean(axis=0)

  # Get top 5 features based on mean absolute SHAP values
  top_features = np.argsort(mean_shap_values)[-5:][::-1]
  top_features_names = X.columns[top_features]

  return top_features_names

m2b_top5 = get_top_features(m2b_model, X1)
v2a_top5 = get_top_features(v2a_model, X2)
v2s_top5 = get_top_features(v2s_model, X3)

print(get_top_features(m2b_model, X1))
print(get_top_features(v2a_model, X2))
print(get_top_features(v2s_model, X3))

#drop insignificant columns
#X1 = X1[m2b_top5]
#X2 = X2[v2a_top5]
#X3 = X3[v2s_top5]

#v_x1 = v_x1[m2b_top5]
#v_x2 = v_x2[v2a_top5]
#v_x3 = v_x3[v2s_top5]

#t_x1 = t_x1[m2b_top5]
#t_x2 = t_x2[v2a_top5]
#t_x3 = t_x3[v2s_top5]

"""# Exploring LSTM"""

nn_df = nn_df.rename(columns={'proﬁtability_an': 'profitability_an'})

nn_df = nn_df.sort_values(by='year')

nn_df = nn_df[nn_df['year'] >= 2021]

nn_df.head()

nn_df.shape

nn_df.tail()

# Exclude target variables and industry column
features = nn_df.columns[~nn_df.columns.isin(['ML_lnm2b', 'ML_lnv2a', 'ML_lnv2s', 'industry','year'])]

# Calculate standard deviation of features
std_devs = nn_df[features].std(ddof=0)

# Select features with standard deviation greater than 1 and greater than the mean
selected_features = std_devs[(std_devs > 1) & (std_devs > nn_df[features].mean())].index.tolist()

# Normalize selected features using MinMaxScaler
scaler = MinMaxScaler()
nn_df[selected_features] = scaler.fit_transform(nn_df[selected_features])

nn_df.head()

nn_m2b_df = nn_df[['profitability_an','rd_sale_wr','industry','roe_wr','roic_an','gprof_wr','roce_wr','aftret_invcapx_wr','CAPMbeta','chbe_an','ML_lnm2b']]
nn_v2a_df = nn_df[['sales2rec_an','roic_an','industry','rd_sale_wr','rect_turn_wr','roa_wr','chbe_an','CAPMbeta','lt_ppent_wr','Accrual_wr','ML_lnv2a']]
nn_v2s_df = nn_df[['at_turn_wr','opleverage_an','gpm_wr','ebitda2revenue_an','rd_sale_wr','sales2cash_an','npm_wr','chbe_an','CAPMbeta','sale_ac','ML_lnv2s']]

# Convert DataFrame to a 3D array for LSTM
def create_time_series(df, n_years=3):
    X, y = [], []
    for i in range(0, len(df), n_years):
        if i + n_years <= len(df):
            X.append(df.iloc[i:i+n_years, 0:-1].values)  # Features (excluding target)
            y.append(df.iloc[i+n_years-1, -1])  #target
    return np.array(X), np.array(y)

X_m2b, y_m2b = create_time_series(nn_m2b_df)
X_v2a, y_v2a = create_time_series(nn_v2a_df)
X_v2s, y_v2s = create_time_series(nn_v2s_df)

# Verify the shapes
print("Shape of X:", X_m2b.shape)  # (samples, timesteps, features)
print("Shape of y:", y_m2b.shape)  # (samples,)

# Split data into training and testing sets
split_ratio = 0.8
m2b_train, m2b_test, m2b_y_train, m2b_y_test = train_test_split(X_m2b, y_m2b, test_size=1-split_ratio, random_state=31)

v2a_train, v2a_test, v2a_y_train, v2a_y_test = train_test_split(X_v2a, y_v2a, test_size=1-split_ratio, random_state=31)

v2s_train, v2s_test, v2s_y_train, v2s_y_test = train_test_split(X_v2s, y_v2s, test_size=1-split_ratio, random_state=31)

# Verify the shapes of the training and testing sets
print("Shape of X_train:", m2b_train.shape)
print("Shape of y_train:", m2b_y_train.shape)
print("Shape of X_test:", m2b_test.shape)
print("Shape of y_test:", m2b_y_test.shape)

!pip install scikeras

!pip install scikit-learn

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam, RMSprop

# Define the LSTM model
def build_model(input_shape,units):
    model = Sequential()
    model.add(LSTM(units, activation='tanh', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def input_shape(X_train):
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    return input_shape

# Build the models
nn_m2b_model = build_model(input_shape(m2b_train),150)
nn_v2a_model = build_model(input_shape(v2a_train),150)
nn_v2s_model = build_model(input_shape(v2s_train),150)

# Train the models
nn_m2b_model.fit(m2b_train, m2b_y_train, epochs=20, batch_size=10, verbose=0)
nn_m2b_model.fit(v2a_train, v2a_y_train, epochs=20, batch_size=20, verbose=0)
nn_m2b_model.fit(v2s_train, v2s_y_train, epochs=20, batch_size=10, verbose=0)


# Evaluate the model
def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)

  #rmse, pearson_corr, r_squared = get_metrics(y_test, y_pred)
  mse = np.mean((y_pred - y_test)**2)
  rmse = np.sqrt(mse)

  r_squared = r2_score(y_test, y_pred)

  err = perc_val_err (y_test, y_pred)
  avg_err = np.mean(err)

  return rmse, r_squared, avg_err


rmse1, r_squared1, avg_err1 = evaluate_model(nn_m2b_model, m2b_test, m2b_y_test)
rmse2, r_squared2, avg_err2 = evaluate_model(nn_v2a_model, v2a_test, v2a_y_test)
rmse3, r_squared3, avg_err3 = evaluate_model(nn_v2s_model, v2s_test, v2s_y_test)


print("m2b RMSE:", rmse1)
#print("m2b Pearson Correlation:", pearson_corr1)
print("m2b R-squared:", r_squared1)
print("m2b Average Percentage Error:", avg_err1,"\n")

print("v2a RMSE:", rmse2)
#print("v2a Pearson Correlation:", pearson_corr2)
print("v2a R-squared:", r_squared2)
print("v2a Average Percentage Error:", avg_err2,"\n")


print("v2s RMSE:", rmse3)
#print("v2s Pearson Correlation:", pearson_corr3)
print("v2s R-squared:", r_squared3)
print("v2s Average Percentage Error:", avg_err3,"\n")

from sklearn.model_selection import cross_val_score

#define study objective for units and activation
def build_model(units=100, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(v2s_train.shape[1], v2s_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

def objective(trial):
    # Suggest hyperparameters to be optimized by Optuna
    units = trial.suggest_categorical('units', [50, 100, 150])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])

    # Create the model with the suggested hyperparameters
    model = KerasRegressor(build_fn=build_model, units=units, activation=activation, verbose=0)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, v2s_train, v2s_y_train, cv=3, scoring='neg_mean_squared_error')
    return -scores.mean()

#initialise study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best parameters from Optuna: {best_params}")

# Create the model with the best hyperparameters from Optuna
model = KerasRegressor(build_fn=build_model, units=best_params['units'], activation=best_params['activation'], verbose=0)

# Define the parameter grid for GridSearchCV
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [10, 20],
    'epochs': [10, 20]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(v2s_train, v2s_y_train)

# Print the best parameters and score
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
