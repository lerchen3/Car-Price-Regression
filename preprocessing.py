import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

training_data = pd.read_csv('train.csv')  # 188533 rows, 13 columns - last column is price
test_data = pd.read_csv('test.csv')  # 125690 rows

training_data = training_data.fillna('unknown')
test_data = test_data.fillna('unknown')
def encode_columns(df):
    df['hp'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
    quantiles = [0.04 * cnt for cnt in range(26)]
    bin_edges = df['hp'].quantile(quantiles).values
    df['hp_bin'] = pd.cut(df['hp'], bins=bin_edges, labels=False, include_lowest=True) #bucket into 11 unique (was originally 348)
    df['cylinder'] = df['engine'].str.extract(r'(\d+\.?\d*) Cylinder').astype(float) #7 unique

    df = df.drop(columns=['engine', 'hp'])
    
    df['got_V'] = df['model'].str.extract(r'(\d+\.?\d*) V').notna().astype(int)
    
    return df

training_data = encode_columns(training_data)
test_data = encode_columns(test_data)

test_data['price'] = 0  
all_data = pd.concat([training_data, test_data], ignore_index=True)

nan_cols = ['brand', 'accident', 'hp_bin', 'cylinder']
all_data['nans'] = 0              
for nan_col_idx in range(4):
    nan_col = nan_cols[nan_col_idx]
    all_data['nans'] += all_data[nan_col].isna().astype(int) * (2 ** nan_col_idx) #4 different types apparently (categorical)

all_data.fillna({'brand': 'unknown', 'accident': 'unknown', 'hp_bin' : -1, 'cylinder': -1}, inplace=True)

categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'hp_bin', 'cylinder', 'got_V', 'nans']

all_train = all_data.head(188533)

for column in categorical_features:
    all_data[column] = pd.Categorical(all_data[column])
    mean_values = all_train.groupby(column)['price'].mean()
    all_data[column + '_num'] = all_data[column].map(mean_values)

X_data = all_data.drop(columns=['price'])

scaler = StandardScaler()
numerical = ['model_year', 'milage', 'brand_num', 'model_num', 'fuel_type_num', 'transmission_num', 'ext_col_num', 'int_col_num', 'accident_num', 'clean_title_num', 'hp_bin_num', 'cylinder_num', 'got_V_num', 'nans_num']
X_data[numerical] = pd.DataFrame(scaler.fit_transform(X_data[numerical]), columns=X_data[numerical].columns)