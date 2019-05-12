import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge

from pandas_sklearn import simple_imputer, variance_threshold_selector, select_k_best, select_from_model
random_state = 2019
np.random.RandomState(random_state)

from regression import *

target_class = "SalePrice"
cols_must_drop = ["Id", "BsmtFinSF2", "PoolQC"]
# BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2  "YrSold", "MoSold"
log_transform = ["SalePrice", "LotArea"]

must_try_include = []#, "KitchenQual" "OverallQual", "OverallCond", "Year.Built", "YearRemodAdd","TotalBsmtSF", "SaleCondition"]


home_data = pd.read_csv("data/train.csv")
home_test = pd.read_csv("data/test.csv")


# Outliers?
home_data = home_data[home_data.GrLivArea < 4500]
home_data.reset_index(drop=True, inplace=True)

# Remove rows with missing target
home_data.dropna(axis=0, subset=[target_class], inplace=True)
home_data.reset_index(drop=True, inplace=True)


# features set as 'must drop'
if len(cols_must_drop) > 0:
    home_data.drop(cols_must_drop, axis=1, inplace=True)
    home_test.drop(cols_must_drop, axis=1, inplace=True)


# log(1+x)
if len(log_transform) > 0:
    for feature in log_transform:
        home_data[feature] = np.log1p(home_data[feature])
        if feature in home_test.columns:
            home_test[feature] = np.log1p(home_test[feature])

y = home_data[target_class]
home_data.drop([target_class], axis=1, inplace=True)

data = pd.concat([home_data, home_test], sort=True).reset_index(drop=True)

# Change data type
data['MSSubClass'] = data['MSSubClass'].apply(str)
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)

data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Get names of columns with too much missing values
cols_with_missing = [col for col in data.columns
                    if data[col].isnull().sum()/len(data) > 0.05]

# Drop columns in training and validation data
data.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in data.columns if
                    data[cname].nunique() < 10 and 
                    data[cname].dtype == "object"]

# Select numerical columns
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_cols = [cname for cname in data.columns if 
                data[cname].dtype in numeric_dtypes]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
data = data[my_cols]   

# Preprocessing
fill_by_freq = [x for x in ["Functional","Electrical","KitchenQual"] if x in data.columns]

data = simple_imputer(df=data, cols=fill_by_freq, strategy='most_frequent')
data = simple_imputer(df=data, cols=numerical_cols, strategy='constant', fill_value=0)
data = simple_imputer(df=data, cols=categorical_cols, strategy='constant', fill_value=None)


# Creating new features
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['1stFlrSF'] + data['2ndFlrSF'])

data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                              data['EnclosedPorch'] + data['ScreenPorch'] +
                              data['WoodDeckSF'])

# simplified features
data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# OneHotEncoder
data = pd.get_dummies(data, prefix_sep='_', drop_first=True)

# Remove features with low variance
data = variance_threshold_selector(df=data, threshold=0.005)

# Separate target from predictors
X_train = data.iloc[:len(y), :]
X_test = data.iloc[len(X_train):, :]


# BestK - univariate
from sklearn.feature_selection import f_regression, mutual_info_regression # for regression

# f_regression
# best15_f_regression = select_k_best(X_train, y, f_regression, 15)
# best25_f_regression = select_k_best(X_train, y, f_regression, 25)
best70_f_regression = select_k_best(X_train, y, f_regression, 70)

# best15_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 15)
# best25_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 25)
best70_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 70)


mean_from_random_forest = select_from_model(X_train, y, RandomForestRegressor(n_estimators=81, random_state=random_state, n_jobs=4), threshold='mean')

#median_from_random_forest = select_from_model(X_train, y, svr, threshold='mean')

mean_from_ridge = select_from_model(X_train, y, ridge, threshold='mean')

mean_from_lasso = select_from_model(X_train, y, lasso, threshold='mean')

mean_from_elasticnet = select_from_model(X_train, y, elasticnet, threshold='mean')

features_space = {
    "rf" : mean_from_random_forest,
    "lasso" : mean_from_lasso,
    "elasticnet": mean_from_elasticnet,
    "ridge": mean_from_ridge,
    "f_regr": best70_f_regression,
    "mutual_info": best70_mutual_info_regression,
    "all": set(X_train.columns)
}

from sklearn.model_selection import cross_val_score



scores = []
models = {
    'Ridge': ridge,
    'Lasso': lasso,
    'SVR': svr,
    'Elastic Net': elasticnet,
    'XGB': xgbr
}

for model_name in models:
    model_scores = []
    model = models[model_name]
    for name in features_space:
        score = cross_val_score(model, np.array(X_train[features_space[name]]), np.array(y), cv=10, scoring='neg_mean_squared_error')
        scores.append([model_name, name, len(features_space[name]), np.mean(score), np.median(score), np.std(score)])
        print(name, len(features_space[name]), np.mean(score), np.median(score))
        model_scores += score
    print("\nModel ", model_name, ": ", np.mean(model_scores), np.median(model_scores), "\n")
print("\n\n\n")

result = pd.DataFrame(scores, columns=["Model name", "Method","N","Mean","Median", "Std"]).sort_values(by=['Median'])

result.to_csv("feature_selection_evaluation.csv")