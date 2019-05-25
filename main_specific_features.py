
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV, train_test_split, cross_validate
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

from pandas_sklearn import *

random_state = 2019
np.random.RandomState(random_state)

from regression import *

target_class = "SalePrice"
cols_must_drop = ["Id", "MoSold"]
# BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2  "YrSold", 
log_transform = ["SalePrice"]

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

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

features_to_log = []
for feature in home_data.columns:
    if feature != target_class and home_data[feature].dtype in numeric_dtypes:
        cor1 = np.corrcoef(home_data[feature],home_data[target_class])[0][1]
        cor2 = np.corrcoef(np.log1p(home_data[feature]),home_data[target_class])[0][1]
        if (abs(cor2) - abs(cor1)) > 0.05:
            features_to_log.append(feature)
            print("Applied Log to: ", feature, " due to corr improv. of: ", abs(cor2) - abs(cor1))

y = home_data[target_class]
home_data.drop([target_class], axis=1, inplace=True)

data = pd.concat([home_data, home_test], sort=True).reset_index(drop=True)

#Thanks Alex Lekov for the ideas shared at 
# https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17#L189
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
                    data[cname].nunique() < 15 and 
                    data[cname].dtype == "object"]

# Select numerical columns
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

data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data["BsmtFinSF2"] + data['1stFlrSF'] + data['2ndFlrSF'])

data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF'])

# simplified features
data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
data.drop(['PoolArea'], axis=1, inplace=True)

data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

for feature in features_to_log:
    data[feature] = np.log1p(data[feature])
    data[feature] = np.log1p(data[feature])

# OneHotEncoder
data = pd.get_dummies(data, prefix_sep='_', drop_first=True)

# Remove features with low variance
data = variance_threshold_selector(df=data, threshold=0.00)

# Separate target from predictors
X_train = data.iloc[:len(y), :]
X_test = data.iloc[len(X_train):, :]


#%% 
param_grid = {
    'objective':['regression'],
    'num_leaves': [4, 5, 9],
    'max_bin': [200],
    'n_estimators': [1000, 2000, 3000],
    'feature_fraction': [0.2, 0.7, 0.9],
    'bagging_fraction': [0.2, 0.75, 0.9], 
    'bagging_seed': [7],
    'reg_alpha': [0.0, 0.0001, 0.1],
    'bagging_freq': [3, 5],
    'num_boost_round': [20, 50],
    'learning_rate': [0.01, 0.02, 0.03],
    'max_depth': [3, 5, 7],
    'eval_metric': ['l1'],
    'verbose': [-1]
}

from lightgbm import LGBMRegressor
lgb_estimator = LGBMRegressor()
kfolds = KFold(n_splits=3, shuffle=True, random_state=random_state)

gsearch = GridSearchCV(estimator=lgb_estimator, 
                        param_grid=param_grid, 
                        cv=kfolds,
                        scoring='neg_mean_absolute_error') 

lgb_model = gsearch.fit(X=X_train, 
                        y=y)

print(lgb_model.best_params_, lgb_model.best_score_)




#%%

from xgboost import XGBRegressor
from sklearn.model_selection import KFold

param_grid = {'learning_rate':[0.01, 0.03, 0.05], 
                'n_estimators':[1000, 4000],
                    'max_depth': [3, 4, 6],
                    'min_child_weight': [0],
                    'gamma': [0, 0.1] , 
                    'subsample': [0.3, 0.75],
                    'colsample_bytree': [0.3, 0.75],
                    'objective':['reg:linear'], 
                    'nthread':[2],
                    'scale_pos_weight':[1],
                    'seed':[27],
                    'reg_alpha': [0.0001, 0.01] 
                }

estimator = XGBRegressor()

kfolds = KFold(n_splits=3, shuffle=True, random_state=random_state)

gsearch = GridSearchCV(estimator=estimator, 
                    param_grid=param_grid, 
                    cv=kfolds, 
                    scoring='neg_mean_absolute_error',
                    n_jobs=3)

lgb_model = gsearch.fit(X=X_train, 
                        y=y)

print(lgb_model.best_params_, lgb_model.best_score_)



#%% 
# BestK - univariate
from sklearn.feature_selection import f_regression, mutual_info_regression # for regression

features_rfe_lasson_u_lgbm = ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa']
#--------------------------------

features_space = {
# f_regression
    'best15_f_regression': select_k_best(X_train, y, f_regression, 15).tolist(),
    'best35_f_regression': select_k_best(X_train, y, f_regression, 35).tolist(),
    'best25_f_regression': select_k_best(X_train, y, f_regression, 80).tolist(),
    'best120_f_regression': select_k_best(X_train, y, f_regression, 120).tolist(),
    '#best145_f_regression': select_k_best(X_train, y, f_regression, 145).tolist(),

    'best15_mutual_info_regression': select_k_best(X_train, y, mutual_info_regression, 15).tolist(),
    'best35_mutual_info_regression': select_k_best(X_train, y, mutual_info_regression, 35).tolist(),
    'best80_mutual_info_regression': select_k_best(X_train, y, mutual_info_regression, 80).tolist(),
    'best120_mutual_info_regression': select_k_best(X_train, y, mutual_info_regression, 120).tolist(),
    'best145_mutual_info_regression': select_k_best(X_train, y, mutual_info_regression, 145).tolist(),

    #'median_from_random_forest': select_from_model(X_train, y, rf, threshold='median').tolist(),
    #'median_from_ridge': select_from_model(X_train, y, ridge, threshold='median').tolist(),
    'median_from_lasso': select_from_model(X_train, y, lasso, threshold='median').tolist(),
    #'median_from_elasticnet': select_from_model(X_train, y, elasticnet, threshold='median').tolist(),

    #'mean_from_ridge': select_from_model(X_train, y, ridge, threshold='mean').tolist(),
    'mean_from_lasso': select_from_model(X_train, y, lasso, threshold='mean').tolist(),
    #'mean_from_elasticnet': select_from_model(X_train, y, elasticnet, threshold='mean').tolist(),

    #'default_ridge': select_from_model(X_train, y, ridge).tolist(),
    'default_lasso': select_from_model(X_train, y, lasso).tolist(),
    #'default_elasticnet': select_from_model(X_train, y, elasticnet).tolist(),

    "rfe": features_rfe_lasson_u_lgbm,
    "all": list(set(X_train.columns))
}

print(features_space) # copy/paste to save the result in the cell below 

#%%

# Results from code above
features_space = {
    'best15_f_regression': ['1stFlrSF', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'ExterQual_TA', 'KitchenQual_TA'], 'best35_f_regression': ['1stFlrSF', 'BsmtFinSF1', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'LotFrontage', 'MasVnrArea', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasfireplace', 'BsmtFinType1_GLQ', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'HeatingQC_TA', 'KitchenQual_Gd', 'KitchenQual_TA', 'MSZoning_RM', 'MasVnrType_None', 'SaleCondition_Partial', 'SaleType_New'], 'best25_f_regression': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_Rec', 'BsmtFinType1_missing_value', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Electrical_FuseF', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_2Story', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LotConfig_CulDSac', 'LotShape_Reg', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD'], 'best120_f_regression': ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition2_Feedr', 'Condition2_PosN', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD', 'Street_Pave'], '#best145_f_regression': ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_Po', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_11', 'MoSold_4', 'MoSold_5', 'MoSold_9', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007'], 'best15_mutual_info_regression': ['1stFlrSF', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'ExterQual_Gd', 'ExterQual_TA', 'KitchenQual_TA'], 'best35_mutual_info_regression': ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasfireplace', 'BsmtFinType1_GLQ', 'BsmtQual_Gd', 'BsmtQual_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'HeatingQC_TA', 'KitchenQual_Gd', 'KitchenQual_TA'], 'best80_mutual_info_regression': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_Duplex', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_GLQ', 'BsmtFinType1_Rec', 'BsmtFinType1_missing_value', 'BsmtFinType2_missing_value', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition2_Norm', 'Electrical_FuseF', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2Story', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LotConfig_CulDSac', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD', 'YrSold_2009'], 'best120_mutual_info_regression': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_GLQ', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Electrical_FuseF', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Functional_Min2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_2', 'MoSold_7', 'MoSold_8', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009'], 'best145_mutual_info_regression': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Functional_Maj2', 'Functional_Min2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_2', 'MoSold_3', 'MoSold_4', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'SaleType_missing_value', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009'], 'median_from_ridge': ['BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_FuseP', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotShape_IR2', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Mansard', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2009'], 'median_from_lasso': ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'YearBuilt', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Electrical_SBrkr', 'ExterCond_TA', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_2', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofStyle_Gable', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'], 'median_from_elasticnet': ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Electrical_SBrkr', 'ExterCond_TA', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_2', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'], 'mean_from_ridge': ['Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtFinType2_BLQ', 'BsmtFinType2_Unf', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'ExterQual_Fa', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Sev', 'LotConfig_CulDSac', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MasVnrType_Stone', 'MoSold_5', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2009'], 'mean_from_lasso': ['Fireplaces', 'GarageCars', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'hasgarage', 'BldgType_Twnhs', 'BsmtExposure_Gd', 'BsmtFinType2_BLQ', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'ExterCond_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'HeatingQC_TA', 'HouseStyle_2.5Fin', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Sev', 'LotConfig_CulDSac', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_Stone', 'PavedDrive_Y', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleType_ConLD', 'SaleType_New', 'Street_Pave', 'YrSold_2009'], 'mean_from_elasticnet': ['GarageCars', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'hasgarage', 'BldgType_Twnhs', 'BsmtExposure_Gd', 'BsmtFinType2_BLQ', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_TA', 'HouseStyle_2.5Fin', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Sev', 'LotConfig_CulDSac', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_Stone', 'PavedDrive_Y', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleType_ConLD', 'SaleType_New', 'Street_Pave'], 'default_ridge': ['BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_FuseP', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotShape_IR2', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Mansard', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2009'], 'default_lasso': ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'YearBuilt', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Electrical_SBrkr', 'ExterCond_TA', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_2', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofStyle_Gable', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'], 'default_elasticnet': ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Electrical_SBrkr', 'ExterCond_TA', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_2', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'], 'rfe': ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa'], 'all': ['TotalSF', 'Foundation_CBlock', 'RoofMatl_WdShngl', 'RoofStyle_Shed', 'BsmtCond_Gd', 'haspool', 'LandContour_HLS', 'OverallCond', 'OpenPorchSF', 'HouseStyle_SFoyer', 'BsmtHalfBath', 'BsmtQual_Gd', 'Condition2_Feedr', 'RoofStyle_Gambrel', 'TotRmsAbvGrd', 'HouseStyle_2.5Unf', 'Street_Pave', 'LotShape_IR3', 'HeatingQC_Fa', 'BldgType_2fmCon', 'Condition1_RRNn', 'BsmtUnfSF', 'RoofMatl_Membran', 'BedroomAbvGr', 'RoofStyle_Mansard', 'Total_sqr_footage', 'BsmtFinType2_LwQ', 'Total_porch_sf', 'Condition1_PosN', 'Heating_Grav', 'RoofStyle_Hip', '1stFlrSF', 'BsmtFinType2_BLQ', 'SaleType_ConLw', 'MoSold_7', 'ScreenPorch', 'BsmtCond_missing_value', 'SaleType_WD', 'ExterCond_Fa', 'YearBuilt', 'BldgType_TwnhsE', 'Condition1_PosA', 'MasVnrType_BrkFace', 'SaleType_missing_value', 'ExterQual_Fa', 'LotArea', 'YrSold_2010', 'MoSold_12', 'BsmtExposure_Gd', 'BldgType_Duplex', 'HeatingQC_Gd', 'BsmtExposure_Mn', 'Condition2_Norm', 'Foundation_Stone', 'hasbsmt', 'LandSlope_Mod', 'SaleType_ConLI', 'KitchenAbvGr', 'Heating_GasW', 'YrSold_2009', 'ExterCond_TA', 'hasfireplace', 'MoSold_5', 'SaleType_CWD', 'BsmtFinType2_missing_value', 'HouseStyle_1.5Unf', 'SaleType_Con', 'BsmtFinType1_Unf', 'HeatingQC_Po', 'MoSold_11', 'MasVnrType_None', 'LotConfig_CulDSac', 'KitchenQual_Gd', 'MoSold_4', 'SaleCondition_Normal', 'Condition2_RRNn', 'YrSold_2007', 'YearRemodAdd', 'BsmtQual_Fa', 'CentralAir_Y', 'HouseStyle_SLvl', 'Foundation_Wood', 'SaleCondition_AdjLand', 'YrSold_2008', 'RoofMatl_Metal', 'BsmtFinType2_Rec', '3SsnPorch', 'BsmtCond_Po', 'MoSold_10', 'MSZoning_FV', 'Utilities_missing_value', 'MSZoning_RH', 'SaleCondition_Family', 'Condition1_Norm', 'LotConfig_FR3', 'MasVnrArea', 'KitchenQual_Fa', 'BsmtFinType1_missing_value', 'Condition1_RRNe', 'Foundation_Slab', 'TotalBsmtSF', 'RoofMatl_Tar&Grv', '2ndFlrSF', 'BsmtQual_TA', 'Heating_OthW', 'BsmtQual_missing_value', 'Condition1_RRAn', 'BsmtFinType1_BLQ', 'WoodDeckSF', 'BsmtFinType1_GLQ', 'ExterCond_Po', 'LotConfig_FR2', 'ExterCond_Gd', 'LowQualFinSF', 'HouseStyle_2.5Fin', 'HeatingQC_TA', 'GarageCars', 'ExterQual_TA', 'MiscVal', 'RoofMatl_Roll', 'Total_Bathrooms', 'KitchenQual_TA', 'FullBath', 'SaleType_ConLD', 'BsmtExposure_missing_value', 'BldgType_Twnhs', 'BsmtExposure_No', 'Functional_Min1', 'MasVnrType_missing_value', 'LotFrontage', 'Condition2_RRAe', 'Heating_Wall', 'HouseStyle_1Story', 'MasVnrType_Stone', 'BsmtFinType1_Rec', 'BsmtFinType2_GLQ', 'HalfBath', 'BsmtFinType1_LwQ', 'LandContour_Lvl', 'LandSlope_Sev', 'BsmtFinType2_Unf', 'ExterQual_Gd', 'Condition2_RRAn', 'Functional_Maj2', 'LotShape_IR2', 'Foundation_PConc', 'Electrical_Mix', 'BsmtCond_TA', 'LandContour_Low', 'PavedDrive_Y', 'Electrical_FuseF', 'EnclosedPorch', 'MoSold_3', 'GarageArea', 'BsmtFinSF2', 'Functional_Mod', 'MSZoning_RM', 'SaleType_New', 'SaleType_Oth', 'PavedDrive_P', 'Functional_Typ', 'Functional_Sev', 'Condition2_PosA', 'LotConfig_Inside', 'Electrical_SBrkr', 'MoSold_9', 'MSZoning_RL', 'MoSold_2', 'SaleCondition_Alloca', 'BsmtFullBath', 'LotShape_Reg', 'Heating_GasA', 'Electrical_FuseP', 'SaleCondition_Partial', 'MoSold_6', 'Functional_Min2', 'Condition2_PosN', 'MoSold_8', 'HouseStyle_2Story', 'RoofMatl_WdShake', 'Utilities_NoSeWa', 'hasgarage', 'has2ndfloor', 'BsmtFinSF1', 'Condition1_RRAe', 'Fireplaces', 'RoofStyle_Gable', 'OverallQual', 'Condition1_Feedr', 'GrLivArea']}



#%%

models = {
    'Ridge': ridge,
    'Lasso': lasso,
    'SVR': svr,
    'Elastic Net': elasticnet,
    'XGB': xgbr,
    'LightGBM': lightgbm,
    'Stack1': stack1,
    'Stack2': stack2
}

#%%
scores = []
model_performance = []

from sklearn.model_selection import cross_val_score
# ! don't forget to set this to False if you want to run CV to select the best feature set
skip = False
k_cross = 5
for model_name in models:
    
    model = models[model_name]

    min_error = float('inf')
    selected_features = None
    selected_features_score = None
    print("Computing: ", model_name)

    if not skip:
        for name in features_space:
            print("    using ",name, " n=",len(features_space[name]))

            scoring = 'neg_mean_absolute_error'
            result = cross_validate(model, np.array(X_train[features_space[name]]), np.array(y),                            cv=k_cross,
                                    scoring=scoring,
                                    return_train_score=True)
            score = result['test_score']
            score_train = result['train_score']

            if scoring == 'neg_mean_absolute_error':
                score = score * -1
                score_train = score_train * -1

            scores.append([model_name, name, len(features_space[name]), np.mean(score), np.mean(score_train), np.mean(score_train)-np.mean(score)])

            # select the best features to work with the current Model
            if np.mean(score) < min_error:
                selected_features = features_space[name]
                selected_features_score = np.mean(score)
                selected_features_name = name
                min_error = np.mean(score)
                print("        score improved!")
                
    else:
        selected_features_score = 1
        selected_features = features_space[list(features_space.keys())[0]]

    model_performance.append(
        [model_name, selected_features_score, selected_features, selected_features_name]
    )
print("\n\n\n")



#%%
result = pd.DataFrame(scores, 
    columns=["Model name", "Method","N","Test","Train", "Train-Test"]).sort_values(by=['Test'])

result.to_csv("feature_selection_evaluation.csv")

print(model_performance)
#%%

model_performance =[
    ['Ridge', 0.08309356774133776, ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa'], 'rfe'], ['Lasso', 0.08274344244989851, ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa'], 'rfe'], ['SVR', 0.0854499966114312, ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasfireplace', 'BsmtFinType1_GLQ', 'BsmtQual_Gd', 'BsmtQual_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'HeatingQC_TA', 'KitchenQual_Gd', 'KitchenQual_TA'], 'best35_mutual_info_regression'], ['Elastic Net', 0.08271876066523719, ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa'], 'rfe'], ['XGB', 0.08143655304986412, ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa'], 'rfe'], ['LightGBM', 0.12618117519697358, ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Functional_Maj2', 'Functional_Min2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_2', 'MoSold_3', 'MoSold_4', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'SaleType_missing_value', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009'], 'best145_mutual_info_regression'], ['Stack1', 0.08159861660480185, ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Functional_Maj2', 'Functional_Min2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_2', 'MoSold_3', 'MoSold_4', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'SaleType_missing_value', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009'], 'best145_mutual_info_regression']]


#%%
def get_weights(results, transforming_weights, exp_w=1):
    total_error = 0.0
    for result in results:
        total_error += result[1]

    weights = []
    names = []
    for result in results:
        error = result[1]
        name = result[0]

        # the bigger w, the better /  the lower error, the better
        w = 1-(error/total_error) 

        w = w**exp_w

        # give more/less weight based on dict transforming_weights
        w = w * transforming_weights[name] 

        weights.append(w)        
        names.append(name)

    weights = np.round(weights, decimals=2)
    weights = np.array(weights)
    weights = weights/np.sum(weights)
    
    

    for i in range(len(weights)):
        print(names[i], ": ", weights[i])

    print("\n-- Weights Sum to 1.0? --")
    print(np.sum(weights))

    print("\n\n")
    
    return weights



#%%
# Use only the CV error (1*cv_weights)
cv_weights = {}
for name in models:
    cv_weights[name] = 1

# Modify the CV errors to give more weight to these
stack1_pref_weights = {}
for name in models:
    stack1_pref_weights[name] = 1
stack1_pref_weights['Stack1'] = 2
stack1_pref_weights['LightGBM'] = 0.4
stack1_pref_weights['Ridge'] = 0.6
stack1_pref_weights['XGB'] = 4

good = ["Lasso", "XGB", "Ridge", "Elastic Net", "Stack1", "XGB"]
XGB_stack_pref_weights = {}
for name in models:
    if name in good:
        XGB_stack_pref_weights[name] = 1
    else:
        XGB_stack_pref_weights[name] = 0   
XGB_stack_pref_weights['Stack1'] = 4
XGB_stack_pref_weights['XGB'] = 4

good = ["Lasso", "XGB", "Elastic Net", "Stack1", "XGB"]
XGB_stack_pref2_weights = {}
for name in models:
    if name in good:
        XGB_stack_pref2_weights[name] = 1
    else:
        XGB_stack_pref2_weights[name] = 0.5  
XGB_stack_pref2_weights['Stack1'] = 2
XGB_stack_pref2_weights['XGB'] = 9


stack_only_weights = {}
for name in models:
   stack_only_weights[name] = 0
stack_only_weights['Stack1'] = 1

XGB_only_weights = {}
for name in models:
   XGB_only_weights[name] = 0
XGB_only_weights['XGB'] = 1


stack2_only_weights = {}
for name in models:
   stack2_only_weights[name] = 0
stack2_only_weights['Stack2'] = 1

flexible = {}
for name in models:
    if name in good:
        flexible[name] = 1
    else:
        flexible[name] = 0.5  
# flexible['Stack1'] = 1.5
# flexible['Stack2'] = 1.5
flexible['XGB'] = 2.5

transform_weights = []
#weights = get_weights(model_performance, cv_weights, exp_w=1) # 13....
#weights = get_weights(model_performance, cv_weights, exp_w=5) # 13....
#weights = get_weights(model_performance, stack1_pref_weights) # 12987.19663
#weights = get_weights(model_performance, XGB_stack_pref_weights) # 13040.32423
#weights = get_weights(model_performance, XGB_stack_pref2_weights) # 13173.07132
#weights = get_weights(model_performance, stack_only_weights) # 
#weights = get_weights(model_performance, XGB_only_weights) # 
#weights = get_weights(model_performance, stack2_only_weights) # 
weights = get_weights(model_performance, flexible) # 

# if onde of these is better... check if its ok to do RFE... 


#%%

prediction = np.zeros(len(X_test))
models_errors = []
predic_train_voting = np.zeros(len(X_train))
for i in range(len(model_performance)):
    result = model_performance[i]

    model = models[result[0]]
    features = result[2]
    weight = weights[i]

    # predicting Test
    model.fit(np.array(X_train[features]), np.array(y))
    pred = np.expm1(model.predict(np.array(X_test[features])))
    prediction += pred*weight

    # predicting Train
    predic_train = np.expm1(model.predict(np.array(X_train[features])))
    y_true = np.expm1(y)
    predic_train_voting += predic_train*weight

    # Model[i] Error
    mae = mean_absolute_error(predic_train, y_true)
    models_errors.append([result[0], len(features), result[3], mae])

# Voting Error
mae = mean_absolute_error(predic_train_voting, y_true)
models_errors.append(["Voting", 0, "combinations", mae])

result = pd.DataFrame(models_errors, columns=["Model name", "N features","F Sel.", "Error"]).sort_values(by=['Error'])
result.to_csv("train_pred_scores.csv")


submission = pd.read_csv("data/sample_submission.csv")
submission.iloc[:, 1] = prediction
submission.to_csv("submission.csv", index=False)
