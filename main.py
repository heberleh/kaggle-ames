
#%%
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

from pandas_sklearn import *

random_state = 2019
np.random.RandomState(random_state)

from regression import *

target_class = "SalePrice"
cols_must_drop = ["Id"]
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
        if (abs(cor2) - abs(cor1)) > 0.15:
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

#%%
# f_regression
# best15_f_regression = select_k_best(X_train, y, f_regression, 15)
# best25_f_regression = select_k_best(X_train, y, f_regression, 25)
best100_f_regression = select_k_best(X_train, y, f_regression, 100)

best25_f_regression = select_k_best(X_train, y, f_regression, 35)
best80_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 80)
best120_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 120)
best145_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 145)

median_from_random_forest = select_from_model(X_train, y, rf, threshold='median')
median_from_ridge = select_from_model(X_train, y, ridge, threshold='median')
median_from_lasso = select_from_model(X_train, y, lasso, threshold='median')
median_from_elasticnet = select_from_model(X_train, y, elasticnet, threshold='median')

#%%

# Results from code above
features_space = {
    #'rf': ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_Po', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_10', 'MoSold_11', 'MoSold_12', 'MoSold_2', 'MoSold_3', 'MoSold_4', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'MoSold_9', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleType_missing_value', 'Street_Pave', 'Utilities_NoSeWa', 'Utilities_missing_value', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010'], 
    'lasso': ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'YearBuilt', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Electrical_SBrkr', 'ExterCond_TA', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_2', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'MoSold_8', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofStyle_Gable', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'],
    'elasticnet': ['BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'TotRmsAbvGrd', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Electrical_SBrkr', 'ExterCond_TA', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_2', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_ConLD', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'], 
    'ridge': ['BsmtHalfBath', 'Fireplaces', 'GarageCars', 'HalfBath', 'KitchenAbvGr', 'OverallCond', 'OverallQual', 'Total_Bathrooms', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_FuseP', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotShape_IR2', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MasVnrType_None', 'MasVnrType_Stone', 'MoSold_12', 'MoSold_5', 'MoSold_6', 'MoSold_7', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Membran', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Mansard', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2009'], 
    'f_regr': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Electrical_FuseF', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Functional_Maj2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD'], 
    'mutual_info80': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFullBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_Duplex', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_GLQ', 'BsmtFinType1_Rec', 'BsmtFinType1_missing_value', 'BsmtFinType2_missing_value', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Electrical_FuseF', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_2Story', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_Lvl', 'LotConfig_CulDSac', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD', 'YrSold_2007', 'YrSold_2008'], 
    'mutual_info120': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_BLQ', 'BsmtFinType2_Unf', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Electrical_FuseF', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Functional_Maj2', 'Functional_Min2', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Lvl', 'LandSlope_Mod', 'LotConfig_CulDSac', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_2', 'MoSold_3', 'MoSold_7', 'MoSold_8', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2009'], 
    'mutual_info145': ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_TA', 'BsmtCond_missing_value', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No', 'BsmtExposure_missing_value', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType1_missing_value', 'BsmtFinType2_GLQ', 'BsmtFinType2_missing_value', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtQual_missing_value', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_PosN', 'Condition2_RRAn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_Po', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Functional_Maj2', 'Functional_Min2', 'Functional_Sev', 'Functional_Typ', 'Heating_GasA', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LotConfig_CulDSac', 'LotConfig_FR3', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MasVnrType_missing_value', 'MoSold_2', 'MoSold_3', 'MoSold_7', 'MoSold_8', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'RoofStyle_Shed', 'SaleCondition_AdjLand', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_New', 'SaleType_WD', 'SaleType_missing_value', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009'], 
    'rfe': ['1stFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'EnclosedPorch', 'Fireplaces', 'GarageArea', 'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'ScreenPorch', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf', 'hasgarage', 'hasfireplace', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_No', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Electrical_SBrkr', 'ExterCond_Gd', 'ExterCond_TA', 'ExterQual_Gd', 'ExterQual_TA', 'Foundation_PConc', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'Heating_GasW', 'Heating_Grav', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MasVnrType_None', 'MasVnrType_Stone', 'PavedDrive_Y', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Hip', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'YrSold_2007', 'YrSold_2008', 'YrSold_2009', 'YrSold_2010', '2ndFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'FullBath', 'TotalBsmtSF', 'ExterCond_Fa'], 
    'all': ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']
}

features_space.keys()
#%% define what is going to be used
features_space = {
    #"rf" : median_from_random_forest.tolist(),
    "lasso" : median_from_lasso.tolist(),
    "elasticnet": median_from_elasticnet.tolist(),
    "ridge": median_from_ridge.tolist(),
    "f_regr": best100_f_regression.tolist(),
    "mutual_info80": best80_mutual_info_regression.tolist(),    
    "mutual_info120": best120_mutual_info_regression.tolist(),
    "mutual_info145": best145_mutual_info_regression.tolist(),
    "rfe": features_rfe_lasson_u_lgbm,
    "all": list(set(X_train.columns))
}


#%%

models = {
    'Ridge': ridge,
    'Lasso': lasso,
    'SVR': svr,
    'Elastic Net': elasticnet,
    'XGB': xgbr,
    'LightGBM': lightgbm,
    'Stack1': stack1    
}

#%%
scores = []
model_performance = []

from sklearn.model_selection import cross_val_score
# ! don't forget to set this to False if you want to run CV to select the best feature set
skip = False
k_cross = 3
for model_name in models:
    model_scores = []
    model = models[model_name]

    min_error = 999999999999
    selected_features = None
    selected_features_score = None
    print("Computing: ", model_name)

    if not skip:
        for name in features_space:
            print("    using ",name, " n=",len(features_space[name]))

            scoring = 'neg_mean_absolute_error'
            score = cross_val_score(model, np.array(X_train[features_space[name]]), np.array(y),                            cv=k_cross,
                                    scoring=scoring)
            if scoring == 'neg_mean_absolute_error':
                score = score * -1

            scores.append([model_name, name, len(features_space[name]), np.mean(score), np.median(score), np.std(score)])

            model_scores += score.tolist()

            # select the best features to work with the current Model
            if np.mean(score) < min_error:
                selected_features = features_space[name]
                selected_features_score = np.mean(score)    
    else:
        selected_features_score = 1
        selected_features = features_space[list(features_space.keys())[0]]

    model_performance.append(
        [model_name, selected_features_score, selected_features]
    )
print("\n\n\n")

#%%
result = pd.DataFrame(scores, 
    columns=["Model name", "Method","N","Mean","Median", "Std"]).sort_values(by=['Model name'])

result.to_csv("feature_selection_evaluation.csv")

print(model_performance)
#%%

model_performance = [
    ['Ridge', 0.0847263973353327, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']], ['Lasso', 0.08425160697856493, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']], ['SVR', 0.11305233446416384, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']], ['Elastic Net', 0.08432047247822715, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']], ['XGB', 0.08175491473120101, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']], ['LightGBM', 0.12760970331453278, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']], ['Stack1', 0.08234429648944037, ['BsmtCond_Po', 'KitchenAbvGr', 'Condition2_RRAn', 'BldgType_TwnhsE', 'SaleType_missing_value', 'RoofStyle_Shed', 'SaleCondition_Normal', 'SaleType_ConLI', 'hasfireplace', 'MoSold_3', 'LandContour_Lvl', 'Electrical_SBrkr', 'SaleType_Oth', 'BldgType_Twnhs', 'Condition2_Norm', 'BsmtFinType2_Unf', 'HeatingQC_Fa', 'MSZoning_FV', 'BsmtFinType1_GLQ', 'HouseStyle_2.5Unf', 'Condition2_PosN', 'Functional_Min1', 'YrSold_2007', 'BsmtQual_Gd', 'GarageArea', 'Condition1_RRNe', 'Foundation_Stone', 'BsmtQual_missing_value', 'HouseStyle_2Story', 'BsmtFinType1_Unf', 'HouseStyle_SLvl', 'KitchenQual_TA', 'LandSlope_Mod', 'MoSold_12', 'BsmtFinType2_BLQ', 'Total_Bathrooms', 'KitchenQual_Fa', 'KitchenQual_Gd', 'SaleCondition_AdjLand', 'BsmtFinType2_missing_value', 'has2ndfloor', 'HouseStyle_1.5Unf', 'Condition2_RRAe', 'SaleType_New', '3SsnPorch', 'YrSold_2009', 'LotConfig_FR3', 'Heating_OthW', 'haspool', 'SaleType_CWD', '1stFlrSF', 'LotConfig_CulDSac', 'Functional_Maj2', 'LotFrontage', 'Condition1_RRAn', 'LowQualFinSF', 'OverallCond', 'TotalBsmtSF', 'Condition1_PosN', 'ExterCond_TA', 'HeatingQC_TA', 'hasbsmt', 'OpenPorchSF', 'BldgType_Duplex', 'BsmtFinSF2', 'LotShape_IR2', 'Utilities_NoSeWa', 'Foundation_Wood', 'WoodDeckSF', 'BsmtFinType1_Rec', 'ExterCond_Fa', 'HouseStyle_2.5Fin', '2ndFlrSF', 'BsmtFinType1_LwQ', 'RoofStyle_Hip', 'SaleCondition_Family', 'RoofMatl_Membran', 'BsmtHalfBath', 'MSZoning_RL', 'MSZoning_RH', 'FullBath', 'RoofMatl_Roll', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'Foundation_CBlock', 'LandSlope_Sev', 'RoofMatl_WdShake', 'RoofMatl_Tar&Grv', 'MasVnrType_None', 'CentralAir_Y', 'MoSold_5', 'BsmtExposure_No', 'BsmtFinType2_Rec', 'Street_Pave', 'BsmtExposure_Mn', 'Heating_Wall', 'BsmtFinType2_GLQ', 'SaleType_ConLD', 'TotalSF', 'LandContour_Low', 'HeatingQC_Po', 'BsmtUnfSF', 'Total_sqr_footage', 'MoSold_2', 'MoSold_6', 'Condition1_RRAe', 'RoofStyle_Mansard', 'SaleType_Con', 'Functional_Mod', 'BsmtExposure_missing_value', 'LotConfig_Inside', 'SaleType_ConLw', 'SaleType_WD', 'ExterQual_Gd', 'YrSold_2008', 'ExterCond_Gd', 'SaleCondition_Alloca', 'MasVnrArea', 'HeatingQC_Gd', 'MasVnrType_missing_value', 'Functional_Min2', 'PavedDrive_P', 'MSZoning_RM', 'Fireplaces', 'ExterCond_Po', 'LandContour_HLS', 'PavedDrive_Y', 'Condition1_PosA', 'Electrical_FuseF', 'MoSold_10', 'YearBuilt', 'MoSold_9', 'Condition1_RRNn', 'MoSold_4', 'hasgarage', 'BsmtQual_TA', 'BsmtCond_Gd', 'Foundation_Slab', 'Utilities_missing_value', 'Heating_GasA', 'Condition2_Feedr', 'BsmtFinType1_BLQ', 'Heating_Grav', 'BsmtFinType1_missing_value', 'MasVnrType_BrkFace', 'Functional_Sev', 'Functional_Typ', 'MasVnrType_Stone', 'HouseStyle_SFoyer', 'RoofMatl_Metal', 'LotShape_IR3', 'YrSold_2010', 'LotArea', 'Condition1_Norm', 'MoSold_7', 'BsmtQual_Fa', 'Foundation_PConc', 'HouseStyle_1Story', 'BsmtExposure_Gd', 'MoSold_8', 'ExterQual_TA', 'HalfBath', 'EnclosedPorch', 'Total_porch_sf', 'Condition1_Feedr', 'OverallQual', 'LotConfig_FR2', 'Condition2_PosA', 'MoSold_11', 'SaleCondition_Partial', 'BsmtCond_missing_value', 'BsmtFullBath', 'YearRemodAdd', 'BldgType_2fmCon', 'LotShape_Reg', 'MiscVal', 'RoofStyle_Gambrel', 'ScreenPorch', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtFinSF1', 'Electrical_FuseP', 'GarageCars', 'BedroomAbvGr', 'BsmtFinType2_LwQ', 'ExterQual_Fa', 'Heating_GasW', 'BsmtCond_TA', 'Electrical_Mix', 'Condition2_RRNn']]
]

#%%
def get_weights(results, transforming_weights):
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

        # give more/less weight based on dict transforming_weights
        w = w * transforming_weights[name] 

        weights.append(w)        
        names.append(name)

    weights = np.array(weights)
    weights = weights/np.sum(weights)
    
    print("\n\n-- Weights --")
    for i in range(len(weights)):
        print(names[i], ": ", weights[i])
    print("\n\n")
    
    return weights


# Use only the CV error (1*cv_weights)
cv_weights = {}
for name in models:
    cv_weights[name] = 1

# Modify the CV errors to give more weight to these
stack1_pref_weights = {}
for name in models:
    stack1_pref_weights[name] = 1
stack1_pref_weights['Stack1'] = 3
stack1_pref_weights['Ridge'] = 0.6
stack1_pref_weights['LightGBM'] = 1.2
stack1_pref_weights['XGB'] = 1.2


transform_weights = []
print("----------- Pure CV [current in use] -----------")
weights = get_weights(model_performance, transform_weights=cv_weights)

print("----------- Transformed CV -----------")
get_weights(model_performance, transform_weights=stack1_pref_weights)


#%%

prediction = np.zeros(len(X_test))
models_errors = []
predic_train_voting = np.zeros(len(X_train))
for i in range(len(weights)):
    result = model_performance[i]
    model = models[result[0]]
    features = result[2]
    weight = weights[i]

    # predicting Test
    model.fit(np.array(X_train[features]), np.array(y))
    pred = np.floor(np.expm1(model.predict(np.array(X_test[features]))))
    prediction += pred*weight

    # predicting Train
    predic_train = np.floor(np.expm1(model.predict(np.array(X_train[features]))))
    y_true = np.floor(np.expm1(y))
    predic_train_voting += predic_train*weight

    # Model[i] Error
    mae = mean_absolute_error(predic_train, y_true)
    models_errors.append([result[0], len(features), mae])

# Voting Error
mae = mean_absolute_error(predic_train_voting, y_true)
models_errors.append(["Voting", len(features), mae])

result = pd.DataFrame(models_errors, columns=["Model name", "N features", "Error"]).sort_values(by=['Error'])
result.to_csv("train_pred_scores.csv")


submission = pd.read_csv("data/sample_submission.csv")
submission.iloc[:, 1] = prediction
submission.to_csv("submission.csv", index=False)
