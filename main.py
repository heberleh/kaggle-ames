
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


#%% define what is going to be used
features_space = {
    "rf" : median_from_random_forest,
    "lasso" : median_from_lasso,
    "elasticnet": median_from_elasticnet,
    "ridge": median_from_ridge,
    "f_regr": best100_f_regression,
    "mutual_info80": best80_mutual_info_regression,    
    "mutual_info120": best120_mutual_info_regression,
    "mutual_info145": best145_mutual_info_regression,
    "rfe": features_rfe_lasson_u_lgbm,
    "all": set(X_train.columns)
}
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
k_cross = 4
for model_name in models:
    model_scores = []
    model = models[model_name]

    min_error = 999999999999
    selected_features = None
    print("Computing: ", model_name)

    if not skip:
        for name in features_space:
            print("    using ",name, " n=",len(features_space[name]))
            score = cross_val_score(model, np.array(X_train[features_space[name]]), np.array(y),                            cv=k_cross, scoring='neg_mean_absolute_error', n_jobs=4)

            scores.append([model_name, name, len(features_space[name]), np.mean(score), np.median(score), np.std(score)])
            #print("    ",name, len(features_space[name]), np.mean(score), np.median(score))
            model_scores += score.tolist()

            # select the best features to work with the current Model
            if np.mean(score) < min_error:
                selected_features = features_space[name]    
    else:
        model_scores = [1,1]
        selected_features = features_space[list(features_space.keys())[0]]

    model_performance.append(
        [model_name, np.mean(model_scores)**2, selected_features]
    )
print("\n\n\n")

result = pd.DataFrame(scores, columns=["Model name", "Method","N","Mean","Median", "Std"]).sort_values(by=['Model name'])
result.to_csv("feature_selection_evaluation.csv")

total_error = 0.0
for result in model_performance:
    total_error += result[1]
estimators = []
weights = []
names = []
for result in model_performance:
    estimators.append(models[result[0]])
    w = 1-(result[1]/total_error)
    if result[0] in ['Ridge']:
        w = w*0.6
    if result[0] in ['Stack1']:
        w = w*2.3
    if result[0] in ['LightGBM']:
        w = w*1.2
    if result[0] in ['XGB']:
        w = w*1.2
    weights.append(w)
    names.append(result[0])
weights = np.array(weights)
weights = weights/np.sum(weights)

print("\n\nWeights")
for i in range(len(names)):
    print(names[i],": ",weights[i])
print("\n\n")

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
