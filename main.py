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
cols_must_drop = ["Id", "BsmtFinSF2", "PoolQC", "MoSold"]
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

# Change data type
data['MSSubClass'] = data['MSSubClass'].apply(str)
data['YrSold'] = data['YrSold'].astype(str)
#data['MoSold'] = data['MoSold'].astype(str)

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
data = variance_threshold_selector(df=data, threshold=0.001)

# Separate target from predictors
X_train = data.iloc[:len(y), :]
X_test = data.iloc[len(X_train):, :]


# BestK - univariate
from sklearn.feature_selection import f_regression, mutual_info_regression # for regression

# f_regression
# best15_f_regression = select_k_best(X_train, y, f_regression, 15)
# best25_f_regression = select_k_best(X_train, y, f_regression, 25)
best100_f_regression = select_k_best(X_train, y, f_regression, 100)

# best15_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 15)
# best25_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 25)
best100_mutual_info_regression = select_k_best(X_train, y, mutual_info_regression, 100)


#median_from_random_forest = select_from_model(X_train, y, RandomForestRegressor(n_estimators=81, random_state=random_state, n_jobs=4), threshold='median')

#median_from_random_forest = select_from_model(X_train, y, svr, threshold='mean')

#median_from_ridge = select_from_model(X_train, y, ridge, threshold='median')

#median_from_lasso = select_from_model(X_train, y, lasso, threshold='median')

#median_from_elasticnet = select_from_model(X_train, y, elasticnet, threshold='median')

features_space = {
    #"rf" : median_from_random_forest,
    #"lasso" : median_features_spacefrom_lasso,
    #"elasticnet": median_from_elasticnet,
    #"ridge": median_from_ridge,
    "f_regr": best100_f_regression,
    "mutual_info": best100_mutual_info_regression,
    "all": set(X_train.columns)
}

from sklearn.model_selection import cross_val_score



models = {
    'Ridge': ridge,
    'Lasso': lasso,
    'SVR': svr,
    'Elastic Net': elasticnet,
    'XGB': xgbr,
    'Stack1': stack1,
    'Stack2': stack2,
    'Stack3': stack3,
    'Stack4': stack4
}

scores = []
model_performance = []

for model_name in models:
    model_scores = []
    model = models[model_name]

    min_error = 999999999999
    selected_features = None
    print("Computing: ", model_name)
    for name in features_space:
        print("    using ",name)
        score = cross_val_score(model, np.array(X_train[features_space[name]]), np.array(y), cv=10, scoring='neg_mean_squared_error')
        scores.append([model_name, name, len(features_space[name]), np.mean(score), np.median(score), np.std(score)])
        print(name, len(features_space[name]), np.mean(score), np.median(score))
        model_scores += score.tolist()

        # select the best features to work with the current Model
        if np.mean(score) < min_error:
            selected_features = features_space[name]    

    model_performance.append(
        [model_name, np.mean(model_scores)**2, selected_features]
    )
print("\n\n\n")

result = pd.DataFrame(scores, columns=["Model name", "Method","N","Mean","Median", "Std"]).sort_values(by=['Median'])
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
    weights.append(w)
    names.append(result[0])
weights = np.array(weights)
weights = weights/np.sum(weights)

print("\n\nWeights")
for i in range(len(names)):
    print(names[i],": ",weights[i])
print("\n\n")

prediction = np.zeros(len(X_test))
for i in range(len(weights)):
    result = model_performance[i]
    model = models[result[0]]
    features = result[2]
    weight = weights[i]

    model.fit(np.array(X_train[features]), np.array(y))
    pred = np.floor(np.expm1(model.predict(np.array(X_test[features]))))
    prediction += pred*weight

submission = pd.read_csv("data/sample_submission.csv")
submission.iloc[:, 1] = prediction
submission.to_csv("submission.csv", index=False)
