import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.svm import SVR
from xgboost import XGBRegressor

from pandas_sklearn import simple_imputer, variance_threshold_selector, select_k_best, select_from_model

target_class = "SalePrice"
cols_must_drop = ["Id"]
log_transform = ["SalePrice"]

random_state = 2019
np.random.RandomState(random_state)

home_data = pd.read_csv("data/train.csv")
home_val = pd.read_csv("data/test.csv")

# features set as 'must drop'
if len(cols_must_drop) > 0:
    home_data.drop(cols_must_drop, axis=1, inplace=True)
    home_val.drop(cols_must_drop, axis=1, inplace=True)

if len(log_transform) > 0:
    for feature in log_transform:
        home_data[feature] = np.log(home_data[feature])
        if feature in home_val.columns:
            home_val[feature] = np.log(home_val[feature])

X_val = home_val

# Remove rows with missing target
home_data.dropna(axis=0, subset=[target_class], inplace=True)

print(home_data)

# Separate target from predictors
y = home_data[target_class]
X = home_data.copy()
X.drop(['SalePrice'], axis=1, inplace=True)


rs = ShuffleSplit(n_splits=1, train_size=0.9, test_size=.10, random_state=random_state)
rep = 0
results = {}
for train_index, test_index in rs.split(X):

    print("--------- Rep ", rep, "---------")
    rep += 1

    dropped_features = set()
    selected_features = set()

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_val_rep = X_val.copy()

    # Get names of columns with too much missing values
    cols_with_missing = [col for col in X_train.columns
                        if X_train[col].isnull().sum()/len(X_train) > 0.20]   
    for col in cols_with_missing:
        dropped_features.add(col)

    # Drop columns in training and validation data
    print("Droping cols due to NAs: ", cols_with_missing)
    X_train.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)
    X_val_rep.drop(cols_with_missing, axis=1, inplace=True)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train.columns if
                        X_train[cname].nunique() < 10 and 
                        X_train[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train.columns if 
                    X_train[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()    
    X_test = X_test[my_cols].copy()
    X_val_rep = X_val_rep[my_cols].copy()

    # Preprocessing    
    X_train = simple_imputer(df=X_train, cols=numerical_cols, strategy='mean')
    X_test = simple_imputer(df=X_test, cols=numerical_cols, strategy='mean')
    X_val_rep = simple_imputer(df=X_val_rep, cols=numerical_cols, strategy='mean')

    X_train = simple_imputer(df=X_train, cols=categorical_cols, strategy='most_frequent')
    X_test = simple_imputer(df=X_test, cols=categorical_cols, strategy='most_frequent')
    X_val_rep = simple_imputer(df=X_val_rep, cols=categorical_cols, strategy='most_frequent')

    # OneHotEncoder
    X_train = pd.get_dummies(X_train, prefix_sep='_', drop_first=True)
    X_test = pd.get_dummies(X_test, prefix_sep='_', drop_first=True)
    X_val_rep = pd.get_dummies(X_val_rep, prefix_sep='_', drop_first=True)

    # Remove features with low variance
    cols_before_var = set(X_train.columns)
    var_threshold = 0.05
    
    X_train = variance_threshold_selector(df=X_train, threshold=var_threshold)  #fit and transform
    X_test = X_test[X_train.columns]   # transform
    X_val_rep = X_val_rep[X_train.columns]


    # ==== Feature selection ====

    # BestK - univariate
    from sklearn.feature_selection import f_regression, mutual_info_regression # for regression

    # f_regression
    best20_f_regression = select_k_best(X_train, y_train, f_regression, 20)

    best20_mutual_info_regression = select_k_best(X_train, y_train, mutual_info_regression, 20)

    # median_from_svr_linear = select_from_model(X_train, y_train, SVR(kernel="linear"), threshold='median')
    # median_from_svr_rbf = select_from_model(X_train, y_train, SVR(kernel="rbf"), threshold='median')
    # median_from_svr_poly = select_from_model(X_train, y_train, SVR(kernel="poly"), threshold='median')

    median_from_random_forest = select_from_model(X_train, y_train, RandomForestRegressor(n_estimators=128, random_state=random_state, n_jobs=8), threshold='median')

    selected_features_intersection = set(best20_f_regression).intersection(
        set(best20_mutual_info_regression)).intersection(
            set(median_from_random_forest))
    
    selected_features_union = set(best20_f_regression).union(
        set(best20_mutual_info_regression)).union(
            set(median_from_random_forest))

    # Various hyper-parameters to tune
    xgb1 = XGBRegressor()
    parameters = {'nthread':[2], #when use hyperthread, xgboost may become slower
                'objective':['reg:linear'],
                'learning_rate': [.03, 0.05, .07], #so called `eta` value
                'max_depth': [5, 6, 7],
                'min_child_weight': [4],
                'silent': [1],
                'booster': ['gbtree', 'gblinear', 'dart'],
                'subsample': [0.7],
                'colsample_bytree': [0.7],
                'n_estimators': [10]}#[512, 1024, 1536]}

    xgb_grid_sel_f_i = GridSearchCV(xgb1,
                            parameters,
                            cv = 5,
                            scoring = 'neg_mean_absolute_error',
                            n_jobs = 4,
                            verbose=False)

    xgb_grid_sel_f_u = GridSearchCV(xgb1,
                            parameters,
                            cv = 5,
                            scoring = 'neg_mean_absolute_error',
                            n_jobs = 4,
                            verbose=False)

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv = 5,
                            scoring = 'neg_mean_absolute_error',
                            n_jobs = 4,
                            verbose=False)

    # log y ... 

    # X_sel_var
    xgb_grid_sel_f_i.fit(X_train[selected_features_intersection], y_train)
    xgb_grid_sel_f_u.fit(X_train[selected_features_union], y_train)

    # X_all_var
    xgb_grid.fit(X_train, y_train)



    print("Selected features score u: ", xgb_grid_sel_f_u.best_score_)
    print(xgb_grid.best_params_)
    mae_u = mean_absolute_error(xgb_grid_sel_f_u.predict(X_test[selected_features_union]), y_test)
    print("MAE test: ", mae_u)
    print("Number of features: ", len(X_train[selected_features_union].columns))

    print("Selected features score i: ", xgb_grid_sel_f_i.best_score_)
    print(xgb_grid.best_params_)
    mae_i = mean_absolute_error(xgb_grid_sel_f_i.predict(X_test[selected_features_intersection]), y_test)
    print("MAE test: ", mae_u)
    print("Number of features: ", len(X_train[selected_features_intersection].columns))

    print("All features score: ", xgb_grid.best_score_)
    print(xgb_grid.best_params_)
    mae = mean_absolute_error(xgb_grid.predict(X_test), y_test)
    print("MAE test: ", mae_u)
    print("Number of features: ", len(X_train.columns))


