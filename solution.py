import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pipeline import PipelineFeatureImportances
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.svm import SVR

from pandas_sklearn import simple_imputer, variance_threshold_selector

target_class = "SalePrice"
cols_must_drop = ["Id"]
log_transform = ["SalePrice"]

random_state = 2019
np.random.seed = random_state

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
for train_index, test_index in rs.split(X):

    print("--------- Rep ", rep, "---------")
    rep += 1

    dropped_features = set()
    selected_features = set()

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Get names of columns with too much missing values
    cols_with_missing = [col for col in X_train.columns
                        if X_train[col].isnull().sum()/len(X_train) > 0.20]   
    for col in cols_with_missing:
        dropped_features.add(col)

    # Drop columns in training and validation data
    print("Droping cols due to NAs: ", cols_with_missing)
    X_train.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

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

    # Preprocessing    
    X_train = simple_imputer(df=X_train, cols=numerical_cols, strategy='mean')
    X_test = simple_imputer(df=X_test, cols=numerical_cols, strategy='mean')
    
    X_train = simple_imputer(df=X_train, cols=categorical_cols, strategy='most_frequent')
    X_test = simple_imputer(df=X_test, cols=categorical_cols, strategy='most_frequent')

    # OneHotEncoder
    X_train = pd.get_dummies(X_train, prefix_sep='_', drop_first=True)
    X_test = pd.get_dummies(X_test, prefix_sep='_', drop_first=True)
        
    # Remove features with low variance
    cols_before_var = set(X_train.columns)
    var_threshold = 0.05
    X_train = variance_threshold_selector(df=X_train, threshold=var_threshold)    
    X_test = variance_threshold_selector(df=X_test, threshold=var_threshold)    
    print("VarFilter removed:", len(cols_before_var - set(X_train.columns)))
    print("Remaining: ", len(X_train.columns))



