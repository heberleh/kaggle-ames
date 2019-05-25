
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
import numpy as np

# Make the feature selection and other methods from Sklearn 
# actually work with features (with their context/name)

def variance_threshold_selector(df, threshold=0.0):
    selector = VarianceThreshold(threshold)
    selector.fit(df)    
    return df[df.columns[selector.get_support(indices=True)]]


def simple_imputer(df, df_test=None, cols=[], strategy=None, fill_value=None):
    df2 = df.copy()
    if len(cols) == 0:        
        cols = df2.columns    
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    df2[cols] = imputer.fit_transform(df2[cols])

    if df_test:
        df_test2 = df_test.copy()
        df_test2[cols] = imputer.transform(df_test2[cols])
        return (df2, df_test2)
    return df2


from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import f_regression, mutual_info_regression # for regression
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif # for classification

def select_k_best(df_X, df_y, scoring_function, k):
    selector = SelectKBest(scoring_function, k=k)    
    selector.fit(df_X, df_y)
    return df_X.columns[selector.get_support(indices=True)]


from sklearn.pipeline import Pipeline
def select_from_model(df_X, df_y, model, threshold="median"):
    if isinstance(model, Pipeline):        
        model = model.steps[1][1]        
    selector = SelectFromModel(estimator=model, threshold=threshold)    
    selector.fit(df_X, df_y)
    return df_X.columns[selector.get_support(indices=True)]    


def rfe_cv(X_train, y, model, scoring='neg_mean_absolute_error', step=1, cv=KFold(n_splits=5), n_jobs = -1):
    # if isinstance(model, Pipeline):        
    #     model = model.steps[1][1]
    rfe = RFECV(model, scoring=scoring, step=step, cv=cv, n_jobs = n_jobs-1)
    rfe.fit(X_train, y)
    print(rfe.ranking_)
    print(np.mean(rfe.ranking_), np.median(rfe.ranking_))
    return X_train.columns[rfe.get_support(indices=True)]