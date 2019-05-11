
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

# Make the feature selection and other methods from Sklearn 
# actually work with features (with their context/name)

def variance_threshold_selector(df, threshold=0.0):
    selector = VarianceThreshold(threshold)
    selector.fit(df)    
    return df[df.columns[selector.get_support(indices=True)]]


def simple_imputer(df, cols=[], strategy=None):
    df2 = df.copy()
    if len(cols) == 0:        
        cols = df2.columns    
    imputer = SimpleImputer(strategy=strategy)
    df2[cols] = imputer.fit_transform(df2[cols])
    return df2