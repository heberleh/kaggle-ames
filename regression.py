from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor


random_state = 2019

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

kfolds = KFold(n_splits=10, shuffle=True, random_state=2019)

ridge = make_pipeline(RobustScaler(),
                      RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(),
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))

svr = make_pipeline(RobustScaler(),
                    SVR(C=20, epsilon=0.008, gamma=0.0003, ))

xgbr = XGBRegressor(learning_rate=0.01, n_estimators=3000,
                       max_depth=4, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear', nthread=8,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

rf = RandomForestRegressor(n_estimators=81, random_state=random_state, n_jobs=4)

stack1 = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, svr, xgbr),
                                meta_regressor=xgbr,
                                use_features_in_secondary=True)

stack2 = StackingCVRegressor(regressors=(ridge, elasticnet, lasso),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)

stack3 = StackingCVRegressor(regressors=(ridge, lasso, xgbr),
                                meta_regressor=xgbr,
                                use_features_in_secondary=True)

stack4 = StackingCVRegressor(regressors=(ridge, lasso, elasticnet),
                                meta_regressor=lasso,
                                use_features_in_secondary=True)