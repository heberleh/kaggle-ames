from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_ 

random_state = 2019

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

kfolds = KFold(n_splits=10, shuffle=True, random_state=2019)

ridge = Mypipeline([("scaler", RobustScaler()),
                        ("model", RidgeCV(alphas=alphas_alt, cv=kfolds))])

lasso = Mypipeline([("scaler", RobustScaler()),
                        ("model", LassoCV(max_iter=1e7, alphas=alphas2,
                              random_state=42, cv=kfolds))])
                                                              
elasticnet = Mypipeline([("scaler", RobustScaler()),
                        ("model", ElasticNetCV(max_iter=1e7, alphas=e_alphas,
                                        cv=kfolds, l1_ratio=e_l1ratio))])

svr = Mypipeline([("scaler", RobustScaler()),
                ("model", SVR(C=20, epsilon=0.008, gamma=0.0003))])

grid_xgbr_params = {'colsample_bytree': 0.3, 'gamma': 0, 'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 0, 'n_estimators': 1000, 'nthread': 2, 'objective': 'reg:linear', 'reg_alpha': 0.01, 'scale_pos_weight': 1, 'seed': 27, 'subsample': 0.75} # -0.08109561292018139

original_xgbr_params = { 'learning_rate':0.01, 'n_estimators':4000,'max_depth':3, 'min_child_weight':0, 'gamma':0, 'subsample':0.7,'colsample_bytree':0.7,'objective':'reg:linear', 'nthread':2,'scale_pos_weight':1, 'seed':27,'reg_alpha':0.00006}

xgbr = XGBRegressor(grid_xgbr_params)

rf = RandomForestRegressor(n_estimators=1000, max_depth=3, random_state=random_state, n_jobs=8)


gridcv_lgbm_params = {'bagging_fraction': 0.75, 'bagging_freq': 3, 'bagging_seed': 7, 'eval_metric': 'l1', 'feature_fraction': 0.7, 'learning_rate': 0.03, 'max_bin': 200, 'max_depth': 5, 'n_estimators': 1000, 'num_boost_round': 50, 'num_leaves': 9, 'objective': 'regression', 'reg_alpha': 0.0, 'verbose': -1}

original_lgbm_params = {'objective':'regression','num_leaves':4,'learning_rate':0.01,'n_estimators':3000,'max_bin':200,'bagging_fraction':0.75,'bagging_freq':5,'bagging_seed':7,'feature_fraction':0.2,'feature_fraction_seed':7,'verbose':-1}

lightgbm = LGBMRegressor(gridcv_lgbm_params)      

stack1 = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                            xgbr, lightgbm),
                                meta_regressor=xgbr,
                                use_features_in_secondary=True)

stack2 = StackingCVRegressor(regressors=(elasticnet, lasso, xgbr, lightgbm),
                                meta_regressor=xgbr,
                                use_features_in_secondary=True)