from sklearn.pipeline import Pipeline

class Mypipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_ 
    
    # def fit(self, X, y=None, **fit_params):
    #     super(Mypipeline, self).fit(X, y, **fit_params)
    #     self.feature_importances_ = self.steps[-1][-1].feature_importances_
    #     return self