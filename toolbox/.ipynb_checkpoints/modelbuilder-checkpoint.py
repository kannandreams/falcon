import pandas as pd
import time
import collections
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

class Modeler:
    """
    Useful Methods for modeling
    """
    def __init__(self):
        print("Model building initialized")
    
    def build(self,estimator,X_train, y_train, X_val, y_val,algo_group):
        if algo_group=='classification':
            Results = collections.namedtuple('Results', 'model_name,model,training_time,y_pred,y_pred_probs')
            print("**********************************************")
            print("Started : Estimator :",estimator)
            start_time = time.time()
            model = estimator.fit(X_train, y_train)
            training_time=(time.time() - start_time)
            print("Training Completed in %s seconds:" %(time.time() - start_time))
            y_pred = model.predict(X_val)
            y_pred_probs = model.predict_proba(X_val)
            result = Results(type(model).__name__,model,training_time,y_pred,y_pred_probs)
            return result
        else:
            print("Under construction...")
    