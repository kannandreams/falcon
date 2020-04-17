from sklearn.metrics import classification_report,precision_score,accuracy_score
from sklearn.metrics import recall_score,roc_curve,roc_auc_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error,r2_score
import collections


class Evaluator:
    """
    Useful Methods for modeling
    """
    def __init__(self):
        print("Metrics process initialized....")
        
    
    def classifier_metrics(self,model,X_train,y_train,X_val,y_val,y_pred,y_pred_probs):
        Results = collections.namedtuple('Results', 'model, \
                                         training_score,validation_score,auc_score,f1_score,avg_precision_score')
        
        train_acc_score=model.score(X_train, y_train)
        val_acc_score=model.score(X_val, y_val)
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred))
        auc = roc_auc_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        ap = average_precision_score(y_val, y_pred)
        print("Metrics generated for :",type(model).__name__)
        
        result = Results(type(model).__name__,train_acc_score,val_acc_score,auc,f1,ap)
        return result