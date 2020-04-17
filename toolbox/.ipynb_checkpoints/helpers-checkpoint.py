import pandas as pd
import numpy as np
import gc
import missingno as msno
from math import radians, cos, sin, asin, sqrt


class ToolBox:
    """
    Useful Methods for data preparation
    """
    def __init__(self):
        print("ToolBox initialized")
    
    def read_from_file(self,file_path,file_name,file_type="csv",sep=","):
        file = file_path+file_name
        df = pd.read_csv(file,sep=",")
        df.columns = map(str.lower, df.columns)
        return df
    
    def class_counts(self,df,tgt_var_name):
        print('Class Counts')
        print(df.groupby(tgt_var_name).size())
        print("-"*50)
    
    def save_as_checkpoint(self,df,path,name):
        df.to_csv("{}/{}.csv".format(path,name),index=False)
        
    def generate_date_dim(self,df,date_col,alias_name,required_dims):
        #from datetime import datetime
        df_date=pd.DataFrame()
        if 'date' in required_dims:
            df_date[alias_name+'date'] = pd.to_datetime(df[date_col]).dt.date
            
        if 'year' in required_dims:
            df_date[alias_name+'year'] = pd.DatetimeIndex(df[date_col]).year
    
        if 'month' in required_dims:
            df_date[alias_name+'month'] = pd.DatetimeIndex(df[date_col]).month
    
        if 'month_name' in required_dims:
            df_date[alias_name+'month_name'] = pd.DatetimeIndex(df[date_col]).strftime("%b")
        
        if 'month_year' in required_dims:        
            df_date[alias_name+'month_year'] = pd.DatetimeIndex(df[date_col]).to_period('M')
     
        if 'weekday_name' in required_dims: 
            df_date[alias_name+'wkday'] = pd.DatetimeIndex(df[date_col]).weekday_name
    
        if 'dayofweek' in required_dims: 
            df_date[alias_name+'dayofweek'] = pd.DatetimeIndex(df[date_col]).dayofweek
    
        if 'weekend' in required_dims: 
            df_date[alias_name+'weekend'] = ((pd.DatetimeIndex(df[date_col]).dayofweek) // 5 == 1).astype(int)
        
        return df_date
    
    def create_flag(self,df,column_name):
        df[column_name+'_flag'] = 0
        df.loc[df[column_name] > 0, column_name+'_flag'] = 1
        return df

    def missing_value_stats(self,df):
        total = df.isnull().sum()
        percent = (df.isnull().sum()/df.isnull().count())
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data = missing_data[(missing_data.Total > 0) ].sort_values('Total',ascending=False)
        return missing_data
    
    def one_hot_encode(self,df, column_list,drop_first_flag=True):
        for x in column_list:
            dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False,drop_first=drop_first_flag)
            df = df.drop(x, 1)
            df = pd.concat([df, dummies], axis=1)
        return df
    
    def rearrange_label(self,df,label):
        df = df.drop(columns=[label]).assign(is_cancelled=df[label])
        df.columns = map(str.lower, df.columns)
        return df
    
    def select_features(self,method,df,features):
        if method == 'Correlation':
            df = df[features]
        else:
            df = df[features]
        return df
    
    def split_into_x_y(self,df,label_column):
        if label_column in df:
            y = df.pop(label_column)
        print('Predictor name:', np.array(df.columns))
        feature_names=np.array(df.columns)
        X = df
        return (X,y,feature_names)
    
    def split_stats(self,X,X_train, X_val, y_train, y_val):
        split_stats=[]
        split_stats.append(["Total","Train set",
                   "Test Set","Class Dist. in Train set",
                    "Class Dist. in Validation set"])
        split_stats.append([X.shape,X_train.shape,X_val.shape,
                    y_train.value_counts(),y_val.value_counts()])
        stats = pd.DataFrame(split_stats)
        stats = stats.rename(columns=stats.iloc[0]).drop(stats.index[0])
        return stats
    
    def memory_cleanup(self,df_list):
        del [df_list]
        gc.collect()

