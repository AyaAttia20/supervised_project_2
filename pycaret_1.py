
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn  import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

class classification:
    def set(data,target,numeric_features: Optional[List[str]] = None,categorical_features: str = "mode",polynomial_features: bool = False):
       data=st.write('shape of the original data',data.shape)
       target=st.write('target column ' ,target)

       return data,target,numeric_features,categorical_features,polynomial_features

    def compare(x_scal_poly,y):
                x_train,x_test,y_train,y_test=train_test_split(x_scal_poly,y,test_size=0.2)
                
                clf=LogisticRegression()
                clf.fit(x_train,y_train)
                train_clf=clf.predict(x_train)
                test_clf=clf.predict(x_test)
                ConfusionMatrixDisplay.from_estimator(clf,x_test,y_test)
                plt.show()

                clf_1=SVC()
                clf_1.fit(x_train,y_train)
                train_svc=clf_1.predict(x_train)
                test_svc=clf_1.predict(x_test)

            #     st.write('accuracy score train for logistic Regression : ', accuracy_score(y_train,train_clf))
            #     st.write('accuracy score test for logistic Regression : ',accuracy_score(y_test,test_clf))
            #     st.write('accuracy score train for svm : ',accuracy_score(y_train,train_svc))
            #     st.write('accuracy score test for svm : ',accuracy_score(y_test,test_svc))
                st.subheader('evaluate the performance of model')
                if (accuracy_score(y_train,train_clf) and accuracy_score(y_test,test_clf))>(accuracy_score(y_train,train_svc) and accuracy_score(y_test,test_svc)):
                      best=clf
                      st.write('best model')
                      st.write(best ) 
                      st.write('accuracy score for the best model: ','train ',accuracy_score(y_train,train_clf) , 'test',accuracy_score(y_test,test_clf))
                      st.write('precision_score for the best model: ','train ',precision_score(y_train,train_clf) ,'test ',precision_score(y_test,test_clf))
                else:
                      best=clf_1 
                      st.write('best model')
                      st.write(best ) 
                      st.write('accuracy score for the best model: ','train ',accuracy_score(y_train,train_svc) ,'test ', accuracy_score(y_test,test_svc))
                      st.write('precision_score for the best model: ','train ',precision_score(y_train,train_svc) , 'test ',precision_score(y_test,test_svc))    

            #     st.write('best model')
            #     return st.write(best )  

#     def create_model(best):
#           st.write('model created')
          
          
            
    
class regrression:
    def set(data,target,numeric_features: Optional[List[str]] = None,categorical_features: str = "mode",polynomial_features: bool = False):
       data=st.write('shape of the original data', data.shape)
       target=st.write('target column ' ,target)

       return data,target,numeric_features,categorical_features,polynomial_features
    
    def compare(x_scal_poly,y):
           
            x_train,x_test,y_train,y_test=train_test_split(x_scal_poly,y,test_size=0.2)    

            lr=LinearRegression()
            lr.fit(x_train,y_train)
            train_pred=lr.predict(x_train)
            test_pred=lr.predict(x_test)

 #    GradientBoostingRegressor
            params = {
                        "n_estimators": 500,
                        "max_depth": 4,
                        "min_samples_split": 5,
                        "learning_rate": 0.01,
                        "loss": "squared_error"}
            # ensemble
            reg = ensemble.GradientBoostingRegressor(**params)
            reg.fit(x_train, y_train)
            train_reg=reg.predict(x_train)
            test_reg=reg.predict(x_test) 
            # st.write('r2 score train for linear regreesion  : ',r2_score(y_train,train_pred))
            # st.write('r2 score test  for linear regreesion : ',r2_score(y_test,test_pred))
            # st.write('r2 score train for GradientBoostingRegressor : ',r2_score(y_train,train_reg))
            # st.write('r2 score test for GradientBoostingRegressor : ',r2_score(y_test,test_reg)) 

            
            if (r2_score(y_train,train_pred) and r2_score(y_test,test_pred)) >(r2_score(y_train,train_reg) and r2_score(y_test,test_reg)):
                    best=lr
                    st.write('the best is linear regreesion')
                    st.subheader('evaluate the performance of model')
                    st.write('r2 score for the best model: ','train ',(r2_score(y_train,train_pred) ,'test', r2_score(y_test,test_pred)))
                   
            else:        
                    best=reg
                    st.write('the best is GradientBoostingRegressor')
                    st.subheader('evaluate the performance of model')
                    st.write('r2 score for the best model: ','train ',(r2_score(y_train,train_reg)) ,'test ', r2_score(y_train,train_reg))
                 
            return st.write(best)    

                  
st.header(" My Own Package Like pycaret ")
Select = st.sidebar.selectbox("Select Option",('Package','show code'))
if Select=='Package':
    data_set=st.file_uploader('Upload File',type=['csv','txt','xlsx'])
        # Select = st.sidebar.selectbox("Select Option", ('Exploratory Data Analysis ','Machine Learning Model','show Code'))
    if data_set is not None:
            df=pd.read_csv(data_set)

    col=df.columns
    target_col = st.multiselect("Select Target variable",col)
    list_missing=df.isna().sum()
            
    i=0 
    list_of_missing=[]


    for i in range(len(list_missing)):
        if list_missing.iloc[i]!=0:
            list_of_missing.append(df.columns[i])
        i+=1 

    mean_impute=SimpleImputer(strategy='mean',missing_values=np.nan)       
    mode_impute=SimpleImputer(strategy='most_frequent',missing_values=np.nan)   


    d_type=df.dtypes
    num_feature=[]
    cat_feature=[]
    for j in range(len(d_type)):
        if d_type.iloc[j]=='object':
            cat_feature.append(df.columns[j])
            
        elif d_type.iloc[j]=='float64'or d_type.iloc[j]=='int64' :
            num_feature.append(df.columns[j])    

    i=0
    j=0
    for i in range(len(list_of_missing)):
        if list_of_missing[i]==num_feature[i]:
            for j in  range(df[list_of_missing].shape[0]):
                    df[list_of_missing[i]]=mean_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1,1))
                    j+=1
        elif   list_of_missing[i]==cat_feature[i]:
            for j in  range(df[list_of_missing].shape[0]):
                df[list_of_missing[i]]=mode_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1, 1))[:,0]
                j+=1

        i+=1  
    le=LabelEncoder()
    for i in range(len(cat_feature)):
        df[cat_feature[i]]=le.fit_transform(df[cat_feature[i]])
        i+=1


    mxc= MinMaxScaler()
    x=df.drop(target_col,axis=1)
    y=df[target_col]
    x_scl=mxc.fit_transform(x)
    # polynomial feature
    poly=PolynomialFeatures(degree=3)
    x_poly=poly.fit_transform(x_scl)
    x_scal_poly=mxc.fit_transform(x_poly)

    alg=st.radio('select supervised algorithm ',options=['Classification','Regression'])
    if alg=='Classification':
        st.subheader('Initialize : setup function')
        classification.set(data=df,target=target_col,numeric_features=df[num_feature],categorical_features=df[cat_feature],polynomial_features=poly)
        st.write('Data type of Target variable : ',df[target_col].dtypes)
        st.write('numeric features ',len(df[num_feature].columns))
        st.write('categorical features ',len(df[cat_feature].columns))
        st.write(x)
        
        st.subheader('Train : compare models function')
        classification.compare(x_scal_poly,y)
        # classification.create_model()

    if alg=='Regression':
        st.subheader('Initialize : setup function')
        regrression.set(data=df,target=target_col,numeric_features=df[num_feature],categorical_features=df[cat_feature],polynomial_features=poly)
        st.write('Data type of Target variable : ',df[target_col].dtypes)
        st.write('numeric features ',len(df[num_feature].columns))
        st.write('categorical features ',len(df[cat_feature].columns))
        st.write(x)
        st.subheader('Train : compare models function')
        regrression.compare(x_scal_poly,y)

if Select=='show code':
     st.subheader('Code Of APP') 
     code=""" 

import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn  import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

class classification:
    def set(data,target,numeric_features: Optional[List[str]] = None,categorical_features: str = "mode",polynomial_features: bool = False):
       data=st.write('shape of the original data',data.shape)
       target=st.write('target column ' ,target)

       return data,target,numeric_features,categorical_features,polynomial_features

    def compare(x_scal_poly,y):
                x_train,x_test,y_train,y_test=train_test_split(x_scal_poly,y,test_size=0.2)
                
                clf=LogisticRegression()
                clf.fit(x_train,y_train)
                train_clf=clf.predict(x_train)
                test_clf=clf.predict(x_test)
                ConfusionMatrixDisplay.from_estimator(clf,x_test,y_test)
                plt.show()

                clf_1=SVC()
                clf_1.fit(x_train,y_train)
                train_svc=clf_1.predict(x_train)
                test_svc=clf_1.predict(x_test)

            #     st.write('accuracy score train for logistic Regression : ', accuracy_score(y_train,train_clf))
            #     st.write('accuracy score test for logistic Regression : ',accuracy_score(y_test,test_clf))
            #     st.write('accuracy score train for svm : ',accuracy_score(y_train,train_svc))
            #     st.write('accuracy score test for svm : ',accuracy_score(y_test,test_svc))
                st.subheader('evaluate the performance of model')
                if (accuracy_score(y_train,train_clf) and accuracy_score(y_test,test_clf))>(accuracy_score(y_train,train_svc) and accuracy_score(y_test,test_svc)):
                      best=clf
                      st.write('best model')
                      st.write(best ) 
                      st.write('accuracy score for the best model: ','train ',accuracy_score(y_train,train_clf) , 'test',accuracy_score(y_test,test_clf))
                      st.write('precision_score for the best model: ','train ',precision_score(y_train,train_clf) ,'test ',precision_score(y_test,test_clf))
                else:
                      best=clf_1 
                      st.write('best model')
                      st.write(best ) 
                      st.write('accuracy score for the best model: ','train ',accuracy_score(y_train,train_svc) ,'test ', accuracy_score(y_test,test_svc))
                      st.write('precision_score for the best model: ','train ',precision_score(y_train,train_svc) , 'test ',precision_score(y_test,test_svc))    

            #     st.write('best model')
            #     return st.write(best )  

#     def create_model(best):
#           st.write('model created')
          
          
            
    
class regrression:
    def set(data,target,numeric_features: Optional[List[str]] = None,categorical_features: str = "mode",polynomial_features: bool = False):
       data=st.write('shape of the original data', data.shape)
       target=st.write('target column ' ,target)

       return data,target,numeric_features,categorical_features,polynomial_features
    
    def compare(x_scal_poly,y):
           
            x_train,x_test,y_train,y_test=train_test_split(x_scal_poly,y,test_size=0.2)    

            lr=LinearRegression()
            lr.fit(x_train,y_train)
            train_pred=lr.predict(x_train)
            test_pred=lr.predict(x_test)

 #    GradientBoostingRegressor
            params = {
                        "n_estimators": 500,
                        "max_depth": 4,
                        "min_samples_split": 5,
                        "learning_rate": 0.01,
                        "loss": "squared_error"}
            # ensemble
            reg = ensemble.GradientBoostingRegressor(**params)
            reg.fit(x_train, y_train)
            train_reg=reg.predict(x_train)
            test_reg=reg.predict(x_test) 
            # st.write('r2 score train for linear regreesion  : ',r2_score(y_train,train_pred))
            # st.write('r2 score test  for linear regreesion : ',r2_score(y_test,test_pred))
            # st.write('r2 score train for GradientBoostingRegressor : ',r2_score(y_train,train_reg))
            # st.write('r2 score test for GradientBoostingRegressor : ',r2_score(y_test,test_reg)) 

            
            if (r2_score(y_train,train_pred) and r2_score(y_test,test_pred)) >(r2_score(y_train,train_reg) and r2_score(y_test,test_reg)):
                    best=lr
                    st.write('the best is linear regreesion')
                    st.subheader('evaluate the performance of model')
                    st.write('r2 score for the best model: ','train ',(r2_score(y_train,train_pred) ,'test', r2_score(y_test,test_pred)))
                   
            else:        
                    best=reg
                    st.write('the best is GradientBoostingRegressor')
                    st.subheader('evaluate the performance of model')
                    st.write('r2 score for the best model: ','train ',(r2_score(y_train,train_reg)) ,'test ', r2_score(y_train,train_reg))
                 
            return st.write(best)    

                  
st.header(" My Own Package Like pycaret ")
Select = st.sidebar.selectbox("Select Option",('Package','show code'))
if Select=='Package':
    data_set=st.file_uploader('Upload File',type=['csv','txt','xlsx'])
        # Select = st.sidebar.selectbox("Select Option", ('Exploratory Data Analysis ','Machine Learning Model','show Code'))
    if data_set is not None:
            df=pd.read_csv(data_set)

    col=df.columns
    target_col = st.multiselect("Select Target variable",col)
    list_missing=df.isna().sum()
            
    i=0 
    list_of_missing=[]


    for i in range(len(list_missing)):
        if list_missing.iloc[i]!=0:
            list_of_missing.append(df.columns[i])
        i+=1 

    mean_impute=SimpleImputer(strategy='mean',missing_values=np.nan)       
    mode_impute=SimpleImputer(strategy='most_frequent',missing_values=np.nan)   


    d_type=df.dtypes
    num_feature=[]
    cat_feature=[]
    for j in range(len(d_type)):
        if d_type.iloc[j]=='object':
            cat_feature.append(df.columns[j])
            
        elif d_type.iloc[j]=='float64'or d_type.iloc[j]=='int64' :
            num_feature.append(df.columns[j])    

    i=0
    j=0
    for i in range(len(list_of_missing)):
        if list_of_missing[i]==num_feature[i]:
            for j in  range(df[list_of_missing].shape[0]):
                    df[list_of_missing[i]]=mean_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1,1))
                    j+=1
        elif   list_of_missing[i]==cat_feature[i]:
            for j in  range(df[list_of_missing].shape[0]):
                df[list_of_missing[i]]=mode_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1, 1))[:,0]
                j+=1

        i+=1  
    le=LabelEncoder()
    for i in range(len(cat_feature)):
        df[cat_feature[i]]=le.fit_transform(df[cat_feature[i]])
        i+=1


    mxc= MinMaxScaler()
    x=df.drop(target_col,axis=1)
    y=df[target_col]
    x_scl=mxc.fit_transform(x)
    # polynomial feature
    poly=PolynomialFeatures(degree=3)
    x_poly=poly.fit_transform(x_scl)
    x_scal_poly=mxc.fit_transform(x_poly)

    alg=st.radio('select supervised algorithm ',options=['Classification','Regression'])
    if alg=='Classification':
        st.subheader('Initialize : setup function')
        classification.set(data=df,target=target_col,numeric_features=df[num_feature],categorical_features=df[cat_feature],polynomial_features=poly)
        st.write('Data type of Target variable : ',df[target_col].dtypes)
        st.write('numeric features ',len(df[num_feature].columns))
        st.write('categorical features ',len(df[cat_feature].columns))
        st.write(x)
        
        st.subheader('Train : compare models function')
        classification.compare(x_scal_poly,y)
        # classification.create_model()

    if alg=='Regression':
        st.subheader('Initialize : setup function')
        regrression.set(data=df,target=target_col,numeric_features=df[num_feature],categorical_features=df[cat_feature],polynomial_features=poly)
        st.write('Data type of Target variable : ',df[target_col].dtypes)
        st.write('numeric features ',len(df[num_feature].columns))
        st.write('categorical features ',len(df[cat_feature].columns))
        st.write(x)
        st.subheader('Train : compare models function')
        regrression.compare(x_scal_poly,y)




 """    
     st.code(code, language='python')
     
  