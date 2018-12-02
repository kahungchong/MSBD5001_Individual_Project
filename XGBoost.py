import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics 
from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split  
from sklearn.feature_selection import SelectKBest,VarianceThreshold,f_regression
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,LabelEncoder
import matplotlib.pylab as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings 
warnings.filterwarnings('ignore')

poly = PolynomialFeatures(degree=1)
varSel=VarianceThreshold(threshold=.08)
min_max_scaler =  MinMaxScaler()
kbSel = SelectKBest(score_func=f_regression,k=10)
le = LabelEncoder()

#read file and fill na
train_data=pd.read_csv("train.csv")
train_fillna = train_data.fillna(0)
train_dpID = train_fillna.drop(["id"],axis=1)

#feature and label data
feature_df = train_fillna[train_fillna.columns[1:14]]
label_df = train_fillna[train_fillna.columns[14]]

#encode label
le.fit(feature_df['penalty'])
feature_df["penalty"]=le.transform(feature_df["penalty"])

#data preprocessing
min_max_scaler.fit(feature_df)
feature_df = min_max_scaler.transform(feature_df)

varSel.fit(feature_df)
feature_df = varSel.transform(feature_df)

feature_df = poly.fit_transform(feature_df)


#train test data split
train_X,test_X,train_Y,test_Y = train_test_split(feature_df,label_df,test_size=0.25,random_state=14)





def modelfit(alg,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain =xgb.DMatrix(train_X,label=train_Y)
        xgtest = xgb.DMatrix(test_X)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds,show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])#cvresult.shape[0]和alg.get_params()['n_estimators']值一样

    #Fit the algorithm on the data
    alg.fit(train_X, train_Y,eval_metric='rmse')
    #Predict training set:
    dtrain_predictions = alg.predict(train_X)
    #Print model report:
    print(" Score (Train): %f" % mean_squared_error(train_Y.values, dtrain_predictions))
    #Predict on testing data:
    dtest_predictions = alg.predict(test_X)
    print("Score (Test): %f" % mean_squared_error(test_Y.values, dtest_predictions))




XGBmodel = XGBRegressor(booster='gbtree',
                    objective= 'reg:linear',
                    eval_metric='rmse',
                    gamma = 0.1,
                    min_child_weight= 1.1,
                    max_depth= 5,
                    subsample= 0.7,
                    colsample_bytree= 0.7,
                    tree_method= 'exact',
                    learning_rate=0.1,
                    n_estimators=300,
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27
                    )


modelfit(XGBmodel)

#adjust parameters

param_test1 = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch1 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',
                    objective= 'reg:linear',
                    eval_metric='rmse',
                    gamma = 0.2,
                    min_child_weight= 3,
                    max_depth= 3,
                    subsample= 0.8,
                    colsample_bytree= 0.8,
                    tree_method= 'exact',
                    learning_rate=0.1,
                    n_estimators=100,
                    nthread=4,
                    scale_pos_weight=1,
                    reg_alpha=0.001,
                    seed=27),
                       param_grid = param_test1, scoring=None,n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_X,train_Y)

print(gsearch1.best_params_, gsearch1.best_score_)


#fit model
all_set = xgb.DMatrix(feature_df,label=label_df)
XGBmodel=xgb.train(XGBmodel.get_params(),all_set,num_boost_round=10000)

#generate result
test_data = pd.read_csv("test.csv")
test_drID = test_data.drop(["id"],axis=1)
test_drID["penalty"] = le.transform(test_drID["penalty"])
test_drID = test_drID.fillna(0)
test_set = min_max_scaler.transform(test_drID)
test_set = varSel.transform(test_set)
test_set = poly.fit_transform(test_set)
test_set = xgb.DMatrix(test_set)
xgbResult = XGBmodel.predict(test_set)
for i in range(0,len(xgbResult)):
    if xgbResult[i]<0:
        xgbResult[i]=0.2
xgbResult = xgbResult * 1.25
result=pd.DataFrame({'Id':test_data['id'],"time":xgbResult})
result.to_csv("result.csv",index=False)

