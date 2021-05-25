# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np


import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import os
from tqdm.notebook import tqdm
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
# %matplotlib inline
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn import metrics
# -

TARGET_COL = "diabetes_mellitus"
df = pd.read_csv("../input/widsdatathon2021/TrainingWiDS2021.csv")
print(df.shape)
test = pd.read_csv("../input/widsdatathon2021/UnlabeledWiDS2021.csv")
print(test.shape)
df['label']='train'
test['label']='test'
frames = [df,test]
join_df = pd.concat(frames, keys=['x', 'y'])
assert len(join_df) == len(df) + len(test)
lst = join_df.isna().sum()/len(join_df)
p = pd.DataFrame(lst)
p.reset_index(inplace=True)
p.columns = ['a','b']
low_count = p[p['b']>0.8]
todelete=low_count['a'].values
join_df.drop(todelete,axis=1,inplace=True)
join_df.head()


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# +
join_df.drop(['Unnamed: 0','encounter_id'],inplace=True,axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = join_df.select_dtypes(include=numerics)
numeric_cols = newdf.columns

# Need to do column by column due to memory constraints
categorical_cols =  ['elective_surgery','hospital_id','icu_id',
 'ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type','aids','cirrhosis','hepatic_failure','immunosuppression',
 'leukemia','lymphoma','solid_tumor_with_metastasis','elective_surgery','apache_post_operative','arf_apache','fio2_apache','gcs_unable_apache','gcs_eyes_apache',
 'gcs_motor_apache','gcs_verbal_apache','intubated_apache','ventilated_apache','solid_tumor_with_metastasis']
for i, v in tqdm(enumerate(categorical_cols)):
    join_df[v] = join_df[v].fillna(join_df[v].value_counts().index[0])
for i, v in tqdm(enumerate([numeric_cols])):
    join_df[v] =join_df.groupby(['ethnicity','gender'], sort=False)[v].apply(lambda x: x.fillna(x.mean()))
join_df[categorical_cols].isna().sum()

# +
from sklearn.preprocessing import OrdinalEncoder

# In loop to minimize memory use
for i, v in tqdm(enumerate(categorical_cols)):
    join_df[v] = OrdinalEncoder(dtype="int").fit_transform(join_df[[v]])
    

gc.collect()

train = join_df[join_df['label']=="train"]
predict = join_df[join_df['label']=='test']

train.reset_index(inplace=True)
train.drop(['level_0','level_1','label'],inplace=True,axis =1 )

predict.reset_index(inplace=True)
predict.drop(['level_0','level_1','diabetes_mellitus','label'],inplace=True,axis=1)
features = train.columns
num_feature = [col for col in features if col not in categorical_cols]


# +
num_feature = [col for col in features if col not in categorical_cols and train[col].dtype != 'object']
drop_columns=[]
corr = train[num_feature].corr()
# Drop highly correlated features 
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >=0.999 :
            if columns[j]:
                columns[j] = False
                print('FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(train[num_feature].columns[i] , train[num_feature].columns[j], corr.iloc[i,j]))
        elif corr.iloc[i,j] <= -0.995:
            if columns[j]:
                columns[j] = False
# -

drop_columns = train[num_feature].columns[columns == False].values
print('drop_columns',len(drop_columns),drop_columns)

train.drop(drop_columns,inplace=True,axis =1 )
predict.drop(drop_columns,inplace=True,axis =1 )
train[TARGET_COL].value_counts()/len(train)

# +
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
df_majority = train[train['diabetes_mellitus']==0]
df_minority = train[train['diabetes_mellitus']==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=83798,    # to match majority class
                                 random_state= 303) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.diabetes_mellitus.value_counts()
train = df_upsampled
# -

X_train, X_test, y_train, y_test = train_test_split(
     train[[c for c in train if TARGET_COL != c]], train[TARGET_COL], test_size=0.20, random_state=42)
print(X_train.shape,X_test.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

optuna=False

# +
import optuna.integration.lightgbm as lgb
if optuna:
# old prams
    params={'learning_rate':0.1,
                          'num_leaves':50,
                          'n_estimators':2000,
                          'metric': 'auc',
                          'objective': 'binary'}
        
#            'random_state':33,'early_stopping_rounds':100,
#            'min_data_per_group':5,'boosting_type':'gbdt','num_leaves':151,'max_dept':-1,
#            'learning_rate':0.002, 'subsample_for_bin':200000, 
#            'min_split_gain':0.0, 'min_child_weight':0.001,
#            'min_child_samples':20, 'subsample':1.0, 'subsample_freq':0, 
#            'colsample_bytree':.75, 'reg_alpha':1.3, 'reg_lambda':0.1,  
#            'n_jobs':- 1, 'cat_smooth':1.0,
#            'silent':True, 'importance_type':'split','metric': 'auc'
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(params, dtrain, valid_sets=[dval], verbose_eval=100, early_stopping_rounds=100, )
    params = model.params
    params
else:
  params={
 'random_state': 33,
 'min_data_per_group': 5,
 'boosting_type': 'gbdt',
 'num_leaves': 125,
 'max_dept': -1,'max_bin':63,
 'learning_rate': 0.01,
 'subsample_for_bin': 200000,
 'lambda_l1': 0.012792604554947268,
 'lambda_l2': 0.02701650012625809,
 'n_jobs': -1,
 'cat_smooth': 1.0,
 'silent': True,
 'importance_type': 'split',
 'metric': 'auc',
 'feature_pre_filter': False,
 'bagging_fraction': 0.9947921066721309,
 'min_data_in_leaf': 30,
 'min_sum_hessian_in_leaf': 0.01,
 'bagging_freq': 1,
 'feature_fraction': 0.5,
 'min_gain_to_split': 0.2,
 'min_child_samples': 20,'path_smooth':200,'extra_trees':True}

 

# +
# model= LGBMClassifier(
#                               random_state=33,
#                               early_stopping_rounds = 250,
#                               n_estimators=10000,min_data_per_group=5, # reduce overfitting when using categorical_features
#                               boosting_type='gbdt', num_leaves=151, max_depth=- 1, learning_rate=0.02, subsample_for_bin=200000, 
#                               min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, 
#                               colsample_bytree=.75, reg_alpha=1.3, reg_lambda=0.1,  n_jobs=- 1,cat_smooth=1.0, 
#                               silent=True, importance_type='split')

# +
# params={'learning_rate': 0.05,
#  'num_leaves': 256,
#  'metric': 'auc',
#  'objective': 'binary',
#  'feature_pre_filter': False,
#  'lambda_l1': 1.003183229713461e-08,
#  'lambda_l2': 1.2171923409569426e-08,
#  'feature_fraction': 0.8,
#  'bagging_fraction': 1.0,
#  'bagging_freq': 0,
#  'min_child_samples': 10,
#  'num_iterations': 10000}
# -

params

params['num_iterations'] = 2000
params['learning_rate'] = 0.01
model= LGBMClassifier(**params)
X=train[[c for c in train if TARGET_COL != c]]
y= train[TARGET_COL]
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric = 'auc',
    verbose=1
)

print(f"accuracy score is {accuracy_score(y_test, model.predict(X_test))}") #0.02- 0.8423
print(metrics.classification_report(y_test, model.predict(X_test), labels=[0, 1]))

pred = model.predict_proba(predict, num_iteration=model.best_iteration_)[:,1]
test[TARGET_COL] = pred
test[["encounter_id","diabetes_mellitus"]].to_csv("submission.csv",index=False)




