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

# + _uuid="839c7bce-0993-4ebf-960a-9b15814d3f64" _cell_guid="8ddc423d-92a2-4166-9e90-7479336017ce" jupyter={"outputs_hidden": false}
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

# + _uuid="25d3e3e2-e671-437f-afb9-18a73d6f57d9" _cell_guid="b2adc1d5-ed7e-40f5-be62-2304415fc780" jupyter={"outputs_hidden": false}
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


# + _uuid="bbe423f9-fb4c-4c49-9223-3f7e71f7a9cb" _cell_guid="d8ae00d1-a70b-4480-87c5-814fa41b7fe9" jupyter={"outputs_hidden": false}
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


# + _uuid="0ffc4f17-4f2e-4475-942e-6560821410e3" _cell_guid="e206a114-2e37-477a-bf2e-bccf10e70372" jupyter={"outputs_hidden": false}
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

# + _uuid="6ca7cce3-e174-4104-9898-e7beebfa2092" _cell_guid="2eaeaadb-349c-4b63-8b81-3ce70c4dd0a5" jupyter={"outputs_hidden": false}
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

# + _uuid="f406fbac-840e-4783-9cd4-f5f53fc52e9c" _cell_guid="a2a0268e-f755-4ba9-a043-446b81ab6b82" jupyter={"outputs_hidden": false}
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

# + _uuid="9dd5f8a6-4266-480e-90a0-929a9b8cd5b0" _cell_guid="17bb6c88-9a0f-4a2e-b62f-9cbf434db3f4" jupyter={"outputs_hidden": false}
drop_columns = train[num_feature].columns[columns == False].values
print('drop_columns',len(drop_columns),drop_columns)

# + _uuid="de5fe708-14e1-49dd-bfe3-3b17617665ad" _cell_guid="6390b12c-9040-4e3b-9f03-46c44de19b55" jupyter={"outputs_hidden": false}
train.drop(drop_columns,inplace=True,axis =1 )
predict.drop(drop_columns,inplace=True,axis =1 )
train[TARGET_COL].value_counts()/len(train)

# + _uuid="a4881031-9699-4b64-b7a7-c844a8894afa" _cell_guid="d7e41dac-8971-47e3-9e93-a2c8a6b79a62" jupyter={"outputs_hidden": false}
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

# + _uuid="4013b932-f90a-4d6b-aad7-b7c053e97e11" _cell_guid="a2fca7c4-f957-493a-8337-ab3a38261730" jupyter={"outputs_hidden": false}
X_train, X_test, y_train, y_test = train_test_split(
     train[[c for c in train if TARGET_COL != c]], train[TARGET_COL], test_size=0.20, random_state=42)
print(X_train.shape,X_test.shape)

# + _uuid="179aeb45-25b7-40b3-9799-771456acc74a" _cell_guid="541070b5-00ed-4823-a276-7f92fcc55a81" jupyter={"outputs_hidden": false}
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# + _uuid="f9d59e83-6d8b-45dd-bfb0-7149f62d8934" _cell_guid="7d4be66b-bc5c-450a-97e7-55b021b35e55" jupyter={"outputs_hidden": false}
optuna=False

# + _uuid="8fd56098-79ed-406e-8895-f6191a4147df" _cell_guid="9cef25a6-3ecc-4f8d-a240-7e69e2ca361c" jupyter={"outputs_hidden": false}
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

# + _uuid="2584eb08-dfde-4d9d-b972-4e9c731464e9" _cell_guid="68096ecd-6609-421d-931c-25a1f3f6e990" jupyter={"outputs_hidden": false}
# model= LGBMClassifier(
#                               random_state=33,
#                               early_stopping_rounds = 250,
#                               n_estimators=10000,min_data_per_group=5, # reduce overfitting when using categorical_features
#                               boosting_type='gbdt', num_leaves=151, max_depth=- 1, learning_rate=0.02, subsample_for_bin=200000, 
#                               min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, 
#                               colsample_bytree=.75, reg_alpha=1.3, reg_lambda=0.1,  n_jobs=- 1,cat_smooth=1.0, 
#                               silent=True, importance_type='split')

# + _uuid="ba1467c2-271e-4f14-88e3-5ca1fc931575" _cell_guid="a2d6cae2-8509-4dbc-a02a-28e5ce35719a" jupyter={"outputs_hidden": false}
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

# + _uuid="885fee29-2b09-4d21-a0fb-44ddff9871e3" _cell_guid="dcd28697-f9fd-4d91-86af-7600d996f80b" jupyter={"outputs_hidden": false}
params

# + _uuid="220729d3-a702-408a-8693-11b6d6a7e048" _cell_guid="8cb340a6-d5b8-4dfc-bd3c-78f2b7c1fd56" jupyter={"outputs_hidden": false}
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

# + _uuid="512be25b-6243-4ac0-98bf-07ee507fef85" _cell_guid="073582ba-c651-4a79-9705-c5e5d563f7cb" jupyter={"outputs_hidden": false}
print(f"accuracy score is {accuracy_score(y_test, model.predict(X_test))}") #0.02- 0.8423
print(metrics.classification_report(y_test, model.predict(X_test), labels=[0, 1]))

# +
#ROC Curve
from sklearn.metrics import roc_curve
y_pred_prob1 = model.predict_proba(X_test)[:,1]
fpr1 , tpr1, thresholds1 = roc_curve(y_test, y_pred_prob1)

# y_pred_prob2 = classifier2.predict_proba(X_test)[:,1]
# fpr2 , tpr2, thresholds2 = roc_curve(Y_test, y_pred_prob2)


# y_pred_prob3 = classifier3.predict_proba(X_test)[:,1]
# fpr3 , tpr3, thresholds3 = roc_curve(Y_test, y_pred_prob3)

# y_pred_prob4 = classifier4.predict_proba(X_test)[:,1]
# fpr4 , tpr4, thresholds4 = roc_curve(Y_test, y_pred_prob4)


plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, label= "LGBM classifier")
# plt.plot(fpr2, tpr2, label= "Poly")
# plt.plot(fpr3, tpr3, label= "RBF")
# plt.plot(fpr4, tpr4, label= "Sigmoid")
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Receiver Operating Characteristic')
plt.show()

# +
from sklearn.metrics import plot_roc_curve

classifiers = [model]
ax = plt.gca()
for i in classifiers:
    plot_roc_curve(i, X_test, y_test, ax=ax)
# -



# + _uuid="5eec7b8a-230d-4a78-b351-42c23e81a3dc" _cell_guid="15c8701a-9c75-4210-a726-0249aa61d722" jupyter={"outputs_hidden": false}
pred = model.predict_proba(predict, num_iteration=model.best_iteration_)[:,1]
test[TARGET_COL] = pred
test[["encounter_id","diabetes_mellitus"]].to_csv("submission.csv",index=False)

# + _uuid="9c643d3c-1e2c-4fbf-8b95-cd620cf19423" _cell_guid="fdce1f8d-e631-4e65-99d9-17fa0035df44" jupyter={"outputs_hidden": false}


# + _uuid="c8fec668-1b7e-4ab6-bdfd-00d458bc8ef0" _cell_guid="465ac966-c90f-437b-9424-9707f54556fc" jupyter={"outputs_hidden": false}

