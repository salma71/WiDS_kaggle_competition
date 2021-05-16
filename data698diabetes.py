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

# # ICU Data Diabetes Classifiers

# +
import numpy as np 
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn import metrics
# -

# %%time
raw_data = pd.read_csv("../input/diabetes-data/diabetes_data.csv")
data = raw_data.copy()

# drop useless columns
data.drop(columns=['Unnamed: 0','encounter_id'], inplace=True)

# # Data Description

data.shape

data.head(20)

# Variables in **bold** are considered important, all others likely are irrelevant
#
# Three types of variables
# - continuous numeric variables (bmi, h1 and d1 vars)
# - binary numeric variables
# - categorical variables
#
# Encoded as strings:
# - apache_2_diagnosis, apache_3j_diagnosis (diagnoses codes)
# - bmi (miscoded, should be numeric and recomputed based on height and weight columns)
# - **ethnicity**, gender (categorical, to be one-hot encoded)
# - hospital_admit_source
# - icu_admit_source
# - icu_admit_type
# - icu_stay_type
# - icu_type
#
# Encoded as integers:
# - encounter_id, hospital_id, icu_id (indexes)
# - gcs_eyes_apache, gcs_motor_apache, gcs_verbal_apache
#
# Encoded as binary:
# - aids, cirrhosis, hepatic_failure, **immunosuppression (some diabetes patients take steroids)**, leukemia, lymphoma, solid_tumor_with_metastasis (comorbidities)
# - apache_post_operative, **arf_apache (diabetes can cause renal failure)**, gcs_unable_apache, intubated_apache, ventilated_apache
# - elective_surgery, readmission_status (irrelevant)
# - **diabetes_mellitus (target)**
#
# Encoded as numeric (all others):
# - **age, height, weight**
# - **urineoutput_apache**
# - **d1_bun_max, d1_bun_min, h1_bun_max, h1_bun_min (all related to kidney failure)**
# - **h1_glucose_max, h1_glucose_min, d1_glucose_max, d1_glucose_min**
# - all others

# # Data Preprocessing

# ## Sparse Variables

# Dealing with missing values
missing_vals = pd.DataFrame(data.isna().sum()/len(data))
missing_vals = missing_vals.reset_index()
missing_vals.columns = ['var','missing']
missing_vals = missing_vals.sort_values(by=['missing'], ascending=False)

missing_vals.plot.bar(x='var', y='missing', rot=0)

# Using a threshold of 80% missing values, we can discard 32 variables
missing_vals[missing_vals['missing']>0.8].count()
sparse_vars = missing_vals[missing_vals['missing']>0.5]['var'].tolist()
sparse_vars

data = data.drop(sparse_vars, axis=1)
data.shape

# ## Data Types

# Identify variables to be one-hot encoding. Most others will be dropped
data.select_dtypes(include=object).columns.tolist()

data.select_dtypes(include=np.number).columns.tolist()

# Delete useless and irrelevant variables
generic = ['hospital_admit_source','icu_admit_source','icu_stay_type','icu_type']
irrelevant = ['aids','cirrhosis','hepatic_failure','leukemia','lymphoma','solid_tumor_with_metastasis',
              'elective_surgery','readmission_status']
data.drop(columns=generic+irrelevant, inplace=True)

data.shape

# # Data Processing

# Recalculate bmi
bmi = data[['bmi','weight','height']].copy()
bmi['bmi_calc'] = bmi['weight']/((bmi['height']/100)**2)
bmi['diff'] = bmi['bmi_calc'] - bmi['bmi']

bmi.sort_values(by='diff' ,ascending=False)

data['bmi'] = bmi['bmi_calc']

# ## Missing Value Imputation
#
# - Impute missing continuous numeric data by mean
# - Impure missing categorical data by mode
#

categoricals = ['ethnicity','gender']
for i, v in enumerate(categoricals):
    data[v] = data[v].fillna(data[v].value_counts().index[0])
numerics = data.select_dtypes(include=np.number).columns.tolist()
for i, v in enumerate(numerics):
    data[v] = data.groupby(['ethnicity','gender'], sort=False)[v].apply(lambda x: x.fillna(x.mean()))

# ## Encoding of Categorical Data

# One-hot encoding (ethnicity)
data = pd.get_dummies(data, drop_first=True)
data.shape

# +
# from sklearn.feature_selection import VarianceThreshold

# selector = VarianceThreshold(threshold=500)
# d = selector.fit_transform(data)
# data.shape
# -

# ## Balancing of Training Data

# +
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
df_majority = data[data['diabetes_mellitus']==0]
df_minority = data[data['diabetes_mellitus']==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=83798,    # to match majority class
                                 random_state= 303) # reproducible results
 
# Combine majority class with upsampled minority class
data_upsampled = pd.concat([df_majority, df_minority_upsampled])
# -

data_upsampled.shape

# ## Train Test Split

y = data_upsampled['diabetes_mellitus']
X = data_upsampled.drop(['diabetes_mellitus',], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# ## Data Scaling

from sklearn import preprocessing
# scaling for logistic regression
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# +
# subset based on variable importance did not yield better logistic regression results
important_vars = ['glucose_apache',
 'd1_glucose_max',
 'bmi',
 'age',
 'd1_creatinine_min',
 'd1_glucose_min',
 'weight',
 'icu_id',
 'pre_icu_los_days',
 'heart_rate_apache']

X_train_subset = X_train[important_vars]
X_test_subset = X_test[important_vars]

scaler2 = preprocessing.StandardScaler().fit(X_train_subset)
X_train_subset_scaled = scaler2.transform(X_train_subset)
X_test_subset_scaled = scaler2.transform(X_test_subset)
# -

# # Modeling

# +
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from scipy.stats import randint

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

# +
# models = {}

# def add_model(model_name, model, X_test=X_test):
#     y_pred = model.predict(X_test)
#     y_pred_prob = model.predict_proba(X_test)[:,1]
    
#     acc_train = model.score(X_train, y_train)
#     acc_test = model.score(X_test, y_test)
    
#     models[model_name] = {'model_name' : model_name, 'y_pred': y_pred, 'y_pred_prob': y_pred_prob, 
#                           'acc_train': acc_train, 'acc_test': acc_test}

# +
cols = ['name','acc_train','acc_test','auc','prec','tpr','tnr']
models = pd.DataFrame(columns=cols)

def add_model(metrics):

    model = pd.DataFrame(data=metrics, index=[0])
    return pd.concat([models, model], ignore_index=True)



# -

def get_metrics(model_name, model):
    
    if isinstance(model, LogisticRegression):
        Xtrain = X_train_scaled
        Xtest = X_test_scaled
#     elif isinstance(model, LGBMClassifier):
#         Xtrain = X_train_se
#         Xtest = X_valid
#         y_train = y_train_se
#         y_test = y_valid
    else:
        Xtrain = X_train
        Xtest = X_test
    
    y_pred = model.predict(Xtest)
    y_pred_prob = model.predict_proba(Xtest)[:,1]
    
    acc_train = round(model.score(Xtrain, y_train),4)
    acc_test = round(model.score(Xtest, y_test),4)
    auc = round(roc_auc_score(y_test, y_pred_prob),4)
    
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[1,0]
    fn = cm[0,1]
    prec = round(tp/(tp+fp),4)
    tpr = round(tp/(tp+fn),4)
    tnr = round(tn/(tn+fp),4)
    
    metrics = {'name': model_name, 'acc_train': acc_train, 'acc_test': acc_test, 'auc': auc,
              'prec': prec, 'tpr': tpr, 'tnr': tnr}
    
    return metrics


# set to True to run crossvalidation and tuning procedures
tune = False

# # Decision Trees
#
# 1. Optimized depth tree
# 2. Simple tree
# 3. Tuned tree
#

# ### Optimizing for depth size
#
# The decision tree starts to overfit the training data as the tree grows in depth as expected.
#
# Cost complexity pruning yielded a decrease in train and test accuracy

if tune:
    max_depth = np.arange(5, 100, 10)
    train_accuracy = np.empty(len(max_depth))
    test_accuracy = np.empty(len(max_depth))

    for i, m in enumerate(max_depth):
        dt = DecisionTreeClassifier(max_depth=m)  # Setup DT Classifier with m max_depth
        dt.fit(X_train, y_train)                       # Fit the classifier to the training data
        train_accuracy[i] = dt.score(X_train, y_train) # Compute accuracy on the training set
        test_accuracy[i] = dt.score(X_test, y_test)    # Compute accuracy on the testing set

    dt_opt = pd.DataFrame({'max_depth':max_depth, 'train_acc':train_accuracy, 'test_acc':test_accuracy})
    plt.plot(max_depth, train_accuracy, marker='o', color='blue', linewidth=1)
    plt.plot(max_depth, test_accuracy, marker='o', color='olive', linewidth=1)

# +
dt_best_depth = DecisionTreeClassifier(max_depth=25).fit(X_train, y_train)
y_pred_dtbest = dt_best_depth.predict(X_test)
y_pred_prob_dtbest = dt_best_depth.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred_dtbest))
print(classification_report(y_test, y_pred_dtbest, target_names=['0','1']))
print(get_metrics('DT Simple Tuning', dt_best_depth))

models = add_model(get_metrics('DT Simple Tuning', dt_best_depth))
# -

# ## Tuned Decision Tree
#
# Tuned Decision Tree Parameters: {'max_features': 40, 'max_depth': 50, 'criterion': 'entropy'}

# +
# %%time

if tune:
    # specify classifier
    clf = DecisionTreeClassifier()

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [5,25,50,75],
                  "max_features": [4,20,40,60],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=5, n_iter=n_iter_search, random_state=42)
    random_search.fit(X_train, y_train)

    print("Tuned Decision Tree Parameters: {}".format(random_search.best_params_))
    print("Best score is {}".format(random_search.best_score_))

    dt_tuned = random_search.best_estimator_
else:
    dt_tuned = DecisionTreeClassifier(max_features=40, max_depth=50, criterion='entropy').fit(X_train, y_train)

print(dt_tuned)

# +
y_pred_dt_tuned = dt_tuned.predict(X_test)
y_pred_prob_dt_tuned = dt_tuned.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred_dt_tuned))
print(classification_report(y_test, y_pred_dt_tuned, target_names=['0','1']))
print(get_metrics('DT CV Tuning', dt_tuned))

models = add_model(get_metrics('DT CV Tuning', dt_tuned))
# -

# variable importance
importance = dt_tuned.feature_importances_
sorted_top_imps = sorted(zip(importance, X_train.columns), reverse=True)[:10]
sorted_top_imps

# cost complexity tuning, does not improve performance
if tune:
    alpha = np.arange(0, 0.02, 0.002)
    train_accuracy = np.empty(len(alpha))
    test_accuracy = np.empty(len(alpha))

    for i, a in enumerate(alpha):
        dt = DecisionTreeClassifier(max_features=40, max_depth=50, criterion='entropy', ccp_alpha=a)  # Setup DT Classifier with a alpha
        dt.fit(X_train, y_train)                       # Fit the classifier to the training data
        train_accuracy[i] = dt.score(X_train, y_train) # Compute accuracy on the training set
        test_accuracy[i] = dt.score(X_test, y_test)    # Compute accuracy on the testing set

    dt_opt = pd.DataFrame({'alpha':alpha, 'train_acc':train_accuracy, 'test_acc':test_accuracy})
    plt.plot(alpha, train_accuracy, marker='o', color='blue', linewidth=1)
    plt.plot(alpha, test_accuracy, marker='o', color='olive', linewidth=1)

# # Logistic Regression via sklearn
#
# ### No Penalty

# +
log_reg = LogisticRegression(max_iter=500, solver='saga', penalty='none').fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_prob_lr = log_reg.predict_proba(X_test_scaled)[:,1]

print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=['0','1']))
print(get_metrics('LR', log_reg))

models = add_model(get_metrics('LR', log_reg))
# -

# ### L1 Penalty

# cost complexity tuning, does not improve performance
if tune:
    C = [0.001, 0.01, 0.1, 1,10,100]
    train_accuracy = np.empty(len(C))
    test_accuracy = np.empty(len(C))

    for i, c in enumerate(C):
        lr = LogisticRegression(max_iter=500, solver='saga', penalty='l1', C=c) 
        lr.fit(X_train_scaled, y_train)                       # Fit the classifier to the training data
        train_accuracy[i] = lr.score(X_train_scaled, y_train) # Compute accuracy on the training set
        test_accuracy[i] = lr.score(X_test_scaled, y_test)    # Compute accuracy on the testing set

    lr_opt = pd.DataFrame({'C':C, 'train_acc':train_accuracy, 'test_acc':test_accuracy})
    plt.plot(C, train_accuracy, marker='o', color='blue', linewidth=1)
    plt.plot(C, test_accuracy, marker='o', color='olive', linewidth=1)

# +
log_reg_l1 = LogisticRegression(max_iter=500, solver='saga', penalty='l1', C=1).fit(X_train_scaled, y_train)
y_pred_l1 = log_reg_l1.predict(X_test_scaled)
y_pred_prob_l1 = log_reg_l1.predict_proba(X_test_scaled)[:,1]
print(confusion_matrix(y_test, y_pred_l1))
print(classification_report(y_test, y_pred_l1, target_names=['0','1']))
print(get_metrics('LR L1 Penalty', log_reg_l1))

models = add_model(get_metrics('LR L1 Penalty', log_reg_l1))
# -

# ### L2 Penalty

# cost complexity tuning, does not improve performance
if tune:
    C = [0.001, 0.01, 0.1, 1,10,100]
    train_accuracy = np.empty(len(C))
    test_accuracy = np.empty(len(C))

    for i, c in enumerate(C):
        lr = LogisticRegression(max_iter=500, solver='saga', penalty='l2', C=c) 
        lr.fit(X_train_scaled, y_train)                       # Fit the classifier to the training data
        train_accuracy[i] = lr.score(X_train_scaled, y_train) # Compute accuracy on the training set
        test_accuracy[i] = lr.score(X_test_scaled, y_test)    # Compute accuracy on the testing set

    lr_opt = pd.DataFrame({'C':C, 'train_acc':train_accuracy, 'test_acc':test_accuracy})
    plt.plot(C, train_accuracy, marker='o', color='blue', linewidth=1)
    plt.plot(C, test_accuracy, marker='o', color='olive', linewidth=1)

# +
log_reg_l2 = LogisticRegression(max_iter=500, solver='saga', penalty='l2', C=1).fit(X_train_scaled, y_train)
y_pred_l2 = log_reg_l2.predict(X_test_scaled)
y_pred_prob_l2 = log_reg_l2.predict_proba(X_test_scaled)[:,1]
print(confusion_matrix(y_test, y_pred_l2))
print(classification_report(y_test, y_pred_l2, target_names=['0','1']))
print(get_metrics('LR L2 Penalty', log_reg))

models = add_model(get_metrics('LR L2 Penalty', log_reg))
# -

# # Random Forest
#
# Parameters to tune:
# - n_estimators
# - max_depth
# - max_features
#
# Best RSCV params:
#
# {'n_estimators': 75,
#  'min_samples_split': 2,
#  'min_samples_leaf': 4,
#  'max_features': 'auto',
#  'max_depth': 30,
#  'bootstrap': False}
#
# ## Simple RF

if tune:
    max_depth = np.arange(5,50,5)
    n_estimators = np.arange(5,100,10)
    train_accuracy = np.empty(len(n_estimators))
    test_accuracy = np.empty(len(n_estimators))

    for i, n in enumerate(n_estimators):
        #rf = RandomForestClassifier(max_features='sqrt', max_depth=m, random_state=698)
        rf = RandomForestClassifier(max_features='sqrt', n_estimators=n, random_state=698)
        rf.fit(X_train, y_train)                       # Fit the classifier to the training data
        train_accuracy[i] = rf.score(X_train, y_train) # Compute accuracy on the training set
        test_accuracy[i] = rf.score(X_test, y_test)    # Compute accuracy on the testing set

    rf_opt = pd.DataFrame({'n_estimators':n_estimators, 'train_acc':train_accuracy, 'test_acc':test_accuracy})

    plt.plot(n_estimators, train_accuracy, marker='o', color='blue', linewidth=1)
    plt.plot(n_estimators, test_accuracy, marker='o', color='olive', linewidth=1)

# +
rf = RandomForestClassifier(max_features='sqrt', n_estimators=75, max_depth=30, random_state=698).fit(X_train, y_train)

y_pred_simplerf = rf.predict(X_test)
y_pred_prob_simplerf = rf.predict_proba(X_test)[:,1]
print(confusion_matrix(y_test, y_pred_simplerf))
print(classification_report(y_test, y_pred_simplerf, target_names=['0','1']))
print(get_metrics('RF Simple Tuning', rf))

models = add_model(get_metrics('RF Simple Tuning', rf))
# -

# ## Random Forest Cross Validation
#
# Tuned Decision Tree Parameters: {'n_estimators': 75, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 50, 'bootstrap': False}

if tune:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 75, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 3, 
                                   verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    print("Tuned Decision Tree Parameters: {}".format(rf_random.best_params_))
    print("Best score is {}".format(rf_random.best_score_))

    rf_tuned = rf_random.best_estimator_
else:
    rf_tuned = RandomForestClassifier(n_estimators=75, min_samples_split=2, min_samples_leaf=2,  
                                      max_features='auto', max_depth=50, 
                                      bootstrap=False, random_state=42).fit(X_train, y_train)
print(rf_tuned)

# +
y_pred_rf_tuned = rf_tuned.predict(X_test)
y_pred_prob_rf_tuned = rf_tuned.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred_rf_tuned))
print(classification_report(y_test, y_pred_rf_tuned, target_names=['0','1']))
print(get_metrics('RF CV Tuning', rf_tuned))

models = add_model(get_metrics('RF CV Tuning', rf_tuned))
# -

# variable importance
importances = rf_tuned.feature_importances_
sorted_top_imps = sorted(zip(importance, X_train.columns), reverse=True)[:10]
sorted_top_imps

# +

#plt.barh(sorted_top_imps)
plt.scatter(*zip(*sorted_top_imps))
plt.title('Variable Importance')
plt.show()

# -

# ## Bagging

# +
base = dt_tuned
bag = BaggingClassifier(base_estimator=base, n_estimators=30, random_state=42)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_test)
y_pred_prob_bag = bag.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred_bag))
print(classification_report(y_test, y_pred_bag, target_names=['0','1']))
print(get_metrics('DT Bagging', bag))

models = add_model(get_metrics('DT Bagging', bag))
# -

if tune:    
    n_estimators = np.arange(5,50,5)
    train_accuracy = np.empty(len(n_estimators))
    test_accuracy = np.empty(len(n_estimators))

    for i, n in enumerate(n_estimators):
        #rf = RandomForestClassifier(max_features='sqrt', max_depth=m, random_state=698)
        bdt = BaggingClassifier(base_estimator=base, n_estimators=n, random_state=42)
        bdt.fit(X_train, y_train)                       # Fit the classifier to the training data
        train_accuracy[i] = bdt.score(X_train, y_train) # Compute accuracy on the training set
        test_accuracy[i] = bdt.score(X_test, y_test)    # Compute accuracy on the testing set

    bdt_opt = pd.DataFrame({'n_estimators':n_estimators, 'train_acc':train_accuracy, 'test_acc':test_accuracy})

    plt.plot(n_estimators, train_accuracy, marker='o', color='blue', linewidth=1)
    plt.plot(n_estimators, test_accuracy, marker='o', color='olive', linewidth=1)
    
    bdt_opt

# ## Boosting

X_train_se = pd.read_csv("../input/x-train/x_train.csv")
X_valid = pd.read_csv("../input/data-se/x_valid.csv")
y_train_se = pd.read_csv("../input/data-se/y_train.csv")
y_valid = pd.read_csv("../input/data-se/y_valid.csv")

# +
from lightgbm import LGBMClassifier
import optuna.integration.lightgbm as lgb

optuna=False

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
    dval = lgb.Dataset(X_test, label=y_test)

    boost = lgb.train(params, dtrain, valid_sets=[dval], verbose_eval=100, early_stopping_rounds=100, )
    params = boost.params
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

 
# -

params['num_iterations'] = 2000
params['learning_rate'] = 0.01
boost = LGBMClassifier(**params).fit(
    X_train,
    y_train,
    verbose=0
)

# +
y_pred_boost = boost.predict(X_test)
y_pred_prob_boost = boost.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred_boost))
print(classification_report(y_test, y_pred_boost, target_names=['0','1']))
print(get_metrics('DT Gradient Boosting', boost))

models = add_model(get_metrics('DT Gradient Boosting', boost))
# -

# ## Results

models

# RF CV Tuning best overall classifier
models.style.background_gradient(cmap='Blues', low=0.7, high=0.8)

# +
classifiers = [dt_best_depth, dt_tuned, log_reg, log_reg_l1, log_reg_l2, rf, rf_tuned, bag, boost]
plt.figure(figsize=(12, 8))
plt.plot([0,1],[0,1], 'k--')
plt.title('ROC Curves')
plt.legend(prop={'size': 12})
ax = plt.gca()

for (n,c) in zip(models['name'], classifiers):
    if isinstance(c, LogisticRegression):
        Xtest = X_test_scaled
    else:
        Xtest = X_test

    plot_roc_curve(c, Xtest, y_test, ax=ax, name=n)

# +
# zoomed in view
classifiers = [dt_best_depth, dt_tuned, log_reg, log_reg_l1, log_reg_l2, rf, rf_tuned, bag, boost]
plt.figure(figsize=(12, 8))
plt.plot([0,1],[0,1], 'k--')
plt.title('ROC Curves')
ax = plt.gca()
ax.set_xlim(0, 0.3)
ax.set_ylim(0.6, 1)

for (n,c) in zip(models['name'], classifiers):
    if isinstance(c, LogisticRegression):
        Xtest = X_test_scaled
    else:
        Xtest = X_test

    plot_roc_curve(c, Xtest, y_test, ax=ax, name=n)
