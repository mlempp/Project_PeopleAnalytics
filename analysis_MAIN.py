
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from  modules.numerical_plot_attrition  import  numerical_plot_attrition
from  modules.categorical_plot_attrition  import  categorical_plot_attrition
from  modules.plot_3d2  import  plot_3d2
from  modules.evaluate_model  import  evaluate_model


'''
The data originates from the IBM kaggle dataset about HR Analytics Employee Attrition & Performance
uploaded by pavansubhash
'''


# =============================================================================
# get the data and check for empty values and the data
# =============================================================================

data_DF = pd.read_csv('data/HR-Employee-Attrition.csv')




'''
EDA
'''
# =============================================================================
# 1. Check the datatypes ad if we have missing data in the dataset.
# =============================================================================
print('Which datatyps can we find?')
print(data_DF.dtypes)
print(' ')


print('Do we have empty entries?')
col_with_emp = []
for x in data_DF:
    resp1 = any(data_DF[x].isnull())
    if resp1:
        print ('%s has %d empty entries' %(x, data_DF[x].isnull().sum()))

if col_with_emp ==[]:
    print('No column has empty entries.')
    
num_values = ['int64', 'int32', 'int16', 'float64', 'float32', 'float16']
cat_values = ['object']

# =============================================================================
# 2. Transform the attrition to a numerical value (1 Yes, 0 No) and drop some features (Over18, EmployeeCount, StandardHours).
# =============================================================================
data_DF.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)

data_DF = data_DF.drop('Over18', axis=1)
data_DF = data_DF.drop('EmployeeCount', axis=1)
data_DF = data_DF.drop('StandardHours', axis=1)
data_DF = data_DF.drop('EmployeeNumber', axis=1)

num_columns = [x for x in data_DF if str(data_DF[x].dtype) in num_values]
cat_columns = [x for x in data_DF if str(data_DF[x].dtype) in cat_values]

# =============================================================================
# 3. Make exploatory plots.
# =============================================================================

#1 plot label imbalance
plt.figure()
sns.set(style="darkgrid",font_scale = 0.6)
sns.countplot(x='Attrition', data = data_DF)
perc_Att = data_DF.Attrition.value_counts()[1] / len(data_DF.Attrition) * 100
plt.title('Attrition: ' + str(round(perc_Att, 2))+ '%')

#2 plot every numerical value for the people that did leave or not
fig = plt.figure()
for i,x in enumerate(num_columns):
    ax = fig.add_subplot(4, int(np.ceil(len(num_columns)/4)), i+1)
    numerical_plot_attrition(data_DF, x, ax)
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.xlabel(x)

fig.tight_layout()
plt.legend()

#3 plot every categoical value for the people that did leave or not together with the resulting turnover in this category
fig = plt.figure()
for i,x in enumerate(cat_columns):
    ax = fig.add_subplot(4, int(np.ceil(len(cat_columns)/4)), i+1)
    categorical_plot_attrition(data_DF, x, ax)
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation =45)
    plt.xlabel(x)
# =============================================================================
# 4. Transform the categories to numericals
# =============================================================================

for x in data_DF:
    if data_DF[x].dtype == 'O':
        data_DF[x] = data_DF[x].astype('category')

data_DF_transform = data_DF.copy()
for x in data_DF_transform:
    if str(data_DF_transform[x].dtypes) == 'category':
        data_DF_transform[x]=data_DF_transform[x].cat.codes
        
# =============================================================================
# 5. Check the correlation across the different features.
# =============================================================================

corr = data_DF_transform.corr()
plt.figure()
sns.heatmap(corr,  xticklabels=corr.columns, yticklabels=corr.columns)

# =============================================================================
# 6. Check if the coocurance of different features at different levels influences the attrition
# =============================================================================
combos = [['Age','MonthlyIncome'],
          ['OverTime','MonthlyIncome'],
          ['WorkLifeBalance','RelationshipSatisfaction'],
          ['BusinessTravel','DistanceFromHome']]

for combo in combos:
    x = combo[0]
    y = combo[1]
    plot_3d2(x,y,data_DF_transform)



'''
prepare the data for the model
'''
# =============================================================================
# prepare data for model 
# =============================================================================

Y           = data_DF_transform.Attrition.values
X_features  = list(data_DF_transform.drop('Attrition', axis=1).columns)
X           = data_DF_transform.drop('Attrition', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

oversampler = SMOTE(random_state=0)
smote_X_train, smote_y_train = oversampler.fit_sample(X_train,y_train)



'''
train and optimize 3 different gradient-boosted tree based models
'''
# =============================================================================
# 1. RandomForestClassifier 
# =============================================================================
print('Model 1: RandomForestClassifier')
RF_model = RandomForestClassifier(random_state = 42, verbose = 0)

random_grid = {'n_estimators':      [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)],
               'max_features':      ['auto', 'sqrt'],
               'max_depth':         [int(x) for x in np.linspace(5, 40, num = 10)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf':  [1, 2, 4]}

RF_model_random = RandomizedSearchCV(estimator = RF_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1, scoring='roc_auc')
RF_model_random.fit(smote_X_train, smote_y_train)

RF_model_best = RF_model_random.best_estimator_

AUC_test    = evaluate_model(RF_model_best, X_test, y_test, train = False, plot = False)
AUC_train   = evaluate_model(RF_model_best, X_train, y_train, train = True, plot = False)

# =============================================================================
# 2. LightGBM
# =============================================================================
print('Model 1: LightGBM')

LGB_model = lgb.LGBMClassifier(learning_rate = 0.125, 
                               metric = 'l1', 
                               n_estimators = 100, 
                               num_leaves = 38, 
                               objective = 'binary',
                               lambda_l1 = 0.2,
                               verbosity = 0)

param_grid = {'n_estimators':   [int(x) for x in np.linspace(start = 4, stop = 20, num = 10)],
              'learning_rate':  [0.10, 0.125, 0.15, 0.175, 0.2, 0.25],
              'num_leaves':     [4,6,8],
              'max_bin':       [int(x) for x in np.linspace(start = 10, stop = 100, num = 20)]}


LGB_model_random = RandomizedSearchCV(estimator = LGB_model, param_distributions = param_grid, n_iter = 200, cv = 3, verbose=0, random_state=42, n_jobs = -1, scoring='roc_auc')
LGB_model_random.fit(smote_X_train, smote_y_train)

LGB_model_best = LGB_model_random.best_estimator_

AUC_test    = evaluate_model(LGB_model_best, X_test, y_test, train = False, plot = False)
AUC_train   = evaluate_model(LGB_model_best, X_train, y_train, train = True, plot = False)

# =============================================================================
# 3. XGbBoost
# =============================================================================
print('Model 1: XGBoost')

XGB_model = xgb.XGBClassifier(learning_rate=0.02, 
                              n_estimators=600, 
                              objective='binary:hinge',
                              silent=True, 
                              nthread=1,
                              verbosity = 0)

param_grid = {'min_child_weight':   [1, 5, 10],
              'gamma':              [0.5, 1, 1.5, 2, 5],
              'subsample':          [0.6, 0.8, 1.0],
              'colsample_bytree':   [0.6, 0.8, 1.0],
              'max_depth':          [3, 4, 5]}


XGB_model_random = RandomizedSearchCV(estimator = XGB_model, param_distributions = param_grid, n_iter = 200, cv = 3, verbose=0, random_state=42, n_jobs = -1, scoring='roc_auc')
XGB_model_random.fit(smote_X_train, smote_y_train)

XGB_model_best = XGB_model_random.best_estimator_

AUC_test    = evaluate_model(XGB_model_best, X_test, y_test, train = False, plot = False)
AUC_train   = evaluate_model(XGB_model_best, X_train, y_train, train = True, plot = False)

# =============================================================================
# 4. Bagging the estimations of te three models
# =============================================================================

rfc_pred = RF_model_best.predict(X_test)
rfc_auc = metrics.accuracy_score(y_test, rfc_pred)
rfc_f1 = metrics.f1_score(y_test, rfc_pred, average='weighted')
print('Randomforest Classifier AUC: %f F1: %f' %(rfc_auc*100, rfc_f1*100))


lgb_pred = LGB_model_best.predict(X_test)
lgb_auc = metrics.accuracy_score(y_test, lgb_pred)
lgb_f1 = metrics.f1_score(y_test, lgb_pred, average='weighted')
print('LightGBM Classifier AUC: %f F1: %f' %(lgb_auc*100, lgb_f1*100))

xgb_pred = XGB_model_best.predict(X_test)
xgb_auc = metrics.accuracy_score(y_test, xgb_pred)
xgb_f1 = metrics.f1_score(y_test, xgb_pred, average='weighted')
print('XGbBoost Classifier AUC: %f F1: %f' %(xgb_auc*100, xgb_f1*100))

bag_pred = np.mean([rfc_pred, lgb_pred, xgb_pred], axis = 0)
bag_pred = np.array([1 if x > 0.49 else 0 for x in bag_pred]) 
bag_auc = metrics.accuracy_score(y_test, bag_pred)
bag_f1 = metrics.f1_score(y_test, bag_pred, average='weighted')
print('Bagged Model Classifier AUC: %f F1: %f' %(bag_auc*100, bag_f1*100))


# =============================================================================
# 5. Plot the metrics
# =============================================================================

fig, axs = plt.subplots(1, 2)
aucs = [rfc_auc, lgb_auc, xgb_auc, bag_auc]
f1s = [rfc_f1, lgb_f1, xgb_f1, bag_f1]
axs[0].bar(['Random Forest', 'LigthGBM', 'XGBoost', 'Ensemble'], aucs)
axs[0].set_title('Accuracy')
axs[1].bar(['Random Forest', 'LigthGBM', 'XGBoost', 'Ensemble'], f1s)
axs[1].set_title('F1 score')


# =============================================================================
# 5. Plot the feature importance for each model
# =============================================================================

rfc_imp = pd.DataFrame(data = RF_model_best.feature_importances_, index = X_features)
lgb_imp = pd.DataFrame(data = LGB_model_best.feature_importances_, index = X_features)
xgb_imp = pd.DataFrame(data = XGB_model_best.feature_importances_, index = X_features)

fig, axs = plt.subplots(3, 1)
fig.suptitle('feature importance')
rfc_imp.sort_values(0, ascending = True).plot(kind = 'barh', ax = axs[0], legend = False)
axs[0].set_title('Random Forest')
lgb_imp.sort_values(0, ascending = True).plot(kind = 'barh', ax = axs[1], legend = False)
axs[1].set_title('LigthGBM')
xgb_imp.sort_values(0, ascending = True).plot(kind = 'barh', ax = axs[2], legend = False)
axs[2].set_title('XGBoost')

