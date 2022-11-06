#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
from sklearn.metrics import confusion_matrix as CM 
from sklearn import model_selection
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
import bisect
from scipy.stats import mstats


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid")
input_path = './'
onput_path = input_path
higgs = pd.read_csv(input_path+'higgs.csv.gz')


# In[42]:


higgs.describe()


# In[43]:


higgs0 = higgs.drop(['KaggleSet', 'KaggleWeight', 'EventId'], axis=1)
higgs0['Label'] = higgs0['Label'].replace('b',0)
higgs1 = higgs0.replace(-999,0)
higgs1['Label']= higgs1['Label'].replace('s',1)
higgs1.head()


# In[44]:


sns.heatmap(higgs1.corr())


# In[45]:


higgs1.plot(kind='scatter', x='DER_mass_MMC',y='DER_mass_vis')


# In[46]:


higgs1.plot(kind='scatter', x='DER_mass_jet_jet',y='DER_prodeta_jet_jet')


# In[47]:


#-------------------------------------------
# Separate the predictors (signals) from the response
#-------------------------------------------
signals = higgs1[[c for c in higgs1.columns if c != 'Label']]
responses = higgs1['Label']
#-------------------------------------------

#-------------------------------------------
# Performing standardization and PCA
#-------------------------------------------
var_threshold = 0.98 # minimum percentage of variance we want to be described by the resulting transformed components
pca_obj = PCA(n_components=var_threshold) # Create PCA object
signals_Transformed = pca_obj.fit_transform(StandardScaler().fit_transform(signals)) # Transform the initial features
columns = ['comp_' + str(n) for n in range(1,signals_Transformed.shape[1]+1)] #create a list of columns
transf_signals_df = pd.DataFrame(signals_Transformed, columns=columns) # Create a data frame from the PCA'd data
transf_input_df = transf_signals_df.copy()
transf_input_df['Label'] = responses #create a full dataframe (including the response) out of the transformed features
#-------------------------------------------
print (signals_Transformed.shape)


# In[48]:


sns.heatmap(transf_signals_df.corr())


# In[49]:


n_positive = len(responses[responses==1])
resampled_df = transf_input_df[transf_input_df.Label==1].copy()
df = transf_input_df[transf_input_df.Label==0].sample(n_positive, replace=False)
resampled_df = resampled_df.append(df, ignore_index=True)

#---------------------------
# Resample (undersampling to allow speedy computation)
#---------------------------

resampled_signals_df = resampled_df[[c for c in resampled_df.columns if c != 'Label']].values
resampled_response_sr = resampled_df['Label'].values


# In[58]:


test_size_rt = 0.2
train_signals, test_signals, train_labels, test_labels = train_test_split(resampled_signals_df, resampled_response_sr, test_size=test_size_rt)

#---------------------------
# Setting the classifier
#---------------------------
random_state = np.random.RandomState(0) # Define a random state
Classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)


# In[61]:


Classifier.fit(train_signals, train_labels) #fitting the classifier
predicted_responses = Classifier.predict(test_signals) #applying predictions
precision = precision_score(test_labels, predicted_responses, average='binary')
recall = recall_score(test_labels, predicted_responses, average='binary')
print ('Precision = ', '{:.2f}'.format(precision))
print ('Recall = ', '{:.2}'.format(recall))


# In[62]:


from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
RF_display = RocCurveDisplay.from_estimator(Classifier, resampled_signals_df,resampled_response_sr, ax=ax, alpha=0.8)


# In[63]:


Classifier1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, criterion='squared_error')
Classifier1.fit(train_signals, train_labels) #fitting the classifier
predicted_responses_1 = Classifier1.predict(test_signals) #applying predictions
precision_1 = precision_score(test_labels, predicted_responses_1, average='binary')
recall_1 = recall_score(test_labels, predicted_responses_1, average='binary')
print ('Precision = ', '{:.2f}'.format(precision))
print ('Recall = ', '{:.2}'.format(recall))


# In[64]:


from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
GB_display = RocCurveDisplay.from_estimator(Classifier1, resampled_signals_df,resampled_response_sr, ax=ax, alpha=0.8)
RF_display.plot(ax=ax, alpha=0.8)
plt.show()


# In[65]:


from sklearn.ensemble import AdaBoostClassifier
Classifier2 = AdaBoostClassifier(n_estimators=100, algorithm='SAMME', learning_rate=1.0)
Classifier2.fit(train_signals, train_labels) #fitting the classifier
predicted_responses_2 = Classifier2.predict(test_signals) #applying predictions
precision_2 = precision_score(test_labels, predicted_responses_2, average='binary')
recall_2 = recall_score(test_labels, predicted_responses_2, average='binary')
print ('Precision = ', '{:.2f}'.format(precision_2))
print ('Recall = ', '{:.2}'.format(recall_2))


# In[66]:


from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
GB_display = RocCurveDisplay.from_estimator(Classifier1, resampled_signals_df,resampled_response_sr, ax=ax, alpha=0.8)
AB_display = RocCurveDisplay.from_estimator(Classifier2, resampled_signals_df,resampled_response_sr, ax=ax, alpha=0.8)
RF_display.plot(ax=ax, alpha=0.8)
plt.show()


# In[84]:


def plot_confusion_matrix(CM, labels, Norm='True', Cmap=plt.cm.Blues, Fig_counter=1, Title='Confusion Matrix'):
    if Norm == 'True':
        CM = CM.astype('float')/CM.sum(axis=0)[np.newaxis,:]
        plt.figure(Fig_counter,figsize=(7,5))
        plt.imshow(CM, interpolation='nearest', cmap=Cmap) #create the graph and set the interpolation
        plt.title(Title) #adding the title
        plt.colorbar() #additing the colorbar
        if Norm == 'True':
            plt.clim(0,1) #Set the colorbar limits    
            tick_marks = np.arange(len(labels)) #defininig the tick marks
            plt.xticks(tick_marks, labels) #apply the labels to marks
            plt.yticks(tick_marks, labels) #apply the labels to marks
            plt.ylabel('True label') #adding the y-axis title
            plt.xlabel('Predicted label') #adding the x-axis title


# In[86]:


conf_mat = CM(test_labels, predicted_responses) #building the confusion matrix
labels = np.unique(train_labels.astype(int).astype(str)).tolist() #extracting the labels
sns.set_style('ticks') 
plot_confusion_matrix(conf_mat, labels, Norm='True', Cmap=plt.cm.gray_r, Fig_counter=1, Title='Random Forest Confusion Matrix')


# In[69]:


conf_mat = CM(test_labels, predicted_responses_1) #building the confusion matrix
labels = np.unique(train_labels.astype(int).astype(str)).tolist() #extracting the labels
sns.set_style('white') #setting the plotting style
plot_confusion_matrix(conf_mat, labels, Norm='True', Cmap=plt.cm.gray_r, Fig_counter=1, Title='Gradient Boosting Confusion Matrix')


# In[70]:


conf_mat = CM(test_labels, predicted_responses_2) #building the confusion matrix
labels = np.unique(train_labels.astype(int).astype(str)).tolist() #extracting the labels
sns.set_style('white') #setting the plotting style
plot_confusion_matrix(conf_mat, labels, Norm='True', Cmap=plt.cm.gray_r, Fig_counter=1, Title='ADBoost Matrix')


# In[87]:


feat_ls = resampled_df.keys().tolist()
importance_ls = Classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in Classifier.estimators_], axis=0)
importance_ls, feat_ls, std = (list(t) for t in zip(*sorted(zip(importance_ls, feat_ls, std), reverse=True)))
plt.figure(figsize=(7,5))
plt.title("Feature importances")
plt.bar(range(len(feat_ls[0:6])), importance_ls[0:6], color='gray', yerr=std[0:6], ecolor='black', align="center")
plt.xticks(range(len(feat_ls[0:6])), feat_ls[0:6], rotation='vertical')
plt.xlim([-0.5, len(feat_ls[0:6])-0.5])
plt.savefig(onput_path + 'FeatureImportance.png', dpi=600, bbox_inches='tight')

