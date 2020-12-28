#!/usr/bin/env python
# coding: utf-8

# ### 1. Reading Dataset

# In[1]:


#List of packages that is used in EDA and preprocessing.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


# In[2]:


#Reading the dataset
data = pd.read_excel("Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
data


# ### 2. Data Preprocessing
# Converting the data into usable format.
# 
#     a. Hot encoding categorical columns.
#     b. Marking Window 0-2 as 1 if the patient was admitted to ICU in any of the future windows.
#     c. Removing all the records of the windows in which patients were actually admitted to the ICU in Window 0-2.
#     d. Keep only first row of the patient and fill the NaN values of window 0-2 with the mean values of all windows.
#     e. Removing unnecessary columns.
#     f. Removing all the rows still having NaN values.

# In[3]:


print(data.info())


# ###### a. Hot encoding categorical columns.

# In[4]:


no_ICU_column = data.drop('ICU', axis = 1)     #seperating the target ("ICU") column
ICU_column = data['ICU']

hotcode_columns = data.select_dtypes(object).columns     #finding columns that are not of float or integer
hotcode_columns


# In[5]:


no_ICU_column = pd.get_dummies(no_ICU_column, columns = hotcode_columns)     #performing hotcoding
no_ICU_column.head()


# In[6]:


data_extend = pd.concat([no_ICU_column, ICU_column], axis = 1)         #adding the ICU column again at the last position
data_extend.head(5)


# ###### b. Marking Window 0-2 as 1 if the patient was admitted to ICU in any of the future windows.

# In[7]:


column_names = data_extend.columns
lst = data_extend.to_numpy()
print(lst)


# In[8]:


# create loop to record the rows in which patient is admitted to the ICU and adding 1 label to the previous rows.
i=0
ICU_admitted_rows = []
while(i<len(lst)):
    for j in range(5):
        if(lst[i+j][-1]==1):
            for k in range(j):
                lst[i+k][-1]=1
            for toremove in range(i+j,i+5):
                ICU_admitted_rows.append(toremove)
            break
    i+=5
print(ICU_admitted_rows)


# ###### c. Removing all the records of the windows in which patients were actually admitted to the ICU in Window 0-2.

# In[9]:


#removing the rows in which patient was admitted to the ICU
deletedcount = 0
for rowToRemove in ICU_admitted_rows:             
    lst = np.delete(lst, rowToRemove-deletedcount, axis=0)
    deletedcount+=1
df = pd.DataFrame(lst, columns = column_names)
df.head(10)


# ###### d. Keep only first row of the patient and fill the NaN values of window 0-2 with the help of mean of values in all the windows of that patient.

# In[10]:


#keeping only the first window that is 0-2 for every patient and filling NaN values with mean of all windows.
pd.options.mode.chained_assignment = None 
edited_dfs_list = []
max_patient_id = df['PATIENT_VISIT_IDENTIFIER'].max()
for i in range(int(max_patient_id)):                
    tempdf = df[df['PATIENT_VISIT_IDENTIFIER']==i]
    if(len(tempdf)!=0):
        tempdf.fillna(tempdf.mean(), inplace=True)
        tempdf = tempdf.iloc[[0]]
        edited_dfs_list.append(tempdf)

final_data = pd.concat(edited_dfs_list)
final_data.head(10)


# ###### e. Removing unnecessary columns.

# In[11]:


# Drop unnecessary columns.
final_data = final_data.drop(['GENDER','PATIENT_VISIT_IDENTIFIER','WINDOW_0-2',	'WINDOW_2-4',
                              'WINDOW_4-6','WINDOW_6-12','WINDOW_ABOVE_12'],axis = 1)
final_data.head()


# ###### f. Removing all the rows still having NaN values.

# In[12]:


# Drop remaining rows with NAN values as there is no data in any window.
final_data = final_data.dropna(axis = 0)
final_data.isnull().sum()


# In[13]:


final_data.describe()


# ### 3. Data Visualization

# ###### a. Distribution analysis

# In[14]:


# Distribution of ICU and non-ICU patients.

ICU_admission = final_data['ICU'].value_counts()

print("Total number of patient: ", sum(ICU_admission))
print("Distribution of ICU admissions")
print("Patients who were not admitted to ICU: ",ICU_admission[0])
print("Patients who were admitted to ICU: ",ICU_admission[1])

labels= ['Admitted to ICU', 'Not Admitted to ICU']
colors=['orange', 'lightgreen']
sizes= [ICU_admission[1], ICU_admission[0]]
plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')
plt.title("ICU Distribution of data")
plt.axis('equal')
plt.show()


# In[15]:


# Distribution of patient age.

Age_distr = final_data['AGE_ABOVE65'].value_counts()
print("Age Distribution")
print("Patients below age 65: ",Age_distr[0])
print("Patients above age 65: ",Age_distr[1])
labels= ['Below 65', 'Above 65']
colors=['turquoise', 'violet']
sizes= [Age_distr[0], Age_distr[1]]
plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.title("Age Distribution of data")
plt.show()


# In[16]:


# Distribution of ICU admitted patients by age.

ICU_Admitted_data = final_data[final_data['ICU']==1]
Age_distr = ICU_Admitted_data['AGE_ABOVE65'].value_counts()
print("Age Distribution")
print("Patients below age 65: ",Age_distr[0])
print("Patients above age 65: ",Age_distr[1])
labels= ['Below 65', 'Above 65']
colors=['yellow', 'tomato']
sizes= [Age_distr[0], Age_distr[1]]
plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.title("Age Distribution of ICU Admitted patients")
plt.show()


# ###### b. Correlation analysis

# In[17]:


import seaborn as sns
corr = final_data.corr()
plt.subplots(figsize=(50,50))
sns.heatmap(corr, cmap="YlGnBu")


# As we can see, it impossible to see the correlation between each variable through the heatmap as there were many variables. Thus, we will focus on those feature that have high correlation with our target variable ("ICU").

# In[18]:


# ICU correlation information
corr.shape
ICU_corr = corr.iloc[236]
ICU_corr.describe()


# In[19]:


#Choosing features with high correlation with ICU.
ICU_corr = np.array(ICU_corr)
selected = []
for i in ICU_corr:
    if(i):
        if(i>0.11):
            selected.append(True)
        elif(i<-0.12):
            selected.append(True)
        else:
            selected.append(False)
    else:
        selected.append(False)

print(len(selected), selected.count(True))
selected = np.array(selected)
selected_final_data = final_data.loc[:, selected]
selected_final_data.head()


# In[20]:


# Removing all median and relative difference columns

selected_final_data = selected_final_data[['AGE_ABOVE65', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3', 'DISEASE GROUPING 4',
                                           'HTN', 'BIC_VENOUS_MEAN', 'CALCIUM_MEAN' , 'CREATININ_MEAN', 'GLUCOSE_MEAN', 'INR_MEAN',
                                           'LACTATE_MEAN', 'LEUKOCYTES_MEAN', 'LINFOCITOS_MEAN', 'NEUTROPHILES_MEAN', 'PC02_VENOUS_MEAN',
                                           'PCR_MEAN', 'PLATELETS_MEAN', 'SAT02_VENOUS_MEAN', 'SODIUM_MEAN', 'UREA_MEAN', 'BLOODPRESSURE_DIASTOLIC_MEAN',
                                           'RESPIRATORY_RATE_MEAN', 'TEMPERATURE_MEAN', 'OXYGEN_SATURATION_MEAN', 'BLOODPRESSURE_SISTOLIC_MIN',
                                           'HEART_RATE_MIN', 'RESPIRATORY_RATE_MIN', 'TEMPERATURE_MIN', 'BLOODPRESSURE_DIASTOLIC_MAX', 'BLOODPRESSURE_SISTOLIC_MAX',
                                           'HEART_RATE_MAX', 'OXYGEN_SATURATION_MAX', 'BLOODPRESSURE_DIASTOLIC_DIFF', 'BLOODPRESSURE_SISTOLIC_DIFF', 
                                           'HEART_RATE_DIFF', 'RESPIRATORY_RATE_DIFF', 'TEMPERATURE_DIFF', 'OXYGEN_SATURATION_DIFF', 
                                           'AGE_PERCENTIL_10th', 'AGE_PERCENTIL_20th', 'AGE_PERCENTIL_80th', 'AGE_PERCENTIL_90th', 'ICU']]

selected_final_data.head()


# In[21]:


corr = selected_final_data.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corr, cmap="YlGnBu")


# ###### c. Comparing the vital signs and lab test results of ICU & non-ICU patients.

# In[22]:


Non_ICU_Admitted_data = selected_final_data[selected_final_data['ICU']==0]
ICU_Admitted_data = selected_final_data[selected_final_data['ICU']==1]


# In[23]:


Vital_Non_ICU_Admitted_data = Non_ICU_Admitted_data[['BLOODPRESSURE_DIASTOLIC_MEAN',
       'RESPIRATORY_RATE_MEAN', 'TEMPERATURE_MEAN', 'OXYGEN_SATURATION_MEAN',
       'BLOODPRESSURE_SISTOLIC_MIN', 'HEART_RATE_MIN', 'RESPIRATORY_RATE_MIN',
       'TEMPERATURE_MIN', 'BLOODPRESSURE_DIASTOLIC_MAX',
       'BLOODPRESSURE_SISTOLIC_MAX', 'HEART_RATE_MAX', 'OXYGEN_SATURATION_MAX',
       'HEART_RATE_DIFF', 'RESPIRATORY_RATE_DIFF', 'TEMPERATURE_DIFF']]

Vital_ICU_Admitted_data = ICU_Admitted_data[['BLOODPRESSURE_DIASTOLIC_MEAN',
       'RESPIRATORY_RATE_MEAN', 'TEMPERATURE_MEAN', 'OXYGEN_SATURATION_MEAN',
       'BLOODPRESSURE_SISTOLIC_MIN', 'HEART_RATE_MIN', 'RESPIRATORY_RATE_MIN',
       'TEMPERATURE_MIN', 'BLOODPRESSURE_DIASTOLIC_MAX',
       'BLOODPRESSURE_SISTOLIC_MAX', 'HEART_RATE_MAX', 'OXYGEN_SATURATION_MAX',
       'HEART_RATE_DIFF', 'RESPIRATORY_RATE_DIFF', 'TEMPERATURE_DIFF']]

Lab_Non_ICU_Admitted_data = Non_ICU_Admitted_data[['HTN', 'BIC_VENOUS_MEAN', 'CALCIUM_MEAN',
       'CREATININ_MEAN', 'GLUCOSE_MEAN', 'INR_MEAN', 'LACTATE_MEAN',
       'LEUKOCYTES_MEAN', 'LINFOCITOS_MEAN', 'NEUTROPHILES_MEAN',
       'PC02_VENOUS_MEAN', 'PCR_MEAN', 'PLATELETS_MEAN', 'SAT02_VENOUS_MEAN',
       'SODIUM_MEAN', 'UREA_MEAN']]

Lab_ICU_Admitted_data = ICU_Admitted_data[['HTN', 'BIC_VENOUS_MEAN', 'CALCIUM_MEAN',
       'CREATININ_MEAN', 'GLUCOSE_MEAN', 'INR_MEAN', 'LACTATE_MEAN',
       'LEUKOCYTES_MEAN', 'LINFOCITOS_MEAN', 'NEUTROPHILES_MEAN',
       'PC02_VENOUS_MEAN', 'PCR_MEAN', 'PLATELETS_MEAN', 'SAT02_VENOUS_MEAN',
       'SODIUM_MEAN', 'UREA_MEAN']]


# In[24]:


# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(20, 10)) 
   
vital_non_ICU = np.array(Vital_Non_ICU_Admitted_data.mean(axis=0)) 
vital_ICU = np.array(Vital_ICU_Admitted_data.mean(axis=0)) 
   
# Set position of bar on X axis 
br1 = np.arange(len(vital_ICU)) + (barWidth*0.5)
br2 = [x + barWidth for x in br1]  
   
# Make the plot 
plt.bar(br2, vital_ICU, color ='r', width = barWidth, edgecolor ='grey', label ='ICU Admitted') 
plt.bar(br1, vital_non_ICU, color ='b', width = barWidth, edgecolor ='grey', label ='NOT Admitted') 

   
plt.xlabel('Features', fontweight ='bold') 
plt.ylabel('Normalized Values', fontweight ='bold') 
plt.xticks([r + barWidth for r in range(len(vital_ICU))], ['BLOODPRESSURE_DIASTOLIC_MEAN',
       'RESPIRATORY_RATE_MEAN', 'TEMPERATURE_MEAN', 'OXYGEN_SATURATION_MEAN',
       'BLOODPRESSURE_SISTOLIC_MIN', 'HEART_RATE_MIN', 'RESPIRATORY_RATE_MIN',
       'TEMPERATURE_MIN', 'BLOODPRESSURE_DIASTOLIC_MAX',
       'BLOODPRESSURE_SISTOLIC_MAX', 'HEART_RATE_MAX', 'OXYGEN_SATURATION_MAX',
       'HEART_RATE_DIFF', 'RESPIRATORY_RATE_DIFF', 'TEMPERATURE_DIFF'], rotation = 90) 

plt.legend()
plt.title("Vital Signs of Covid19 Patients")
plt.show()


# From the bar chart above, we can see that most of the vital signs are different between the ICU admitted patients and non-ICU patients. These differences can help in better classifying the patient's condition whether he/she is needed to be admitted to ICU. Since all variable showed difference reading for ICU and non-ICU, so no variable will be removed.

# In[25]:


# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(20, 10)) 
   
lab_non_ICU = np.array(Lab_Non_ICU_Admitted_data.mean(axis=0)) 
lab_ICU = np.array(Lab_ICU_Admitted_data.mean(axis=0)) 
   
# Set position of bar on X axis 
br1 = np.arange(len(lab_ICU)) + (barWidth*0.5)
br2 = [x + barWidth for x in br1]  
   
# Make the plot 
plt.bar(br2, lab_ICU, color ='r', width = barWidth, edgecolor ='grey', label ='ICU Admitted') 
plt.bar(br1, lab_non_ICU, color ='b', width = barWidth, edgecolor ='grey', label ='NOT Admitted') 

   
plt.xlabel('Features', fontweight ='bold') 
plt.ylabel('Normalized Value', fontweight ='bold') 
plt.legend()
plt.xticks([r + barWidth for r in range(len(lab_ICU))], ['HTN', 'BIC_VENOUS_MEAN', 'CALCIUM_MEAN',
       'CREATININ_MEAN', 'GLUCOSE_MEAN', 'INR_MEAN', 'LACTATE_MEAN',
       'LEUKOCYTES_MEAN', 'LINFOCITOS_MEAN', 'NEUTROPHILES_MEAN',
       'PC02_VENOUS_MEAN', 'PCR_MEAN', 'PLATELETS_MEAN', 'SAT02_VENOUS_MEAN',
       'SODIUM_MEAN', 'UREA_MEAN'], rotation = 90) 
plt.title("Lab Test Results of Covid19 patients")
plt.show()


# Similarly for the blood test result, all the variable has different reading between ICU and non-ICU patients. So, none of the variable will be removed.

# ###### d. Dimentionality reduction

# In[26]:


#Create features set and target set
X_data = selected_final_data.drop(['ICU'], axis = 1)
Y_data = selected_final_data.ICU
print(X_data.shape)
print(Y_data.shape)


# In[27]:


#Plotting t-SNE for dimentionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 

model = TSNE(n_components = 2, random_state = 0) 
  
tsne_data = model.fit_transform(X_data) 

tsne_data = np.vstack((tsne_data.T, Y_data)).T 
tsne_df = pd.DataFrame(data = tsne_data, 
     columns =("Dim_1", "Dim_2","label"))

# Ploting the result of tsne 
sns.FacetGrid(tsne_df, hue ="label", size = 6).map( 
       plt.scatter, 'Dim_1', 'Dim_2', s = 100).add_legend() 
  
plt.show() 


# ### 4. Modelling
# 
# Machine learning algorithms chose:
# 
#     a. Gaussian Naive Bayes
#     b. Logistic Regression 
#     c. Support Vector Machine
#     d. Random Forest

# In[28]:


# Load evaluation libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn_evaluation import plot
import matplotlib.pyplot as eval_chart


# In[29]:


selected_final_data.head()


# In[30]:


selected_final_data.ICU.value_counts()


# Since the data is unbalanced, we will focus on the F1-score for model performance.

# In[31]:


# Split data to train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.20, random_state=1)


# ###### a. Gaussian Naive Bayes

# In[32]:


from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,Y_train)
y_pred=gnb.predict(X_test)


# In[33]:


print ('Accuracy:', accuracy_score(Y_test,y_pred))
print ('F1 score:', f1_score(Y_test,y_pred, pos_label=1))
print ('Recall:', recall_score(Y_test,y_pred, pos_label=1))
print ('Precision:', precision_score(Y_test,y_pred, pos_label=1))
print ('ROC:', roc_auc_score(Y_test,y_pred))

plot.confusion_matrix(Y_test,y_pred,target_names=[0,1])
eval_chart.show()

print(classification_report(Y_test,y_pred, labels = [0,1]))


# ###### b. Logistic Regression

# In[34]:


from sklearn.linear_model import LogisticRegression

lgc=LogisticRegression(random_state=0)
lgc.fit(X_train, Y_train)
y_pred1=lgc.predict(X_test)


# In[35]:


print ('Accuracy:', accuracy_score(Y_test,y_pred1))
print ('F1 score:', f1_score(Y_test,y_pred1, pos_label=1))
print ('Recall:', recall_score(Y_test,y_pred1, pos_label=1))
print ('Precision:', precision_score(Y_test,y_pred1, pos_label=1))
print ('ROC:', roc_auc_score(Y_test,y_pred1))

plot.confusion_matrix(Y_test,y_pred1,target_names=[0,1])
eval_chart.show()

print(classification_report(Y_test,y_pred1, labels = [0,1]))


# ###### c. Support Vector Machine (SVM)

# In[36]:


from sklearn import svm

svmc = svm.SVC(kernel='linear')
svmc.fit(X_train,Y_train)
y_pred2=svmc.predict(X_test)


# In[37]:


print ('Accuracy:', accuracy_score(Y_test,y_pred2))
print ('F1 score:', f1_score(Y_test,y_pred2, pos_label=1))
print ('Recall:', recall_score(Y_test,y_pred2, pos_label=1))
print ('Precision:', precision_score(Y_test,y_pred2, pos_label=1))
print ('ROC:', roc_auc_score(Y_test,y_pred2))

plot.confusion_matrix(Y_test,y_pred2,target_names=[0,1])
eval_chart.show()

print(classification_report(Y_test,y_pred2, labels = [0,1]))


# ###### d. Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(criterion='gini',random_state=23,max_depth=6,bootstrap=True)
rfc.fit(X_train,Y_train)
y_pred3=rfc.predict(X_test)


# In[40]:


print ('Accuracy:', accuracy_score(Y_test,y_pred3))
print ('F1 score:', f1_score(Y_test,y_pred3, pos_label=1))
print ('Recall:', recall_score(Y_test,y_pred3, pos_label=1))
print ('Precision:', precision_score(Y_test,y_pred3, pos_label=1))
print ('ROC:', roc_auc_score(Y_test,y_pred3))

plot.confusion_matrix(Y_test,y_pred3,target_names=[0,1])
eval_chart.show()

print(classification_report(Y_test,y_pred3, labels = [0,1]))


# In[41]:


# Features Importance
# grab feature importances from the model and feature name 
importances = rfc.feature_importances_
feature_names = X_data.columns

# sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

for i in indices:
    print(feature_names[i], ':', importances[i])


# In[44]:


estimator = rfc.estimators_[5]


# In[45]:


# Visualizing Decision Tree from Random Forest classifier
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
    
dotfile = StringIO()
export_graphviz(estimator, out_file=dotfile, feature_names=feature_names)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_png("dtree.png")


# <img src="dtree.png">

# Classifiers Performance Comparison:
# 
# <table class="table table-bordered">
#     <thead>
#         <tr>
#             <th>Classifier</th>
#             <th>Precision</th>
#             <th>F1 Score</th>
#             <th>Recall</th>
#             <th>Accuracy</th>
#             <th>ROC Curve</th>
#         </tr>
#     </thead>
#     <tbody>
#         <tr>
#             <td>Gaussian Naive Bayes</td>
#             <td>0.789</td>
#             <td>0.769</td>
#             <td>0.750</td>
#             <td>0.847</td>
#             <td>0.824</td>
#         </tr>
#         <tr>
#             <td>Logistic Regression</td>
#             <td>0.882</td>
#             <td>0.811</td>
#             <td>0.750</td>
#             <td>0.881</td>
#             <td>0.849</td>
#         </tr>
#         <tr>
#             <td>SVM</td>
#             <td>0.937</td>
#             <td>0.833</td>
#             <td>0.750</td>
#             <td>0.898</td>
#             <td>0.862</td>
#         </tr>
#         <tr>
#         <tr style="color: blue">
#             <td>Random Forest</td>
#             <td>0.904</td>
#             <td>0.927</td>
#             <td>0.950</td>
#             <td>0.949</td>
#             <td>0.949</td>
#         </tr>
#      </tbody>
# </table>
# 
# From the result above, the overall best performing classifier is Random Forest.
