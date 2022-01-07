#!/usr/bin/env python
# coding: utf-8

# ### Importing Required Libraries

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date

import random
import pickle

import statistics
from scipy import stats
from statsmodels.stats import weightstats as stests
from scipy.stats import shapiro
from statsmodels.stats import power

import category_encoders as ce
from category_encoders import TargetEncoder, OneHotEncoder 

from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
from sklearn.metrics import log_loss, confusion_matrix, classification_report, cohen_kappa_score, accuracy_score, f1_score, roc_curve, roc_auc_score 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
import pydotplus

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold, cross_val_score

# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None


from sklearn.utils import resample

from wordcloud import WordCloud
from wordcloud import STOPWORDS

import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from nltk.stem.porter import PorterStemmer

import string
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


# ### Reading the file - 2021VAERSSYMPTOMS

# In[3]:


df_symp = pd.read_csv('2021VAERSSYMPTOMS.csv')


# ### Reading the file - 2021VAERSVAX

# In[4]:


df_vax = pd.read_csv('2021VAERSVAX.csv',encoding_errors='ignore')


# ### Reading the file - 2021VAERSDATA

# In[5]:


df_data= pd.read_csv('2021VAERSDATA.csv', encoding_errors='ignore')


# ### Analysis on the reported events from 2016 - 2021

# In[6]:


df_data['VAX_DATE']= pd.to_datetime(df_data['VAX_DATE'], format='%m/%d/%Y')
df_data['ONSET_DATE']= pd.to_datetime(df_data['ONSET_DATE'], format='%m/%d/%Y')


# In[7]:


df_6yrs_data = df_data.loc[(df_data['VAX_DATE'] >= '2016-01-01')]


# ### Merging 2021VAERSSYMPTOMS and 2021VAERSVAX

# In[8]:


df_symvax = pd.merge(df_symp, df_vax, on='VAERS_ID', how='inner', indicator= False)


# ### Merging with 2021VAERSDATA to get the final merged data

# In[9]:


df_merged = pd.merge(df_symvax, df_6yrs_data, on='VAERS_ID', how= 'inner', indicator= False) 


# ### Finding and Removing Duplicate Records

# In[10]:


# Dropping duplicate records keeping the first record: 

df_new = df_merged.drop_duplicates(subset= ['VAERS_ID'], keep= 'first')


# ### Count and List of Numerical & Categorical Variables

# In[11]:


cat = []
num = []

for i in df_new.columns:
    if df_new[i].dtype==object:
        cat.append(i)
    else:
        num.append(i)
        
print("The number of numerical features are:",len(num))
print()
print("The numerical features are:\n\n",num)
print("\n\n")
print("The number of categorical features are:", len(cat)) 
print()
print("The categorical features are:\n\n",cat)


# ### Setting the 'VAERS_ID' which is an unique event ID as index

# In[12]:


df_new.set_index('VAERS_ID', inplace= True)


# ### Dropping redundant columns

# In[13]:


# Dropping redundant columns: which have unique values / those which are not required for the analysis

# 'VAX_LOT', 'VAX_ROUTE', 'VAX_SITE', 'VAX_NAME', 
# 'SYMPTOMVERSION1', 'SYMPTOMVERSION2', 'SYMPTOMVERSION3', 'SYMPTOMVERSION4', 'SYMPTOMVERSION5', 
# 'RECVDATE', 'RPT_DATE', 'LAB_DATA', 'V_FUNDBY', 'SPLTTYPE', 'FORM_VERS', 'CAGE_MO', 'DATEDIED', 'HOSPDAYS', 'TODAYS_DATE', 'OTHER_MEDS', 'RECOVD'


# In[14]:


df_v1 = df_new.drop(columns=['VAX_LOT', 'VAX_ROUTE', 'VAX_SITE', 'VAX_NAME', 'SYMPTOMVERSION1', 'SYMPTOMVERSION2', 'SYMPTOMVERSION3', 'SYMPTOMVERSION4', 'SYMPTOMVERSION5', 'RECVDATE', 'RPT_DATE', 'LAB_DATA', 'V_FUNDBY', 'SPLTTYPE', 'FORM_VERS', 'CAGE_MO', 'DATEDIED', 'HOSPDAYS', 'TODAYS_DATE', 'OTHER_MEDS', 'RECOVD']) 


# ### Dropping redundant records from 'VAX_TYPE' and 'VAX_MANU'

# In[15]:


# dropping rows with unknown vaccine type - UNK 

df_vax_type_unk = df_v1[df_v1['VAX_TYPE'] == 'UNK'].index
df_v1.drop(df_vax_type_unk, inplace=True)


# In[16]:


# dropping rows with unknown vaccine manufacturers - UNK 

df_vax_manu_unk = df_v1[df_v1['VAX_MANU'] == 'UNKNOWN MANUFACTURER'].index
df_v1.drop(df_vax_manu_unk, inplace=True)


# ### Combining ER_VISIT & ER_ED_VISIT variables

# In[17]:


# Since the variables - 'ER_ED_VISIT' and 'ER_VISIT' represent same data, we are grouping the data into a single variable. 

# Replacing null values in 'ER_ED_VISIT' with values in 'ER_VISIT':

df_v1['ER_ED_VISIT'] = df_v1['ER_ED_VISIT'].fillna(df_v1['ER_VISIT'])


# In[18]:


# dropping the column - 'ER_VISIT' 

df_v1 = df_v1.drop('ER_VISIT', axis= 1)


# ### Replacing Null values as per VAERS DataUserGuide

# In[19]:


# Died - Replace 'nan' with 'N'
df_v1['DIED'] = df_v1['DIED'].replace({np.nan: 'N'})

# L_Threat - Replace 'nan' with 'N'
df_v1['L_THREAT'] = df_v1['L_THREAT'].replace({np.nan: 'N'})

# ER_ED_Visit - Replace 'nan' with 'N'
df_v1['ER_ED_VISIT'] = df_v1['ER_ED_VISIT'].replace({np.nan: 'N'}) 

# Ofc_Visit - Replace 'nan' with 'N'
df_v1['OFC_VISIT'] = df_v1['OFC_VISIT'].replace({np.nan: 'N'}) 

# Hospital - Replace 'nan' with 'N'
df_v1['HOSPITAL'] = df_v1['HOSPITAL'].replace({np.nan: 'N'}) 

# X_Stay - Replace 'nan' with 'N'
df_v1['X_STAY'] = df_v1['X_STAY'].replace({np.nan: 'N'})

# Disable - Replace 'nan' with 'N' 
df_v1['DISABLE'] = df_v1['DISABLE'].replace({np.nan: 'N'}) 

# Birth_Defect - Replace 'nan' with 'N'
df_v1['BIRTH_DEFECT'] = df_v1['BIRTH_DEFECT'].replace({np.nan: 'N'}) 


# ### Feature Engineering - Creating Target Column

# In[20]:


df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'N') & (df_v1['ER_ED_VISIT'] == 'N') & (df_v1['HOSPITAL'] == 'N') & (df_v1['X_STAY'] == 'N') & (df_v1['OFC_VISIT'] == 'N') & (df_v1['DISABLE'] == 'N') & (df_v1['BIRTH_DEFECT'] == 'Y')), 'ADVERSE_EFFECT'] = 'Birth_Defect'

df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'N') & (df_v1['ER_ED_VISIT'] == 'N') & (df_v1['HOSPITAL'] == 'N') & (df_v1['X_STAY'] == 'N') & (df_v1['OFC_VISIT'] == 'N') & (df_v1['DISABLE'] == 'Y')), 'ADVERSE_EFFECT'] = 'Disabled' 

df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'N') & (df_v1['ER_ED_VISIT'] == 'N') & (df_v1['HOSPITAL'] == 'N') & (df_v1['X_STAY'] == 'N') & (df_v1['OFC_VISIT'] == 'Y')), 'ADVERSE_EFFECT'] = 'Clinic_Visit' 

df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'N') & (df_v1['ER_ED_VISIT'] == 'N') & (df_v1['X_STAY'] == 'N')  & (df_v1['HOSPITAL'] == 'Y')), 'ADVERSE_EFFECT'] = 'Hospitalized' 

df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'N') & (df_v1['ER_ED_VISIT'] == 'N') & (df_v1['X_STAY'] == 'Y')), 'ADVERSE_EFFECT'] = 'Prolonged_Hospitalization' 

df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'N') & (df_v1['ER_ED_VISIT'] == 'Y')), 'ADVERSE_EFFECT'] = 'ER_Visit' 

df_v1.loc[((df_v1['DIED'] == 'N') & (df_v1['L_THREAT'] == 'Y')), 'ADVERSE_EFFECT'] = 'Life_Threat' 

df_v1.loc[((df_v1['DIED'] == 'Y')), 'ADVERSE_EFFECT'] = 'Died' 


# In[21]:


df_v1['ADVERSE_EFFECT'] = df_v1['ADVERSE_EFFECT'].fillna('No_Adverse_Effect')


# ### Concatenating Symptoms

# In[22]:


# Replacing null values:

df_v1['SYMPTOM2'] = df_v1['SYMPTOM2'].replace({np.nan: 'Not_Applicable'}) 
df_v1['SYMPTOM3'] = df_v1['SYMPTOM3'].replace({np.nan: 'Not_Applicable'}) 
df_v1['SYMPTOM4'] = df_v1['SYMPTOM4'].replace({np.nan: 'Not_Applicable'}) 
df_v1['SYMPTOM5'] = df_v1['SYMPTOM5'].replace({np.nan: 'Not_Applicable'}) 
df_v1['SYMPTOM_TEXT'] = df_v1['SYMPTOM_TEXT'].replace({np.nan: 'Not_Applicable'})


# In[23]:


# Concatenating symptoms: 

df_v1['SYMPTOMS POST VACCINATION'] = df_v1['SYMPTOM1'] + ", " + df_v1['SYMPTOM2'] + ", " + df_v1['SYMPTOM3'] + ", " + df_v1['SYMPTOM4'] + ", " + df_v1['SYMPTOM5'] + ", " + df_v1['SYMPTOM_TEXT'] 


# In[24]:


# Dropping the individual columns: 

df_v2 = df_v1.drop(columns=['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5', 'SYMPTOM_TEXT']) 


# In[25]:


# Dropping the individual columns from which target feature was engineered: 

df_v3 = df_v2.drop(columns=['DIED', 'L_THREAT', 'HOSPITAL', 'DISABLE', 'OFC_VISIT', 'ER_ED_VISIT', 'X_STAY', 'BIRTH_DEFECT']) 


# In[26]:


# Dropping columns - 'CUR_ILL', 'HISTORY', 'ALLERGIES':

df_v4 = df_v3.drop(columns= ['CUR_ILL', 'HISTORY', 'ALLERGIES', 'PRIOR_VAX'])


# ### Replacing missing values in 'AGE_YRS' with corresponding values in 'CAGE_YR'

# In[27]:


df_v4['AGE_YRS'] = df_v4['AGE_YRS'].fillna(df_v4['CAGE_YR'])


# In[28]:


# Dropping 'CAGE_YR' column:

df_v6 = df_v4.drop(columns= ['CAGE_YR'])


# ### Replacing ONSET_DATE lesser than VAX_DATE with VAX_DATE

# In[29]:


df_v6.loc[(df_v6['ONSET_DATE'] < df_v6['VAX_DATE']), 'ONSET_DATE'] = df_v6['VAX_DATE']


# ### Imputing Missing Values in 'NUMDAYS' based on 'VAX_DATE' & 'ONSET_DATE'

# In[30]:


df_v6['NUMDAYS'] = (df_v6['ONSET_DATE'] - df_v6['VAX_DATE']) / np.timedelta64(1, 'D')


# ### Pre-Processing of variable - 'NUMDAYS'

# In[31]:


# Renaming a column - 'NUMDAYS': 

df_v6.rename(columns = {'NUMDAYS': 'NUMDAYS BETWEEN VAX_DATE & ONSET_DATE'}, inplace = True)


# In[32]:


df_v6 = df_v6[~(df_v6['NUMDAYS BETWEEN VAX_DATE & ONSET_DATE']>365)]


# In[33]:


# Dropping 'VAX_DATE' & 'ONSET_DATE' columns as we consider only the 'NUMDAYS BETWEEN VAX_DATE & ONSET_DATE' for analysis:

df_v7 = df_v6.drop(columns= ['VAX_DATE', 'ONSET_DATE'])


# In[34]:


# Since the percentage of missing rows is less than 5% in the columns, Drop by row approach is done: 

df_pros = df_v7.dropna()


# In[35]:


print("The percentage of data reduction is:", ((len(df_v7) - len(df_pros)) / len(df_v7)) * 100,"\n")
print("The percentage of data remaining is:", 100 - (((len(df_v7) - len(df_pros)) / len(df_v7)) * 100))


# ### Analysis of variable - 'AGE_YRS'
AGE_YRS denotes the age of the patient. We form an additional column by grouping the AGE_YRS as follows:

0 < AGE_YRS <= 12   -- 'Child'
12 < AGE_YRS <= 18  -- 'Adolescents'
18 < AGE_YRS <= 30  -- 'Young_Adult'
30 < AGE_YRS <= 59  -- 'Senior_Adult'
AGE_YRS > 59        -- 'Senior_Citizen'
# In[36]:


df_pros.loc[((df_pros['AGE_YRS'] >= 0) & (df_pros['AGE_YRS'] <= 12)), 'AGE_GROUP'] = 'Child'
df_pros.loc[((df_pros['AGE_YRS'] > 12) & (df_pros['AGE_YRS'] <= 18)), 'AGE_GROUP'] = 'Adolescent'
df_pros.loc[((df_pros['AGE_YRS'] > 18) & (df_pros['AGE_YRS'] <= 30)), 'AGE_GROUP'] = 'Young_Adult'
df_pros.loc[((df_pros['AGE_YRS'] > 30) & (df_pros['AGE_YRS'] <= 59)), 'AGE_GROUP'] = 'Senior_Adult'
df_pros.loc[df_pros['AGE_YRS'] > 59, 'AGE_GROUP'] = 'Senior_Citizen'


# In[37]:


df_f1 = df_pros.drop(['AGE_YRS'], axis= 1)


# In[38]:


# changing column order: 

df_final = df_f1.iloc[:, [0,1,2,3,4,9,6,8,5,7]]


# ### Class Imbalance Treatment

# #### Down-sample Majority Classes

# In[39]:


# Separate majority and minority classes
df_minority = df_final.loc[(df_final['ADVERSE_EFFECT'] != 'No_Adverse_Effect') & (df_final['ADVERSE_EFFECT'] != 'Clinic_Visit')]

cls = ['No_Adverse_Effect', 'Clinic_Visit']

for i in cls:
    df_majority = df_final[df_final['ADVERSE_EFFECT'] == i]
 
    # Upsample minority class
    df_majority_downsampled = resample(df_majority, 
                                       replace= True,     # sample with replacement
                                       n_samples= 60752,    # to match majority class
                                       random_state= 1) # reproducible results
 
    # Combine majority class with upsampled minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    df_minority = df_downsampled


# #### Up-sample Minority Classes

# In[40]:


# Separate majority and minority classes
df_majority = df_downsampled.loc[(df_downsampled['ADVERSE_EFFECT'] == 'No_Adverse_Effect') | (df_downsampled['ADVERSE_EFFECT'] == 'Clinic_Visit')]

cls = ['ER_Visit', 'Hospitalized', 'Life_Threat', 'Died', 'Disabled', 'Prolonged_Hospitalization', 'Birth_Defect']

for i in cls:
    df_minority = df_downsampled[df_downsampled['ADVERSE_EFFECT'] == i]
 
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace= True,     # sample with replacement
                                     n_samples= 60752,    # to match majority class
                                     random_state= 1) # reproducible results
 
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    df_majority = df_upsampled


# #### Analysis after Class Imbalance Treatment

# In[41]:


df_treated = df_upsampled.copy()


# In[42]:


df_treated.head()


# ### Target Encoding of Independent Categorical Variables

# In[43]:


enc= ce.OneHotEncoder().fit(df_treated['ADVERSE_EFFECT'])
y_onehot= enc.transform(df_treated['ADVERSE_EFFECT'])
y_onehot.head(10)


# In[44]:


for k in df_treated.drop(['ADVERSE_EFFECT', 'SYMPTOMS POST VACCINATION'], axis= 1).columns: 
        # target encoding all independent variables except 'Symptoms' feature as it needs to be processed using NLP
        for i in y_onehot.columns: 
            encoder = ce.TargetEncoder()
            df_treated[k] = encoder.fit_transform(df_treated[k], y_onehot[i])


# In[45]:


df_treated['VAX_TYPE_orig'] = df_upsampled['VAX_TYPE']
df_treated['VAX_MANU_orig'] = df_upsampled['VAX_MANU'] 
df_treated['VAX_DOSE_SERIES_orig'] = df_upsampled['VAX_DOSE_SERIES'] 
df_treated['STATE_orig'] = df_upsampled['STATE'] 
df_treated['SEX_orig'] = df_upsampled['SEX'] 
df_treated['AGE_GROUP_orig'] = df_upsampled['AGE_GROUP'] 
df_treated['V_ADMINBY_orig'] = df_upsampled['V_ADMINBY']


# In[46]:


orig_col = ['VAX_TYPE_orig', 'VAX_MANU_orig', 'VAX_DOSE_SERIES_orig', 'STATE_orig', 'SEX_orig', 'AGE_GROUP_orig', 'V_ADMINBY_orig'] 

enc_col = ['VAX_TYPE', 'VAX_MANU', 'VAX_DOSE_SERIES', 'STATE', 'SEX', 'AGE_GROUP', 'V_ADMINBY']

temp_dict = {}

for subcol,col in zip(enc_col, orig_col): 
    x = df_treated[[col, subcol]].value_counts().reset_index()
    temp_dict[subcol] = dict(zip(x[col], x[subcol]))
    
    
print(temp_dict)


# ### Label Encoding the Target Variable

# In[47]:


from sklearn.preprocessing import OrdinalEncoder 
ode = OrdinalEncoder(categories= [['No_Adverse_Effect', 'Birth_Defect', 'Disabled', 'Clinic_Visit', 'Hospitalized', 'Prolonged_Hospitalization', 'ER_Visit', 'Life_Threat', 'Died']]) 
df_treated['ADVERSE_EFFECT'] = ode.fit_transform(df_treated['ADVERSE_EFFECT'].values.reshape(-1,1))


# In[48]:


df_treated.head()


# ### NLP - 'SYMPTOMS POST VACCINATION' variable

# In[49]:


def clean_punc(word):
    cleaned = re.sub(r'[?|!|\'|#|]', r'', word)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r'', cleaned)
    return cleaned

final_string = [] 
s = ''

def text_preprocess(text_data): 
    vect = CountVectorizer(stop_words='english')
    stop = list(vect.get_stop_words())
    
    global final_string
    global s
    
    for i, sentence in text_data.iteritems():
        filtered_sentence = []
        for word in sentence.split():
            for cleaned_word in clean_punc(word).split():
                if (cleaned_word.isalpha() and (len(cleaned_word) > 1) and cleaned_word not in stop):
                    s = cleaned_word.lower()
                    filtered_sentence.append(s)
                else:
                    continue
                
        strl = ""' '.join(filtered_sentence)
        wordnet = WordNetLemmatizer()
        final_string.append("".join([wordnet.lemmatize(word) for word in strl]))
        
    
    return final_string


# In[50]:


df_treated['PROCESSED_SYMPTOMS'] = df_treated[['SYMPTOMS POST VACCINATION']].apply(text_preprocess)


# #### Tokenization & Vectorization of Text Column

# In[51]:


X_sym = df_treated[['PROCESSED_SYMPTOMS']] 
y_sym = df_treated['ADVERSE_EFFECT']


# #### Build Document-term Matrix (DTM) - 1000 features

# In[52]:


# import and instantiate CountVectorizer (with max 1000 features)
cvect = CountVectorizer(max_features= 1000)


# In[53]:


#Feed SMS data to CountVectorizer
cvect.fit(X_sym['PROCESSED_SYMPTOMS'])


# In[54]:


#Convert Training SMS messages into Count Vectors
X_ct = cvect.transform(X_sym['PROCESSED_SYMPTOMS'])


# In[55]:


X_ct_df = pd.DataFrame(X_ct.toarray(), index= X_sym.index)


# ### Combining processed text feature and other features

# In[56]:


y = df_treated['ADVERSE_EFFECT']
X = df_treated.drop(['ADVERSE_EFFECT', 'SYMPTOMS POST VACCINATION', 'PROCESSED_SYMPTOMS', 'VAX_TYPE_orig', 'VAX_MANU_orig', 'VAX_DOSE_SERIES_orig', 'STATE_orig', 'SEX_orig', 'AGE_GROUP_orig', 'V_ADMINBY_orig'], axis= 1)


# In[57]:


# Scaling only the 'NUMDAY BETWEEN VAXDATE & ONSETDATE' Variable as the range of values in it is high:

mm = MinMaxScaler() 
mm_arr = mm.fit_transform(X[['NUMDAYS BETWEEN VAX_DATE & ONSET_DATE']]) 
X['NUMDAYS BETWEEN VAX_DATE & ONSET_DATE'] = mm_arr


# In[58]:


X_full = pd.concat([X, X_ct_df], axis= 1)


# In[59]:


X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y, random_state= 10, test_size= 0.3)

print("X_train_full: ", X_train_full.shape)
print("X_test_full: ", X_test_full.shape)
print("y_train:", y_train_full.shape)
print("y_test:", y_test_full.shape)


# ### Analysis using Ensemble Models

# #### Random Forest Classifier

# In[60]:


rf = RandomForestClassifier(random_state= 10) 
rf.fit(X_train_full, y_train_full)


# In[61]:


pickle.dump(rf, open('model.pkl', 'wb'))


# In[62]:


model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:




