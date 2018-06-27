
# coding: utf-8

# In[2]:


Team4TF = 'Shiwen (GiGi) Wang AND Zhicheng (Jason) Xue'

print('\nTeam members for x433.7 Introduction to TensorFlow Using Python are %s' %Team4TF )
Team4TFemail = 'jingjingwsw@gmail.com; emailxjason@gmail.com'
print('\nOur emails are %s' %Team4TFemail )


# ## Import all libraries needed for the project

# In[146]:


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
plotly.offline.init_notebook_mode()
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix
from sklearn import preprocessing
from tensorflow.contrib.factorization import KMeans
import os
np.random.seed(2018)


# # PART I Data Load and Preprocessing

# ## Load csv data into pandas data frame for easy processing

# In[3]:


df_full = pd.read_csv('loan.csv')


# ## Check columns 19 and 55 based on warning message

# In[4]:


column_list=df_full.columns.tolist()
print('There are', len(column_list), 'columns in the dataframe')
print('Col 19 is', column_list[19], 'and data type is', df_full[column_list[19]].dtypes)
print('Col 55 is', column_list[55], 'and data type is', df_full[column_list[55]].dtypes)


# In[5]:


print(df_full.iloc[:,19:20].head(10))


# In[6]:


print(df_full.iloc[:,55:56].head(10))


# In[7]:


print(df_full.verification_status_joint.unique())


# #### Comment: both columns are string columns with NaN used for missing values and that is reason the warning showed up during data import step

# ## Check all available data types in the full dataframe

# In[8]:


df_full.dtypes.unique()


# ### a.Check columns whose data type is int64

# In[9]:


columns_int64=df_full.select_dtypes(include='int64').columns.tolist()
print('Number of int64 columns are: ', len(columns_int64))
print('int64 columns are: ',columns_int64)


# ####  Comment: id and member_id columns will not be useful for this analysis, therefore both will be removed from the dataframe

# ### b. Check columns whose data type is float64

# In[10]:


columns_float64 = df_full.select_dtypes(include='float64').columns.tolist()
print('Number of float64 columns are: ', len(columns_float64))
print('float64 columns are: ',columns_float64)


# In[11]:


columns_float64_na_pct={}
for col in columns_float64: 
    columns_float64_na_pct[col]=np.sum(np.isnan(df_full[col]))/np.size(df_full[col])


# In[12]:


columns_float64_na_drop=[]
for key, value in columns_float64_na_pct.items():
    if columns_float64_na_pct[key]>=0.95:
        columns_float64_na_drop.append(key)
print(columns_float64_na_drop)


# In[13]:


columns_float64_keep = [x for x in columns_float64 if (x not in columns_float64_na_drop)]
print('Float64 columns to keep:',columns_float64_keep)


# #### Comment: During this step, float64 columns with high percentage of missing values were dropped as they add little value to the analysis afterwards

# ### c. Check columns whose data type is string

# In[14]:


columns_string=df_full.select_dtypes(include='O').columns.tolist()
print('Number of string columns are: ', len(columns_string))
print('string columns are: ',columns_string)


# In[15]:


columns_string_na_drop=[] # columns that should be dropped due to high % of NAs
columns_string_other_drop=[] # columns that should be dropped due to large number of distinct values or other reasons
for col in columns_string:
    if df_full[col].isnull().sum(axis=0)/np.size(df_full[col])>=0.95:
        columns_string_na_drop.append(col)
    elif (len(df_full[col].unique())>=100) | ('_d' in col):
        columns_string_other_drop.append(col)


# In[16]:


print('Columns that should be dropped due to high % of NAs:', columns_string_na_drop)
print('Columns that should be dropped due to other reasons:', columns_string_other_drop)


# In[17]:


columns_string_keep_temp = [
                        x for x in columns_string 
                       if (x not in columns_string_na_drop)&(x not in columns_string_other_drop)
                    ]


# In[18]:


print(columns_string_keep_temp)


# #### Comment: string columns with high percentage of missing values or other reasons such as large number of distinct values were dropped due to the little value AND/OR complexity they can add to the analysis afterwards

# In[19]:


for col in columns_string_keep_temp:
    print('String column name is:',col,'with number of unique values',len(df_full[col].unique()))
    print(df_full[col].isnull().sum(axis=0))
    print(df_full[col].value_counts())


# ## Generate target variable based on loan status for developing machine learning model

# In[20]:


def CreateArrays(col):
    values = df_full[col].value_counts().index.values
    counts = df_full[col].value_counts().values
    return values,counts


# In[21]:


def targetFunc(var):
    if var in np.array(['Default','Late (16-30 days)','Late (31-120 days)','Charged Off']):
        target = 1
    elif var in np.array(['Current']):
        target = 0
    else:
        target = -1
    return target

df_full['target'] = df_full['loan_status'].apply(lambda x: targetFunc(x))


# In[22]:


def targetLabelFunc(var):
    if var in np.array(['Default','Late (16-30 days)','Late (31-120 days)','Charged Off']):
        target_label = 'Default/Late/ChargeOff'
    elif var in np.array(['Current']):
        target_label = 'Current'
    else:
        target_label = 'Others'
    return target_label

df_full['target_label'] = df_full['loan_status'].apply(lambda x: targetLabelFunc(x))


# In[23]:


df_full['loan_status'].value_counts()


# In[24]:


loan_status_values, loan_status_counts = CreateArrays('loan_status')
print('loan_status_values:', loan_status_values)


# In[25]:


target_label = np.array(['Current', 'Others', 'Default/Late/ChargeOff'])
print(target_label)


# In[26]:


target_values, target_counts = CreateArrays('target')
print(target_values);print(target_counts)


# In[27]:


plt.figure('Distribution of Target')
plt.axes([0.035, 0.035, 0.9, 0.9])
c = ['green', 'orange', 'red']
e = [0, 0, 0.05]
plt.cla()
plt.pie(target_counts, explode = e, labels = target_label, 
        colors = c, radius = .75, autopct='%1.2f%%', shadow = True, startangle = 15)
plt.axis('equal')
plt.xticks(()); plt.yticks(())
plt.title('Distribution of Target')
plt.show()


# ### Comments: 2/3 of the sample are current on their loan payment and about 7% of the loans are in some type of default/late payment/charge off status. This is an unbalanced dataset for supervised training.

# #### Comment: In all of the following analysis, I will exclude "Others"

# In[28]:


df = df_full[df_full['target']!=-1].copy()


# # PART II Exploratory Data Analysis

# ## Exploratory Analysis on Numerical Variables

# In[124]:


print(columns_float64_keep)


# ### Annual income

# In[135]:


sns.distplot(df.annual_inc[df.annual_inc<df.annual_inc.quantile(0.9)], hist=True, kde=True, 
             bins=50, color = 'blue',
             hist_kws={'edgecolor':'black'})
plt.title('Histogram of Annual Income')
plt.xlabel('Income')
plt.ylabel('Borrowers')


# In[143]:


df.annual_inc.describe()


# In[148]:


bins_annual_inc = [0, 45000, 65000, 90000, 150000, 300000, 500000, 1000000]
labels_annual_inc = ['A_LT45k','B_45kTO65K', 'C_65KTO90K', 'D_90KTO150K', 'E_150KTO300K', 'F_300KTO500K', 'G_Above1M']
df['annual_inc_binned'] = pd.cut(df['annual_inc'], bins=bins_annual_inc, labels=labels_annual_inc)


# In[179]:


annual_inc_df = pd.DataFrame(df.pivot_table(values='target',index=['annual_inc_binned'],aggfunc=lambda x: len(x),dropna=True))
annual_inc_df = annual_inc_df.rename(columns={'target': 'NumberOfLoans'})
annual_inc_df.reset_index(level=0, inplace=True)
annual_inc_df.drop([7],inplace=True)
annual_inc_df = annual_inc_df.sort_values('annual_inc_binned')


# In[180]:


annual_inc_df_bad = pd.DataFrame(df.pivot_table(values='target',index=['annual_inc_binned'],aggfunc=lambda x: np.mean(x),dropna=True))
annual_inc_df_bad = annual_inc_df_bad.rename(columns={'target': 'Percentage of Bad Loans'})
annual_inc_df_bad.reset_index(level=0, inplace=True)
annual_inc_df_bad.drop([7],inplace=True)
annual_inc_df_bad = annual_inc_df_bad.sort_values('annual_inc_binned')


# In[184]:


annual_inc_df_final = pd.merge(annual_inc_df, annual_inc_df_bad,  how='left', left_on=['annual_inc_binned'], right_on = ['annual_inc_binned'])


# In[185]:


annual_inc_df_final


# ### comment: There is a negative correlation between annual income and chance of default, which matches our intuition. Higher income people are less likely to default on their loans

# ### DTI

# In[140]:


sns.distplot(df.dti[df.dti<df.dti.quantile(0.9)], hist=True, kde=True, 
             bins=50, color = 'red',
             hist_kws={'edgecolor':'black'})
plt.title('Histogram of Debt to Income Ratio')
plt.xlabel('DTI')
plt.ylabel('Borrowers')


# In[145]:


df[df.dti != 9999].dti.describe()


# ### comment: the distribution of DTI is right skewed with a long tail. Based on the distribution, the data should represent percentage of income (DTI=18.75 represents 18.75% of income is tied with debts).

# ## Exploratory Analysis on Categorical Variables

# In[29]:


def CreateArrays(df,col):
    values = df[col].value_counts().index.values
    counts = df[col].value_counts().values
    return values,counts


# In[30]:


print(columns_string_keep_temp)


# ### Term

# In[31]:


term_values0,term_counts0 = CreateArrays(df[df['target']==0],'term')
term_values1,term_counts1 = CreateArrays(df[df['target']==1],'term')


# In[32]:


plt.axes([0.075, 0.075, .88, .88])
p1 = plt.bar(term_values0,term_counts0,color='green')
p2 = plt.bar(term_values1,term_counts1,color='red')
plt.ylabel('Number of Loans')
plt.title('Number of Loans by Term and Target')
plt.legend((p1, p2), ('Current','Default/Late/ChargeOff'))

plt.show()


# ### Comment: More than half of the loans have terms 36 months rather than 60 months

# ### Grade

# In[33]:


temp = df.pivot_table(values='target',index=['grade'],aggfunc=lambda x: 100*x.mean())
print ('\nProbility of Not Current for each Credit Grade:') 
print (temp)


# In[34]:


fig = plt.figure()
temp.plot(kind = 'bar', title='Probability of Default by Credit Grade',legend=None)
plt.xlabel('Credit Grade')
plt.ylabel('% Probability of Default')
fig.tight_layout()
plt.show()
plt.clf()
plt.close()


# ### Comment: clearly "grade" will be an useful feature for predicting likelihood of default

# ## State

# In[35]:


state_list=df.addr_state.value_counts().index.values
print(state_list)


# In[36]:


state_df = pd.DataFrame(df.pivot_table(values='target',index=['addr_state'],aggfunc=lambda x: len(x)))
# print ('\nNumber of Loans for each US state:') 
state_df = state_df.rename(columns={'target': 'NumberOfLoans'})
state_df['state'] = state_df.index
state_df.reset_index(level=0, inplace=True)


# In[37]:


for col in state_df.columns:
    state_df[col] = state_df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

# state_df['text'] = state_df['state']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_df['state'],
        z = state_df['NumberOfLoans'].astype(float),
        locationmode = 'USA-states',
        text = state_df['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Number of Loans"
        )
    ) ]

layout = dict(
        title = 'Number of Lending Club loans by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict( data=data, layout=layout )

url = plotly.offline.plot( fig, filename='state-cloropleth-map.html' )


# In[38]:


get_ipython().run_cell_magic('HTML', '', '<iframe src="state-cloropleth-map.html" width=700 height=350></iframe>')


# ### Comment: majority of Lending Club's loans are concentrated in big 4 states measured by economy size- California, Texas, Florida and New York.

# ## Home ownership & Employment length

# In[39]:


home_emp_df = pd.DataFrame(
    df.pivot_table(
        values='target',index=['emp_length'], columns='home_ownership', 
#         aggfunc=lambda x: (np.mean(x)*100).astype(str) + '%'
#         aggfunc=lambda x: str(((np.round(np.mean(x),decimals=2))*100))+'%'
        aggfunc=lambda x: np.mean(x)
    )
)
print('\nBad Rate of Loans By Home Ownership and Employment Length:') 
home_emp_df.style.format({
    'Any': '{:,.2f}'.format,
    'Mortgage': '{:,.2f}'.format,
    'NONE': '{:,.2f}'.format,
    'OTHER': '{:,.2f}'.format,
    'OWN': '{:,.2f}'.format,
    'RENT': '{:,.2f}'.format,
})

home_emp_df = home_emp_df.reindex(index = [
    '< 1 year', '1 year','2 years', '3 years', '4 years','5 years',  
    '6 years', '7 years', '8 years', '9 years', '10+ years'])
home_emp_df[['RENT','OWN','MORTGAGE','OTHER','ANY','NONE']]
home_emp_df.index
# state_df = state_df.rename(columns={'target': 'NumberOfLoans'})
# home_emp_df['emp_length'] = home_emp_df.index
# home_emp_df.reset_index(level=0, inplace=True)


# In[40]:


plt.figure(figsize=(10,6), dpi=120)

plt.plot(home_emp_df.index,home_emp_df['RENT'],color='red', linestyle='--', label='Rent')
plt.plot(home_emp_df.index,home_emp_df['OWN'],color='green', linestyle='-', label='Own')
plt.plot(home_emp_df.index,home_emp_df['MORTGAGE'],color='blue', linestyle='-.', label='Mortgage')
plt.ylabel('Bad Rate')
plt.xlabel('Employment Length')

plt.legend(loc='best')
plt.title('Bad Rate By Home Ownership and Employment Length')
plt.grid()
plt.show()
plt.close()


# ### Comment: Borrowers who rent has higher risk comparing to those who own; those who have about 6 years of employment history tend to show higher risk comparing to others

# # PART III Data Preparation for Machine Learning

# ## Missing Value Processing

# In[41]:


print('Numerical variables to be considered:',columns_float64_keep)
columns_string_keep_temp.remove('loan_status')
print('Categorical variables to be considered:',columns_string_keep_temp)


# In[42]:


columns_float64_na_add_dict={}
for var in columns_float64_keep:
    if columns_float64_na_pct[var] > 0:
        columns_float64_na_add_dict[var] = var+'_na'
print(columns_float64_na_add_dict)


# In[43]:


for var in columns_string_keep_temp:
    print(var, (df[var].isnull().sum(axis=0))/np.size(df[var]))


# In[44]:


columns_string_na_add_dict={}
for var in columns_string_keep_temp:
    if df[var].isnull().sum(axis=0)/np.size(df[var])>0:
        columns_string_na_add_dict[var] = var+'_na'
print(columns_string_na_add_dict)


# ### Create a copy of DF and only keep needed columns

# In[45]:


var_list_pre = columns_float64_keep + columns_string_keep_temp
var_list_pre.append('target')


# In[47]:


df_final = df[var_list_pre].copy()


# In[49]:


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
na_dict = merge_two_dicts(columns_float64_na_add_dict, columns_string_na_add_dict)


# ### Create dummy variables for columns with NAs to prepare for later stage modeling exercise

# In[50]:


print(na_dict)


# In[51]:


for key, value in na_dict.items():
    print('key:', key, 'value:', na_dict[key])
    
    df_final[na_dict[key]] = np.where(df_final[key].isnull(),1,0)
    
    if key in columns_float64_keep:
        
        df_final[key] = np.where(df_final[key].isnull(),0,df_final[key])
    else:
        df_final[key] = np.where(df_final[key].isnull(),'MV',df_final[key])
        
    print('Transformation Done!')


# ### Mapping categorical variable values to right format with space

# In[53]:


for var in columns_string_keep_temp:
    print(var)
    df_final[var].str.strip()


# In[54]:


df_final.replace('\s+', '_',regex=True,inplace=True)


# In[56]:


df_final['term'].replace('_36_months', '36_months',regex=True,inplace=True)
df_final['term'].replace('_60_months', '60_months',regex=True,inplace=True)


# In[57]:


df_final['emp_length'].replace('<_1_year', 'LT_1_year',regex=True,inplace=True)


# In[58]:


df_final['emp_length']=np.where(df_final['emp_length']=='10+_years','GT_10_years',df_final['emp_length'])


# ## Dummy variables coding 

# In[61]:


df_final = pd.get_dummies(df_final, dummy_na=True,
               columns=[
                        'term', 
                        'grade', 
                        'sub_grade', 
                        'emp_length',
                        'home_ownership', 
                        'verification_status', 
                        'pymnt_plan', 
                        'purpose',
                        'addr_state', 
                        'initial_list_status', 
                        'application_type'                                
])


# In[62]:


features = [n for n in df_final.columns.tolist() if n != 'target']


# In[63]:


label = ['target']


# In[64]:


df_final[features].dtypes.unique()


# In[65]:


columns_string_final=df_final[features].select_dtypes(include='O').columns.tolist()
print('Number of string columns are: ', len(columns_string_final))
print('string columns are: ',columns_string_final)


# ## Split dataset into train and test dataset with stratified sampling since this is an unbalanced dataset

# In[67]:


# Split dataset into train and test dataset
train_x, test_x, train_y, test_y = train_test_split(df_final[features], df_final[label],
                                                    train_size=0.7, test_size=0.3, stratify=df_final[label],
                                                   random_state=2018)


# In[68]:


# Train and Test dataset size details
print("Train_x Shape :: ", train_x.shape)
print("Train_y Shape :: ", train_y.shape)
print("Test_x Shape :: ", test_x.shape)
print("Test_y Shape :: ", test_y.shape)


# ## Data preprocessing

# ### Standardize all features because they were on different scales

# In[69]:


scaler = preprocessing.StandardScaler().fit(train_x)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)


# # PART IV Benchmark Model Using Scikit Learn

# ## Train random forest model

# In[70]:


# param_grid = {'n_estimators': [10, 100], 'max_features': ['sqrt', 'log2']}
# CV_rf_model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring=make_scorer(accuracy_score))
# CV_rf_model.fit(train_x_scaled,train_y.values.ravel())
# print(CV_rf_model.best_params_)


# In[71]:


rf_model = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', max_features='sqrt', random_state=2018)


# In[72]:


rf_model.fit(train_x_scaled,train_y.values.ravel())


# In[73]:


y_pred = rf_model.predict(test_x_scaled)
print('Accuracy:', accuracy_score(test_y,y_pred))


# ### Feature dimension reduction using feature importance output from random forest

# In[74]:


feature_imp = pd.Series(
    rf_model.feature_importances_,
    index = train_x.columns).sort_values(ascending=False)
print(feature_imp[feature_imp.values>=0.01])


# In[75]:


# Creating a bar plot
sns.barplot(x=feature_imp[feature_imp.values>=0.01], y=feature_imp[feature_imp.values>=0.01].index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.grid()
plt.show()


# In[186]:


tf_features=feature_imp[feature_imp.values>=0.01].index.values


# In[187]:


print(tf_features.shape)


# In[188]:


train_x.columns


# In[189]:


tf_target = 'target'


# ### export train_x, test_x, train_y, test_y to prepare for TensorFlow run

# In[190]:


train_x_scaled.shape


# In[191]:


train_x.shape


# In[192]:


train_y.columns


# In[193]:


df_train_feature_xport = pd.DataFrame(train_x_scaled, columns = train_x.columns)[tf_features]
df_train_label_xport   = pd.DataFrame(train_y, columns = train_y.columns)[tf_target]
df_train_xport_all         = df_train_feature_xport.join(df_train_label_xport)

df_test_feature_xport  = pd.DataFrame(test_x_scaled, columns = test_x.columns)[tf_features]
df_test_label_xport    = pd.DataFrame(test_y, columns = test_y.columns)[tf_target]
df_test_xport_all         = df_test_feature_xport.join(df_test_label_xport)


# In[194]:


print('Shape of training data for TF:', df_train_xport_all.shape)
print('Shape of test data for TF:', df_test_xport_all.shape)


# In[195]:


df_train_xport = df_train_xport_all.sample(frac=1,random_state=2018)
df_test_xport = df_test_xport_all.sample(frac=1,random_state=2018)


# In[196]:


print('Shape of training data for TF:', df_train_xport.shape)
print('Shape of test data for TF:', df_test_xport.shape)


# # PART V TensorFlow Section

# In[198]:


tf.reset_default_graph()


# ### Define useful functions for transformations

# In[199]:


# Inference used for combining inputs
def combine_inputs(X):
    return tf.matmul(X,W)+b

# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X), name='Inference')

# define sigmoid loss function
def loss(X,Y):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y)
    )


# In[200]:


def train(total_loss,learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


# In[201]:


def evaluate(X,Y):
    predicted=tf.cast(inference(X)>0.5,tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted,Y),tf.float32))


# ### Convert Pandas dataframe to matrix

# In[203]:


df_train_feature_xport_m = df_train_feature_xport.values
df_train_label_xport_m   = df_train_label_xport.values

df_test_feature_xport_m = df_test_feature_xport.values
df_test_label_xport_m   = df_test_label_xport.values

train_num = len(df_train_label_xport_m)
test_num = len(df_test_label_xport_m)


# ### Set up constant hyperparameters for model

# In[204]:


learning_rate=0.01
training_steps=1000


# ### Run my TensorFlow Graph and compute accuracy

# In[207]:


graph = tf.Graph()
with graph.as_default():
    
    with tf.name_scope('Input'):
        
        x_train_ = tf.convert_to_tensor(df_train_feature_xport_m, dtype=tf.float32)
        y_train_ = tf.reshape(tf.convert_to_tensor(df_train_label_xport_m, dtype=tf.float32),[train_num,1])
        
        x_test_ = tf.convert_to_tensor(df_test_feature_xport_m, dtype=tf.float32)
        y_test_ = tf.reshape(tf.convert_to_tensor(df_test_label_xport_m, dtype=tf.float32),[test_num,1])
        
    with tf.name_scope('Variables'):
        # params and variables initialization
        # there are 21 features and 1 outcome
        W = tf.Variable(tf.zeros([21,1]), name='Weights') # 21 represents 21 features
        b = tf.Variable(0., name='Bias',)
        
    with tf.name_scope('Inference'):
        y = inference(x_train_)
        
    with tf.name_scope('Cross_Entropy'):
        cross_entropy = loss(x_train_,y_train_)
        
    with tf.name_scope('Train_Op') as scope:
        train_op = train(cross_entropy, learning_rate)
        
    with tf.name_scope('Evaluator') as scope:
        accuracy = evaluate(x_train_,y_train_)
        #correct_prediction is True/False boolean vector, cast converts to 1/0
    
    with tf.name_scope('Summaries'):
        summ_W = tf.summary.histogram('weights', W)
        summ_b = tf.summary.histogram('biases', b)
        summ_ce = tf.summary.scalar('cross_entropy', cross_entropy)
        summ_acc = tf.summary.scalar('accuracy', accuracy)

        summ_merged = tf.summary.merge([summ_W, summ_b, summ_ce, summ_acc])  
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./test11',sess.graph)
 
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
        
        for step in np.arange(training_steps):
            sess.run(train_op)
            
            #Finally, continuously print our progress and the final accuracy of the test data
            
            if step % 100 == 0:
                summary_str = sess.run(summ_merged)
                writer.add_summary(summary_str, step)
                print('Step =',step,' Accuracy =',sess.run(accuracy))
              
        print('Final Accuracy: ', sess.run(accuracy))
        print('done')
        

    
    coord.request_stop()
    coord.join(threads)
    
    writer.close()
    sess.close()


# ![TensorBoard](logistic.png)

# ![TensorBoard](logistic_summaries.png)

# ![TensorBoard](distributions.png)

# ![TensorBoard](histograms.png)

# ### Comment: Using full train/test and only the top 21 features with the highest feature importance from random forest model, this model in TensorFlow using sigmoid  function for cross entropy was able to reach a final accuracy of 93.58%, which is below the accuracy of the random forest model

# ## Running unsupervised K-Means on the full train/test dataset

# ### Create a new graph object

# In[209]:


tf.reset_default_graph()
graph = tf.Graph()


# ### Set parameters for graph run

# In[214]:


num_steps = 1000 # Total steps to train
k = 2 # The number of clusters
num_classes = 2 # 2 class outcomes
num_features = 21 # 9 features with highest feature importance from random forest


# ### Input features/labels(for assigning a label to a centroid and testing)

# In[211]:


X_kmeans = tf.placeholder(tf.float32, shape=[None, num_features])
Y_kmeans = tf.placeholder(tf.float32, shape=[None, num_classes])


# ### Define K-Means

# In[212]:


# K-Means Parameters
kmeans = KMeans(inputs=X_kmeans, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6: 
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)


# ### Initialize variables and sessions

# In[213]:


# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars, feed_dict={X_kmeans: df_train_feature_xport_m})
sess.run(init_op, feed_dict={X_kmeans: df_train_feature_xport_m})


# ### Actual K-means training

# In[215]:


for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X_kmeans: df_train_feature_xport_m})
    if i % 100 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += df_train_label_xport_m[i]
    
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)


# ### K-means model evaluation

# In[216]:


cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx) # Lookup: centroid_id -> label

# Compute accuracy
correct_prediction_kmeans = tf.equal(cluster_label, tf.cast(tf.argmax(Y_kmeans, 1), tf.int32))
accuracy_kmeans = tf.reduce_mean(tf.cast(correct_prediction_kmeans, tf.float32))

# Test Model
test_x_kmeans, test_y_kmeans = df_test_feature_xport_m ,pd.get_dummies(df_test_label_xport_m).values
    
print("K Means Test Accuracy:", sess.run(accuracy_kmeans, feed_dict={X_kmeans: test_x_kmeans, Y_kmeans: test_y_kmeans}))


# In[217]:


# Open a SummaryWriter to save summaries
writer = tf.summary.FileWriter('./test_kmeans2', sess.graph)
# Write the summaries to disk
writer.flush()      
writer.close()
sess.close()


# ![TensorBoard](kmeans.png)

# ### Comment: Using full train/test and only the top 21 features with the highest feature importance from random forest model, K-means model in TensorFlow was able to reach a final accuracy of 90.88%, which is below the accuracy of both random forest model and TensorFlow logistic regression

# # PART VI Final Conclusion

# ### For this Lending Club loan dataset, after all the data transformation and cleaning steps, we have attempted 3 different approches focusing on predicting which active borrowers are unlikely to be current on their loan obligations. Our findings are supervised learning algorithm outperforms unsupervised learning algorithm and random forest model implemented in Scikit Learn outperforms the logistic regression model implemented in TensorFlow.
