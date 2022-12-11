import pandas as pd
import pickle
import numpy as np

df=pd.read_csv("carclaims.csv")
df1=pd.get_dummies(df,columns=['AccidentArea','Sex','Fault','PoliceReportFiled','WitnessPresent'],drop_first=True)

from sklearn.preprocessing import LabelEncoder
# le=LabelEncoder()
# df1['Make']=le.fit_transform(df1['Make'])
# df1['MaritalStatus']=le.fit_transform(df1['MaritalStatus'])
# df1['VehicleCategory']=le.fit_transform(df1['VehicleCategory'])
# df1['BasePolicy']=le.fit_transform(df1['BasePolicy'])
marital_dict={'Divorced':1,'Married':2,'Single':3,'Widow':4}
make_dict={'Accura':1,'BMW':2,'Chevrolet':3,'Dodge':4,'Ferrari':5,'Ford':6, 'Honda':7, 'Jaguar':8, 'Lexus':9,'Mazda':10,'Mecedes':11, 'Mercury':12,'Nisson':13,'Pontiac':14,'Porche':15,'Saab':16,'Saturn':17,'Toyota':18,'VW':19 }
AgentType_dict={'Internal':1,'External':0}
month_dict={'Dec':12, 'Jan':1, 'Oct':10, 'Jun':6, 'Feb':2, 'Nov':12, 'Apr':4, 'Mar':3, 'Aug':8,
       'Jul':7, 'May':5, 'Sep':9}
dow_dict={'Wednesday':3, 'Friday':5, 'Saturday':6, 'Monday':1, 'Tuesday':2, 'Sunday':7,
       'Thursday':4}
dowc_dict={'Wednesday':3, 'Friday':5, 'Saturday':6, 'Monday':1, 'Tuesday':2, 'Sunday':7,
       'Thursday':4}
monthc_dict={'Dec':12, 'Jan':1, 'Oct':10, 'Jun':6, 'Feb':2, 'Nov':12, 'Apr':4, 'Mar':3, 'Aug':8,
       'Jul':7, 'May':5, 'Sep':9}

df1['Month']=df1['Month'].map(month_dict).astype(float)
df1['DayOfWeek']=df1['DayOfWeek'].map(dow_dict).astype(float)
df1['DayOfWeekClaimed']=df1['DayOfWeekClaimed'].map(dowc_dict).astype(float)
df1['MonthClaimed']=df1['MonthClaimed'].map(monthc_dict).astype(float)
df1['AgentType']=df1['AgentType'].map(AgentType_dict).astype(float)
df1['Make']=df1['Make'].map(make_dict).astype(float)
df1['MaritalStatus']=df1['MaritalStatus'].map(marital_dict).astype(float)

vp_dict={'more than 69,000':6, '20,000 to 29,000':2, '30,000 to 39,000':3,
       'less than 20,000':1, '40,000 to 59,000':4, '60,000 to 69,000':5}
dpa_dict={'more than 30':5, '15 to 30':4, 'none':1, '1 to 7':2, '8 to 15':3}
dpc_dict={'more than 30':4, '15 to 30':3, 'none':1, '8 to 15':2}
pnoc_dict={'none':1, '1':2, '2 to 4':3, 'more than 4':4}
aov={'3 years':3, '6 years':6, '7 years':7, 'more than 7':8, '5 years':5, 'new':1,
       '4 years':4, '2 years':2}
aoph={'26 to 30':4, '31 to 35':5, '41 to 50':7, '51 to 65':8, '21 to 25':3,
       '36 to 40':6, '16 to 17':1, 'over 65':9, '18 to 20':2}
nos={'none':1, 'more than 5':4, '3 to 5':3, '1 to 2':2}
acc={'1 year':2, 'no change':1, '4 to 8 years':4, '2 to 3 years':3,
       'under 6 months':5}
noc={'3 to 4':3, '1 vehicle':1, '2 vehicles':2, '5 to 8':4, 'more than 8':5}
f={'No':0, 'Yes':1}
pt={'Sport - Liability':0, 'Sport - Collision':1, 'Sedan - Liability':4,
       'Utility - All Perils':9, 'Sedan - All Perils':6, 'Sedan - Collision':5,
       'Utility - Collision':8, 'Utility - Liability':7, 'Sport - All Perils':3}

df1['VehiclePrice']=df['VehiclePrice'].map(vp_dict).astype(float)
df1['Days:Policy-Accident']=df['Days:Policy-Accident'].map(dpa_dict).astype(float)
df1['Days:Policy-Claim']=df['Days:Policy-Claim'].map(dpc_dict).astype(float)
df1['PastNumberOfClaims']=df['PastNumberOfClaims'].map(pnoc_dict).astype(float)
df1['AgeOfVehicle']=df['AgeOfVehicle'].map(aov).astype(float)
df1['AgeOfPolicyHolder']=df['AgeOfPolicyHolder'].map(aoph).astype(float)
df1['NumberOfSuppliments']=df['NumberOfSuppliments'].map(nos).astype(float)
df1['AddressChange-Claim']=df['AddressChange-Claim'].map(acc).astype(float)
df1['NumberOfCars']=df['NumberOfCars'].map(noc).astype(float)
df1['FraudFound']=df['FraudFound'].map(f).astype(float)
df1['PolicyType']=df['PolicyType'].map(pt).astype(float)



df1.drop(['VehicleCategory','BasePolicy','Age','Year','AgentType'],axis=1,inplace=True)
df1.drop(df1[df1['PolicyNumber']==1517].index,inplace=True)
df1=df1.reset_index()
df1.drop('index',axis=1,inplace=True)

df1.drop(['WeekOfMonth','DayOfWeek','DayOfWeekClaimed',"PolicyNumber","DriverRating","NumberOfCars",'WitnessPresent_Yes'],axis=1,inplace=True)
df1.rename(columns = {'Days:Policy-Accident':'DaysPolicyAccident'}, inplace = True)
df1.rename(columns = {'Days:Policy-Claim':'DaysPolicyClaim'}, inplace = True)
df1.rename(columns = {'AddressChange-Claim':'AddressChangeClaim'}, inplace = True)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,KFold
x = df1.drop(['FraudFound'], axis=1)
y = df1['FraudFound']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

mod=RandomForestClassifier()
mod=mod.fit(x_train,y_train)
skfold=StratifiedKFold(n_splits=12,shuffle=True,random_state=40)
cv_res=cross_val_score(mod,x_train,y_train,cv=skfold,scoring='precision')


# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# x = df1.drop(['FraudFound'], axis=1)
# y = df1['FraudFound']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# from sklearn.linear_model import LogisticRegression
# lr=LogisticRegression()

# model_lr =lr.fit(x_train,y_train)
pickle.dump(mod,open('clm1.pkl', 'wb'))
