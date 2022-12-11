from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('clm1.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
   return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
   Month = request.form['Month']
   Make = request.form['Make']
   MonthClaimed = request.form['MonthClaimed']
   WeekOfMonthClaimed = request.form['WeekOfMonthClaimed']
   MaritalStatus = request.form['MaritalStatus']
   PolicyType = request.form['PolicyType']
   VehiclePrice = request.form['VehiclePrice']
   RepNumber= request.form['RepNumber']
   Deductible = request.form['Deductible']
   DaysPolicyAccident = request.form['DaysPolicyAccident']
   DaysPolicyClaim = request.form['DaysPolicyClaim']
   PastNumberOfClaims = request.form['PastNumberOfClaims']
   AgeOfVehicle = request.form['AgeOfVehicle']
   AgeOfPolicyHolder = request.form['AgeOfPolicyHolder']
   NumberOfSuppliments = request.form['NumberOfSuppliments']
   AddressChangeClaim = request.form['AddressChangeClaim']
   AccidentArea = request.form['AccidentArea']
   Sex = request.form['Sex']
   Fault = request.form['Fault']
   PoliceReportFiled = request.form['PoliceReportFiled']
   AgentType = request.form['AgentType']

   
#Mapping
   marital_dict={'Divorced':1,'Married':2,'Single':3,'Widow':4}
   make_dict={'Accura':1,'BMW':2,'Chevrolet':3,'Dodge':4,'Ferrari':5,'Ford':6, 'Honda':7, 'Jaguar':8, 'Lexus':9,'Mazda':10,'Mecedes':11, 'Mercury':12,'Nisson':13,'Pontiac':14,'Porche':15,'Saab':16,'Saturn':17,'Toyota':18,'VW':19 }
   month_dict={'Dec':12, 'Jan':1, 'Oct':10, 'Jun':6, 'Feb':2, 'Nov':12, 'Apr':4, 'Mar':3, 'Aug':8,
       'Jul':7, 'May':5, 'Sep':9}
#    dow_dict={'Wednesday':3, 'Friday':5, 'Saturday':6, 'Monday':1, 'Tuesday':2, 'Sunday':7,
#        'Thursday':4}
#    dowc_dict={'Wednesday':3, 'Friday':5, 'Saturday':6, 'Monday':1, 'Tuesday':2, 'Sunday':7,
#        'Thursday':4}
   monthc_dict={'Dec':12, 'Jan':1, 'Oct':10, 'Jun':6, 'Feb':2, 'Nov':12, 'Apr':4, 'Mar':3, 'Aug':8,
       'Jul':7, 'May':5, 'Sep':9}
   sex_dict={'Male':1,'Female':0}
   AccidentArea_dict={'Rural':0,'Urban':1}
   PoliceReportFiled_dict={'Yes':1,'No':0}
   AgentType_dict={'Internal':1,'External':0}

   Fault_dict={'Third Party':1,'Policy Holder':0}

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
#    noc={'3 to 4':3, '1 vehicle':1, '2 vehicles':2, '5 to 8':4, 'more than 8':5}
#    f={'No':0, 'Yes':1}
   pt={'Sport - Liability':0, 'Sport - Collision':1, 'Sedan - Liability':4,
       'Utility - All Perils':9, 'Sedan - All Perils':6, 'Sedan - Collision':5,
       'Utility - Collision':8, 'Utility - Liability':7, 'Sport - All Perils':3}

   MaritalStatus=marital_dict.get(MaritalStatus)
   Make=make_dict.get(Make)
   PoliceReportFiled=PoliceReportFiled_dict.get(PoliceReportFiled)
   AccidentArea=AccidentArea_dict.get(AccidentArea)
   Sex=sex_dict.get(Sex)
   Month=month_dict.get(Month)
   Fault=Fault_dict.get(Fault)
#    DayOfWeek=dow_dict.get(DayOfWeek)
#    DayOfWeekClaimed=dowc_dict.get(DayOfWeekClaimed)
   MonthClaimed=monthc_dict.get(MonthClaimed) 
   VehiclePrice=vp_dict.get(VehiclePrice)
   DaysPolicyAccident=dpa_dict.get(DaysPolicyAccident)
   DaysPolicyClaim=dpc_dict.get(DaysPolicyClaim)
   PastNumberOfClaims=pnoc_dict.get(PastNumberOfClaims)
   AgeOfVehicle=aov.get(AgeOfVehicle)
   AgentType=AgentType_dict.get(AgentType)
   AgeOfPolicyHolder=aoph.get(AgeOfPolicyHolder)
   NumberOfSuppliments=nos.get(NumberOfSuppliments)
   AddressChangeClaim=acc.get(AddressChangeClaim)
#    NumberOfCars=noc.get(NumberOfCars)
#    FraudFound=f.get(FraudFound)
   PolicyType=pt.get(PolicyType)
   pred=0
   arr = np.array([[Month,Make, MonthClaimed, WeekOfMonthClaimed,MaritalStatus,PolicyType,VehiclePrice,RepNumber,Deductible,DaysPolicyAccident,DaysPolicyClaim,PastNumberOfClaims,AgeOfVehicle,AgeOfPolicyHolder,NumberOfSuppliments,AddressChangeClaim,AccidentArea,Sex,Fault,PoliceReportFiled,AgentType]])
#    pred = model.predict(arr)
   f={0:'_______', 1:'Not Fraud'}
   pred=f.get(pred)
   return render_template('after.html', data=pred)


if __name__ == "__main__":
     app.run(debug=True)