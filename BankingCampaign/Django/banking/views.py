
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages

from django.contrib.auth.models import User

import sys
from subprocess import run,PIPE

from .forms  import Form1
from django.contrib.auth.decorators import login_required

from joblib import load
classifier = load('./SavedModels/classifier.joblib')

# Create your views here.
def home(request):
    return render(request,'banking/home.html')

#def about(request):
#    return render(request,'banking/about.html')

@login_required
def form1(request):
    if request.method == 'POST':
        form1 = Form1(request.POST)
        if form1.is_valid():
            age = form1.cleaned_data.get('age')
            marital_status = form1.cleaned_data.get("marital_status")
            if marital_status == 'married':
                marital_married = 1.0
                marital_single = 0.0
            elif marital_status == 'single':
                marital_married = 0.0
                marital_single = 1.0
            else:
                marital_married = 0.0
                marital_single = 0.0

            region_name_east = 0.0
            region_name_west = 0.0
            region_name_south = 0.0
            region = form1.cleaned_data.get("region")
            if region=='east':
                region_name_east = 1.0
            if region =='west':
                region_name_west = 1.0
            if region == 'south':
                region_name_south = 1.0

            

            default_yes = form1.cleaned_data.get("has_credit")
            housing_yes = form1.cleaned_data.get("housing_loan")
            loan_yes = form1.cleaned_data.get("personal_loan")
            contact_telephone = form1.cleaned_data.get("contact_mode")
            duration = form1.cleaned_data.get("duration")
            campaign = form1.cleaned_data.get("campaign")
            pdays = form1.cleaned_data.get("pdays")
            previous = form1.cleaned_data.get("previous")
            job = form1.cleaned_data.get("job")
            month = form1.cleaned_data.get("month")
            day_of_week = form1.cleaned_data.get("day_of_week")
            education = form1.cleaned_data.get("education")
            city_name = form1.cleaned_data.get("city_name")
            state_name = form1.cleaned_data.get("state_name") 
            poutcome = form1.cleaned_data.get("poutcome")
            poutcome_success = 0.0
            poutcome_nonexistent = 0.0
            if poutcome == 'success':
                poutcome_success = 1.0
            elif poutcome == 'nonexistent':
                poutcome_nonexistent = 1.0

            emp_var = form1.cleaned_data.get("emp_var")
            cons_price_idx = form1.cleaned_data.get("cons_price_idx")
            cons_conf_idx = form1.cleaned_data.get("cons_conf_idx")
            euribor3m = form1.cleaned_data.get("euribor3m")
            nr_employed = form1.cleaned_data.get("nr_employed")
            y_pred = classifier.predict([[age, job, education, month, day_of_week, duration, campaign, pdays, previous, emp_var,
                              cons_price_idx, cons_conf_idx, euribor3m, nr_employed, city_name, state_name, marital_married, marital_single, 
                              loan_yes,contact_telephone, default_yes, housing_yes, region_name_east, region_name_south,
                              region_name_west, poutcome_nonexistent, poutcome_success]])
            
            if y_pred[0] == 1:
                y_pred = 'yes'
            else:
                y_pred = 'no'
            return render(request,'banking/display.html',{'result' : y_pred})
    else:
        form1 = Form1()  
    context = {
        'form1': form1
    }
    return render(request, 'banking/form1.html',context)

def display(request, self):
   return render(request, 'banking/display.html')