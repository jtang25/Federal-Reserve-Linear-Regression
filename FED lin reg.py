import requests
import seaborn as sns
from datetime import datetime
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functools import reduce
import numpy as np
import statsmodels.api as sm
import math 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew
import mplcyberpunk

plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 7), dpi=80)
plt.style.use('cyberpunk')

def predict(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1)
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    display(regression_model.coef_)
    y_predict = regression_model.predict(X_test)
    plt.scatter(X_test, y_test,  color='gainsboro')
    plt.plot(X_test, y_predict, color='royalblue', linewidth = 3, linestyle= '-')
    plt.legend()

CPI = pd.read_csv("CPI.csv")
CPI['CPI'] = CPI['CPI'].diff()
CPI_Date = pd.to_datetime(CPI['DATE'])
FFR = pd.read_csv("FFR.csv")
FFR['FFR'] = FFR['FFR'].diff()
FFR = FFR[FFR.FFR!=0]
FFR_Date = pd.to_datetime(FFR['DATE'])
PAYROLL = pd.read_csv("PAYROLL.csv")
PAYROLL_Date = pd.to_datetime(PAYROLL['DATE'])
GDP = pd.read_csv("GDP.csv")
GDP = GDP.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5'], axis=1)
GDP['GDP'] = GDP['GDP']*0.3
GDP_Date = pd.to_datetime(GDP['DATE'])
# monthly % from a year ago
M2 = pd.read_csv("M2.csv")
M2_Date = pd.to_datetime(M2['DATE'])
# % from a year ago
HOUSING = pd.read_csv("HOUSING.csv")
HOUSING_Date = pd.to_datetime(HOUSING['DATE'])
HOUSING['HOUSING'] = HOUSING['HOUSING']*0.02
# % from a year ago
UNEM = pd.read_csv("UNEM.csv")
UNEM_Date = pd.to_datetime(UNEM['UNEM'])
# % from a year ago
RETAIL = pd.read_csv("RETAIL.csv")
RETAIL_Date = pd.to_datetime(RETAIL['RETAIL'])
# % from a year ago
PCE = pd.read_csv("PCE.csv")
PCE_Date = pd.to_datetime(PCE['PCE'])
# % from a year ago
PPI = pd.read_csv("PPI.csv")
PPI_Date = pd.to_datetime(PPI['PPI'])
# % change
PPIP = pd.read_csv("PPIP.csv")
PPIP_Date = pd.to_datetime(PPIP['PPIP'])
# % from a year ago
MANU = pd.read_csv("MANU.csv")
MANU_Date = pd.to_datetime(MANU['MANU'])
# % change month over month
SNP = pd.read_csv("SNP.csv")
SNP_Date = pd.to_datetime(SNP['DATE'])
SNP.drop(['PRICE'], axis=1)
SNP['SNP'] = SNP['PRICE'].pct_change()*100

var1 = input()
var2 = input()
if(var1=='GDP'):
    var1_data = GDP['GDP']
    var1_Date = GDP_Date
elif (var1=='CPI'):
    var1_data = CPI['CPI']
    var1_Date = CPI_Date
elif(var1=='FFR'):
    var1_data = FFR['FFR']
    var1_Date = FFR_Date
elif(var1=='PAYROLL'):
    var1_data = PAYROLL['PAYROLL']
    var1_Date = PAYROLL_Date
if(var2=='GDP'):
    var2_data = GDP['GDP']
    var2_Date = GDP_Date
elif(var2=='CPI'):
    var2_data = CPI['CPI']
    var2_Date = CPI_Date
elif(var2=='FFR'):
    var2_data = FFR['FFR']
    var2_Date = FFR_Date
elif(var2=='PAYROLL'):
    var2_data = PAYROLL['PAYROLL']
    var2_Date = PAYROLL_Date
elif(var2=='M2'):
    var2_data = M2['M2']
    var2_Date = M2_Date
elif(var2=='HOUSING'):
    var2_data = HOUSING['HOUSING']
    var2_Date = HOUSING_Date
elif(var2=='UNEM'):
    var2_data = UNEM['UNEM']
    var2_Date = UNEM_Date
elif(var2=='RETAIL'):
    var2_data = RETAIL['RETAIL']
    var2_Date = RETAIL_Date
elif(var2=='PCE'):
    var2_data = PCE['PCE']
    var2_Date = PCE_Date
elif(var2=='PPI'):
    var2_data = PPI['PPI']
    var2_Date = PPI_Date
elif(var2=='PPIP'):
    var2_data = PPIP['PPIP']
    var2_Date = PPIP_Date
elif(var2=='MANU'):
    var2_data = MANU['MANU']
    var2_Date = MANU_Date
elif(var2=='SNP'):
    var2_data = SNP['SNP']
    var2_Date = SNP_Date

#GDP Moving Average
length_var1 = int(input())
MA_var1 = var1_data.rolling(length_var1).mean()
MA_var1_date = var1_Date.copy()
for x in range(length_var1-1):
    MA_var1_date.drop(index=MA_var1_date.index[0], axis=0, inplace=True)

#Payroll Moving Average
length_var2 = int(input())
MA_var2 = var2_data.rolling(length_var2).mean()
MA_var2_date = var2_Date.copy()
for x in range(length_var2-1):
    MA_var2_date.drop(index=MA_var2_date.index[0], axis=0, inplace=True)
# MA_var2_date = MA_var2_date -  pd.to_timedelta(120, unit='days')

# X = GDP_PAYROLL[['Payroll']]
# Y = GDP_PAYROLL.drop(['Payroll','datenum'],axis=1)
# g=plt.figure(2)
# predict(X,Y)
# g.show()

i=plt.figure(2,figsize=(20, 10), dpi=250)
dtFmt = mdates.DateFormatter('%Y-%b')
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.xticks(rotation=45, fontweight='light',  fontsize='small')
plt.plot(MA_var1_date, MA_var1, label = var1)
plt.plot(MA_var2_date, MA_var2, label = var2)
plt.xlabel('Date')
plt.ylabel('% change')
plt.title(var1+' vs '+var2)
plt.legend()
# plt.plot(GDP_Date,GDP['GDP'])
# plt.plot(PAYROLL_Date,PAYROLL['PAYROLL'])
