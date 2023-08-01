#Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf
from sklearn import metrics
from tabulate import tabulate
from scipy import stats
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor



#PART A : Access the dataset
path = r"C:\Users\dhruv\OneDrive\Desktop\Project Nikhil Sir\states90.xlsx"
df = pd.read_excel(path)
print(df)
    


#PART B : Describing the dataset
print(df.info())


'''#PART C : Codebook
## region : Geographic Region
## csat : Mean composite SAT score
## percent : % High School Graduates taking SAT
## expense : Per pupil expenditures; primary and secondary
## income : Median Household income
## high : % of people over 25 with High School Diploma
## college :  % of people over 25 with Bachelor's Degree '''


#PART D : Scatterplot matrix and boxplots of csat by region

#Scatterplot Matrix




#Boxplots
sns.boxplot(x=df['region'],y=df['csat'])
plt.title("Boxplot of csat vs region")
plt.show()

#PART E : Fitting SLRs

#fitting SLRs for expense, percent, income, high, college to observe their individual effect on csat

#expense
X = df["expense"]
Y = df["csat"]

#adding the constant
const = sm.add_constant(X)

#fitting the model
model = sm.OLS(Y,const).fit()
print(model.summary())

#visualizing
plt.scatter(X,Y,color = "blue")
plt.plot(X, 1060.7324 - 0.0223*X, 'r')
plt.xlabel("expense")
plt.ylabel("csat")
plt.title("Linear Regression for expense vs csat")
plt.show()




#percent
X = df["percent"]
Y = df["csat"]

#adding the constant
const = sm.add_constant(X)

model = sm.OLS(Y,const).fit()
print(model.summary())


#visualizing
plt.scatter(X,Y,color = "blue")
plt.plot(X, 1024.1429 - 2.2381*X, 'r')
plt.xlabel("percent")
plt.ylabel("csat")
plt.title("Linear Regression for percent vs csat")
plt.show()




#income
X = df["income"]
Y = df["csat"]

#adding the constant
const = sm.add_constant(X)

model = sm.OLS(Y,const).fit()
print(model.summary())

#visualizing
plt.scatter(X,Y,color = "blue")
plt.plot(X, 1110.8575 - 0.0049*X, 'r')
plt.xlabel("income")
plt.ylabel("csat")
plt.title("Linear Regression for income vs csat")
plt.show()



#high
X = df["high"]
Y = df["csat"]

#adding the constant
const = sm.add_constant(X)

model = sm.OLS(Y,const).fit()
print(model.summary())

#visualizing
plt.scatter(X,Y,color = "blue")
plt.plot(X, 865.7524 + 1.0273*X, 'r')
plt.xlabel("high")
plt.ylabel("csat")
plt.title("Linear Regression for high vs csat")
plt.show()



#college
X = df["college"]
Y = df["csat"]

#adding the constant
const = sm.add_constant(X)
model = sm.OLS(Y,const).fit()
print(model.summary())

#visualizing
plt.scatter(X,Y,color = "blue")
plt.plot(X, 1064.0749 - 5.9924*X, 'r')
plt.xlabel("college")
plt.ylabel("csat")
plt.title("Linear Regression for college vs csat")
plt.show()



#Dropping unnececary columns
df1 = df.drop(["state","pop","area","density","vsat","msat"], axis=1)


#Forming dummy variables
df2 = pd.get_dummies(df1)
df3 = df2.drop(["region_West"], axis = 1)


#region_Midwest
X = df3["region_Midwest"]
Y = df["csat"]
const = sm.add_constant(X)
model = sm.OLS(Y,const).fit()
print(model.summary())
    

#region_South
X = df3["region_South"]
Y = df["csat"]
const = sm.add_constant(X)
model = sm.OLS(Y,const).fit()
print(model.summary())
    
    
#region_NEast
X = df3["region_NEast"]
Y = df["csat"]
const = sm.add_constant(X)
model = sm.OLS(Y,const).fit()
print(model.summary())


# FIT MLR

#Defining independent and dependent variables
dep_var = "csat"
ind_var = df3.columns.tolist()
ind_var.remove("csat")

#Assigning X (independent variables) and Y (dependent variable)
X = df3[ind_var]
Y = df1[dep_var]


#Splitting the dataset into TRAIN and TEST
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Transforming data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Fitting MLR
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Fit an OLS model with intercept
const = sm.add_constant(X)
est = sm.OLS(Y, const).fit() 
print(est.summary())


#intercept and respective coefficients
print("Intercept is : ",regressor.intercept_)
print("Coefficients are : ")
table = list(zip(ind_var,regressor.coef_))
headers = ["Variables", "Coefficients"]
print(tabulate(table, headers=headers, floatfmt=".4f"))


#predicting command
Y_pred = regressor.predict(X_test)
meanAbsErr = metrics.mean_absolute_error(Y_test, Y_pred)
meanSqErr = metrics.mean_squared_error(Y_test, Y_pred)
rootMeanSqErr = np.sqrt(meanSqErr)
print("Mean Absolute Error : ", meanAbsErr)
print("Mean Squared Error : ", meanSqErr)
print("Root Mean Squared Error : ", rootMeanSqErr)



#Nested Models

# FIT MLR WITH STANDARDIZED COEFFICIENTS

#Standardizing dataframe
df_z = df3.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)

#Fitting regression
formula = 'csat ~ percent + expense + income + high + college + region_Midwest + region_NEast + region_South'
result = smf.ols(formula, data = df_z).fit()

#Checking results
print("The following table shows how much each variable individually affects csat : ",result.summary())


#Calculating Predicted Values
result2 = pd.DataFrame({"Actual csat scores" : Y_test, "Predicted csat scores" : Y_pred, "Difference" : abs(Y_test - Y_pred)})
print(result2)



#CALCULATE SE AND 95% CI FOR PREDICTED VALUES
ci_pred_values = st.t.interval(alpha=0.90, df=len(list(Y_pred))-1,
                               loc=np.mean(list(Y_pred)),
                               scale=st.sem(list(Y_pred)))
print("The Confidence Interval for predicted values is : ", ci_pred_values)


#INTERACTION (EFFECT MODIFICATION) region * percent(percentage of HS Graduates taking SAT)

formula = 'csat ~ percent + expense + income + high + college + region_Midwest*percent + region_NEast*percent + region_South*percent'
result3 = smf.ols(formula, data = df3).fit()
print(result3.summary())


#FITTING THE QUADRATIC TERM

percent2 = df["percent"]**2
df2 = pd.get_dummies(df1)
df4 = df3.insert(2, "percentsq", percent2, True)

#Defining independent and dependent variables
dep_var = "csat"
ind_var = df3.columns.tolist()
ind_var.remove("csat")
ind_var.remove("percent")


#Assigning X (independent variables) and Y (dependent variable)
X = df3[ind_var]
Y = df1[dep_var]


#Splitting the dataset into TRAIN and TEST
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Transforming data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Fitting MLR
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Fit an OLS model with intercept
const = sm.add_constant(X)
est = sm.OLS(Y, const).fit() 
print(est.summary())



#interaction 2
formula = 'csat ~ percent + percentsq + expense + income + high + college + region_Midwest*percent + region_NEast*percent + region_South*percent'
result3 = smf.ols(formula, data = df3).fit()
print(result3.summary())


    
X = df3["percentsq"]
Y = df["csat"]
#adding the constant
const = sm.add_constant(X)
model = sm.OLS(Y,const).fit()
print(model.summary())
    

X = df3["percent"]
Y = df["csat"]
#adding the constant
const = sm.add_constant(X)
model = sm.OLS(Y,const).fit()
plt.scatter(X,Y,color = "blue")
mymodel = np.poly1d(np.polyfit(list(X), list(Y), 3))
myline = np.linspace(0, 90, 1000)
plt.plot(myline, mymodel(myline), color = "red")
plt.xlabel("percent square")
plt.ylabel("csat")
plt.title("Linear Regression for percent squared vs csat")
plt.show()




#Defining independent and dependent variables
dep_var = "csat"
ind_var = df3.columns.tolist()
ind_var.remove("csat")

#Assigning X (independent variables) and Y (dependent variable)
X = df3[ind_var]
Y = df1[dep_var]


#Splitting the dataset into TRAIN and TEST
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Transforming data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Fitting MLR
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Fit an OLS model with intercept
const = sm.add_constant(X)
est = sm.OLS(Y, const).fit() 
print(est.summary())




#Variance Inflation Factor
# the independent variables set
X = df3[ind_var]
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)







