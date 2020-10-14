# How to avoid a rejection on your credit card application ?

> #### Rejection hurts...!  
![moneytips.com](/images/moneytips_rejection.jpeg)
Rejections are the most common emotional feeling that everyone has to sustain in daily life. Be it on an application for a basic/secured credit card or for a new premium credit card. It is human nature to expect approvals in every encounter. In this project, we are going to peform data analysis on [credit card data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI's Machine Learning Repository, construct a machine learning model that can help an applicant on `how to avoid rejections on their credit card applications`.

### Background
Financial institutions, such as retail, commercial, internet bank, and credit unions, receive on an average few hundred applications everyday. Commercial Banks, with a significantely larger footprint, generally deploy automated credit card approval/rejection systems to decrease the workload on their employees. The basic working principle behind these automated credit card approval/rejection systems looks simple but is inherently complex. As a credit card applicant, have a basic understanding on how these automated systems work will help us a lot. Our prime motive, as a credit card applicant, is to avoid getting rejections on our credit card applications. One may wonder why I mentioned the term 'avoiding rejection', and not 'getting approvals' in my previous sentence. My understanding of the financial space is that, 'today's rejection impacts tomorrow's credit application". So, avoiding rejection on a credit card application at any cost will help in long term in strenthening ones financial credibility.

### Gathering data from UCI's ML repository
Let's start our analysis by gathering credit card applications data. It is a good practise to automate the data downloading/updating process. Such small automations will enhance our productivity as a Data Scientist and more importantly reliability of our results. We do this by calling the **check_download** function from `src/utils-gather-asses.py` file. In this case, credit card dataset is small enough to see any noticeable impact of above suggested gathering process. Two files downloaded [credit card data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) to our local  `dataset` folder are:

1. `crx.data` - Masked Credit Card Applications Data File
This file contains masked (to protect confidentiality of applicants) data of credit card applications. This dataset has a good mix of continuous, nominal (small and large) attributes with few missing values. 

2. `crx.names` - Information file on the credit card dataset
This file provides us with a brief overview about the dataset uploaded to UCI's ML repository. Details about missing attribute values help us speedup the data cleaning process.


### Assessing Credit Card Data
In brief this small crdit card applications dataset has a total of 690 instances and 15+ class attributes. 

| # |   Feature/Column  | Count |  Dtype |    
|---|  ---------------- | ------| -------|  
| 0 |  Gender           | 690   | object |
| 1 |  Age              | 690   | float64|
| 2 |  Debt             | 690   | float64|
| 3 |  Married          | 690   | object |
| 4 |  BankCustomer     | 690   | object |
| 5 |  EducationLevel   | 690   | object |
| 6 |  Ethnicity        | 690   | object |
| 7 |  YearsEmployed    | 690   | float64|
| 8 |  PriorDefaulter   | 690   | object |
| 9 |  Employed         | 690   | object |
| 10|  CreditScore      | 690   | int64  |
| 11|  DriversLicense   | 690   | object |
| 12|  Citizen          | 690   | object |
| 13|  ZipCode          | 690   | object |
| 14|  Income           | 690   | int64  |
| 15|  **Approved**     | 690   | object |

We followed the same naming convention for the masked features/columns similar to the one followed by [Ryan Kuhn in his Credit Card Analysis of the same dataset](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html). Finally, in the assessment stage, after working on correcting the datatypes and fixing missing values, we end up with below datatypes in the dataset `crx.data_clean.csv`

| # |   Dtype  | Count |    List of Features/Columns |   
|---|  ------- | ------| ----------------------------|
| 0 |  Integer |   2   | YearsEmployed, CreditScore  | 
| 1 |  Float   |   3   | Age, Debt, Income           | 
| 2 |  Object  |  11   | Gender, Married, BankCustomer, EducationLevel, Ethnicity, PriorDefaulter, Employed, DriversLicense, Citizen, ZipCode, Approved |  



### Exploratory Analysis on Credit Card Data 
- Univariate
- Bivariate


### Cleaning Credit Card Data
Three stages
1. Drop missing values - save to _drop.csv
2. Replace - save to _replace.csv 
3. KNN - _knn.csv

### Models on Credit Card dataset
Features that are identified to be related to Approval are:


### Try it yourself - Approval | Rejection on your Credit Card Application 
A simple UI, where a user can input few pre-determined features and get a prediction from the model about the possibility of an approval or rejection on the credit card application.  

