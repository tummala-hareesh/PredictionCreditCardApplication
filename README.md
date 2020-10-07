# How not to get a rejection on your credit card application ?

> #### Rejection hurts...!  
![moneytips.com](/images/moneytips_rejection.jpeg)
Rejections are the most common emotional feeling that everyone has to sustain in daily life. Be it on an application for a basic/secured credit card or for a new premium credit card. It is human nature to expect approvals in every encounter. In this project, we are going to peform data analysis on [credit card data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI's Machine Learning Repository, construct a machine learning model using our observation on the data, and finally build a dashboard that provides suggestions on `how not to get a rejection on their credit card applications`.


### Background
Financial institutions, such as retail, commercial, internet bank, and credit unions, receive on an average few hundred applications everyday. Commercial Banks, with a significantely larger footprint, generally deploy automated credit card approval/rejection systems to decrease the workload on their employees. The basic working principle behind these automated credit card approval/rejection systems looks simple but is inherently complex in nature. As a credit card applicant, have a basic understanding on how these automated systems work will help us a lot. Our prime motive, as a credit card applicant, is to avoid getting rejections on our credit card applications. One may wonder why I mentioned the term 'avoiding rejection', and not about 'getting approvals' in my previous sentence. My understanding of the financial space is that, 'today's rejection impacts tomorrow's credit application". So, avoiding an application rejection at any cost will help in long term in strenthening our financial credibility.


### Gathering data from UCI's ML repository
Let's start our analysis by gathering credit card data. It is a good practise to verify (the presence of the files locally!) before downloading/updating. We do this by calling the **check_download** function from `src/utils-gather-asses.py` file. In the current case, credit card dataset is small enough to see any noticeable impact of above suggested gathering process. Two files downloaded [credit card data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) to our local  `dataset` folder are:

1. `crx.data` - Masked Credit Card Applications Data File
This file contains masked (to protect confidentiality of applicants) data of credit card applications. This dataset has a good mix of continuous, nominal (small and large) attributes with few missing values. 

2. `crx.names` - Information file on the credit card dataset
This file provides us with a brief overview about the dataset uploaded to UCI's ML repository. Details about missing attribute values help us speedup the data cleaning process.


### Assessing Credit Card Data
In brief this small dataset has a total of 690 instances and 15+ class attributes. 

- Attach a table with first few rows.... named table!!!


In the data assessing stage, we organized our findings in two data files based on **Quality** and **Tidiness**.


### Exploring Credit Card Data 
Data


### Cleaning Credit Card Data
Three stages
1. Drop missing values - save to _drop.csv
2. Replace - save to _replace.csv 
3. KNN - _knn.csv


### Models on Credit Card dataset


### Interactive Dashboard - Credit Card Applications 
Dashboard using bokeh that display a suggestion according to user input. 







