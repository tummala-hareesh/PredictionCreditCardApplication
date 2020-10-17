# Predicting outcome on a credit card application

### Rejection hurts...!!!  
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
To get an idea about the dataset, we perform exploratory data analysis in different stages as below: 

#### **Univariate Analysis**
During this stage, we develop our first senses on the data using both non-graphical and graphical methods. Below is a bar chart showing distribution of credit card applications according to their education qualification. Further, conclusions such as the highest no. of applications are coming from applicants with education level of type c, followed by q, w, i and so on, are drawn by sorting all the education categories.  

![Education Level of Credit Card Applicants](/images/uv_education.png)

Histogram plots on continuous features, such as `CreditScore`  can be used to interpret the skewness of the feature data. `CreditScore` data is not normally distributed. It is heavily skewed to right.  

![Credit Score of Credit Card Applicants](/images/uv_creditscore.png)

#### **Bivariate Analysis**
To understand the relationship between each categorical feature and credit card application approval rating, we take advantage of 6 different types of plots on the `Age` feature as shown below: 

![Interpretation of Age of Credit Card Applicants](/images/bv_age.png)

| 			Bivarite Analysis 	   | 		Types of plots			   |
|----------------------------------|-----------------------------------|
|![Scatter-1](/images/bv_age1.png) | ![Scatter-2](/images/bv_age2.png) |
|![Distrib-1](/images/bv_age3.png) | ![Distrib-2](/images/bv_age4.png) |
|![Estimat-1](/images/bv_age5.png) | ![Estimat-2](/images/bv_age6.png) |

We gain vital information from every plot type. For example, Scatter plots show the distribution of each feature value, which can help in isolating irrelevant outliers. In contrast, Distrib-1 and Distrib-2 pack a lot of information. Median, IQR values and outliers (in some cases) are clearly evident in these plot types. Estimat-1 and Estimat-2 provide us with the mean age information on approved and rejected applicats. Depending on the project we work on, we are free to choose any plot that provide information needed.   
 

#### **Multivariate Analysis**
Additional relationship between variables is drawn from correlation heatmap and pairplot, respectively. For credit card dataset, heatmap indicates a strong correlation of Accepted applciations with CreditScore and YearsEmployed.

![Correlation-Heatmap](/images/mv_corr.png)


A step-by-step EDA performed in `2-Explore-Clean-NamedCreditCard.ipynb` notebook suggests that `CreditScore`, `Income`, `YearsEmployed`, `PriorDefaulter` are good indicators to predict the outcome of any new credit card application.  So, we make sure that these features are well addressed in our model to predict target feature.. 


### Cleaning Credit Card Data
We address missing values in the credit card dataset in three different methods. For each possible method, we build a ML model and check against training data.  

1. Method of dropping rows with missing values - saved to `datasets/crx.data_drop.csv`
In this method, we simply delete the rows with missing values. This results in an abridged dataset with 653 rows for building predictive models.   

2. Method of replacing missing values - saved to `datasets/crx.data_replace.csv` 
This is one of the most commonly used methods in data science. Depending on the datatype, missing values in a column/feature are replaced with either mean/median or mode value.  

3. KNN - saved to `datasets/crx.data_knn.csv`
K-nearest neighbors method is more advanced way of dealing with missing values. In this method, we organize data into clusters and missing value are replaced from the cluster closest to the missing value's row. There are more advanced methods avaiable for dealing with missing values. But, for this project, we use the first two methods described in this section.   

dataframe
### Predictive Model for Credit Card Applications 
Features that are identified to be related to credit card approval are used for first training, and later testing the model.




### Try it yourself - Approval | Rejection on your Credit Card Application 
A simple UI, where a user can input his/her details into the model and get a prediction on the target feature, the credit card application approval/rejection, is under development.  

