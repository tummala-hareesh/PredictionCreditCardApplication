""" 
	utils-models.py: Collection of python functions used in Data Science Projects
"""
#!/usr/bin/env python3

# Load modules
import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,SGDRegressor
from sklearn.linear_model import ElasticNet,Lars,Lasso,LassoLars,OrthogonalMatchingPursuit
from sklearn.linear_model import ARDRegression, BayesianRidge
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def apply_encoder_one_hot(df_cat):
	"""
		Performs one hot encoding on categorical data and returns new num df
		- One-hot = For categorical variables where no ordinal relationship exists
	"""
	# Initialize encoder
	encoder = OneHotEncoder(sparse=False)

	# fit_transform
	one_hot = encoder.fit_transform(df_cat)

	return one_hot


def apply_encoder_ordinal(df_cat):
	"""
		Performs ordinal encoding on categorical data and returns new num df
		- Ordinal = Mapping each unique category value is assigned an integer value.
	"""
	# Initialize encoder
	encoder = OrdinalEncoder()

	# fit_transform
	ordinal = encoder.fit_transform(df_cat)

	return ordinal


def apply_encoder_label(df_cat):
	"""
		Performs label encoding on categorical data and returns new num df
		- Ordinal = this transformer should be used to encode target values, i.e. y, and not the input X.
	"""
	# Initialize encoder
	encoder = LabelEncoder()

	# fit_transform
	label = encoder.fit_transform(df_cat)

	return label


def apply_standard_scaling(df):

	scaler = StandardScaler()

	df_scaled = scaler.fit_transform(df)

	return df_scaled


def get_classifiers_basic(nmodels='all', lscaler=False):
	"""
		Returns one of or all basic classifiers
	"""
	if (not lscaler):
		# 1. Logistic Regression
		clr1 = LogisticRegression()

		# 2. SVC
		clr2 = SVC()

		# 3. Randome Forest Classfier
		clr3 = RandomForestClassifier()
	else:
		# 1. Logistic Regression
		clr1 = make_pipeline(StandardScaler(), LogisticRegression())

		# 2. SVC
		clr2 = make_pipeline(StandardScaler(), SVC())

		# 3. Randome Forest Classfier
		clr3 = make_pipeline(StandardScaler(), RandomForestClassifier())

	if (nmodels == 'all'):
		models = [clr1, clr2,clr3]
	else:
		models = ['clr'+str(nmodels)]

	return models


def get_classifiers_neural(nmodels='all'):
	"""
		Returns one of or all neural classifiers
	"""
	# 1. MLPClass
	clr1 = MLPClassifier()

	if (nmodels == 'all'):
		models = [clr1]
	else:
		models = ['clr'+str(nmodels)]

	return models


def get_regressors_classical(nmodels='all'):
	"""
		Returns one of or all classical regressors
	"""

	# 1. Linear Regression
	clr1 = LinearRegression()

	# 2. Ridge Regression
	clr2 = Ridge(alpha=0.5)

	# 3. RidgeCV Regression
	clr3 = RidgeCV(alphas=[ia/10 for ia in range(1, 10, 1)])

	# # 4. SGDRegressor
	# clr4 = SGDRegressor(alpha=0.0001)

	if (nmodels == 'all'):
		models = [clr1, clr2, clr3]#, clr4]
	else:
		models = ['clr'+str(nmodels)]

	return models


def get_regressors_variable(nmodels='all'):
	"""
		Returns one of or all variable selection regressors
	"""

	# 1. Elastic net
	lr1 = ElasticNet()

	# 2. Elastic net
	lr2 = Lars()

	# 3. Lasso
	lr3 = Lasso()

	# 4. LassoLars
	lr4 = LassoLars()

	# 5. OrthogonalMatchingPursuit
	lr5 = OrthogonalMatchingPursuit()

	if (nmodels == 'all'):
		models = [lr1, lr2, lr3, lr4, lr5]
	else:
		models = ['lr'+str(nmodels)]

	return models


def get_regressors_bayesian(nmodels='all'):
	"""
		Returns one or all of bayesian regressors
	"""
	# 1. ARDRegression
	lr1 = ARDRegression()

	#2. BayesianRidge
	lr2 = BayesianRidge()

	if (nmodels == 'all'):
		models = [lr1, lr2]
	else:
		models = ['lr'+str(nmodels)]

	return models	


def get_regressors_multitask(nmodels='all'):
	"""
		Returns one or all of Multi-task linear regressors 
	"""
	# 1. MultiTaskElasticNet
	lr1 = MultiTaskElasticNet()

	# 2. MultiTaskLasso
	lr2 = MultiTaskLasso()

	if (nmodels == 'all'):
		models = [lr1, lr2]
	else:
		models = ['lr'+str(nmodels)]

	return models


def get_regressors_outlierrobust(nmodels='all'):
	"""
		Returns one or all of Outlier-Robust linear regressors 
	"""
	# 1. HuberRegressor
	lr1 = HuberRegressor()

	# 2. RANSACRegressor
	lr2 = RANSACRegressor()

	# 3. TheilSenRegressors
	lr3 = TheilSenRegressors()

	if (nmodels == 'all'):
		models = [lr1, lr2, lr3]
	else:
		models = ['lr'+str(nmodels)]

	return models


def get_regressors_generalized(nmodels='all'):
	"""
		Returns one or all of Generalized linear regressors 
	"""
	# 1. PoissonRegressor
	lr1 = PoissonRegressor()

	# 2. TweedieRegressor
	lr2 = TweedieRegressor()

	# 3. GammaRegressor
	lr3 = GammaRegressor()

	if (nmodels == 'all'):
		models = [lr1, lr2, lr3]
	else:
		models = ['lr'+str(nmodels)]

	return models


def combine_two_nparrays(npa1, npa2):
	"""
		Returns a combined numpy array using npa1 and npa2
	"""
	return np.concatenate([npa1, npa2], axis=1)


def get_train_test_dfs(feature_df, target_df, ntrain=0.33, lshuffle=True, irand=19):
	"""
		Returns training and testing (split) dataframes for models
	"""
	if (ntrain < 0.5): print(' Are you sure to train on less data?',ntrain,'\n Recommended: ntrain=0.7')

	feature_train, feature_test, target_train, target_test \
		= train_test_split(feature_df, target_df, train_size=ntrain \
												, shuffle=lshuffle \
												, random_state=irand)

	return feature_train,feature_test,target_train,target_test


def train_model(model, feature_df, target_df, nproc=2, ncv=5, ltrain=True, typeML='classfication'):
	"""
		Returns mean mse and std on each model
		- Scores end with 's'
		- Errors end with 'e'
	"""
	if (typeML == 'classfication'):
		# classification scoring metrics
		scoring_metrics =  {'accs':'accuracy',
							'f1_s':'f1',
							'rocs':'roc_auc'}
	elif (typeML == 'regression'):
		# regression scoring metrics
		scoring_metrics =  {'mse':'neg_mean_squared_error',
							'mae':'neg_mean_absolute_error',
							'r2s':'r2'}


	# cross validation 
	scores_dict = cross_validate(model, feature_df, target_df
								, cv=ncv
								, n_jobs=nproc
								, return_train_score=ltrain
								, scoring=scoring_metrics)

	# mean scores
	mean_scores = {}
	for metric,score in scores_dict.items():
		mean_scores[metric] = np.mean(score)

	return mean_scores


def get_best_model(models_scores, imetric):
	"""
		Retuns a model with best metrics 
	"""
	best_model_score = {}
	for model, metrics_scores in models_scores.items():
		for metric, score in metrics_scores.items():
			if (metric == imetric):
				best_model_score[model] = score

	# best scoring model 
	if (imetric[-1] == 's'): 		# For SCORE metrics
		model = max(best_model_score, key=best_model_score.get)
	elif (imetric[-1] == 'e'): 		# For ERROR metrics
		model = min(best_model_score, key=best_model_score.get)

	return model

# def f_importances(coef, names):
#     imp = coef
#     imp,names = zip(*sorted(zip(imp,names)))
#     plt.barh(range(len(names)), imp, align='center')
#     plt.yticks(range(len(names)), names)
#     plt.show()


def print_models_metrics_scores(models_scores):
	"""
		Prints summary of each model from models & their scores
	""" 
	print('\n')
	print(80*'-')
	for model, metrics_scores in models_scores.items():
		print('\n Model: ', model)
		print('   {0:12}  {1:12}'.format('  Metric', '      Mean'))
		for metric, score in metrics_scores.items():
			print(' - {0:12} :{1:12.5f}'.format(metric, np.round(np.mean(score),7) ))	



def pprint_models_metrics_scores(models_scores):
	"""
		Pretty print of models, metrics and scores in a readable table that can be used for README.md also!
	""" 
	# Templates: table, table-header, table-divide
	table_template  = "{0:1} {1:20} {2:1}"
	header_template = ['|', 'Model'] + list(list(models_scores.values())[0].keys())
	divide_template = ['|', '-'] 

	# Size of metrics_scores 
	len_metrics_scores = len(list(models_scores.values())[0])

	# Intermediatory of template for table
	for il in range(1,len_metrics_scores+1,1):
		# table template adaptive
		table_temp = ' {0}{1}:{2}{3} {4}{5}:{6}{7}'.format('{',2*il+1,12,'}','{',2*il+2,1,'}')
		table_template += table_temp    


	print('\n')
	print(table_template.format(header_template[0], header_template[1], header_template[0]
												  , header_template[2], header_template[0]
												  , header_template[3], header_template[0]
												  , header_template[4], header_template[0]
												  , header_template[5], header_template[0]                
												  , header_template[6], header_template[0]))

	print(table_template.format(divide_template[0], 20*divide_template[1], divide_template[0]
												  , 12*divide_template[1], divide_template[0]
												  , 12*divide_template[1], divide_template[0]
												  , 12*divide_template[1], divide_template[0]
												  , 12*divide_template[1], divide_template[0]                
												  , 12*divide_template[1], divide_template[0]))

	for model, metrics_scores in models_scores.items():
		scores = list(metrics_scores.values())
		print(table_template.format(divide_template[0], str(model)[:20], divide_template[0]
											  , np.round(scores[0],8), divide_template[0]
											  , np.round(scores[1],8), divide_template[0]
											  , np.round(scores[2],8), divide_template[0]
											  , np.round(scores[3],8), divide_template[0]
											  , np.round(scores[4],8), divide_template[0]))
	