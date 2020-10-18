""" 
	utils-explore-clean.py: Collection of python functions used in Data Science Projects
"""
#!/usr/bin/env python3


# Load modules
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def get_version(modulename):
	"""
		Returns version used 
	"""
	# Local module versions dictoronary with abbrevations
	module_abbrv_dict = {'numpy': np, 
						 'pandas': pd, 
						 'seaborn': sns
						 }

	return print(modulename+' version is:',module_abbrv_dict[modulename].__version__)



def split_dataframe_datatypes(df, target_var):
	"""
		Returns two (*-num, *_cat & *_target) dataframes from combined dataframe 
	"""
	df_num = df.select_dtypes(include=np.number)
	df_cat = df.select_dtypes(include=object)

	if target_var in df_num.columns:
		df_tar = df_num.copy() 
		df_tar = df_tar[[target_var]]
		df_num.drop(columns=[target_var], axis=1, inplace=True) 
	elif target_var in df_cat.columns:
		df_tar = df_cat.copy()
		df_tar = df_tar[[target_var]]
		df_cat.drop(columns=[target_var], axis=1, inplace=True) 

	return df_num,df_cat,df_tar


def normalize_feature_zscore(df, colname):
    result = df.copy()
    mean_value = df[colname].mean()
    std_value  = df[colname].std()
    result[colname] = np.log((df[colname] - mean_value) / (std_value))
    return result


def normalize_feature_minmax(df, colname):
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	result = scaler.fit_transform(df[colname])
	return result


def plot_uv_bar(df, colname, colorid=0):
    """
        Returns a bar plot on a catergorical feature.column (Univariate analysis) 
    """
    if (colname in list(df.columns)):
       
        # Set figure size 
        fig, ax = plt.subplots(figsize=(8,6))
    
        # set colorid for bar plot
        base_color = sns.color_palette()[colorid]

        # variable counts to calculate percentage
        cdict_count = df[colname].value_counts().to_dict() 
        total_count = df.shape[0]
        
        
        if (len(list(cdict_count.keys())) > 5):
            # max.count to position the %
            maxcount_pct= np.max(list(cdict_count.values()))*0.125
            # max. no. of categories Vs % rotation 
            rottext_pct = 90        
            # font size for % display
            fontsiz_pct = 12
        else:
            # max.count to position the %
            maxcount_pct= np.max(list(cdict_count.values()))*0.075
            # max. no. of categories Vs % rotation 
            rottext_pct = 0        
            # font size for % display
            fontsiz_pct = 16
                    
            
        # plotting...
        sns.countplot(data = df, x = colname
                               , order = list(cdict_count.keys())
                               , color = base_color
                               , saturation = 0.7)

        # title and labels
        plt.title('Order of '+ colname, fontsize=20)
        plt.xlabel(colname + ' Type', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        
        # x-,y- ticks
        locs, labels = plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # display % count information on each tower of bar plot
        for loc, label in zip(locs, labels):
            count = cdict_count[label.get_text()]
            pct_string = '{:0.1f}%'.format(count*100/total_count)
            plt.text(loc, count-maxcount_pct, pct_string, ha='center', color='w', fontsize=fontsiz_pct, rotation=rottext_pct)

        return plt.show()

    else:
        
        print('  >>>Error:',colname,' is not in DataFrame')




def plot_uv_hist(df, colname, nbins='auto', xlogflag=False, colorid=0):
    """
        Returns a histogram with with automatic labeling  (Univariate Analysis)
    """

    # Set figure size 
    fig, ax = plt.subplots(figsize=(8,6))

    # set colorid for bar plot
    base_color = sns.color_palette()[colorid]
        
    # plotting... histogram
    if (xlogflag):
    	sns.histplot(ax = ax, data = df
                        	, x = colname
                        	, bins = nbins
                        	, color = base_color
    						, log_scale= True)
    else:
    	sns.histplot(ax = ax, data = df
                        	, x = colname
                        	, color = base_color
                        	, bins = nbins)

    
    #plt.legend(prop={'size': 12})
    plt.title('Distribution of '+colname, fontsize=20)
    plt.xlabel(colname+' (units)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    return plt.show()


def plot_bv_reg(df, xcolname, ycolname, xlogflag=False, ylogflag=False, colorid=3):
    """
        Returns a scatter plot with regression-trend line (Bivariate Analysis)
        - x = numerical
        - y = numerical
    """

    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))

    # set colorid for bar plot
    base_color = sns.color_palette()[colorid]
    
    # plotting... scatter+reg line
    sns.regplot(data = df, x = str(xcolname)
                         , y = str(ycolname)
                         , color = base_color
                         , scatter_kws = {'alpha' : 1/3})
    
    # log scale or not
    if (ylogflag): plt.yscale('log')
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()


def plot_bv_bar(df, xcolname, ycolname, icol=0):
    """
        Returns a bivariate bar plot
        - x = categorical
        - y = numerical or categorical
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plotting... box
    sns.barplot(ax=ax, data = df
                     , x = str(xcolname)
                     , y = str(ycolname)
                     , color = sns.color_palette()[icol]);
    
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()


def plot_bv_point(df, xcolname, ycolname, icol=0):
    """
        Returns a bivariate count plot
        - x = categorical
        - y = numerical or categorical
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plotting... box
    sns.pointplot(ax=ax, data = df
                     	, x = str(xcolname)
                     	, y = str(ycolname)
                     	, color = sns.color_palette()[icol]);
    
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()



def plot_bv_box(df, xcolname, ycolname, ylogflag=False, icol=0):
    """
        Returns a bivariate box plot
        - x = categorical
        - y = numerical or categorical
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))


    # log scale or not
    if (ylogflag): plt.yscale('log')

    
    # plotting... box
    sns.boxplot(ax=ax, data = df
                     , x = str(xcolname)
                     , y = str(ycolname)
                     , color = sns.color_palette()[icol]);
    
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    if (ylogflag): plt.ylabel(ycolname+ ' (log units)', fontsize=16) 
    else: plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()


def plot_bv_violin(df, xcolname, ycolname, icol=0):
    """
        Returns a bivariate violin plot
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plotting... box+kde
    sns.violinplot(ax=ax, data = df
                        , x = str(xcolname)
                        , y = str(ycolname)
                        , color = sns.color_palette()[icol]);
    
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()


def plot_bv_strip(df, xcolname, ycolname, icol=1):
    """
        Returns a bivariate swarm plot
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plotting... box+kde
    sns.stripplot(ax=ax, data = df
                        , x = str(xcolname)
                        , y = str(ycolname)
                        , color = sns.color_palette()[icol]);
    
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()



def plot_bv_swarm(df, xcolname, ycolname, icol=1):
    """
        Returns a bivariate swarm plot
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plotting... box+kde
    sns.swarmplot(ax=ax, data = df
                        , x = str(xcolname)
                        , y = str(ycolname)
                        , color = sns.color_palette()[icol]);
    
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()


def plot_bv_join(df, xcolname, ycolname, kindname='hist', xlogflag=False, ylogflag=False, icol=4):
    """
        Returns a scatter plot with regression-trend line 
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plotting... jointplot
    sns.jointplot(ax = ax, data = df
                         , x = str(xcolname)
                         , y = str(ycolname)
                         , kind = kindname
                         , color = sns.color_palette()[icol]);
    

    # log scale or not
    if (ylogflag): plt.yscale('log')
        
    
    # title and labels
    plt.title(xcolname+' Vs '+ycolname, fontsize=20)
    plt.xlabel(xcolname+ ' (units)', fontsize=16)
    plt.ylabel(ycolname+ ' (units)', fontsize=16)
    
    return plt.show()


def plot_bv_facet(df, xcolname, ycolname, ncols=3, xshareflag=False, yshareflag=False, icol=5):
	"""
		Returns facet grid 
	"""
	g = sns.FacetGrid(data=df, col=ycolname
							 , col_wrap=ncols
							 , color=sns.color_palette()[icol]
							 , sharex=xshareflag
							 , sharey=yshareflag)
	g.map(plt.hist, xcolname)

	return plt.show()


def plot_mv_corr(df):
    """
        Returns a heatmap with pearson correlation coefficients as annotations 
    """
    # set plot size
    fig, ax = plt.subplots(figsize=(10,8))

    # plot heatmap with corr. coeff
    sns.heatmap(df.corr(), annot = True, fmt='.1g'
    					 , vmin=-1.0, vmax=1.0, center=0.0
    					 , cmap='coolwarm', annot_kws={"size": 16})

    # Set size and orientation of ticks
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=16, rotation=0)

    return plt.show()