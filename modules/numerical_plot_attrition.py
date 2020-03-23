import seaborn as sns
import numpy as np

def numerical_plot_attrition(df, column, ax):
    data_yes = df[df['Attrition'] == 1]
    data_no  = df[df['Attrition'] == 0]
    
    if df[column].unique().shape[0] > 10:
        bins = int(df[column].unique().shape[0]/5)
        sns.distplot(data_yes[column], ax = ax, label = 'Attrition', axlabel=column, bins = bins, norm_hist = True)
        sns.distplot(data_no[column],  ax = ax, label = 'No Attrition', axlabel=column, bins = bins, norm_hist = True)
    else:
        bins = df[column].unique().shape[0]
        sns.distplot(data_yes[column], ax = ax, label = 'Attrition', axlabel=column, kde = False, bins = bins)
        sns.distplot(data_no[column],  ax = ax, label = 'No Attrition', axlabel=column, kde = False, bins = bins)
    