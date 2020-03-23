import seaborn as sns
import numpy as np



def categorical_plot_attrition(df, column, ax):
    data_yes = df[df['Attrition'] == 1]
    data_no  = df[df['Attrition'] == 0]
    
    cats = df[column].unique()
    
    plot = sns.countplot(x = column, hue = 'Attrition', data = df)
    
    count_yes   = data_yes[column].value_counts().sort_index()
    count_no    = data_no[column].value_counts().sort_index()
    perc_att    = np.round(count_yes / (count_no+count_yes)*100, decimals = 2)
    
    for i,x in enumerate(cats):
        tmp = perc_att[x]
        plot.annotate(format(tmp, '.2f'), (i, int(plot.get_ylim()[1])), ha = 'center')