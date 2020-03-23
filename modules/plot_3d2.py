import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_3d2(x,y,data):
    
    x_quants = np.quantile(data[x].values, [0, 0.25, 0.5, 0.75, 1])
    y_quants = np.quantile(data[x].values, [0, 0.25, 0.5, 0.75, 1])
    
    categ= {'low': [0, 0.25],
            'med-low': [0.25, 0.5],
            'med-high': [0.5, 0.75],
            'high': [0.75, 1]}
    
    fig = plt.figure()
    Attrition   = data[data.Attrition == 1]
    NOAttrition = data[data.Attrition == 0]
    
    count = 1
    for i1, k in enumerate(list(categ.keys())):
        for i2, j in  enumerate(list(categ.keys())):
            
            x_quants = np.quantile(data[x].values, categ[k])
            y_quants = np.quantile(data[y].values, categ[j])
            
            count_att = Attrition[(Attrition[x]>=  x_quants[0]) & 
                                  (Attrition[x]<= x_quants[1]) & 
                                  (Attrition[y]>=  y_quants[0]) & 
                                  (Attrition[y]<= y_quants[1])   ].shape[0]
            
            count_noatt = NOAttrition[(NOAttrition[x]>=  x_quants[0]) & 
                                      (NOAttrition[x]<= x_quants[1]) & 
                                      (NOAttrition[y]>=  y_quants[0]) & 
                                      (NOAttrition[y]<= y_quants[1])   ].shape[0]
            
            prcnt = count_att / (count_att+count_noatt)
            
            ax = fig.add_subplot(4, 4, count)
            if prcnt*100 > 16.21:
                plot = plt.bar(['no_att','att'], [count_noatt, count_att], color ='r')
            else:
                plot = plt.bar(['no_att','att'], [count_noatt, count_att])
            plt.ylabel(k+'_'+x)
            plt.xlabel(j+'_'+y)
            plt.title('Attrition: ' + format(prcnt*100, '.2f') + '%')
            
            count+=1

    fig.tight_layout()
    