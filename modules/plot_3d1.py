import seaborn as sns
import numpy as np

def plot_3d1(x,y,data):
    g = sns.relplot(x=x, y=y, hue='Attrition',
               height=5, data=data)
    g.set_axis_labels(x, y)