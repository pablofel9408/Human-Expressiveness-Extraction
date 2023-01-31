import torch
import numpy as np

def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])

    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
corr_coeff = np.random.rand(100)
hist, bin_edges = np.histogram(corr_coeff)
inds = np.digitize(corr_coeff,bin_edges)
emotion = np.random.randint(0,5,100)
names = {0:"joy",1:"ang",2:"hap",3:"aaa",4:"cccc",5:"ddd"}
emotion =  [*map(names.get, emotion)]
actor = np.random.randint(0,5,100)
key_map_act = {0:"asd",1:"fgh",2:"tyu",3:"qwe",4:"plo",5:"ijh"}
actor =  [*map(key_map_act.get, actor)]
aux_dict = {"corr_coeff":inds,"emotion":emotion,"actor":actor}
dataframe = pd.DataFrame(aux_dict)
grouped_df = dataframe.groupby(["emotion","actor"])
sns.scatterplot(x="corr_coeff",y="emotion",hue="actor", data=dataframe)
plt.show()
# for key, item in grouped_df:
#     grouped_df.get_group(key).plot.scatter(x='corr_coeff',y='emotion',c='actor')
# plt.show()
# hist, bin_edges = np.histogram(aa)
# inds = np.digitize(aa,bin_edges)
# colors = {0:"r",1:"g",2:"b",3:"c",4:"y",5:"rh"}
# names = {0:"joy",1:"ang",2:"hap",3:"aaa",4:"cccc",5:"ddd"}
# actor = {0:"asd",1:"fgh",2:"tyu",3:"qwe",4:"plo",5:"ijh"}
# out = groupby(bb,inds)

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# print(bb)
# c =  [*map(colors.get, bb)]
# markers =  [*map(markers.get, bb)]
# bb =  [*map(names.get, bb)]
# scatter = plt.scatter(inds,bb, marker='o', c=c)
# print(scatter.legend_elements())
# custom = [Line2D([], [], marker='o', color='red', linestyle='None'),
#         Line2D([], [], marker='o', color='red', linestyle='None'),
#         Line2D([], [], marker='o', color='red', linestyle='None'),
#         Line2D([], [], marker='o', color='red', linestyle='None'),
#         Line2D([], [], marker='o', color='blue', linestyle='None')]

# plt.legend(handles = custom, labels=['No View', 'View'], bbox_to_anchor= (1.05, 0.5), loc= "lower left")
# plt.show()

