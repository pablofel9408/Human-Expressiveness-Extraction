import numpy as np
from scipy import interpolate

def samples_interpolation(samples, k_spline_degree, end_time, resample_len):

    time = np.linspace(0, end_time, len(samples))
    t, c, k = interpolate.splrep(time, samples, k=k_spline_degree)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    data_time = np.linspace(0, end_time, resample_len)
    output_spline_data = spline(data_time)

    return output_spline_data

import pandas as pd 

# path = '/Users/Pablo/Documents/GVLAB/Phd/Data Processing/Emotions Dataset/Emotions_Walk_College_de_France/JointAngleData anm_scaled/EMLACOE01.4/LeftFoot.dat'
# df_topex = pd.read_csv(path, header=None, delimiter='\t')
# print(df_topex)

# result = {}
# for k, v in zip((['a','b','c'],'time'), [1,2,4,5]):
#     result[k] = v
# print(result)

import numpy as np

data = np.genfromtxt('Emotions Dataset/Emotions_Walk_College_de_France/anm_scaled/EMLANEE04.4/RightHand.dat',
                     skip_header=0,
                     skip_footer=0,
                     names=True,
                     dtype=None,
                     delimiter='\t')
data = np.array(data.tolist())
print(np.shape(data))
print(np.shape(np.array(data.tolist())))

