import pandas as pd 
import matplotlib.pyplot as plt 
from pylab import *


input_data_df = pd.read_csv("C:\\Users\\posorio\\OneDrive - 国立研究開発法人産業技術総合研究所\\Documents\\Expressive movement\\Modeling\\distribution_and_similarity_train.csv")
print(input_data_df.columns)

# jd_dist = input_data_df[['mse']]
jd_dist = input_data_df[['cosine_similarity_v_x','cosine_similarity_v_y',
                            'cosine_similarity_v_z','cosine_similarity_av_x',
                            'cosine_similarity_av_y','cosine_similarity_av_z']]
# jd_dist = input_data_df[['time_human_js_dist','weight_human_js_dist','flow_human_js_dist','space_human_js_dist']]
# jd_dist = input_data_df[['time_robot_js_dist','weight_robot_js_dist','flow_robot_js_dist','space_robot_js_dist']]
plt.plot(jd_dist, linewidth=3)
# plt.legend(['Time','Weight','Flow','Space'],fontsize=20,loc='lower right', bbox_to_anchor=(0.9, 0.2))
plt.legend([r'$v_x$',r'$v_y$',r'$v_z$',r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'],fontsize=20,loc='upper right')
ax = gca()
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ylabel('Cosine Similarity', fontsize=20)
xlabel('Lambda Values', fontsize=20)
plt.show()
