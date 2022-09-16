import os
import utilities

from loadEmotionData import loadEmotionData
from processDataset import preprocessEmotionData
from postprocessDataset import postprocessEmotionData

def aux_plot(constants, candidates_dict,emotion_dataset_obj):
    for j in constants['candidate']:
        for i in candidates_dict[j].keys():
            for k in constants['new_markers']:
                print(j+'_'+k)
                emotion_dataset_obj.plot_mocap_data(candidate_val=j,traj_val=i, marker=j+'_'+k)

    print(candidates_dict['EMLA'].keys())

def main():

    # dirname = os.path.dirname(__file__)
    # dirname[0] = dirname[0].upper()
    dirname = "C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Main Scripts\\Data acquisition\\Human Expressive Data\\Data  Engineering Emotion Dataset\\Data  Engineering Emotion Dataset"
    filename = os.path.join(dirname, 'config_dataEng.json')
    constants = utilities.load_json(filename)

    if not os.path.exists(os.path.join(dirname,'Dataset\\')):
        os.makedirs(os.path.join(dirname,'Dataset\\'))

    constants['dataset_dirpath'] = os.path.join(dirname,'Dataset\\')

    emotion_dataset_obj = loadEmotionData(constants)
    emotion_dataset_obj.start_main_process()
    candidates_dict = emotion_dataset_obj.return_data()
    # print(candidates_dict)

    ## Auxiliary plot code to see the whole dataset
    # aux_plot(constants, candidates_dict,emotion_dataset_obj)

    preprocess_obj = preprocessEmotionData(constants)
    preprocess_obj.load_data(candidates_dict)
    preprocess_obj.start_preprocess()
    # preprocess_obj.plot_coordinates_sing_ex()
    processed_candidates_dict = preprocess_obj.return_processed_data()

    postprocess_obj = postprocessEmotionData(constants)
    postprocess_obj.load_data(processed_candidates_dict)
    postprocess_obj.start_preprocess()

if __name__=="__main__":
    main()