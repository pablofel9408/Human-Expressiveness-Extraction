import os
import sys
import shutil
import fnmatch
import json

def close_script():
    print("Exit singal sent, closing script.....")
    sys.exit()

def findDirs(path,pattern):
    resultList=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if name == pattern:
                resultList.append(os.path.join(root, name))
    return resultList

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def load_raw_constants(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def gen_archive(dirname):
    files = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    if files:
        path = os.path.join(dirname,'Archive//')
        if not os.path.exists(path):
            os.makedirs(path)
        if  not os.path.exists(os.path.join(path,'Version_0.0/')):
            os.makedirs(os.path.join(path,'Version_0.0/'))
            index_oi = 0.0
        else:
            list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
            list_subfolders_with_paths = [float(path_name.split('/')[-1].split('_')[-1]) for path_name in list_subfolders_with_paths]
            index_oi = max(list_subfolders_with_paths)
            index_oi += 0.1
            os.makedirs(os.path.join(path,'Version_'+str(index_oi)+'/'))
        archive_path = os.path.join(path,'Version_'+str(index_oi)+'/')
        files = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
        for f in files:
            shutil.copy(f, archive_path)
            os.remove(f)






