import torch
import clip
from PIL import Image
import argparse
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import sys
from scipy import linalg
from tqdm import tqdm
from torch.distributions.normal import Normal
from scipy.stats import multivariate_normal
import shutil
import matplotlib.pyplot as plt
from gaussion_test import gaussion_correlation, gaussion_plot

def del_files_in_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return 0
        
        
def calc_log_prob(batch_feature_source_np, batch_feature_test_np):
    mu1 = np.mean(batch_feature_source_np, axis=0)
    sigma1 = np.cov(batch_feature_source_np, rowvar=False)
    m = multivariate_normal(mu1, sigma1)
    prob = m.pdf(batch_feature_test_np)
    return prob
    
def outlier_detection(d, prob, outlier_precent = 0.002):
    sorted_ind = np.argsort(prob)
    ind = int(np.floor(sorted_ind.shape[0] * outlier_precent))
    low_prob = sorted_ind[:ind]
    high_prob = sorted_ind[-ind:]
    # Check whether the specified path exists or not 
    
    if not os.path.exists('./low_prob_clip'):
        os.makedirs('./low_prob_clip')
    else:
        del_files_in_folder('./low_prob_clip')
    
    if not os.path.exists('./high_prob_clip'):
        os.makedirs('./high_prob_clip')
    else:
        del_files_in_folder('./high_prob_clip')
    
    low_object = open('./low_prob_clip/low.txt', 'a')
    high_object = open('./high_prob_clip/high.txt', 'a')
    
    for i in range(ind):
        path_low_prob = d[low_prob[i]]
        path_high_prob = d[high_prob[i]]
        shutil.copy(path_low_prob, './low_prob_clip/')
        shutil.copy(path_high_prob, './high_prob_clip/')
        low_object.write( path_low_prob + " " + str(i) + '\n')
        high_object.write( path_high_prob + " " + str(i) + '\n')

    low_object.close()
    high_object.close()
      
          
def from_imgs_folder_to_process_tensor(path, preprocess, testFlag = False):
    d = {};
    arr = []
    i=0
    
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in tqdm(filenames):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.JPEG'):
                curr_path = dirpath + "/" + filename
                curr_img = preprocess(Image.open(curr_path))
                arr.append(curr_img)
                d[i] = curr_path
                i+=1
                if not testFlag:
                    if i>15000:
                        break
    imgs_tensor = torch.stack(arr)
    return imgs_tensor, d
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-p1', '--path_source', type=str, default="/path/to/dataset")
parser.add_argument('-p2', '--path_test', type=str, default="/path/to/synthetic")
parser.add_argument('-bs', '--batch_size', type=int, default=32)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
path_test = args.path_test 
fname = path_test.split('/')[-1]+"_fcd.txt"

print("Working on SOURCE\n")

tensor_source, _ = from_imgs_folder_to_process_tensor(args.path_source, preprocess)

print("Working on TEST\n")

tensor_test, dic = from_imgs_folder_to_process_tensor(args.path_test, preprocess, testFlag = True)
tensor_source = tensor_source[:tensor_test.shape[0],:,:,:]   
    
batch_size = args.batch_size

dl_source = torch.utils.data.DataLoader(tensor_source, batch_size=batch_size,shuffle=False)
dl_test = torch.utils.data.DataLoader(tensor_test, batch_size=batch_size,shuffle=False)    

fcd_res = []
feature_tensor_source_arr = []
feature_tensor_test_arr = []

print("Calculate CLIP features - Dataset\n")                         
for i, batch_source in enumerate(tqdm(dl_source)):
    with torch.no_grad():
        batch_feature_tensor_source = model.encode_image(batch_source.to(device))
    feature_tensor_source_arr.append(batch_feature_tensor_source.cpu().numpy())
    
print("Calculate CLIP features - Synthetic images\n") 
for i, batch_test in enumerate(tqdm(dl_test)):
    with torch.no_grad():
        batch_feature_tensor_test = model.encode_image(batch_test.to(device))
    feature_tensor_test_arr.append(batch_feature_tensor_test.cpu().numpy())
        
print("Finish Infernce from CLIP\n")
    
feature_np_test = np.vstack(feature_tensor_test_arr)
feature_np_source = np.vstack(feature_tensor_source_arr)

prob = calc_log_prob(feature_np_source, feature_np_test)
a = outlier_detection(dic, prob)
np.savetxt(fname, [prob])

