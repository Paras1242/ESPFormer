import numpy as np
import pandas as pd
import os
import h5py
from sklearn import metrics
from matplotlib import pyplot as plt


def load_csv_file(filename1,filename2, dir1, dir2):
    
    file_path_dir1 = os.path.join(dir1, filename1)
    file_path_dir2 = os.path.join(dir2, filename2)
    if os.path.isfile(file_path_dir1):
        print(f"File {filename1} found in {dir1}.")
        return file_path_dir1
    elif os.path.isfile(file_path_dir2):
        print(f"File not found in {dir1}. Checking {dir2}...")
        print(f"File {filename2} found in {dir2}.")
        return file_path_dir2
    else:
        print("File not found in both directories.")
        return None


base_path = "./results/"
base_path_mit = base_path+"mit/"
base_path_espformer = base_path+"ESPFormer/"
base_path_resnet = base_path + "resnet/"
base_path_longformer = base_path+"longformer/"
base_path_mit_custom = base_path+"mit_custom/"
base_path_espformer_custom = base_path+"ESPFormer_custom/"
base_path_longformer_custom = base_path+"longformer_custom/"
MIT_B0_Path = []
espformer_Path = []
Longfor_Path = []
sph_sop_list = [(1,2),(2,2),(5,2),(1,5),(2,5),(5,5),(1,15),(2,15),(5,15),(1,30),(2,30),(5,30)]
y_prob_valid_list = []
y_true_valid_list = []

for i in range(1,13):
    if i in range(1,10):
        new_i = "0"+str(i)
    j = i-1
    sph = sph_sop_list[j][1]
    sop = sph_sop_list[j][0]
    npy_fname = base_path_resnet + f'tuhszr_sngfld_unscld_unfilt_blcdet_srate256Hz_bmrk{i:02d}_sph{sph:02d}m_sop{sop:02d}m_seg05s_ovr00s_fold00_tuhstd_valid_output.npy'
    csv_fname = base_path_resnet + f'tuhszr_sngfld_unscld_unfilt_blcdet_srate256Hz_bmrk{i:02d}_sph{sph:02d}m_sop{sop:02d}m_seg05s_ovr00s_fold00_tuhstd_valid_labels.csv'
    
    y_prob_valid = np.load(npy_fname)
    y_true_valid = pd.read_csv(csv_fname,header=None).values
    y_prob_valid_list.append(y_prob_valid)
    y_true_valid_list.append(y_true_valid)
    
    file1_prob = f"OG_probablity_BM{new_i}.csv"
    file1_label = f"OG_True_label_BM{new_i}.csv"
    prob_file_name = f"BM{i}_probablity.csv"
    labels_file_name = f"BM{i}_labels.csv"
    
    MIT_B0_Path.append((load_csv_file(file1_prob,prob_file_name,base_path_mit_custom,base_path_mit),
                        load_csv_file(file1_label,labels_file_name,base_path_mit_custom,base_path_mit)))
    
    espformer_Path.append((load_csv_file(file1_prob,prob_file_name,base_path_espformer_custom,base_path_espformer),
                        load_csv_file(file1_label,labels_file_name,base_path_espformer_custom,base_path_espformer)))
    
    Longfor_Path.append((load_csv_file(file1_prob,prob_file_name,base_path_longformer_custom,base_path_longformer),
                         load_csv_file(file1_label,labels_file_name,base_path_longformer_custom,base_path_longformer)))
    
sph = [(0,2),(0,2),(0,2),(1,5),(1,5),(1,5),(2,15),(2,15),(2,15),(3,30),(3,30),(3,30)]
sop = [(0,1),(1,2),(2,5)]*4

num_rows = 4
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3,num_rows*3))

espformer_auc_score = []
mit_auc_score = []
longformer_auc_score = []
resnet_auc_score = []

# need to update range from 0,11 to 0,12
for j in range(0,12):
    y_prob_valid = y_prob_valid_list[j]
    y_true_valid = y_true_valid_list[j]
    roc_auc_resnet = metrics.roc_auc_score(y_true_valid,y_prob_valid)
    resnet_auc_score.append(roc_auc_resnet)
    fpr_list_resnet, tpr_list_resnet, thr_list_resnet = metrics.roc_curve(y_true_valid,y_prob_valid)
    
    Mit_pred_probs = pd.read_csv(MIT_B0_Path[j][0],header=None).values.flatten()
    Mit_true_labels = pd.read_csv(MIT_B0_Path[j][1],header=None).values.flatten()
    roc_auc_mit = metrics.roc_auc_score(Mit_true_labels,Mit_pred_probs)
    mit_auc_score.append(roc_auc_mit)
    fpr_list_Mit, tpr_list_Mit, thr_list_Mit = metrics.roc_curve(Mit_true_labels,Mit_pred_probs)

    espformer_pred_probs = pd.read_csv(espformer_Path[j][0],header=None).values.flatten()
    espformer_true_labels = pd.read_csv(espformer_Path[j][1],header=None).values.flatten()
    roc_auc_espformer = metrics.roc_auc_score(espformer_true_labels,espformer_pred_probs)
    espformer_auc_score.append(roc_auc_espformer)
    fpr_list_espformer, tpr_list_espformer, thr_list_espformer = metrics.roc_curve(espformer_true_labels,espformer_pred_probs)
    

    if j+1 in [4,6,8,9,11,12]:
        Longfor_pred_probs = pd.read_csv(Longfor_Path[j][0]).values.flatten()    
        Longfor_true_labels = pd.read_csv(Longfor_Path[j][1]).values.flatten()
        
        
    else:    
        Longfor_pred_probs = pd.read_csv(Longfor_Path[j][0],header=None).values.flatten() 
        Longfor_true_labels = pd.read_csv(Longfor_Path[j][1],header=None).values.flatten()

    roc_auc_Longfor = metrics.roc_auc_score(Longfor_true_labels,Longfor_pred_probs)
    longformer_auc_score.append(roc_auc_Longfor)
    # if j==10:
        # # print(roc_auc_Longfor,roc_auc_Swinv2,roc_auc_mit)
    fpr_list_Longfor, tpr_list_Longfor, thr_list_Longfor = metrics.roc_curve(Longfor_true_labels,Longfor_pred_probs)

    
    
    # Plot the reference line
    # ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set limits and labels
    ax = axes[sph[j][0], sop[j][0]]
    #(area = {roc_auc_mit:.3f})
    #(area = {roc_auc_Swinv2:.3f})
    #(area = {roc_auc_Longfor:.3f})
    ax.plot(fpr_list_Longfor, tpr_list_Longfor, color='black', lw=3, label=f'LLM',linestyle='-')
    ax.plot(fpr_list_espformer, tpr_list_espformer, color='red', lw=3, label=f'ESPFormer',linestyle='-.')
    ax.plot(fpr_list_Mit, tpr_list_Mit, color='blue', lw=3, label=f'VIT',linestyle='--')
    ax.plot(fpr_list_resnet, tpr_list_resnet, color='green', lw=3, label=f'ResNet',linestyle=':')
    
    
    
    # Set limits and labels
    ax.set_xlim([0.50-0.01,1+0.01])
    # ax.set_ylim([0.75-0.01,1+0.01])
    ax.set_ylim([0.50-0.01,1+0.01])
    ax.set_xticks([0, 0.50, 1.00], labels = ['0.0', '0.5', '1.0'], fontsize="16")
    # ax.set_yticks([0.75, 1.00])
    ax.set_yticks([0, 0.50, 1.00], labels = ['0.0', '0.5', '1.0'], fontsize="16")
    ax.set_xlabel('FPR', fontsize="16")
    ax.set_ylabel('TPR', fontsize="16")
    ax.set_title(f'BM{j+1}', fontsize="16")#\nSOP = {sop[j][1]} mins SPH = {sph[j][1]} mins')
    ax.grid()
    
    # Add legend
    if j==10:
        ax.legend(loc="lower right", fontsize="11")
    
fig.tight_layout()
plt.draw()
plt.savefig('ROC_AUC_CURVE_BM1To12_v01.jpeg', dpi=300)
plt.show()

# create_excel(swin_auc_score,mit_auc_score, longformer_auc_score, resnet_auc_score)   
