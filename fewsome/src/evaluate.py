import torch.nn.functional as F
import torch
from model import *
import os
import numpy as np
import pandas as pd
import argparse
from datasets.main import load_dataset
import random
import time
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, f1_score, cohen_kappa_score
from sklearn import metrics
from scipy import stats



def create_df_of_dists(labels, labels_sev, minimum_dists, means, dataset_name, num_ref_eval, outs, file_names):

    '''create a dataframe where each row represents an image from the test set with columns for the ground truth labels, the minimum ed to the reference set,
    #the mean ed to the reference set and the distances to each feature vector in the reference set.'''


    if dataset_name =='mrart':
        cols = ['file','label','label_sev','minimum_dists', 'means']
        df = pd.concat([pd.DataFrame(file_names, columns = ['file_names']), pd.DataFrame(labels, columns = ['label']), pd.DataFrame(labels_sev, columns = ['label_sev']),  pd.DataFrame(minimum_dists, columns = ['minimum_dists']),  pd.DataFrame(means, columns = ['means'])], axis =1)
    else:
        cols = ['file','label','minimum_dists', 'means']
        df = pd.concat([pd.DataFrame(file_names, columns = ['file_names']),  pd.DataFrame(labels, columns = ['label']), pd.DataFrame(minimum_dists, columns = ['minimum_dists']),  pd.DataFrame(means, columns = ['means'])], axis =1)

    for i in range(0, num_ref_eval):
        df= pd.concat([df, pd.DataFrame(outs['dist_from_ref_vec_{}'.format(i)])], axis =1)
        cols.append('ref_{}'.format(i))

    df.columns=cols
    df = df.sort_values(by='minimum_dists', ascending = False).reset_index(drop=True)

    return df


def create_dict_ref_vecs(ref_dataset, num_ref_eval, anchor, model, freeze, base_ind, dev):
    outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
    ref_images={} #dictionary for feature vectors of reference set
    ind = list(range(0, num_ref_eval))
    random.seed(1)
    np.random.shuffle(ind)

    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
    with torch.no_grad():
        for i in ind:
          img1, _, lab, _ ,_,_= ref_dataset.__getitem__(i)

          if (i == base_ind) & (freeze == True):
            ref_images['images{}'.format(i)] = anchor.to(dev)
          else:
            ref_images['images{}'.format(i)] = model.forward( img1.to(dev).float()).detach()

          del img1
          torch.cuda.empty_cache()
          outs['dist_from_ref_vec_{}'.format(i)] =[]

    return ref_images, outs

def evaluate(anchor, freeze , seed, base_ind, ref_dataset, val_dataset, model, dataset_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval, dev):

    model.eval()

    #create loader for dataset for test set
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)


    ref_images, outs = create_dict_ref_vecs(ref_dataset, num_ref_eval, anchor, model, freeze, base_ind, dev)

    means = [] #stores the mean ED to the reference vectors for a test image
    minimum_dists=[]  #stores the minimum ED to the reference vectors for a test image
    labels=[] #stores the labels of the test set
    labels_sev=[] #stores the severity labels of the test set
    loss_sum =0
    inf_times=[]
    total_times= []
    file_names = []

    #loop through images in the dataloader
    with torch.no_grad():
        for i, data in enumerate(loader):


            image = data[0][0]
            label = data[2].item()
            label_sev = data[4].item()

            file_names.append(data[-1])
            labels.append(label)
            labels_sev.append(label_sev)


            total =0
            mini=torch.Tensor([1e50])
            t1 = time.time()

            out = model.forward(image.to(dev).float()).detach() #get feature vector for test image
            inf_times.append(time.time() - t1)


            #calculate the distance from the test image to each of the feature vectors in the reference set
            for j in range(0, num_ref_eval):
                euclidean_distance = (F.pairwise_distance(out.to(dev), ref_images['images{}'.format(j)].to(dev)) / torch.sqrt(torch.Tensor([out.size()[1]])).to(dev) ) + (alpha*(F.pairwise_distance(out.to(dev), anchor.to(dev)) /torch.sqrt(torch.Tensor([out.size()[1]])).to(dev) ))

                outs['dist_from_ref_vec_{}'.format(j)].append(euclidean_distance.item())

                total += euclidean_distance.item()

                if euclidean_distance.detach().item() < mini:
                  mini = euclidean_distance.item()


            minimum_dists.append(mini)
            means.append(total/len(indexes))
            loss_sum += criterion(out,[ref_images['images{}'.format(j)]], anchor,label, alpha).item()

            total_times.append(time.time()-t1)


            del image
            del out
            del euclidean_distance
            del total
            torch.cuda.empty_cache()



    #create a dataframe where each row represents an image from the test set with columns for the ground truth labels, the minimum ed to the reference set,
    #the mean ed to the reference set and the distances to each feature vector in the reference set.
    df=create_df_of_dists(labels, labels_sev, minimum_dists, means, dataset_name, num_ref_eval, outs, file_names)

    #calculate AUC
    fpr, tpr, thresholds = roc_curve(np.array(df['label']),np.array(df['minimum_dists']))
    auc = metrics.auc(fpr, tpr)


    #calculate threshold and convert scores to binary predictions
    outputs = np.array(df['minimum_dists']).copy()
    if dataset_name == 'mrart':
        t_val = 27
    else:
        t_val = 50
    thres = np.percentile(outputs, t_val)
    outputs[outputs > thres] =1
    outputs[outputs <= thres] =0

    #calculate f1 score and balanced accuracy
    f1 = f1_score(np.array(df['label']),outputs)
    fp = len(df.loc[(outputs == 1 ) & (df['label'] == 0)])
    tn = len(df.loc[(outputs== 0) & (df['label'] == 0)])
    fn = len(df.loc[(outputs == 0) & (df['label'] == 1)])
    tp = len(df.loc[(outputs == 1) & (df['label'] == 1)])
    spec = tn / (fp + tn)
    recall = tp / (tp+fn)
    acc = (recall + spec) / 2
    print('AUC {}'.format(auc))
    print('F1 {}'.format(f1))
    print('Balanced accuracy {}'.format(acc))


    #get severity metrics
    if dataset_name == 'mrart':
        outputs = np.array(df['minimum_dists']).copy()
        thres = np.percentile(outputs, 29)
        thres2 = np.percentile(outputs, 54)
        outputs[outputs > thres2] =2
        outputs[(outputs <= thres2) & (outputs > thres)] =1
        outputs[(outputs <= thres)] =0
        cl0 = len(df.loc[(outputs == 0 ) & (df['label_sev'] == 0)])
        cl1 = len(df.loc[(outputs == 1 ) & (df['label_sev'] == 1)])
        cl2 = len(df.loc[(outputs == 2 ) & (df['label_sev'] == 2)])
        accuracy_sev = (cl0+cl1+cl2) / len(df)
        wck = cohen_kappa_score(df['label_sev'], outputs, weights='linear')
        sprc= stats.spearmanr(outputs.tolist(), df['label_sev'].tolist())[0]
        print('Accuracy with severity {}'.format(accuracy_sev))
        print('WCK {}'.format(wck))
        print('SRC {}'.format(sprc))
    else:
        accuracy_sev = None
        wck=None
        sprc=None

    #calculate average loss
    avg_loss = (loss_sum / num_ref_eval )/ val_dataset.__len__()

    return avg_loss, auc, f1,acc, df, accuracy_sev, wck, sprc, inf_times, total_times
