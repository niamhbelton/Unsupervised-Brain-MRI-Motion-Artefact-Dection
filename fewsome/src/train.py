#from torch.utils.tensorboard import SummaryWriter
import torch
from datasets.main import load_dataset
from model import *
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import evaluate
import random
import time


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, v=0.0,margin=0.9):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.v = v

    def forward(self, output1, vectors, anchor, label, alpha):

        euclidean_distance = torch.FloatTensor([0]).to(args.device)

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1.to(args.device), i.to(args.device))/ torch.sqrt(torch.Tensor([output1.size()[1]])).to(args.device))

        euclidean_distance += alpha*((F.pairwise_distance(output1.to(args.device), anchor.to(args.device))) /torch.sqrt(torch.Tensor([output1.size()[1]])).to(args.device) )

        #calculate the margin
        marg = (len(vectors) + alpha) * self.margin

        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance

        #calculate the loss
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)

        del euclidean_distance
        return loss_contrastive



def create_batches(lst, n):

    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def train(model, lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, dataset_name, freeze, smart_samp, k, eval_epoch,  model_type, bs, num_ref_eval, num_ref_dist, max_patience, shots):
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    train_aucs=[]



    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    if freeze == True:
      np.random.seed(epochs)
      rand_freeze1 = np.random.randint(len(indexes) )
      base_ind = ind[rand_freeze1]
      anchor = init_feat_vec(model,base_ind , train_dataset)


    else:
          anchor = None
          base_ind = -1


    patience = 0
    best_val_auc = 0
    best_f1=0
    best_acc=0
    max_iter = 0
    patience2 = 5
    stop_training = False


    start_time = time.time()

    for epoch in range(epochs):
        model.train()


        loss_sum = 0
        iterations =0
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)

        np.random.shuffle(ind)

        batches = list(create_batches(ind, bs))

        for i in range(int(np.ceil(len(ind) / bs))):


            model.train()
            for batch_ind,index in enumerate(batches[i]):
                iterations+=1
                seed = (epoch+1) * (i+1) * (batch_ind+1)

                img1, img2, labels, base,_ = train_dataset.__getitem__(index, seed, base_ind)



                # Forward
                img1 = img1.to(args.device)
                img2 = img2.to(args.device)
                labels = labels.to(args.device)



                if (freeze == True) & (index ==base_ind):
                  output1 = anchor
                else:
                  output1 = model.forward(img1.float())

                if (smart_samp == 0) & (k>1):

                  vecs=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, k):

                    if (ind2[j] != base_ind) & (index != ind2[j]):

                      output2=model(train_dataset.__getitem__(ind[j], seed, base_ind)[0].to(args.device).float())
                      vecs.append(output2)

                elif smart_samp == 0:

                  if (freeze == True) & (base == True):
                    output2 = anchor
                  else:
                    output2 = model.forward(img2.float())

                  vecs = [output2]

                else:
                  max_eds = [0] * k
                  max_inds = [-1] * k
                  max_ind =-1
                  vectors=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, num_ref_dist):

                    if ((num_ref_dist ==1) & (ind2[j] == base_ind)) | ((num_ref_dist ==1) & (ind2[j] == index)):
                        c = 0
                        while ((ind2[j] == base_ind) | (index == ind2[j])):
                            np.random.seed(seed * c)
                            j = np.random.randint(len(ind) )
                            c = c+1

                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(args.device).float())
                      vectors.append(output2)
                      euclidean_distance = F.pairwise_distance(output1, output2)

                      for b, vec in enumerate(max_eds):
                          if euclidean_distance < vec:
                            max_eds.insert(b, euclidean_distance)
                            max_inds.insert(b, len(vectors)-1)
                            if len(max_eds) > k:
                              max_eds.pop()
                              max_inds.pop()
                            break

                  vecs = []

                  for x in max_inds:
                     vecs.append(vectors[x])

                if batch_ind ==0:
                    loss = criterion(output1,vecs,anchor,labels,alpha)
                else:
                    loss = loss + criterion(output1,vecs,anchor,labels,alpha)

                loss_sum+= loss.item()


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


        train_losses.append((loss_sum / iterations))

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))


        if( epoch > 1) & (eval_epoch == 0):
          decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)

          if decrease <= 1:
            patience += 1


          if  (patience==max_patience) | (epoch == epochs-1):
              stop_training = True


        elif epoch == epochs-1:
            stop_training=True


        if (stop_training == True) | (eval_epoch == 1):

            if (stop_training == True) :
                print("--- %s seconds ---" % (time.time() - start_time))

            training_time = time.time() - start_time


            val_loss, val_auc, f1, acc,df, acc_sev, wck, sprc, inf_times, total_times = evaluate(anchor, freeze, seed, base_ind, train_dataset, val_dataset, model, dataset_name,  model_name, indexes, data_path, criterion, alpha, num_ref_eval,  args.device)
            train_aucs.append(val_auc)
            model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3))
            for f in os.listdir('./outputs/models/'):
              if (model_name in f) :
                  os.remove(f'./outputs/models/{f}')
            torch.save(model.state_dict(), './outputs/models/' + model_name_temp)


            for f in os.listdir('./outputs/ED/'):
              if (model_name in f) :
                  os.remove(f'./outputs/ED/{f}')
            df.to_csv('./outputs/ED/' +model_name_temp)




            cols = ['ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'epoch', 'AUC','training_time', 'f1','acc','acc_sev', 'wck', 'sprc','train loss', 'train_auc' , 'mean_inf_time', 'std_inf_time','total_inf_time', 'std_total_inf_time']
            params = [ args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, alpha, lr, weight_decay, args.vector_size, biases, smart_samp, k, args.v, epoch+1, val_auc, training_time,f1,acc,acc_sev,wck, sprc,train_losses[-1] , train_aucs[-1], np.mean(inf_times), np.std(inf_times), np.mean(total_times), np.std(total_times)]
            try:
                info = pd.concat([info, pd.DataFrame([params], columns = cols)], axis =0)
            except:
                info = pd.DataFrame([params], columns = cols)

            info.to_csv('./outputs/'+ model_name)

            print("Epoch: {}, Validation AUC {}, Validation loss: {}".format(epoch+1, val_auc, val_loss))

            if stop_training == False:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    best_f1 = f1
                    best_acc = acc
                    best_df=df
                    max_iter = 0
                    training_time_temp = training_time


                else:
                    max_iter+=1

                if max_iter == patience2:
                    break

            else:
                break



    print("Finished Training")
    if eval_epoch == 1:
        print("AUC was {} on epoch {}".format(best_val_auc, best_epoch))
        return best_val_auc, best_epoch, best_val_auc, training_time_temp, best_f1, best_acc,train_losses
    else:
        print("AUC was {} on epoch {}".format(val_auc, epoch))
        return val_auc, epoch, val_auc, training_time, f1,acc, train_losses






def init_feat_vec(model,base_ind, train_dataset ):

        model.eval()
        anchor,_,_,_ ,_= train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          anchor = model(anchor.to(args.device).float()).to(args.device)

        return anchor






def create_reference(dataset_name, normal_class,  data_path, download_data, N, seed, task=None):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, task, data_path,download_data= download_data) #get all training data
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0] #get indexes in the training set that are equal to the normal class
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N) #randomly sample N normal data points
    final_indexes = ind[samp]

    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--freeze', default = True)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--num_ref_dist', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--max_patience', type=int, default=4)
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--data_split_path', type=str, default=None)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    model_name = args.model_name
    model_type = args.model_type
    dataset_name = args.dataset
    N = args.num_ref
    seed = args.seed
    freeze = args.freeze
    epochs = args.epochs
    data_path = args.data_path
    download_data = args.download_data
    indexes = args.index
    alpha = args.alpha
    lr = args.lr
    vector_size = args.vector_size
    weight_decay = args.weight_decay
    smart_samp = args.smart_samp
    k = args.k
    weight_init_seed = args.weight_init_seed
    v = args.v
    eval_epoch = args.eval_epoch
    bs = args.batch_size
    biases = args.biases
    num_ref_eval = args.num_ref_eval
    num_ref_dist = args.num_ref_dist
    if num_ref_eval == None:
        num_ref_eval = N
    if num_ref_dist == None:
        num_ref_dist = N


    #create train and test set
    if (dataset_name == 'mrart') :
        normal_class =1
        indexes = create_reference(dataset_name, normal_class,  data_path, download_data, N, seed, task=None)
        train_dataset = load_dataset(dataset_name, indexes,  'train',  data_path, download_data, seed, N=N, data_split_path=args.data_split_path, shots = args.shots)

    elif (dataset_name =='ixi')  :
        train_dataset = load_dataset(dataset_name, indexes,  'train',  data_path, download_data, seed, N=N, data_split_path=args.data_split_path, shots = args.shots)
        indexes = train_dataset.indexes

    val_dataset = load_dataset(dataset_name, indexes,  'test', data_path, download_data=False, seed=seed, N=N, data_split_path=args.data_split_path, shots = args.shots)



    #set the seed
    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)

    #create directories
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    if not os.path.exists('outputs/' + dataset_name):
        os.makedirs('outputs/' + dataset_name)


    if not os.path.exists('outputs/' + dataset_name + '/ED'):
        os.makedirs('outputs/' + dataset_name + '/ED')

    if not os.path.exists('outputs/'+ dataset_name + '/models'):
        os.makedirs('outputs/' + dataset_name + '/models')


    #Initialise the model
    if model_type == 'RESNET':
        model = RESNET_pre(vector_size, biases)
    elif model_type == 'RESNET_batch':
        model =resnet_batch()

    elif model_type == 'ALEXNET':
        model =ALEXNET_pre()


    criterion = ContrastiveLoss(v)
    train(model,lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path,  dataset_name, freeze, smart_samp,k, eval_epoch, model_type, bs, num_ref_eval, num_ref_dist, args.max_patience, args.shots)
