# Unsupervised-Brain-MRI-Motion-Artefact-Dection

# Unsupervised-Motion-Artefact-Detection

## Installation 

```
git clone https://github.com/niamhbelton/Unsupervised-Motion-Artefact-Detection.git
```

## Data 
Each model was run for five seeds, The folder metadata contains the train-test splits for both datasets for each train set size and each seed for sampling the train set, named as 'df_seed_<seed>_n_<train_set_size>'. The files in the 'MR-ART' dataset indicate the subject ID for the training set but it specifies the complete MRI name for the validation data.

### MR-ART
* Download the data from the following link; https://openneuro.org/datasets/ds004173/versions/1.0.2
* Change the file paths in the notebook 'Data_Prep/convert_mrart_to_png.ipynb' and run the notebook to convert each MR-ART MRI to individual slices of type png. The code also splits the data into folders 'ones', 'twos' and 'threes' depending on their quality assessment score as given in scores.tsv.

### IXI
* To generate the synthetic artefacts for the IXI dataset; 
  * download the T2 images from the following link; https://brain-development.org/ixi-dataset/
  * modify the file paths 'genDir' and 't2Path' in Data_Prep/MotionUtils/GenerateMotion.py to the directory where you want to store the generated files and the directory where original data is. Code originally from; https://github.com/antecessor/MRI_Motion_Classification/tree/master/Utils/MotionUtils.
  * change the directory paths in the Data_Prep/move_ixi_files_to_directories.ipynb notebook and run the code to split the generated data into directories 'anom' and 'normal'
  
```
cd <Unsupervised-Motion-Artefact-Detection/Data_Prep/MotionUtils>
python3 GenerateMotion.py
```





## Models

```
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
cd <path-to-algorithms-directory>
pip install -r requirements.txt
```


### FewSOME


### DeepSVDD

Code based on: https://github.com/lukasruff/Deep-SVDD

Run the following commands before running the model. The 'xp_path' is the directory to write the output files to.

```
cd <path-to-Deep-SVDD-directory>
mkdir <xp_path>
cd src
```

The arguments to run the model are shown below;

```
@click.argument('dataset_name', type=click.Choice(['ixi', 'mrart','mnist', 'fmnist', 'cifar10']))
@click.argument('net_name',type=click.Choice(['MVTEC_LeNet', 'FMNIST_LeNet', 'CIFAR10_LeNet', 'MNIST_LeNet']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path',  type=click.Path(exists=True))
@click.option('--pollution',  help='Percentage of training data to be polluted with anomalies.', default=0.0)
@click.option('--n',   help='Number of data instances to train on.' , default=0)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--early_stopping_loss', type=int, default=0,
              help='early stopping depending on training loss')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--eval_epoch', type=int, default=0,
              help='Set to one to evaluate on test set after each epoch.')
@click.option('--data_split_path', type=str, default='df',
              help='Path to metadata to confirm correct train-test split.')
              
```




Below is an example command for training the model on 10 MRIs from the IXI dataset. The model is evaluated on the test set after each epoch.


```
python main.py ixi  MVTEC_LeNet <xp_path> <path_to_data> --data_split_path <path_to_metadata>/ixi --eval_epoch 1  --device cuda:2 --n  10  --objective one-class --seed 1001 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True  --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 1
```

Below is an example command for training the model on 10 MRIs from the MR-ART dataset. The model is evaluated on the test set after each epoch.
```
python main.py mrart  MVTEC_LeNet  <xp_path> <path_to_data> --data_split_path <path_to_metadata>/mrart --eval_epoch 1  --device cuda:0 --n 10  --objective one-class --seed 1001 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True  --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 1
```


#### Output Files

The output for the above command is 
* A file named final_data_n_<training_set_size>_seed_<seed> is output to the xp_path with results based on the epoch with the best AUC. The columns are as follows; 
  * 'output' - anomaly score (based on mean anomaly scores of slices) 
  * 'output1' - anomaly score (based on max anomaly score of slices)
  * 'label' - 0 for normals, 1 for anomalies. 
  * 'label_sev' - ignore for IXI dataset, only applicable to 'MR-ART', 0 for normal, 1 for medium quality and 2 for good quality.
  * 'pred' - converting 'output' to binary value of 0 and 1 using a threshold based on the class balance
  * 'pred2' - converting 'output1' to binary value of 0 and 1 using a threshold based on the class balance

* A log file named log.txt is output to the xp_path with training and testing details. This provides details on the AUC, F1, Balanced accuracy and inference time.



## References
 
 XI Dataset (2019), https://brain-development.org/ixi-dataset/

Mohebbian, M., Walia, E., Habibullah, M., Stapleton, S. and Wahid, K.A., 2021. Classifying MRI motion severity using a stacked ensemble approach. Magnetic Resonance Imaging, 75, pp.107-115.
 
 Nárai,  ́A., Hermann, P., Auer, T., Kemenczky, P., Szalma, J., Homolya, I., Somogyi, E., Vakli, P., Weiss, B. and Vidny ́anszky, Z. (2022), ‘Movement-related artefacts (mr-art) dataset of matched motion-corrupted and clean structural mri brain scans’, Scientific Data 9(1), 1–6.


     
