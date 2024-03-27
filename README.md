# Unsupervised-Brain-MRI-Motion-Artefact-Dection

## Installation 

```
git clone https://github.com/niamhbelton/Unsupervised-Brain-MRI-Motion-Artefact-Detection.git
```

## Data 
The folder metadata contains the train-test splits for both datasets for each train set size and each seed for sampling the train set, named as 'df_seed_<seed>_n\_<train_set_size>'. The files in the 'MR-ART' dataset indicate the subject ID for the training set but it specifies the complete MRI name for the validation data.

### MR-ART
* Download the data from the following link; https://openneuro.org/datasets/ds004173/versions/1.0.2
* Change the file paths in the notebook 'data_prep/convert_mrart_to_png.ipynb' and run the notebook to convert each MR-ART MRI to individual slices of type png. The code also splits the data into folders 'ones', 'twos' and 'threes' depending on their quality assessment score as given in scores.tsv.

### IXI
* To generate the synthetic artefacts for the IXI dataset; 
  * download the T2 images from the following link; https://brain-development.org/ixi-dataset/
  * in data_prep/MotionUtils/motion_sim.py, modify the file paths 'data_path' to where the IXI-T2 data is stored and the 'output_dir_anom' and  'output_dir_normal' paths to where the new data will be written to. Code originally from; [https://github.com/antecessor/MRI_Motion_Classification/tree/master/Utils/MotionUtils](https://github.com/bduffy0/motion-correction).
 


## Models

```
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
cd <path-to-fewsome-directory>
pip install -r requirements.txt
```


### FewSOME

Change the paths in 'fewsome/run_ixi.py' and 'fewsome/run_mrart.py' files and run the below commands;
```
python3 run_ixi.py
python3 run_mrart.py
```

#### Output Files

For each model, 
* a file will be generated in the 'outputs' directory named according to model_name argument containing details of training and evaluation metrics
* the euclidean distance for each test point to each reference vector is stored in '/outputs/ED/'
* the final model is stored in 'outputs/models/'


## References
 
 XI Dataset (2019), https://brain-development.org/ixi-dataset/

Mohebbian, M., Walia, E., Habibullah, M., Stapleton, S. and Wahid, K.A., 2021. Classifying MRI motion severity using a stacked ensemble approach. Magnetic Resonance Imaging, 75, pp.107-115.
 
Nárai,  ́A., Hermann, P., Auer, T., Kemenczky, P., Szalma, J., Homolya, I., Somogyi, E., Vakli, P., Weiss, B. and Vidny ́anszky, Z. (2022), ‘Movement-related artefacts (mr-art) dataset of matched motion-corrupted and clean structural mri brain scans’, Scientific Data 9(1), 1–6.


     
