
from .mrart import MRART
from .ixi import ixi

def load_dataset(dataset_name, indexes, task, data_path,  download_data,seed=None, N=None,data_split_path=None,shots=0,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ( 'mrart')

    dataset = None


    if dataset_name == 'mrart':
        dataset = MRART(indexes = indexes,
                                root=data_path,
                                task = task,
                                seed=seed,
                                N=N,
                                data_split_path=data_split_path)



    if dataset_name == 'ixi':
        dataset = ixi(indexes = indexes,
                                root=data_path,
                                task = task,
                                seed=seed,
                                N=N,
                                data_split_path=data_split_path)


    return dataset
