# transfer-learning-in-plankton-image-classification
Code and models for the paper: "In-domain versus out-of-domain transfer learning in plankton image classification"

The code is supposed to work with many python/pytorch/cuda versions but for our experiments we used: 

```
Python 3.7.6 
PyTorch 1.12.1 
CUDA 10.2
```

we report here a brief description of functions available in the code.

### Downloading the datasets
The datasets can be downloaded by running the script `run_download_dataset.py`. There is the possibility that the urls of datasets will be changed in the future. In this case the user can find the new urls and update the entries in the config file `config/datasets.yaml` where current urls are stored. 

### Download pre-trained models
To download the pre-trained weights of models it is possible to use the script `run_download_pretrained.py`. This is helpful when a multi-gpu training is performed, since if the pretrained weights are not already available, all the processes would start a download of the weights. Running the mentioned script prevents this from happening.

### Split data 
Using the file `run_split_data.py` it is possible to randomly split a dataset into train and test set. In particular the function moves a predefined percentage of images into another folder to build the test set.
The function accepts two arguments: in_path (the path of the full input dataset, that will be the training set), out_path (the path where to move the images to be used as test set).

### Padding the images to squares
The file `run_pad_dataset.py` enables to pad all dataset images to squares with the value passed as `fill_value` (by default 255). Additionally the user should specify the input directory of the dataset to be padded and the output directory (where to save the padded dataset).

### Crop ZooScan
To crop ZooScan automatically how described in the paper it is possible to use the script `run_crop_zooscan.py`.

### Finetuning a model
To fine-tune models the script `run_finetuning.py` can be used. The base configuration of the architectures is store in the `config/architecture.yaml`, while the base configuration for the fine-tuning is stored in the file `config/training.yaml`. The user can try to change these files to experiments with different hyperparameters.
The output of the finetuning will be saved, by default in a file called `finetuning_output.csv`.
```
--dataset_train: PATH TO THE TRAIN DATASET
--dataset_test: PATH TO THE TEST DATASET
--ckpt_load: PATH TO A CKPT TO LOAD BEFORE STARTING THE FINETUNING
--ckpt_save: PATH THE FILE WHERE TO SAVE THE CKPT
--model_name THE NAME OF THE MODEL TO USE
--batch_size: THE BATCH SIZE FOR TRAINING
--accumulation: THE NUMBER OF GRADIENT ACCUMULATIONS TO USE
```

### Evaluate a model
Using the script `run_eval_f1.py` it is possible to evaluate a model. The user needs to set the following arguments to run the function:
```
--model_name: THE MODEL NAME TO USE 
--dataset_name: THE NAME OF THE DATASET (THE TEST SET TO USE)
--ckpt_load: THE PATH TO THE CKPT OF THE MODEL (OUTPUT OF THE FINETUNING FOR EXAMPLE)
--batch_size: THE BATCH SIZE TO USE.
--out: OUTPUT FILE TO SAVE RESULTS.
```
Both the Accuracy and F1-score will be computed (macro and micro averages). 

### Save logits/features
To experiments with ensembles it might be useful to save the features/logits of one of more models. 
The user can use `run_extract_logits.py` and `run_extract_features.py` to accomplish these tasks.
