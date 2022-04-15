# Attention-PHV
This package is used for protein-protein interaction (PPI) prediction

# Features
・Attention-PHV predicts PPI by amino acid sequences alone.    

# Environment
    Python   : 3.8.0
    Anaconda : 4.9.2
※We recommend creating virtual environments by using anaconda.

# Processing
 This CLI system is used for three processing as follows.  
 ・Training of a word2vec embedding model to encode amino acid sequences.  
 ・Training of Attention-PHV model for PPI prediction.  
 ・PPI prediction.  

# preparation and installation
## 0. Preparation of a virtual environment (not necessary)
0-1. Creating a virtual environment.  
    `$conda create -n [virtual environment name] python==3.8.0`
    ex)  
    `$conda create -n attention_phv_network python==3.8.0`
      
0-2. Activating the virtual environment  
    `$ conda activate [virtual environment name]`
    ex)  
    `$ conda activate attention_phv_network`
    
## 1. Installing the Attention-PHV package
Execute the following command in the directory where the package is located.  
`$pip install ./Attention-PHV/dist/Attention-PHV-0.0.1.tar.gz`

## 2. Training of a word2vec embedding model to encode amino acid sequences
A word2vec model can be trained by following command.  
`$aphv train_w2v -i [Training data file path (fasta format)] -o [output dir path]`

ex)  
`$aphv train_w2v -i ~/Attention-PHV/sample_data/w2v_sample_data.fa -o ~/Attention-PHV/w2v_model`

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-i (--import_file)|Path of training data (.fasta)|necessary|-|
|-o (--out_dir)|Directory to save w2v model|necessary|-|
|-k_mer (--k_mer)|Size of k in k_mer|not necessary|4|
|-v_s (--vector_size)|Vector size|not necessary|128|
|-w_s (--window_size)|Window size|not necessary|3|
|-iter (--iteration)|Iteration of training|not necessary|1000|

(Results)  
Model files will be output to the specified directory.  
Filename: AA_model.pt, AA_model.pt.trainables.syn1neg.npy, AA_model.pt.wv.vectors.npy  

|Filename|contents|
|:----:|:----:|
|AA_model.pt|word2vec model file|
|AA_model.pt.trainables.syn1neg.npy|word2vec model file (depending on the model size)|
|AA_model.pt.wv.vectors.npy|word2vec model file (depending on the model size)|

## 3. Training of Attention-PHV model for PPI prediction
Attention-PHV model for PPI prediction can be trained by following command (Promote the use of GPU-enabled environments).  
`$aphv train_deep -t [Training data file path (csv format)] -v [Training data file path (csv format)] -w [word2vec model file path] -o [output dir path]`

ex)  
`$aphv train_deep -t ~/Attention-PHV/sample_data/train.csv -v ~/Attention-PHV/sample_data/val.csv -w ~/Attention-PHV/w2v_model/AA_model.pt -o ~/Attention-PHV/deep_model`

Note that csv files need to contein the following contents (Check the sample data at /Attention-PHV/sample_data)  
First column (human_id):human protein IDs  
Second column (human_seq):human protein sequences   
Third column (virus_id):viral protein IDs 
Forth column (virus_seq):viral protein sequences  
Fifth column (labels):label (1: interact, 0: not interact)  

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-t (--training_file)|Path of training data file (.csv)|necessary|-|
|-v (--validation_file)|Path of validation data file (.csv)|necessary|-|
|-w (--w2v_model_file)|Path of a trained word2vec model|necessary|-|
|-o (--out_dir)|Directory to output results|necessary|-|
|-t_batch (--training_batch_size)|Training batch size|not necessary|32|
|-v_batch (--validation_batch_size)|Validation batch size|not necessary|32|
|-lr (--learning_rate)|Learning rate|not necessary|0.0001|
|-max_epoch (--max_epoch_num)|Maximum epoch number|not necessary|10000|
|-stop_epoch (--early_stopping_epoch_num)|Epoch number for early stopping|not necessary|20|
|-thr (--threshold)|Threshold to determined whether interact or not|not necessary|0.5|
|-k_mer (--k_mer)|Size of k in k_mer|not necessary|4|
|-max_len (--max_len)|Maximum sequence length|not necessary|9000|

(Results)  
Text and model files will be output to the specified directory.  
Filename: model/deep_model and deep_HV_result.txt
|Filename|contents|
|:----:|:----:|
|model/deep_model|Attention-PHV model file|

## 4. PPI prediction
PPI prediction is executed by following command (Promote the use of GPU-enabled environments).  
`$aphv predict -i [data file path (csv format)] -o [output dir path] -w [word2vec model file path] -d [deep learning model file path]`

ex)  
`$aphv predict -i ~/Attention-PHV/sample_data/test.csv -o ~/Attention-PHV/results -w ~/Attention-PHV/w2v_model/AA_model.pt -d ~/Attention-PHV/deep_model/deep_model`

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-i (--import_file)|Path of data file (.csv)|necessary|-|
|-o (--out_dir)|Directory to output results|necessary|-|
|-w (--w2v_model_file)|Path of a trained word2vec model|necessary|-|
|-d (--deep_model_file)|Path of a trained attention-phv model|necessary|-|
|-vec (--vec_index)|Flag whether features output|not necessary|False|
|-thr (--threshold)|Threshold to determined whether interact or not|not necessary|0.5|
|-batch (--batch_size)|Batch size|not necessary|32|
|-k_mer (--k_mer)|Size of k in k_mer|not necessary|4|
|-max_len (--max_len)|Maximum sequence length|not necessary|9000|

Note that csv files need to contein the following contents (Check the sample data at /Attention-PHV/sample_data)  
First column (human_id):human protein IDs  
Second column (human_seq):human protein sequences  
Third column (virus_id):viral protein IDs  
Forth column (virus_seq):viral protein sequences  

(Results)  
CSV files will be output to the specified directory.  
Filename: probs.csv, after_cnn_human.joblib, after_cnn_virus.joblib, feature_vec_human.joblib, feature_vec_virus.joblib, concatenated_feature_vec.joblib

|Filename|contents|
|:----:|:----:|
|probs.csv|Predictive scores|
|after_cnn_human.joblib|Hidden matrixes generated by network in human|
|after_cnn_virus.joblib|Hidden matrixes generated by network in virus|
|feature_vec_human.joblib|Feature vectors generated by network in human|
|feature_vec_virus.joblib|Feature vectors generated by network in virus|
|concatenated_feature_vec.joblib|Concatenated feature vectors|

#  Other contents
We provided sample data, word2vec model, and Attention-PHV model as well as CLI system.
Note that sample data is not the benchmark datasets and this is only present the example.

              














