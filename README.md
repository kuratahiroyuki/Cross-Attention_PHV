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
    
## 1. Installing the LSTM-PHV package
Execute the following command in the directory where the package is located.  
`$pip install ./Attention-PHV/dist/Attention-PHV-0.0.1.tar.gz`

## 2. Training of a word2vec embedding model to encode amino acid sequences
A word2vec model can be trained by following command.  
`$lstmphv train_w2v -i [Training data file path (fasta format)] -o [output dir path]`

ex)  
`$lstmphv train_w2v -i ~/LSTM-PHV/sample_data/w2v_sample_data.fa -o ~/LSTM-PHV/w2v_model`

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

## 3. Training of LSTM-PHV model for PPI prediction
LSTM-PHV model for PPI prediction can be trained by following command (Promote the use of GPU-enabled environments).  
`$lstmphv train_deep -t [Training data file path (csv format)] -v [Training data file path (csv format)] -w [word2vec model file path] -o [output dir path]`

ex)  
`$lstmphv train_deep -t ~/LSTM-PHV/sample_data/sample_training_data.csv -v ~/LSTM-PHV/sample_data/sample_validation_data.csv -w ~/LSTM-PHV/w2v_model/AA_model.pt -o ~/LSTM-PHV/deep_model`

Note that csv files need to contein the following contents (Check the sample data at /LSTM-PHV/sample_data)  
First column:human protein IDs  
Second column:viral protein IDs  
Third column:human protein sequences  
Forth column:viral protein sequences  
Fifth column:label (1: interact, 0: not interact)  

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-t (--training_file)|Path of training data file (.csv)|necessary|-|
|-v (--validation_file)|Path of validation data file (.csv)|necessary|-|
|-w (--w2v_model_file)|Path of a trained word2vec model|necessary|-|
|-o (--out_dir)|Directory to output results|necessary|-|
|-l (--losstype)|Loss type (imbalanced: loss function for imbalanced data, balanced: Loss function for balanced data)|not necessary|imbalanced|
|-t_batch (--training_batch_size)|Training batch size|not necessary|1024|
|-v_batch (--validation_batch_size)|Validation batch size|not necessary|1024|
|-lr (--learning_rate)|Learning rate|not necessary|0.001|
|-max_epoch (--max_epoch_num)|Maximum epoch number|not necessary|10000|
|-stop_epoch (--early_stopping_epoch_num)|Epoch number for early stopping|not necessary|20|
|-thr (--threshold)|Threshold to determined whether interact or not|not necessary|0.5|
|-k_mer (--k_mer)|Size of k in k_mer|not necessary|4|

(Results)  
Text and model files will be output to the specified directory.  
Filename: model/deep_model and deep_HV_result.txt
|Filename|contents|
|:----:|:----:|
|model/deep_model|LSTM-PHV model file|
|deep_HV_result.txt|Documenting the learning process|

## 4. PPI prediction
PPI prediction is executed by following command (Promote the use of GPU-enabled environments).  
`$lstmphv predict -i [data file path (csv format)] -o [output dir path] -w [word2vec model file path] -d [deep learning model file path]`

ex)  
`$lstmphv predict -i ~/LSTM-PHV/sample_data/sample_test_data.csv -o ~/LSTM-PHV/results -w ~/LSTM-PHV/w2v_model/AA_model.pt -d ~/LSTM-PHV/deep_model/deep_model`

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-i (--import_file)|Path of data file (.csv)|necessary|-|
|-o (--out_dir)|Directory to output results|necessary|-|
|-w (--w2v_model_file)|Path of a trained word2vec model|necessary|-|
|-d (--deep_model_file)|Path of a trained lstm-phv model|necessary|-|
|-thr (--threshold)|Threshold to determined whether interact or not|not necessary|0.5|
|-batch (--batch_size)|Batch size|not necessary|1024|
|-k_mer (--k_mer)|Size of k in k_mer|not necessary|4|

Note that csv files need to contein the following contents (Check the sample data at /LSTM-PHV/sample_data)  
First column:human protein IDs  
Second column:viral protein IDs  
Third column:human protein sequences  
Forth column:viral protein sequences  

(Results)  
CSV files will be output to the specified directory.  
Filename: result.csv, human_transformed_vec.csv, viral_transformed_vec.csv, human_protein_attention_weights.csv, viral_protein_attention_weights.csv  

|Filename|contents|
|:----:|:----:|
|result.csv|Predictive scores and labels|
|human_transformed_vec.csv|Transformed vector generated by the LSTM-PHV network while extracting human protein features. Each row contains the transformed vector in a sample|
|viral_transformed_vec.csv|Transformed vector generated by the LSTM-PHV network while extracting viral protein features. Each row contains the transformed vector in a sample|
|human_protein_attention_weights.csv|Attention weights generated by the LSTM-PHV network while extracting human protein features. Each row contains the attention weights in a sample|
|viral_protein_attention_weights.csv|Attention weights generated by the LSTM-PHV network while extracting viral protein features. Each row contains the attention weights in a sample|

#  Other contents
We provided sample data, word2vec model, and LSTM-PHV model as well as CLI system.
Note that sample data is not the benchmark datasets and this is only present the example.

              














