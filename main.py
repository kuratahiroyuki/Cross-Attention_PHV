
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
common_path = os.path.abspath("./")
import argparse

import warnings
warnings.simplefilter('ignore')

from training_w2v_model import training_w2v
from training_model_word2vec import training_main
from test_model_word2vec import pred_main

def training_deep_model(args):
    train_data_path = args.training_file
    val_data_path = args.validation_file
    w2v_model_path = args.w2v_model_file
    out_path = args.out_dir
    t_batch = args.training_batch_size
    v_batch = args.validation_batch_size
    lr = args.learning_rate
    max_epoch = args.max_epoch_num
    stop_epoch = args.early_stopping_epoch_num
    threshold = args.threshold
    k_mer = args.k_mer
    seq_max = args.max_len
    
    training_main(train_data_path, val_data_path, w2v_model_path, out_path, t_batch = t_batch, v_batch = v_batch, lr = lr, max_epoch = max_epoch, stop_epoch = stop_epoch, thr = threshold, k_mer = k_mer, seq_max = seq_max)
    
def prediction(args):
    in_path = args.import_file
    out_path = args.out_dir
    w2v_model_path = args.w2v_model_file
    deep_model_path = args.deep_model_file
    thresh = args.threshold
    batch_size = args.batch_size
    k_mer = args.k_mer
    seq_max = args.max_len
    vec_ind = args.vec_index
    
    pred_main(in_path, out_path, w2v_model_path, deep_model_path, vec_ind, thresh, batch_size, k_mer, seq_max)
    
def training_w2v_model(args):
    data_path = args.import_file
    out_path = args.out_dir
    k_mer = args.k_mer
    vector_size = args.vector_size
    window_size = args.window_size
    iteration = args.iteration

    training_w2v(data_path, out_path, k_mer, vector_size, window_size, iteration)
    
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="sub_command", help="aphv: this is a CLI to use Cross-Attention PHV")
    subparsers.required = True
    
    deep_train_parser = subparsers.add_parser("train_deep", help = "sub-command <train> is used for training a deep learning model for PPI prediction")
    w2v_train_parser = subparsers.add_parser("train_w2v", help = "sub-command <train> is used for training a w2v model to encode amino acid sequences")
    pred_parser = subparsers.add_parser("predict", help = "sub-command <predict> is used for prediction of PPI")
    
    deep_train_parser.add_argument('-t', '--training_file', help = 'Path of training data file (.csv)', required = True)
    deep_train_parser.add_argument('-v', '--validation_file', help = 'Path of validation data file (.csv)', required = True)
    deep_train_parser.add_argument('-w', '--w2v_model_file', help = 'Path of a trained word2vec model', required = True)
    deep_train_parser.add_argument('-o', '--out_dir', help = 'Directory to output results', required = True)
    deep_train_parser.add_argument('-t_batch', '--training_batch_size', help = 'Training batch size', default = 32, type = int)
    deep_train_parser.add_argument('-v_batch', '--validation_batch_size', help = 'Validation batch size', default = 32, type = int)
    deep_train_parser.add_argument('-lr', '--learning_rate', help = 'Learning rate', default = 0.0001, type = float)
    deep_train_parser.add_argument('-max_epoch', '--max_epoch_num', help = 'Maximum epoch number', default = 10000, type = int)
    deep_train_parser.add_argument('-stop_epoch', '--early_stopping_epoch_num', help = 'Epoch number for early stopping', default = 20, type = int)
    deep_train_parser.add_argument('-thr', '--threshold', help = 'Threshold to determined whether interact or not', default = 0.5, type = float)
    deep_train_parser.add_argument('-k_mer', '--k_mer', help = 'Size of k in k_mer', default = 4, type = int)
    deep_train_parser.add_argument('-max_len', '--max_len', help = 'Maximum sequence length', default = 9000, type = int)
    deep_train_parser.set_defaults(handler = training_deep_model)
    
    pred_parser.add_argument('-i', '--import_file', help = 'Path of data file (.csv)', required = True)
    pred_parser.add_argument('-o', '--out_dir', help = 'Directory to output results', required = True)
    pred_parser.add_argument('-w', '--w2v_model_file', help = 'Path of a trained word2vec model', required = True)
    pred_parser.add_argument('-d', '--deep_model_file', help = 'Path of a trained aphv model', required = True)
    pred_parser.add_argument('-vec', '--vec_index', help = 'Flag whether features output', action='store_true', default = False)
    pred_parser.add_argument('-thr', '--threshold', help = 'Threshold to determined whether interact or not', default = 0.5, type = float)
    pred_parser.add_argument('-batch', '--batch_size', help = 'Batch size', default = 32, type = int)
    pred_parser.add_argument('-k_mer', '--k_mer', help = 'Size of k in k_mer', default = 4, type = int)
    pred_parser.add_argument('-max_len', '--max_len', help = 'Maximum sequence length', default = 9000, type = int)
    pred_parser.set_defaults(handler = prediction)
    
    w2v_train_parser.add_argument("-i", "--import_file", help = "Path of training data (.fasta)", required = True)
    w2v_train_parser.add_argument("-o", "--out_dir", help = "Directory to save w2v model", required = True)
    w2v_train_parser.add_argument("-k_mer", "--k_mer", help='size of k in k_mer', default = 4, type = int)
    w2v_train_parser.add_argument("-v_s", "--vector_size", help = "Vector size", default = 128, type = int)
    w2v_train_parser.add_argument("-w_s", "--window_size", help = "Window size", default = 5, type = int)
    w2v_train_parser.add_argument("-iter", "--iteration", type = int, default = 100, help = "Iteration of training")
    w2v_train_parser.set_defaults(handler = training_w2v_model)
    
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    
   



























