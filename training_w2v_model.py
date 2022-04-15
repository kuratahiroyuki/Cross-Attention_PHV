


import os
import pandas as pd
from Bio import SeqIO
from gensim.models import word2vec
import logging

normal_amino_asids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    
    return data

def import_fasta(filename):
    return [str(record.seq) for record in SeqIO.parse(filename, "fasta")]
        

def mer_vec(seq, k_mer = 4):
    return [[seq[i][j:j + k_mer] for j in range(len(seq[i]) - k_mer + 1)] for i in range(len(seq))]

def training_w2v(data_path, out_path, k_mer = 4, vector_size = 128, window_size = 5, iteration = 100):
    seq_list = import_fasta(data_path)
    k_mers_list = mer_vec(seq_list, k_mer = k_mer)
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(k_mers_list, size = vector_size, min_count = 1, window = window_size, iter = iteration, sg = 1)
    
    os.makedirs(out_path, exist_ok = True)
    model.save(out_path + "/AA_model.pt")








































