#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:08:34 2021

@author: kurata
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch import optim
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn
import numpy as np
from Bio import SeqIO
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from deep_network import Transformer_PHV
import warnings
warnings.simplefilter('ignore')
import joblib
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

amino_asids_vector = np.eye(20)
normal_amino_asids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    return data

def import_fasta(filename):
    data = []
    for record in SeqIO.parse(filename, "fasta"):
        data.append(str(record.seq))
    
    return data

def load_joblib(filename):
    with open(filename, "rb") as f:
        data = joblib.load(f)
    return data

def save_joblib(filename, data):
    with open(filename, "wb") as f:
        joblib.dump(data, f, compress = 3)

def w2v_encoding(seq, model, window, step, k_mer):
    vecs = []
    for i in range(0, len(seq), step):
        pep = seq[i: i+ window]
        if(len(pep) >= k_mer):
            vec = np.mean([model.wv[pep[j: j + k_mer]] for j in range(len(pep) - k_mer + 1)], axis = 0)
        vecs.append(vec)
        
    return vecs

def d2v_encoding(seq, model, window, step, k_mer):
    vecs = []
    for i in range(0, len(seq), step):
        pep = seq[i: i+ window]
        vec = model.infer_vector([pep[j: j + k_mer] for j in range(len(pep) - k_mer + 1)])
        vecs.append(vec)
        
    return vecs

def output_csv(filename, data):
    data.to_csv(filename, index=False)

def transform_seq_scale(seq, stride):
    if(seq%stride == 0):
        return int(seq/stride)
    else:
        return int(seq/stride) + 1

def binary_embed(seq):
    return np.identity(len(normal_amino_asids))[[normal_amino_asids.index(s) for s in seq]]

def encoding_protein(protein_seq_human, protein_seq_virus, enc_model, enc_seq_max, window, stride, k_mer):
    protein_mat_human = enc_model[protein_seq_human]
    protein_mat_virus = enc_model[protein_seq_virus]

    mat_len_human, mat_len_virus = len(protein_mat_human), len(protein_mat_virus)
    w2v_seq_max = enc_seq_max - k_mer + 1

    protein_mat_human = torch.nn.functional.pad(protein_mat_human, (0,0,0, w2v_seq_max - (mat_len_human))).float()
    protein_mat_virus = torch.nn.functional.pad(protein_mat_virus, (0,0,0, w2v_seq_max - (mat_len_virus))).float()

    mat_conv_len_human = max(int((mat_len_human - window)/stride) + 1, 1)
    mat_conv_len_virus = max(int((mat_len_virus - window)/stride) + 1, 1)

    max_conv_len = int((w2v_seq_max - window)/stride) + 1

    w2v_attn_mask_human = torch.cat((torch.full((mat_conv_len_human, max_conv_len), 0).long(), torch.full((max_conv_len - mat_conv_len_human, max_conv_len), 1).long())).long().transpose(-1, -2)
    w2v_attn_mask_virus = torch.cat((torch.full((mat_conv_len_virus, max_conv_len), 0).long(), torch.full((max_conv_len - mat_conv_len_virus, max_conv_len), 1).long())).long().transpose(-1, -2)

    return protein_mat_human, protein_mat_virus, w2v_attn_mask_human.bool(), w2v_attn_mask_virus.bool()

class pv_data_sets(data.Dataset):
    def __init__(self, data_sets, enc_model, enc_seq_max = 9000, window = 20, stride = 10, k_mer = 1):
        super().__init__()
        self.human_protein_id = data_sets["human_id"].values.tolist()
        self.virus_protein_id = data_sets["virus_id"].values.tolist()
        self.human_protein = data_sets["human_seq"].values.tolist()
        self.virus_protein = data_sets["virus_seq"].values.tolist()
        self.enc_model = enc_model
        self.enc_seq_max = enc_seq_max
        self.window, self.stride = window, stride
        self.k_mer = k_mer

    def __len__(self):
        return len(self.human_protein)

    def __getitem__(self, idx):
        protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus = encoding_protein(self.human_protein[idx], self.virus_protein[idx], self.enc_model, enc_seq_max = self.enc_seq_max, window = self.window, stride = self.stride, k_mer = self.k_mer)
        return self.human_protein_id[idx], self.virus_protein_id[idx], protein_mat_human.to(device), protein_mat_virus.to(device), attn_mask_human.to(device), attn_mask_virus.to(device)

class DeepNet():
    def __init__(self, out_path, enc_dict, deep_path, model_params, prediction_params, encoding_params, vec_ind):
        self.out_path = out_path
        self.enc_model = enc_dict
        self.deep_path = deep_path

        self.vec_ind = vec_ind

        self.model_params = model_params
        
        self.batch_size = prediction_params["batch_size"]
        self.enc_seq_max = encoding_params["enc_seq_max"]
        self.thresh = prediction_params["thresh"]
        self.k_mer = encoding_params["k_mer"]
        
    def model_training(self, data_sets):
        os.makedirs(self.out_path, exist_ok=True)

        data_all = pv_data_sets(data_sets, enc_model = self.enc_model, enc_seq_max = self.enc_seq_max, window = self.model_params["kernel_size"], stride = self.model_params["stride"], k_mer = self.k_mer)
        loader = DataLoader(dataset = data_all, batch_size = self.batch_size)
       
        self.model = Transformer_PHV(filter_num = self.model_params["filter_num"], kernel_size_w2v = self.model_params["kernel_size"], stride_w2v = self.model_params["stride"], n_heads = self.model_params["n_heads"], d_dim = self.model_params["d_dim"], feature = self.model_params["feature"], pooling_dropout = self.model_params["pooling_dropout"], linear_dropout = self.model_params["linear_dropout"]).to(device)

        self.model.load_state_dict(torch.load(self.deep_path, map_location = device))
       
        probs_all = []
        
        if(self.vec_ind == True):
            h_out_1_list, h_out_2_list, out_1_list, out_2_list, out_list = [], [], [], [], []

        self.model.eval()
        for i, (human_protein_id, virus_protein_id, protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus) in enumerate(loader):
            with torch.no_grad():
                probs = self.model(protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus)
                probs_list = probs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist()
                probs_all.extend(list(zip(human_protein_id, virus_protein_id, probs_list)))
                
                if(self.vec_ind == True):
                    h_out_1_list.extend(self.model.h_out_1.cpu().clone().detach().numpy())
                    h_out_2_list.extend(self.model.h_out_2.cpu().clone().detach().numpy())
                    out_1_list.extend(self.model.out_1.cpu().clone().detach().numpy())
                    out_2_list.extend(self.model.out_2.cpu().clone().detach().numpy())
                    out_list.extend(self.model.out.cpu().clone().detach().numpy())

        probs_all = pd.DataFrame(probs_all, columns = ["human_ids", "virus_ids", "scores"])
        output_csv(self.out_path + "/probs.csv", probs_all)
            
        if(self.vec_ind == True):
            save_joblib(self.out_path + "/after_cnn_human.joblib", np.array(h_out_1_list))
            save_joblib(self.out_path + "/after_cnn_virus.joblib", np.array(h_out_2_list))
            save_joblib(self.out_path + "/feature_vec_human.joblib", np.array(out_1_list))
            save_joblib(self.out_path + "/feature_vec_virus.joblib", np.array(out_2_list))
            save_joblib(self.out_path + "/concatenated_feature_vec.joblib", np.array(out_list))
        
        return True

def create_mat_dict_w2v(seqs, enc_model, k_mer):
    seqs = list(set(seqs))
    seq2mat_dict = {}
    for i in range(len(seqs)):
        seq2mat_dict[seqs[i]] = torch.tensor([enc_model.wv[seqs[i][j: j + k_mer]] for j in range(len(seqs[i]) - k_mer + 1)])

    return seq2mat_dict

def pred_main(in_path, out_path, w2v_model_path, deep_model_path, vec_ind, thresh, batch_size, k_mer, seq_max):
    model_params = {"filter_num": 128, "kernel_size": 20, "stride": 10, "n_heads": 4, "d_dim": 32, "feature": 128, "pooling_dropout": 0.5, "linear_dropout": 0.3}
    prediction_params = {"batch_size": batch_size, "thresh": thresh}
    encoding_params = {"enc_seq_max": seq_max, "k_mer": k_mer}

    print("Loading datasets", flush = True)
    data = file_input_csv(in_path)

    print("Loading word2vec model", flush = True)
    w2v_model = word2vec.Word2Vec.load(w2v_model_path)

    print("Encoding amino acid sequences", flush = True)
    enc_dict = create_mat_dict_w2v(data["human_seq"].values.tolist() + data["virus_seq"].values.tolist(), w2v_model, encoding_params["k_mer"])

    print("Start prediction", flush = True)
    net = DeepNet(out_path, enc_dict, deep_model_path, model_params, prediction_params, encoding_params, vec_ind)
    res = net.model_training(data)

    print("Finish processing", flush = True)
































