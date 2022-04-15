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
#import torch_optimizer as optim
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
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def transform_seq_scale(seq, stride):
    if(seq%stride == 0):
        return int(seq/stride)
    else:
        return int(seq/stride) + 1

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
        self.human_protein = data_sets["human_seq"].values.tolist()
        self.virus_protein = data_sets["virus_seq"].values.tolist()
        self.y = np.array(data_sets["labels"].values.tolist()).reshape([len(data_sets["labels"]),1])
        self.enc_model = enc_model
        self.enc_seq_max = enc_seq_max
        self.window, self.stride = window, stride
        self.k_mer = k_mer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus = encoding_protein(self.human_protein[idx], self.virus_protein[idx], self.enc_model, enc_seq_max = self.enc_seq_max, window = self.window, stride = self.stride, k_mer = self.k_mer)
        return protein_mat_human.to(device), protein_mat_virus.to(device), attn_mask_human.to(device), attn_mask_virus.to(device), torch.tensor(self.y[idx], device=device, dtype=torch.float)

class DeepNet():
    def __init__(self, out_path, enc_dict, model_params, training_params, encoding_params):
        self.out_path = out_path
        self.enc_model = enc_dict
        
        self.model_params = model_params
        
        self.tra_batch_size = training_params["training_batch_size"]
        self.val_batch_size = training_params["validation_batch_size"]
        self.lr = training_params["lr"]
        self.enc_seq_max_train = encoding_params["enc_seq_max_train"]
        self.enc_seq_max_val = encoding_params["enc_seq_max_val"]
        self.max_epoch = training_params["max_epoch"]
        self.early_stop = training_params["early_stopping"]
        self.thresh = training_params["thresh"]
        self.k_mer = encoding_params["k_mer"]
        self.stopping_met = training_params["stopping_met"]
        
    def model_training(self, train_data_sets, val_data_sets):
        os.makedirs(self.out_path + "/data_model", exist_ok=True)
       
        tra_data_all = pv_data_sets(train_data_sets, enc_model = self.enc_model, enc_seq_max = self.enc_seq_max_train, window = self.model_params["kernel_size"], stride = self.model_params["stride"], k_mer = self.k_mer)
        train_loader = DataLoader(dataset = tra_data_all, batch_size = self.tra_batch_size, shuffle=True)

        val_data_all = pv_data_sets(val_data_sets, enc_model = self.enc_model, enc_seq_max = self.enc_seq_max_val, window = self.model_params["kernel_size"], stride = self.model_params["stride"], k_mer = self.k_mer)
        val_loader = DataLoader(dataset = val_data_all, batch_size = self.val_batch_size, shuffle=True)
        
        self.model = Transformer_PHV(filter_num = self.model_params["filter_num"], kernel_size_w2v = self.model_params["kernel_size"], stride_w2v = self.model_params["stride"], n_heads = self.model_params["n_heads"], d_dim = self.model_params["d_dim"], feature = self.model_params["feature"], pooling_dropout = self.model_params["pooling_dropout"], linear_dropout = self.model_params["linear_dropout"]).to(device)
        
        self.opt = optim.Adam(params = self.model.parameters(), lr = self.lr)
        self.criterion = torch.nn.BCELoss()
       
        max_met = 100 
        early_stop_count = 0
        
        for epoch in range(self.max_epoch):
            training_losses, validation_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []
            self.model.train()
            for i, (protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus, labels) in enumerate(train_loader):
                self.opt.zero_grad()
                probs = self.model(protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus)
                
                loss = self.criterion(probs, labels)

                loss.backward()
                self.opt.step()
                training_losses.append(loss)
                train_probs.extend(probs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                train_labels.extend(labels.cpu().clone().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
                
            loss_epoch = self.criterion(torch.tensor(train_probs).float(), torch.tensor(train_labels).float())

            print("=============================", flush = True)
            print("training loss:: " + str(loss_epoch), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](train_labels, train_probs, thresh = self.thresh)
                else:
                    metrics = metrics_dict[key](train_labels, train_probs)
                print("train_" + key + ": " + str(metrics), flush=True) 
            
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_probs, thresh = self.thresh)
            print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)

            print("-----------------------------", flush = True)
            self.model.eval()
            for i, (protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus, labels) in enumerate(val_loader):
                with torch.no_grad():
                    probs = self.model(protein_mat_human, protein_mat_virus, attn_mask_human, attn_mask_virus)
                    
                    loss = self.criterion(probs, labels)

                    validation_losses.append(loss)
                    val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
       
            loss_epoch = self.criterion(torch.tensor(val_probs).float(), torch.tensor(val_labels).float())
            
            print("validation loss:: "+ str(loss_epoch), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](val_labels, val_probs, thresh = self.thresh)
                else:
                    metrics = metrics_dict[key](val_labels, val_probs)
                print("validation_" + key + ": " + str(metrics), flush=True)

            if(self.stopping_met == "loss"):
                epoch_met = loss_epoch
            else:
                epoch_met = 1 - metrics_dict[self.stopping_met](val_labels, val_probs)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_probs, thresh = self.thresh)
            print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), flush=True)
            print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), flush=True)
            print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), flush=True)
            print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), flush=True)
                
            if epoch_met < max_met:
                early_stop_count = 0
                max_met = epoch_met
                os.makedirs(self.out_path + "/data_model", exist_ok=True)
                os.chdir(self.out_path + "/data_model")
                torch.save(self.model.state_dict(), "deep_model")
                final_val_probs = val_probs
                final_val_labels = val_labels
                final_train_probs = train_probs
                final_train_labels = train_labels
                    
            else:
                early_stop_count += 1
                if early_stop_count >= self.early_stop:
                    print('Traning can not improve from epoch {}\tBest {}: {}'.format(epoch + 1 - self.early_stop, self.stopping_met, max_met), flush=True)
                    break

        print(self.thresh, flush=True)
        for key in metrics_dict.keys():
            if(key != "auc" and key != "AUPRC"):
                train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = self.thresh)
                val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = self.thresh)
            else:
                train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
            print("train_" + key + ": " + str(train_metrics), flush=True)
            print("test_" + key + ": " + str(val_metrics), flush=True)

        threshold_1, threshold_2 = cutoff(final_val_labels, final_val_probs)
        print("Best threshold (AUC) is " + str(threshold_1))
        print("Best threshold (PRC) is " + str(threshold_2))

        return ""

def create_mat_dict_w2v(seqs, enc_model, k_mer):
    seqs = list(set(seqs))
    seq2mat_dict = {}
    for i in range(len(seqs)):
        seq2mat_dict[seqs[i]] = torch.tensor([enc_model.wv[seqs[i][j: j + k_mer]] for j in range(len(seqs[i]) - k_mer + 1)])

    return seq2mat_dict

def training_main(train_path, val_path, w2v_path, out_path, t_batch = 32, v_batch = 32, lr = 0.0001, max_epoch = 10000, stop_epoch = 20, thr = 0.5, k_mer = 4, seq_max = 9000):
    print("Setting parameters", flush = True)
    model_params = {"filter_num": 128, "kernel_size": 20, "stride": 10, "n_heads": 4, "d_dim": 32, "feature": 128, "pooling_dropout": 0.5, "linear_dropout": 0.3}
    training_params = {"training_batch_size": t_batch, "validation_batch_size": v_batch, "lr": lr, "early_stopping": stop_epoch, "max_epoch": max_epoch, "thresh": thr, "stopping_met": "auc"}
    encoding_params = {"enc_seq_max_train": seq_max, "enc_seq_max_val": seq_max, "k_mer": k_mer}

    print("Loading datasets", flush = True)
    training_data = file_input_csv(train_path)
    validation_data = file_input_csv(val_path)

    print("Loading word2vec model", flush = True)
    w2v_model = word2vec.Word2Vec.load(w2v_path)

    print("Encoding amino acid sequences", flush = True)
    mat_dict = create_mat_dict_w2v(training_data["human_seq"].values.tolist() + validation_data["human_seq"].values.tolist() + training_data["virus_seq"].values.tolist() + validation_data["virus_seq"].values.tolist(), w2v_model, encoding_params["k_mer"])

    print("Start training a deep neural network model", flush = True)
    net = DeepNet(out_path, mat_dict, model_params, training_params, encoding_params)
    out = net.model_training(training_data, validation_data)

    print("Finish processing", flush = True)





































