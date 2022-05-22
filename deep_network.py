#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:41:16 2021

@author: tsukiyamashou
"""
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    return [copy.deepcopy(module) for _ in range(N)]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_1, d_model_2, n_heads, d_dim):
        super(MultiHeadAttention, self).__init__()
        self.dim  = d_dim
        
        self.n_heads = n_heads
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        
        self.W_Q_dense = nn.Linear(d_model_1, self.dim * self.n_heads, bias=False)
        self.W_K_dense = nn.Linear(d_model_2, self.dim * self.n_heads, bias=False)
        self.W_V_dense = nn.Linear(d_model_2, self.dim * self.n_heads, bias=False)
        
        self.scale_product = ScaledDotProductAttention(self.dim)

        self.out_dense = nn.Linear(self.n_heads * self.dim, self.d_model_1, bias=False)
        
    def forward(self, Q, K, V, attn_mask):
        Q_spare, batch_size = Q, Q.size(0)
        
        q_s = self.W_Q_dense(Q).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)
        k_s = self.W_K_dense(K).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)
        self.v_s = self.W_V_dense(V).view(batch_size, -1, self.n_heads, self.dim).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, self.attn = self.scale_product(q_s, k_s, self.v_s, attn_mask)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim)
        context = self.out_dense(context)
        
        return context + Q_spare
    
class CNN_net(nn.Module):
    def __init__(self, d_model_in, d_model_out, kernel_size, stride, pooling = True, dropout = 0.5):
        super(CNN_net, self).__init__()
        self.pooling = pooling
        self.cnn = nn.Conv1d(d_model_in, d_model_out, kernel_size = kernel_size, stride = stride)
        if(pooling == True):
            self.pool = torch.nn.MaxPool1d(3, stride = 1, padding=1)
        self.relu_func = nn.ReLU() 
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, mat):
        mat = torch.transpose(mat, -1, -2)
        mat = self.cnn(mat)
        mat = self.relu_func(mat)
        mat = self.dropout_layer(mat)
        if(self.pooling == True):
            mat = self.pool(mat)
        mat = torch.transpose(mat, -1, -2)
        
        return mat

class Transformer_PHV(nn.Module):
    def __init__(self, filter_num = 100, kernel_size_w2v = 20, stride_w2v = 10, n_heads = 4, d_dim = 32, feature = 128, pooling_dropout = 0.5, linear_dropout = 0.3):
        super(Transformer_PHV, self).__init__()
        
        self.h_cnn_1 = CNN_net(d_model_in = feature, d_model_out = filter_num, kernel_size = kernel_size_w2v, stride = stride_w2v, dropout = pooling_dropout)
        self.h_cnn_2 = CNN_net(d_model_in = feature, d_model_out = filter_num, kernel_size = kernel_size_w2v, stride = stride_w2v, dropout = pooling_dropout)
    
        self.att_list_1 = MultiHeadAttention(filter_num, filter_num, n_heads, d_dim)   
        self.att_list_2 = MultiHeadAttention(filter_num, filter_num, n_heads, d_dim)  
        
        self.dense_1 = nn.Linear(filter_num * 2, 64)
        self.dense_2 = nn.Linear(64, 16)
        self.dense_3 = nn.Linear(16, 1)
       
        self.dropout_layer_pool = nn.Dropout(pooling_dropout)
        self.dropout_layer_linear = nn.Dropout(linear_dropout)
        self.sigmoid_func = nn.Sigmoid()
        self.relu_func = nn.ReLU()   
 
    def forward(self, input_1, input_2, attn_mask_1, attn_mask_2):

        self.h_out_1 = self.h_cnn_1(input_1)
        self.h_out_2 = self.h_cnn_2(input_2)

        out_1_q, out_2_k, out_2_v = self.h_out_1, self.h_out_2, self.h_out_2
        out_2_q, out_1_k, out_1_v = self.h_out_2, self.h_out_1, self.h_out_1

        self.out_1_temp = self.att_list_1(out_1_q, out_2_k, out_2_v, attn_mask_2)
        self.out_2_temp = self.att_list_2(out_2_q, out_1_k, out_1_v, attn_mask_1)
        
        out_1 = self.dropout_layer_pool(self.out_1_temp)
        out_2 = self.dropout_layer_pool(self.out_2_temp)
        
        self.out_1, _ = torch.max(out_1, dim = 1)
        self.out_2, _ = torch.max(out_2, dim = 1)
        self.out = torch.cat((self.out_1, self.out_2), dim = 1)

        return self.sigmoid_func(self.dense_3(self.dropout_layer_linear(self.relu_func(self.dense_2(self.dropout_layer_linear(self.relu_func(self.dense_1(self.out))))))))









    
