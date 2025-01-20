import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle


class TwoTowerModel(nn.Module):
    def __init__(self, genome_encoder, large_molecule_encoder, small_molecule_encoder):
        super(TwoTowerModel, self).__init__()
        self.genome_encoder = genome_encoder
        self.large_molecule_encoder = large_molecule_encoder
        self.small_molecule_encoder = small_molecule_encoder
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch, device):
        # Example input tensor
        x_genome = batch['genome'].float().to(device)  # Move tensor to GPU if available
        # Forward pass
        _,_,_, genome_embedding = self.genome_encoder(x_genome.squeeze(1))
        largmol_tensor = torch.tensor(batch['drug_embed_large'], dtype=torch.float32).to(device)
        smalmol_tensor = torch.tensor(batch['drug_embed_small'], dtype=torch.float32).to(device)
        large_mol_embedding = self.large_molecule_encoder(largmol_tensor)
        small_mol_embedding = self.small_molecule_encoder(smalmol_tensor)
        treatment_embedding = large_mol_embedding + small_mol_embedding
        patient_embedding = torch.cat((genome_embedding, batch['demographics'].squeeze(1)), dim=1)
        embedding_similarity = self.similarity(patient_embedding, treatment_embedding)
        match_score = self.sigmoid(embedding_similarity)
        return embedding_similarity, match_score

class TwoTowerLoss(nn.Module):
    def __init__(self, regularizer=0.1):
        super(TwoTowerLoss, self).__init__()
        # Define the loss functions for the two towers neural network
        self.regularizer = regularizer
        self.BCEloss = nn.BCELoss(reduction='sum')
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, patient_treatment_similarity, match_score, batch):
        idx = ~torch.isnan(batch['label']).reshape(-1,)
        # calculate the match score bce loss
        match_score_loss = self.BCEloss(match_score[idx], batch['label'].reshape(-1,)[idx])
        # calculate the contrastive loss
        patient_treatment_similarity_all = patient_treatment_similarity.repeat(batch['label'].reshape(-1,).shape[0],1)
        contrastive_score = torch.mul(patient_treatment_similarity_all,patient_treatment_similarity_all.T)
        contrastive_prob = self.softmax(contrastive_score)[idx,:][:,idx]
        contrastive_label = torch.eye(contrastive_prob.shape[0]).to(match_score.device)
        contrastive_loss = self.BCEloss(contrastive_prob.float(), contrastive_label.float())
        return match_score_loss, contrastive_loss