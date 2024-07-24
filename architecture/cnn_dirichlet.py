"""
CNN model used for toy experiments
modified from https://github.com/HannesStark/dirichlet-flow-matching
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Vectorfield

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

class GaussianFourierProjection(nn.Module):
	"""
	Gaussian random features for encoding time steps.
	"""

	def __init__(self, embed_dim, scale=30.):
		super().__init__()
		# Randomly sample weights during initialization. These weights are fixed
		# during optimization and are not trainable.
		self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

	def forward(self, x):
		x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
		return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class CNNModel(Vectorfield):
	def __init__(self, alphabet_size, num_cls=1, mode="dirichlet", classifier=False,
			hidden_dim=128, clean_data=False, cls_expanded_simplex=False,
			num_cnn_stacks=1, dropout=0.0):
		super(Vectorfield, self).__init__()
		#mode = "riemannian" | "dirichlet"
		self.alphabet_size = alphabet_size
		self.clean_data = clean_data
		self.classifier = classifier
		self.num_cls = num_cls
		self.channel_dim = 2

		if self.clean_data:
			self.linear = nn.Embedding(self.alphabet_size, embedding_dim=hidden_dim)
		else:
			self.linear = nn.Conv1d(self.alphabet_size, hidden_dim, kernel_size=9, padding=4)
			self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= hidden_dim),nn.Linear(hidden_dim, hidden_dim))

		self.num_layers = 5 * num_cnn_stacks
		self.convs = [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
									 nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
									 nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, dilation=4, padding=16),
									 nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, dilation=16, padding=64),
									 nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, dilation=64, padding=256)]
		self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(num_cnn_stacks)])
		self.time_layers = nn.ModuleList([Dense(hidden_dim, hidden_dim) for _ in range(self.num_layers)])
		self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])
		self.final_conv = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
								   nn.ReLU(),
								   nn.Conv1d(hidden_dim, hidden_dim if classifier else self.alphabet_size, kernel_size=1))
		self.dropout = nn.Dropout(dropout)
		if classifier:
			self.cls_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
								   nn.ReLU(),
								   nn.Linear(hidden_dim, self.num_cls))

	def counted_forward(self, seq, timesteps, cls = None, return_embedding=False):
		if self.clean_data:
			feat = self.linear(seq)
			feat = feat.permute(0, 2, 1)
		else:
			time_emb = F.relu(self.time_embedder(timesteps))
			feat = seq.permute(0, 2, 1)
			feat = F.relu(self.linear(feat))

		for i in range(self.num_layers):
			h = self.dropout(feat.clone())
			if not self.clean_data:
				h = h + self.time_layers[i](time_emb)[:, :, None]
			h = self.norms[i]((h).permute(0, 2, 1))
			h = F.relu(self.convs[i](h.permute(0, 2, 1)))
			if h.shape == feat.shape:
				feat = h + feat
			else:
				feat = h
		feat = self.final_conv(feat)
		feat = feat.permute(0, 2, 1)
		if self.classifier:
			feat = feat.mean(dim=1)
			if return_embedding:
				embedding = self.cls_head[:1](feat)
				return self.cls_head[1:](embedding), embedding
			else:
				return self.cls_head(feat)
		return feat