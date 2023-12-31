# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from src.utils.utils import get_degree_matrix
from .gcn_perturb import GCNSyntheticPerturb, GCN2LayerPerturb, GCN1LayerPerturb, GCN4LayerPerturb
from src.utils.utils import normalize_adj


class CFExplainer:
	"""
	CF Explainer class, returns counterfactual subgraph
	"""
	def __init__(self, model, sub_adj, sub_feat, n_hid, dropout,
	              sub_labels, y_pred_orig, num_classes, beta, device, model_type):
		super(CFExplainer, self).__init__()
		self.model = model
		self.model.eval()
		self.sub_adj = sub_adj
		self.sub_feat = sub_feat
		self.n_hid = n_hid
		self.dropout = dropout
		self.sub_labels = sub_labels
		self.y_pred_orig = y_pred_orig
		self.beta = beta
		self.num_classes = num_classes
		self.device = device
		self.model_type = model_type

		# Instantiate CF model class, load weights from original model
		if self.model_type == "GCN1Layer_PyG":
			self.cf_model = GCN1LayerPerturb(self.sub_feat.shape[1], n_hid, n_hid,
												self.num_classes, self.sub_adj, dropout, beta)
		elif self.model_type == "GCN2Layer" or self.model_type == "GCN2Layer_TG":
			self.cf_model = GCN2LayerPerturb(self.sub_feat.shape[1], n_hid, n_hid,
												self.num_classes, self.sub_adj, dropout, beta)
		elif self.model_type == "GCNSynthetic" or self.model_type == "GCN3Layer_PyG":
			self.cf_model = GCNSyntheticPerturb(self.sub_feat.shape[1], n_hid, n_hid,
												self.num_classes, self.sub_adj, dropout, beta)
		elif self.model_type == "GCN4Layer_PyG":
			self.cf_model = GCN4LayerPerturb(self.sub_feat.shape[1], n_hid, n_hid,
												self.num_classes, self.sub_adj, dropout, beta)

		print("Accessing the model state_dict")
		# the required architecture for the perturbed model
		for values in self.cf_model.state_dict():
			print(values, "\t", self.cf_model.state_dict()[values].size())
		# the architecture how it leaves from the PyG model
		for values in self.model.state_dict():
			print(values, "\t", self.model.state_dict()[values].size())
		# we change the names of the keys so that the weights and biases can be transferred from model to cf_model
		corrected_dict = self.model.state_dict()
		if 'gc1.lin.weight' in corrected_dict:
			corrected_dict['gc1.weight'] = torch.transpose(corrected_dict['gc1.lin.weight'],0,1)
		if 'gc2.lin.weight' in corrected_dict:
			corrected_dict['gc2.weight'] = torch.transpose(corrected_dict['gc2.lin.weight'],0,1)
		if 'gc3.lin.weight' in corrected_dict:
			corrected_dict['gc3.weight'] = torch.transpose(corrected_dict['gc3.lin.weight'],0,1)

		#print(self.model.state_dict())
		self.cf_model.load_state_dict(corrected_dict, strict=False)
		#print(self.cf_model.state_dict())

		# the architecture ready for perturbations
		for values in self.model.state_dict():
			print(values, "\t", self.model.state_dict()[values].size())

		# Freeze weights from original model in cf_model
		for name, param in self.cf_model.named_parameters():
			if name.endswith("weight") or name.endswith("bias"):
				param.requires_grad = False
		for name, param in self.model.named_parameters():
			print("orig model requires_grad: ", name, param.requires_grad)
		for name, param in self.cf_model.named_parameters():
			print("cf model requires_grad: ", name, param.requires_grad)



	def explain(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
		self.node_idx = node_idx
		self.new_idx = new_idx

		self.x = self.sub_feat
		self.A_x = self.sub_adj
		self.D_x = get_degree_matrix(self.A_x)

		if cf_optimizer == "SGD" and n_momentum == 0.0:
			self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
		elif cf_optimizer == "SGD" and n_momentum != 0.0:
			self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum)
		elif cf_optimizer == "Adadelta":
			self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)
		print("HERE:")
		print(self.cf_model.parameters())
		best_cf_example = []
		best_loss = np.inf
		num_cf_examples = 0
		for epoch in range(num_epochs):
			new_example, loss_total = self.train(epoch)
			if new_example != [] and loss_total < best_loss:
				best_cf_example.append(new_example)
				best_loss = loss_total
				num_cf_examples += 1
		print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
		print(" ")
		return(best_cf_example)


	def train(self, epoch):
		t = time.time()
		self.cf_model.train()
		self.cf_optimizer.zero_grad()

		# output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
		# output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
		output = self.cf_model.forward(self.x, self.A_x)
		output_actual, self.P = self.cf_model.forward_prediction(self.x)

		# Need to use new_idx from now on since sub_adj is reindexed
		y_pred_new = torch.argmax(output[self.new_idx])
		y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

		# loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
		loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
		print(loss_total)
		loss_total.backward()
		clip_grad_norm(self.cf_model.parameters(), 2.0)
		self.cf_optimizer.step()
		print('Node idx: {}'.format(self.node_idx),
		      'New idx: {}'.format(self.new_idx),
			  'Epoch: {:04d}'.format(epoch + 1),
		      'loss: {:.4f}'.format(loss_total.item()),
		      'pred loss: {:.4f}'.format(loss_pred.item()),
		      'graph loss: {:.4f}'.format(loss_graph_dist.item()))
		print('Output: {}\n'.format(output[self.new_idx].data),
		      'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
		      'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new, y_pred_new_actual))
		print(" ")
		cf_stats = []
		print(y_pred_new)
		print(y_pred_new_actual)
		print("__________________")
		if y_pred_new_actual != self.y_pred_orig: #and loss_total.item() > 0 this part was added to make sure edges are removed
			cf_stats = [self.node_idx.item(), self.new_idx.item(),
			            cf_adj.detach().numpy(), self.sub_adj.detach().numpy(),
			            self.y_pred_orig.item(), y_pred_new.item(),
			            y_pred_new_actual.item(), self.sub_labels[self.new_idx].numpy(),
			            self.sub_adj.shape[0], loss_total.item(),
			            loss_pred.item(), loss_graph_dist.item()]

		return(cf_stats, loss_total.item())
