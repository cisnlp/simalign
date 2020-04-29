# coding=utf-8

import logging
import torch
import numpy as np
try:
	import networkx as nx
except ImportError:
	nx = None
from transformers import *
from typing import Dict, List, Text, Tuple
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from simalign.utils import get_logger

LOG = get_logger(__name__)


class EmbeddingLoader(object):
	def __init__(self, model="bert-base-multilingual-cased", device=torch.device('cpu')):
		TR_Models = {
			'bert-base-uncased': (BertModel, BertTokenizer),
			'bert-base-multilingual-cased': (BertModel, BertTokenizer),
			'bert-base-multilingual-uncased': (BertModel, BertTokenizer),
			'xlm-mlm-100-1280': (XLMModel, XLMTokenizer),
			'roberta-base': (RobertaModel, RobertaTokenizer),
			'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer),
			'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer),
		}

		self.model = model
		self.device = device
		self.emb_model = None
		self.tokenizer = None

		if model in TR_Models:
			model_class, tokenizer_class = TR_Models[model]
			self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
			self.emb_model.eval()
			self.emb_model.to(self.device)
			self.tokenizer = tokenizer_class.from_pretrained(model)
			LOG.info("Initialized the EmbeddingLoader with model: {}".format(self.model))
		else:
			raise ValueError("The model '{}' is not recognised!".format(model))

	def get_embed_list(self, sent_pair):
		if self.emb_model is not None:
			sent_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in sent_pair]
			inputs = [self.tokenizer.prepare_for_model(sent, return_token_type_ids=True, return_tensors='pt')['input_ids'] for sent in sent_ids]

			outputs = [self.emb_model(in_ids.to(self.device)) for in_ids in inputs]
			# use vectors from layer 8
			vectors = [x[2][8].cpu().detach().numpy()[0][1:-1] for x in outputs]
			return vectors
		else:
			return None


class SentenceAligner(object):
	def __init__(self, model: str = "bert", token_type: str = "bpe", distortion: float = 0.0, matching_methods: str = "mai", device: str = "cpu"):
		TR_Models = [
			'bert-base-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased',
			'xlm-mlm-100-1280', 'roberta-base', 'xlm-roberta-base', 'xlm-roberta-large']
		all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

		self.model = model
		self.token_type = token_type
		self.distortion = distortion
		self.matching_methods = [all_matching_methods[m] for m in matching_methods]
		self.device = torch.device(device)

		if model == "bert":
			self.model = "bert-base-multilingual-cased"
		elif model == "xlmr":
			self.model = "xlm-roberta-base"
		if self.model not in TR_Models:
			raise ValueError("The model '{}' is not recognised!".format(model))

		self.embed_loader = EmbeddingLoader(model=self.model, device=self.device)

	@staticmethod
	def get_max_weight_match(sim: np.ndarray) -> np.ndarray:
		if nx is None:
			raise ValueError("networkx must be installed to use match algorithm.")
		def permute(edge):
			if edge[0] < sim.shape[0]:
				return edge[0], edge[1] - sim.shape[0]
			else:
				return edge[1], edge[0] - sim.shape[0]
		G = from_biadjacency_matrix(csr_matrix(sim))
		matching = nx.max_weight_matching(G, maxcardinality=True)
		matching = [permute(x) for x in matching]
		matching = sorted(matching, key=lambda x: x[0])
		res_matrix = np.zeros_like(sim)
		for edge in matching:
			res_matrix[edge[0], edge[1]] = 1
		return res_matrix

	@staticmethod
	def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
		return (cosine_similarity(X, Y) + 1.0) / 2.0

	@staticmethod
	def get_alignment_matrix(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		m, n = sim_matrix.shape
		forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
		backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
		return forward, backward.transpose()

	@staticmethod
	def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
		shape = sim_matrix.shape
		if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
			return sim_matrix

		pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
		pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])
		distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

		return np.multiply(sim_matrix, distortion_mask)

	@staticmethod
	def iter_max(sim_matrix: np.ndarray, max_count: int=2) -> np.ndarray:
		alpha_ratio = 0.9
		m, n = sim_matrix.shape
		forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
		backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
		inter = forward * backward.transpose()

		if min(m, n) <= 2:
			return inter

		new_inter = np.zeros((m, n))
		count = 1
		while count < max_count:
			mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
			mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
			mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
			mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
			if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
				mask *= 0.0
				mask_zeros *= 0.0

			new_sim = sim_matrix * mask
			fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
			bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
			new_inter = fwd * bac

			if np.array_equal(inter + new_inter, inter):
				break
			inter = inter + new_inter
			count += 1
		return inter

	def get_word_aligns(self, src_sent: List, trg_sent: List) -> Dict[str, List]:
		l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in src_sent]
		l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in trg_sent]
		bpe_lists = [[bpe for w in sent for bpe in w] for sent in [l1_tokens, l2_tokens]]

		if self.token_type == "bpe":
			l1_b2w_map = []
			for i, wlist in enumerate(l1_tokens):
				l1_b2w_map += [i for x in wlist]
			l2_b2w_map = []
			for i, wlist in enumerate(l2_tokens):
				l2_b2w_map += [i for x in wlist]

		vectors = self.embed_loader.get_embed_list(list(bpe_lists))
		if self.token_type == "word":
			w2b_map = []
			cnt = 0
			w2b_map.append([])
			for wlist in l1_tokens:
				w2b_map[0].append([])
				for x in wlist:
					w2b_map[0][-1].append(cnt)
					cnt += 1
			cnt = 0
			w2b_map.append([])
			for wlist in l2_tokens:
				w2b_map[1].append([])
				for x in wlist:
					w2b_map[1][-1].append(cnt)
					cnt += 1
			new_vectors = []
			for l_id in range(2):
				w_vector = []
				for word_set in w2b_map[l_id]:
					w_vector.append(vectors[l_id][word_set].mean(0))
				new_vectors.append(np.array(w_vector))
			vectors = np.array(new_vectors)

		all_mats = {}
		sim = self.get_similarity(vectors[0], vectors[1])
		sim = self.apply_distortion(sim, self.distortion)

		all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
		all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
		all_mats["mwmf"] = self.get_max_weight_match(sim)
		all_mats["itermax"] = self.iter_max(sim)

		aligns = {x: set() for x in self.matching_methods}
		for i in range(len(vectors[0])):
			for j in range(len(vectors[1])):
				for ext in self.matching_methods:
					if all_mats[ext][i, j] > 0:
						if self.token_type == "bpe":
							aligns[ext].add((l1_b2w_map[i], l2_b2w_map[j]))
						else:
							aligns[ext].add((i, j))
		for ext in aligns:
			aligns[ext] = sorted(aligns[ext])
		return aligns
