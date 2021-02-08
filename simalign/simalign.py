# coding=utf-8

import os
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
try:
	import networkx as nx
	from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
except ImportError:
	nx = None
import torch
from transformers import BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer

from simalign.utils import get_logger

LOG = get_logger(__name__)


class EmbeddingLoader(object):
	def __init__(self, model: str="bert-base-multilingual-cased", device=torch.device('cpu'), layer: int=8):
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
		self.layer = layer
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
			if os.path.isdir(model):
				# try to load model with auto-classes
				config = AutoConfig.from_pretrained(model, output_hidden_states=True)
				self.emb_model = AutoModel.from_pretrained(model, config=config)
				self.emb_model.eval()
				self.emb_model.to(self.device)
				self.tokenizer = AutoTokenizer.from_pretrained(model)
				LOG.info("Initialized the EmbeddingLoader from path: {}".format(self.model))
			else:
				raise ValueError("The model '{}' is not recognised!".format(model))

	def get_embed_list(self, sent_batch: List[List[str]]) -> torch.Tensor:
		if self.emb_model is not None:
			with torch.no_grad():
				if not isinstance(sent_batch[0], str):
					inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
				else:
					inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True, return_tensors="pt")
				outputs = self.emb_model(**inputs.to(self.device))[2][self.layer]

				return outputs[:, 1:-1, :]
		else:
			return None


class SentenceAligner(object):
	def __init__(self, model: str = "bert", token_type: str = "bpe", distortion: float = 0.0, matching_methods: str = "mai", device: str = "cpu", layer: int = 8):
		model_names = {
			"bert": "bert-base-multilingual-cased",
			"xlmr": "xlm-roberta-base"
			}
		all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

		self.model = model
		if model in model_names:
			self.model = model_names[model]
		self.token_type = token_type
		self.distortion = distortion
		self.matching_methods = [all_matching_methods[m] for m in matching_methods]
		self.device = torch.device(device)

		self.embed_loader = EmbeddingLoader(model=self.model, device=self.device, layer=layer)

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
	def average_embeds_over_words(bpe_vectors: np.ndarray, word_tokens_pair: List[List[str]]) -> List[np.array]:
		w2b_map = []
		cnt = 0
		w2b_map.append([])
		for wlist in word_tokens_pair[0]:
			w2b_map[0].append([])
			for x in wlist:
				w2b_map[0][-1].append(cnt)
				cnt += 1
		cnt = 0
		w2b_map.append([])
		for wlist in word_tokens_pair[1]:
			w2b_map[1].append([])
			for x in wlist:
				w2b_map[1][-1].append(cnt)
				cnt += 1

		new_vectors = []
		for l_id in range(2):
			w_vector = []
			for word_set in w2b_map[l_id]:
				w_vector.append(bpe_vectors[l_id][word_set].mean(0))
			new_vectors.append(np.array(w_vector))
		return new_vectors

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

	def get_word_aligns(self, src_sent: Union[str, List[str]], trg_sent: Union[str, List[str]]) -> Dict[str, List]:
		if isinstance(src_sent, str):
			src_sent = src_sent.split()
		if isinstance(trg_sent, str):
			trg_sent = trg_sent.split()
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

		vectors = self.embed_loader.get_embed_list([src_sent, trg_sent]).cpu().detach().numpy()
		vectors = [vectors[i, :len(bpe_lists[i])] for i in [0, 1]]

		if self.token_type == "word":
			vectors = self.average_embeds_over_words(vectors, [l1_tokens, l2_tokens])

		all_mats = {}
		sim = self.get_similarity(vectors[0], vectors[1])
		sim = self.apply_distortion(sim, self.distortion)

		all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
		all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
		if "mwmf" in self.matching_methods:
			all_mats["mwmf"] = self.get_max_weight_match(sim)
		if "itermax" in self.matching_methods:
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
