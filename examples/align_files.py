# coding=utf-8

import regex
import codecs
import argparse
import torch.nn.functional as F

from simalign.simalign import *


def gather_null_aligns(sim_matrix: np.ndarray, inter_matrix: np.ndarray) -> List[float]:
	shape = sim_matrix.shape
	if min(shape[0], shape[1]) <= 2:
		return []
	norm_x = normalize(sim_matrix, axis=1, norm='l1')
	norm_y = normalize(sim_matrix, axis=0, norm='l1')

	entropy_x = np.array([entropy(norm_x[i, :]) / np.log(shape[1]) for i in range(shape[0])])
	entropy_y = np.array([entropy(norm_y[:, j]) / np.log(shape[0]) for j in range(shape[1])])

	mask_x = np.tile(entropy_x[:, np.newaxis], (1, shape[1]))
	mask_y = np.tile(entropy_y, (shape[0], 1))

	all_ents = np.multiply(inter_matrix, np.minimum(mask_x, mask_y))
	return [x.item() for x in np.nditer(all_ents) if x.item() > 0]

def apply_percentile_null_aligns(sim_matrix: np.ndarray, ratio: float=1.0) -> np.ndarray:
	shape = sim_matrix.shape
	if min(shape[0], shape[1]) <= 2:
		return np.ones(shape)
	norm_x = normalize(sim_matrix, axis=1, norm='l1')
	norm_y = normalize(sim_matrix, axis=0, norm='l1')
	entropy_x = np.array([entropy(norm_x[i, :]) / np.log(shape[1]) for i in range(shape[0])])
	entropy_y = np.array([entropy(norm_y[:, j]) / np.log(shape[0]) for j in range(shape[1])])
	mask_x = np.tile(entropy_x[:, np.newaxis], (1, shape[1]))
	mask_y = np.tile(entropy_y, (shape[0], 1))

	ents_mask = np.where(np.minimum(mask_x, mask_y) > ratio, 0.0, 1.0)

	return ents_mask


# --------------------------------------------------------
# --------------------------------------------------------
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Extracts alignments based on different embeddings", epilog="example: python3 main.py path/to/L1/text path/to/L2/text [options]")
	parser.add_argument("L1_path", type=str, help="Lines in the file should be indexed separated by TABs.")
	parser.add_argument("L2_path", type=str, help="Same format as L1 file.")
	parser.add_argument("-model", type=str, default="bert", help="choices: ['bert', 'xlmr', '<transformer_model_name>']")
	parser.add_argument("-distortion", type=float, default=0.0)
	parser.add_argument("--null-align", type=float, default=1.0)
	parser.add_argument("--token-type", type=str, choices=["bpe", "word"], default="bpe")
	parser.add_argument("--matching-methods", type=str, default="mai", help="m: Max Weight Matching (mwmf), a: argmax (inter), i: itermax, f: forward (fwd), r: reverse (rev)")
	parser.add_argument("--num-test-sents", type=int, default=None, help="None means all sentences")
	parser.add_argument("--batch-size", type=int, default=100)
	parser.add_argument("-log", action="store_true")
	parser.add_argument("-device", type=str, default="cpu")
	parser.add_argument("-output", type=str, default="align_out", help="output alignment files (without extension)")
	args = parser.parse_args()

	if args.model == "bert":
		args.model = "bert-base-multilingual-cased"
	elif args.model == "xlmr":
		args.model = "xlm-roberta-base"

	LOG.info("Simalign parameters: " + str(args))

	langs = [args.L1_path, args.L2_path]
	max_sent_id = args.num_test_sents
	convert_to_words = (args.token_type == "word")
	device = torch.device(args.device)

	# --------------------------------------------------------
	embed_loader = EmbeddingLoader(model=args.model, device=device)

	original_paths = [lang for lang in langs]
	original_corpora = []
	for path in original_paths:
		corpus = [l.strip().split("\t")[1] for l in codecs.open(path, 'r', 'utf-8').readlines()]
		corpus = [regex.sub("\\p{C}+", "", regex.sub("\\p{Separator}+", " ", l)).strip() for l in corpus]
		original_corpora.append(corpus[:max_sent_id])

	words_tokens = []
	for sent_id in range(len(original_corpora[0])):
		l1_tokens = [embed_loader.tokenizer.tokenize(word) for word in original_corpora[0][sent_id].split()]
		l2_tokens = [embed_loader.tokenizer.tokenize(word) for word in original_corpora[1][sent_id].split()]
		words_tokens.append([l1_tokens, l2_tokens])

	sentences_bpe_lists = []
	sentences_b2w_map = []
	for sent_id in range(len(words_tokens)):
		sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[sent_id]]
		b2w_map_pair = [[i for i, w in enumerate(sent) for bpe in w] for sent in words_tokens[sent_id]]
		sentences_bpe_lists.append(sent_pair)
		sentences_b2w_map.append(b2w_map_pair)

	corpora_lengths = [len(corpus) for corpus in original_corpora]
	if min(corpora_lengths) != max(corpora_lengths):
		LOG.warning("Mismatch in corpus lengths: " + str(corpora_lengths))
		raise ValueError('Cannot load parallel corpus.')

	# --------------------------------------------------------
	all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}
	matching_methods = [all_matching_methods[m] for m in args.matching_methods]

	out_f = {ext: open('{}.{}'.format(args.output, ext), 'w') for ext in matching_methods}
	if args.log:
		out_log = open('{}.log'.format(args.output), 'w')

	if args.null_align < 1.0:
		entropies = {x: [] for x in matching_methods}
		for sent_id in range(len(original_corpora[0])):
			sent_pair = [original_corpora[i][sent_id] for i in [0, 1]]
			vectors = embed_loader.get_embed_list(sent_pair).cpu().detach().numpy()
			vectors = [vectors[i][:len(sentences_bpe_lists[sent_id][i])] for i in [0, 1]]

			if convert_to_words:
				vectors = SentenceAligner.average_embeds_over_words(vectors, words_tokens[sent_id])

			all_mats = {}
			sim = SentenceAligner.get_similarity(vectors[0], vectors[1])
			sim = SentenceAligner.apply_distortion(sim, args.distortion)

			methods_matrix = {}
			methods_matrix["forward"], methods_matrix["backward"] = SentenceAligner.get_alignment_matrix(sim)
			methods_matrix["inter"] = methods_matrix["forward"] * methods_matrix["backward"]
			if "mwmf" in matching_methods:
				methods_matrix["mwmf"] = SentenceAligner.get_max_weight_match(sim)
			if "itermax" in matching_methods:
				methods_matrix["itermax"] = SentenceAligner.iter_max(sim)

			for m in entropies:
				entropies[m] += gather_null_aligns(sim, methods_matrix[m])
		null_thresh = {m: sorted(entropies[m])[int(args.null_align * len(entropies[m]))] for m in entropies}

	ds = [(idx, original_corpora[0][idx], original_corpora[1][idx]) for idx in range(len(original_corpora[0]))]
	data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
	for batch_id, batch_sentences in enumerate(data_loader):
		batch_vectors_src = embed_loader.get_embed_list(batch_sentences[1])
		batch_vectors_trg = embed_loader.get_embed_list(batch_sentences[2])
		btach_sim = None
		if not convert_to_words:
			batch_vectors_src = F.normalize(batch_vectors_src, dim=2)
			batch_vectors_trg = F.normalize(batch_vectors_trg, dim=2)

			btach_sim = torch.bmm(batch_vectors_src, torch.transpose(batch_vectors_trg, 1, 2))
			btach_sim = ((btach_sim + 1.0) / 2.0).cpu().detach().numpy()

		batch_vectors_src = batch_vectors_src.cpu().detach().numpy()
		batch_vectors_trg = batch_vectors_trg.cpu().detach().numpy()

		for in_batch_id, sent_id in enumerate(batch_sentences[0].numpy()):
			sent_pair = sentences_bpe_lists[sent_id]
			vectors = [batch_vectors_src[in_batch_id, :len(sent_pair[0])], batch_vectors_trg[in_batch_id, :len(sent_pair[1])]]

			if not convert_to_words:
				sim = btach_sim[in_batch_id, :len(sent_pair[0]), :len(sent_pair[1])]
			else:
				vectors = SentenceAligner.average_embeds_over_words(vectors, words_tokens[sent_id])
				sim = SentenceAligner.get_similarity(vectors[0], vectors[1])

			all_mats = {}

			sim = SentenceAligner.apply_distortion(sim, args.distortion)
			if args.null_align < 1.0:
				mask_nulls = {mmethod: apply_percentile_null_aligns(sim, null_thresh[mmethod]) for mmethod in matching_methods}

			all_mats["fwd"], all_mats["rev"] = SentenceAligner.get_alignment_matrix(sim)
			all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
			if "mwmf" in matching_methods:
				all_mats["mwmf"] = SentenceAligner.get_max_weight_match(sim)
			if "itermax" in matching_methods:
				all_mats["itermax"] = SentenceAligner.iter_max(sim)

			if args.null_align < 1.0:
				if "inter" in matching_methods:
					all_mats["inter"] = np.multiply(all_mats["inter"], mask_nulls["inter"])
				if "mwmf" in matching_methods:
					all_mats["mwmf"] = np.multiply(all_mats["mwmf"], mask_nulls["mwmf"])
				if "itermax" in matching_methods:
					all_mats["itermax"] = np.multiply(all_mats["itermax"], mask_nulls["itermax"])

			raw_aligns = {x: [] for x in matching_methods}
			b2w_aligns = {x: set() for x in matching_methods}
			log_aligns = []

			for i in range(len(vectors[0])):
				for j in range(len(vectors[1])):
					for ext in matching_methods:
						if all_mats[ext][i, j] > 0:
							raw_aligns[ext].append('{}-{}'.format(i, j))
							if args.token_type == "bpe":
								b2w_aligns[ext].add('{}-{}'.format(sentences_b2w_map[sent_id][0][i], sentences_b2w_map[sent_id][1][j]))
								if ext == "inter":
									log_aligns.append('{}-{}:({}, {})'.format(i, j, sent_pair[0][i], sent_pair[1][j]))
							else:
								b2w_aligns[ext].add('{}-{}'.format(i, j))

			for ext in out_f:
				if convert_to_words:
					out_f[ext].write(str(sent_id) + "\t" + ' '.join(sorted(raw_aligns[ext])) + "\n")
				else:
					out_f[ext].write(str(sent_id) + "\t" + ' '.join(sorted(b2w_aligns[ext])) + "\n")
			if args.log:
				out_log.write(str(sent_id) + "\t" + ' '.join(sorted(log_aligns)) + "\n")

	if args.log:
		out_log.close()
	for ext in out_f:
		out_f[ext].close()

