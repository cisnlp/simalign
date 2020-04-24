import regex
import codecs
import argparse
from aligner import *


def gather_null_aligns(sim_matrix, inter_matrix):
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

def apply_percentile_null_aligns(sim_matrix, ratio=1.0):
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
	parser.add_argument("L1_path", type=str)
	parser.add_argument("L2_path", type=str)
	parser.add_argument("-model", type=str, default="bert", help="choices: ['bert', 'xlmr', 'tr:<transformer_model_name>']")
	parser.add_argument("-distortion", type=float, default=0.0)
	parser.add_argument("--null-align", type=float, default=1.0)
	parser.add_argument("--token-type", type=str, choices=["bpe", "word"], default="bpe")
	parser.add_argument("--matching-methods", type=str, default="mai", help="m: Max Weight Matching (mwmf), a: argmax (inter), i: itermax, f: forward (fwd), r: reverse (rev)")
	parser.add_argument("--num-test-sents", type=int, default=-1, help="-1 means all sentences")
	parser.add_argument("-log", action="store_true")
	parser.add_argument("-device", type=str, default="cpu")
	parser.add_argument("-output", type=str, default="align_out", help="output alignment files (without extension)")
	args = parser.parse_args()

	TR_Models = [
		'bert-base-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 
		'xlm-mlm-100-1280', 'roberta-base', 'xlm-roberta-base', 'xlm-roberta-large']
	if args.model == "bert":
		args.model = "tr:bert-base-multilingual-cased"
	elif args.model == "xlmr":
		args.model = "tr:xlm-roberta-base"
	if args.model[3:] not in TR_Models:
		print("The model '{}' is not recognised!".format(args.model))
		exit()
	print(args)

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

	sentences = None
	if args.token_type == "bpe" or convert_to_words:
		sentences = []
		for sent_id in range(len(words_tokens)):
			sent_pair = [[x for w in sent for x in w] for sent in words_tokens[sent_id]]
			sentences.append(sent_pair)
	else:
		corpora = []
		corpora.append([l.split() for l in original_corpora[0]])
		corpora.append([l.split() for l in original_corpora[1]])
		sentences = [x for x in zip(*corpora)]

	corpora_lengths = [len(corpus) for corpus in original_corpora]
	if min(corpora_lengths) != max(corpora_lengths):
		print(corpora_lengths)
		raise ValueError('Cannot load parallel corpus.')

	# --------------------------------------------------------
	all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}
	matching_methods = [all_matching_methods[m] for m in args.matching_methods]

	out_f = {ext: open('{}.{}'.format(args.output, ext), 'w') for ext in matching_methods}
	if args.log:
		out_log = open('{}.log'.format(args.output), 'w')

	if args.null_align < 1.0:
		entropies = {x: [] for x in matching_methods}
		for sent_id, sent_pair in enumerate(sentences):
			vectors = embed_loader.get_embed_list(list(sent_pair))
			l1_tokens, l2_tokens = words_tokens[sent_id]

			if convert_to_words:
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
			sim = SentenceAligner.get_similarity(vectors[0], vectors[1])
			sim = SentenceAligner.apply_distortion(sim, args.distortion)

			methods_matrix = {}
			methods_matrix["forward"], methods_matrix["backward"] = SentenceAligner.get_alignment_matrix(sim)
			methods_matrix["inter"] = methods_matrix["forward"] * methods_matrix["backward"]
			methods_matrix["mwmf"] = SentenceAligner.get_max_weight_match(sim)
			methods_matrix["itermax"] = SentenceAligner.iter_max(sim, 1)

			for m in entropies:
				entropies[m] += gather_null_aligns(sim, methods_matrix[m])
		null_thresh = {m: sorted(entropies[m])[int(args.null_align * len(entropies[m]))] for m in entropies}

	for sent_id, sent_pair in enumerate(sentences):
		l1_tokens, l2_tokens = words_tokens[sent_id]
		if args.token_type == "bpe":
			l1_b2w_map = []
			for i, wlist in enumerate(l1_tokens):
				l1_b2w_map += [i for x in wlist]
			l2_b2w_map = []
			for i, wlist in enumerate(l2_tokens):
				l2_b2w_map += [i for x in wlist]

		vectors = embed_loader.get_embed_list(list(sent_pair))

		if convert_to_words:
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
		sim = SentenceAligner.get_similarity(vectors[0], vectors[1])
		sim = SentenceAligner.apply_distortion(sim, args.distortion)
		if args.null_align < 1.0:
			mask_nulls = {mmethod: apply_percentile_null_aligns(sim, null_thresh[mmethod]) for mmethod in matching_methods}

		all_mats["fwd"], all_mats["rev"] = SentenceAligner.get_alignment_matrix(sim)
		all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
		all_mats["mwmf"] = SentenceAligner.get_max_weight_match(sim)
		all_mats["itermax"] = SentenceAligner.iter_max(sim, 1)

		if args.null_align < 1.0:
			all_mats["inter"] = np.multiply(all_mats["inter"], mask_nulls["inter"])
			all_mats["mwmf"] = np.multiply(all_mats["mwmf"], mask_nulls["mwmf"])
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
							b2w_aligns[ext].add('{}-{}'.format(l1_b2w_map[i], l2_b2w_map[j]))
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

