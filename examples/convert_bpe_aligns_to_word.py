import codecs
import regex


src_l, trg_l = "eng", "per"
name = "{}_{}".format(src_l, trg_l)

home_path_m = "/mounts/work/mjalili/"
input_gold_txt_path = home_path_m + "data/" + name + "/"

# data_path = "/mounts/work/philipp/bert_alignment/"
data_path = "/mounts/Users/student/masoud/projects/embedding_aligns/data/" + name + "/"

# input_txt_path = data_path + "max_weight_matching-forward/eng_per/"
input_align_path = data_path
output_path = data_path + "bert_layer_aligns/"
additional_suffix = "layer_concat"

model = "fwd"
model = "rev"
model = "mwmf"
model = "gdfa"
model = "inter"


def get_word_index(bpe_sent_as_list, orig_sent):
	index_map = []

	orig_i = 0
	word_id = 0

	for b_i, bpe in enumerate(bpe_sent_as_list):
		bpe = bpe.replace("##", "")

		if orig_sent[orig_i:].startswith(bpe):
			orig_i += len(bpe)
			index_map.append(word_id)
		elif bpe == "[UNK]":
			index_map.append(word_id)
			orig_i += orig_sent[orig_i:].find(" ")
		else:
			print("\n#####\nBPE not found in the original sentence!\n#####\n")
			print(bpe, orig_sent[orig_i: orig_i + 15])
			print("'" + bpe[0] + "' '" + bpe[1] + "'\t\t'" + orig_sent[orig_i] + "' '" + orig_sent[orig_i+1] + "'")
			print(bpe_sent_as_list, "\n", orig_sent)
			print(regex.sub("\\p{Separator}", "#", orig_sent[orig_i-5: orig_i + 10]))
			print(regex.sub("\\p{C}", "_", orig_sent[orig_i-5: orig_i + 10]))
			return None

		if orig_i < len(orig_sent) and orig_sent[orig_i] == " ":
			while orig_sent[orig_i] == " ":
				orig_i += 1
			word_id += 1
		if orig_i == len(orig_sent) and b_i + 1 < len(bpe_sent_as_list):
			print("\n#####\nBPE sentence is longer than the original!\n#####\n")
			return None

	return index_map

if __name__ == "__main__":
	# f_al = open(input_align_path + name + additional_suffix + "." + model, "r")
	f_al = open(input_align_path + additional_suffix + "." + model, "r")
	# f_s = codecs.open(data_path + src_l + ".bpe", "r", "utf-8")
	# f_t = codecs.open(data_path + trg_l + ".bpe", "r", "utf-8")
	f_s = codecs.open(data_path + src_l + "_gold.bpe", "r", "utf-8")
	f_t = codecs.open(data_path + trg_l + "_gold.bpe", "r", "utf-8")

	# f_os = codecs.open(input_gold_txt_path + "eng_gold_text_ende.txt", "r", "utf-8")
	# f_ot = codecs.open(input_gold_txt_path + "deu_gold_text_ende.txt", "r", "utf-8")
	# f_os = codecs.open(input_gold_txt_path + "eng_gold_text_enfr.txt", "r", "utf-8")
	# f_ot = codecs.open(input_gold_txt_path + "fra_gold_text_enfr.txt", "r", "utf-8")
	f_os = codecs.open(input_gold_txt_path + src_l + "_gold.txt", "r", "utf-8")
	f_ot = codecs.open(input_gold_txt_path + trg_l + "_gold.txt", "r", "utf-8")

	fo = open(output_path + name + "_" + additional_suffix + "." + model, "w")

	count = 0
	for l1, l2, ol1, ol2, al in zip(f_s, f_t, f_os, f_ot, f_al):
		al = al.split("\t")[1]
		# al = al.strip()
		b_al = [[int(p.split("-")[0]), int(p.split("-")[1])] for p in al.strip().split()]

		l1 = l1.split("\t")[1].strip().split()
		l2 = l2.split("\t")[1].strip().split()
		ol1 = ol1.split("\t")[1].strip()
		ol2 = ol2.split("\t")[1].strip()
		ol1 = ol1.strip()
		ol2 = ol2.strip()
		ol1 = regex.sub("\\p{Separator}+", " ", ol1)
		ol2 = regex.sub("\\p{C}+", "", ol2)

		if not len(l1) or not len(l2):
			print(count, l1, l2)
			print("\n\n\n\n=============== NOOOOOOO ===============\n\n\n\n")
			count += 1
			continue

		l1_map = get_word_index(l1, ol1)
		l2_map = get_word_index(l2, ol2)

		# print(l1, "\n", l1_map)
		# print(l2, "\n", l2_map)
		# break

		new_al = set()
		for p in b_al:
			x = l1_map[p[0]]
			y = l2_map[p[1]]
			# new_al |= set([str(x)+"-"+str(y) for x in s_l for y in t_l if "-" not in str(x) + str(y)])
			new_al.add(str(x)+"-"+str(y))

		tmp = [x for x in new_al if "-" not in x]
		if len(tmp): print(tmp)

		fo.write(str(count) + "\t" + " ".join(sorted([x for x in new_al])) + "\n")
		count += 1

	f_al.close()
	f_s.close()
	f_t.close()
	f_os.close()
	f_ot.close()
	fo.close()
