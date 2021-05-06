#!/usr/bin/env python3
import argparse
import collections
import os.path


def load_gold(g_path):
	gold_f = open(g_path, "r")
	pros = {}
	surs = {}
	all_count = 0.
	surs_count = 0.

	for line in gold_f:
		line = line.strip().split("\t")
		line[1] = line[1].split()

		pros[line[0]] = set([x.replace("p", "-") for x in line[1]])
		surs[line[0]] = set([x for x in line[1] if "p" not in x])

		all_count += len(pros[line[0]])
		surs_count += len(surs[line[0]])

	return pros, surs, surs_count

def calc_score(input_path, probs, surs, surs_count):
	total_hit = 0.
	p_hit = 0.
	s_hit = 0.
	target_f = open(input_path, "r")

	for line in target_f:
		line = line.strip().split("\t")

		if line[0] not in probs: continue
		if len(line) < 2: continue
		line[1] = line[1].split()
		if len(line[1][0].split("-")) > 2:
			line[1] = ["-".join(x.split("-")[:2]) for x in line[1]]

		p_hit += len(set(line[1]) & set(probs[line[0]]))
		s_hit += len(set(line[1]) & set(surs[line[0]]))
		total_hit += len(set(line[1]))
	target_f.close()

	y_prec = round(p_hit / max(total_hit, 1.), 3)
	y_rec = round(s_hit / max(surs_count, 1.), 3)
	y_f1 = round(2. * y_prec * y_rec / max((y_prec + y_rec), 0.01), 3)
	aer = round(1 - (s_hit + p_hit) / (total_hit + surs_count), 3)

	return y_prec, y_rec, y_f1, aer


if __name__ == "__main__":
	'''
	Calculate alignment quality scores based on the gold standard.
	The output contains Precision, Recall, F1, and AER.
	The gold annotated file should be selected by "gold_path".
	The generated alignment file should be selected by "input_path".
	Both gold file and input file are in the FastAlign format with sentence number at the start of line separated with TAB.

	usage: python calc_align_score.py gold_file generated_file
	'''

	parser = argparse.ArgumentParser(description="Calculate alignment quality scores based on the gold standard.", epilog="example: python calc_align_score.py gold_path input_path")
	parser.add_argument("gold_path")
	parser.add_argument("input_path")
	args = parser.parse_args()

	if not os.path.isfile(args.input_path):
		print("The input file does not exist:\n", args.input_path)
		exit()

	probs, surs, surs_count = load_gold(args.gold_path)
	y_prec, y_rec, y_f1, aer = calc_score(args.input_path, probs, surs, surs_count)

	print("Prec: {}\tRec: {}\tF1: {}\tAER: {}".format(y_prec, y_rec, y_f1, aer))

