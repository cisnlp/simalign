# Run bert-as-a-service server with
# bert-serving-start \
# -model_dir /Users/philipp/Downloads/multi_cased_L-12_H-768_A-12 \
# -pooling_strategy NONE \
# -cased_tokenization -show_tokens_to_client
#
# In the environment bertservice
# import bert_serving.client
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Text, Tuple
import networkx as nx
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix
import argparse
import os
# import tqdm


class ParallelCorpus(object):
    """docstring for ParallelCorpus"""

    def __init__(self) -> None:
        self.sentences = None

    def load_from_separate_files(self, paths: List[Text]) -> None:
        """Assumes the line number as sentence ID.
        """
        corpora = []
        for path in paths:
            corpus = open(path, 'r', errors='replace').readlines()
            corpora.append(corpus)

        corpora_lengths = [len(corpus) for corpus in corpora]

        if min(corpora_lengths) != max(corpora_lengths):
            raise ValueError('Cannot load parallel corpus.')

        self.sentences = zip(*corpora)


class GoldStandardText(ParallelCorpus):
    """docstring for GoldStandardText"""
    def __init__(self):
        super(GoldStandardText, self).__init__()

    def load_single_file(self, path: Text) -> Dict[Text, Text]:
        data = {}
        with open(path, 'r') as fp:
            for line in fp:
                line = line.replace('\n', '')
                i, sentence = line.split('\t')
                data[i] = sentence
        return data

    def load_from_separate_files(self, path_e: Text, path_f: Text) -> None:
        self.e = self.load_single_file(path_e)
        self.f = self.load_single_file(path_f)
        self.joint_ids = set(self.e.keys()) & set(self.f.keys())


class Alignment(object):
    """docstring for Alignment"""

    def __init__(self) -> None:
        super(Alignment, self).__init__()

    def from_similarity_matrix(self, sim: np.ndarray, method: Text) -> np.ndarray:
        if method == 'greedy':
            m, n = sim.shape
            forward = np.eye(n)[sim.argmax(axis=1)]  # m x n
            backward = np.eye(m)[sim.argmax(axis=0)]  # n x m
        elif method == 'max_weight_matching':
            def permute(edge):
                if edge[0] < sim.shape[0]:
                    return edge[0], edge[1] - sim.shape[0]
                else:
                    return edge[1], edge[0] - sim.shape[0]
            G = from_biadjacency_matrix(csr_matrix(sim))
            matching = nx.max_weight_matching(G, maxcardinality=True)
            matching = [permute(x) for x in matching]
            matching = sorted(matching, key=lambda x: x[0])
            forward = np.zeros_like(sim)
            for edge in matching:
                forward[edge[0], edge[1]] = 1
            backward = forward.transpose()
        elif method == 'max_flow':
            raise NotImplementedError()
        return forward, backward

    def symmetrize(self, forward: np.ndarray, backward: np.ndarray, method: Text) -> np.ndarray:
        if method == 'intersect':
            sym = forward * backward
        return sym

    def print(self, alignment_matrix: np.ndarray,
              sim: np.ndarray = None, e: List[Text] = None, f: List[Text] = None) -> Text:
        result = []
        for i in range(alignment_matrix.shape[0]):
            for j in range(alignment_matrix.shape[1]):
                if alignment_matrix[i, j] > 0:
                    if e is not None and f is not None:
                        left = e[i]
                        right = f[j]
                    else:
                        left = i
                        right = j
                    if sim is not None:
                        result.append('{}-{} ({:.2f})'.format(left, right, sim[i, j]))
                    else:
                        result.append('{}-{}'.format(left, right))
        return ' '.join(result)


class AlignmentWriter(object):
    """docstring for AlignmentWriter"""
    def __init__(self, path: Text) -> None:
        super(AlignmentWriter, self).__init__()
        self.path = path
        os.system('mkdir -p {}'.format(path))

    def create_files(self) -> None:
        self.outfile_alignment = open('{}/alignment.txt'.format(self.path), 'w')
        self.outfile_e = open('{}/e.txt'.format(self.path), 'w')
        self.outfile_f = open('{}/f.txt'.format(self.path), 'w')

    def write(self, myid: Text, alignment: Text, text_e: Text, text_f: Text) -> None:
        self.outfile_alignment.write('{}\t{}\n'.format(myid, alignment))
        self.outfile_e.write('{}\t{}\n'.format(myid, text_e))
        self.outfile_f.write('{}\t{}\n'.format(myid, text_f))

    def close(self) -> None:
        self.outfile_alignment.close()
        self.outfile_e.close()
        self.outfile_f.close()


def get_vectors(sentences: List[Text]) -> Tuple[np.ndarray, List[List[Text]]]:
    bc = bert_serving.client.BertClient()
    vectors, words = bc.encode(sentences, show_tokens=True)
    return vectors, words


def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (cosine_similarity(X, Y) + 1.0) / 2.0


def get_alignment_matrix(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m, n = sim.shape
    forward = np.eye(n)[sim.argmax(axis=1)]  # m x n
    backward = np.eye(m)[sim.argmax(axis=0)]  # n x m
    return forward, backward.transpose()


def symmetrize(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    return forward * backward


def print_alignment(bitext: Tuple[Text, Text],
                    alignment_matrix: np.ndarray,
                    sim: np.ndarray = None) -> None:
    for i, text_a in enumerate(bitext[0]):
        for j, text_b in enumerate(bitext[1]):
            if alignment_matrix[i, j] > 0:
                if sim is not None:
                    print('{}-{} ({:.2f})'.format(text_a, text_b, sim[i, j]))
                else:
                    print('{}-{}'.format(text_a, text_b))


def prepare_alignment_for_write(alignment_matrix: np.ndarray) -> Text:
    result = []
    for i in range(alignment_matrix.shape[0]):
        for j in range(alignment_matrix.shape[1]):
            if alignment_matrix[i, j] > 0:
                result.append((i, j))
    result = [' '.join([str(i) for i in x]) for x in result]
    return ' '.join(result)


def align_corpus_with_bert(path1: Text, path2: Text, outpath: Text):
    pc = ParallelCorpus()
    pc.load_from_separate_files([path1, path2])

    outfile = open(outpath, 'w')
    outfile_de = open(outpath + '_textde', 'w')
    outfile_en = open(outpath + '_texten', 'w')
    for i, sentence_pair in enumerate(pc.sentences):
        if i > 15:
            break
        vectors, words = get_vectors(list(sentence_pair))
        sim = get_similarity(vectors[0], vectors[1])
        forward, backward = get_alignment_matrix(sim)
        intersect = symmetrize(forward, backward)

        outfile.write(prepare_alignment_for_write(intersect))
        outfile_de.write(' '.join(words[0]) + '\n')
        outfile_en.write(' '.join(words[1]) + '\n')
        outfile.write('\n')


def align_with_bert():
    parser = argparse.ArgumentParser(description='Align with BERT.')
    parser.add_argument("--file_e", type=str, help="")
    parser.add_argument("--file_f", type=str, help="")
    parser.add_argument("--outpath", type=str, help="")
    parser.add_argument("--method", type=str, help="'greedy' or 'max_weight_matching'")
    parser.add_argument("--direction", type=str, help="'forward', 'backward', 'symmetric'")

    params = parser.parse_args()

    # parameterize input arguments
    gs = GoldStandardText()
    gs.load_from_separate_files(params.file_e, params.file_f)
    joint_ids = list(gs.joint_ids)
    joint_ids = sorted(joint_ids, key=lambda x: int(x))
    X, Xwords = get_vectors([gs.e[myid] for myid in joint_ids])
    Y, Ywords = get_vectors([gs.f[myid] for myid in joint_ids])

    writer = AlignmentWriter(params.outpath)
    writer.create_files()

    for i, myid in enumerate(joint_ids):
        sim = get_similarity(X[i, :len(Xwords[i]), :], Y[i, :len(Ywords[i]), :])
        # remove similarity with CLS and SEP token
        sim = sim[1:-1, 1:-1]
        al = Alignment()
        fw, bw = al.from_similarity_matrix(sim, method=params.method)
        sym_alignment = al.symmetrize(fw, bw.transpose(), method='intersect')
        if params.direction == 'forward':
            final_alignment = fw
        elif params.direction == 'backward':
            final_alignment = bw
        elif params.direction == 'symmetric':
            final_alignment = sym_alignment

        alignment_to_print = al.print(final_alignment)
        text_to_print_e = ' '.join(Xwords[i][1:-1])
        text_to_print_f = ' '.join(Ywords[i][1:-1])

        writer.write(myid, alignment_to_print, text_to_print_e, text_to_print_f)

    writer.close()


def main():

    sents = ['Die Bundesregierung erlässt ein Verbot von Waffenexporten nach Saudi-Arabien.',
             'The German government prohibits exporting weapons to Saudi Arabia.']

    sents = ['最初 ， 上帝 創造 了 天地 。',
             'In the beginning God created the heaven and the earth .']
    sents = ['We do not believe that we should cherry-pick .',
             'Wir glauben nicht , daß wir nur Rosinen herauspicken sollten .']

    vectors, words = get_vectors(sents)
    sim = get_similarity(vectors[0], vectors[1])
    forward, backward = get_alignment_matrix(sim)
    print('\n\n\nFORWARD')
    print_alignment(words, forward, sim)
    print('\n\n\nBACKWARD')
    print_alignment(words, backward, sim)
    intersect = symmetrize(forward, backward)
    print('\n\n\nINTERSECT')
    print_alignment(words, intersect, sim)

    align_corpus_with_bert('/Users/philipp/Downloads/DeEn/de',
                           '/Users/philipp/Downloads/DeEn/en',
                           '/Users/philipp/Downloads/DeEn/bertalign')


if __name__ == '__main__':
    align_with_bert()
