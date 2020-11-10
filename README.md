SimAlign: Similarity Based Word Aligner
==============

<p align="center">
    <br>
    <img alt="Alignment Example" src="https://raw.githubusercontent.com/cisnlp/simalign/master/assets/example.png" width="300"/>
    <br>
<p>

SimAlign is a high-quality word alignment tool that uses static and contextualized embeddings and **does not require parallel training data**.

The following table shows how it compares to popular statistical alignment models:

|            | ENG-CES | ENG-DEU | ENG-FAS | ENG-FRA | ENG-HIN | ENG-RON |
| ---------- | ------- | ------- | ------- | ------- | ------- | ------- |
| fast-align | .78     | .71     | .46     | .84     | .38     | .68     |
| eflomal    | .85     | .77     | .63     | .93     | .52     | .72     |
| mBERT-Argmax | .87     | .81     | .67     | .94     | .55     | .65     |

Shown is F1, maximum across subword and word level. For more details see the [Paper](https://arxiv.org/pdf/2004.08728.pdf).


Installation and Usage
--------

Tested with Python 3.7, Transformers 3.1.0, Torch 1.5.0. Networkx 2.4 is optional (only required for Match algorithm). 
For full list of dependencies see `setup.py`.
For installation of transformers see [their repo](https://github.com/huggingface/transformers#installation).

Download the repo for use or alternatively install with pip

`pip install --upgrade git+https://github.com/cisnlp/simalign.git#egg=simalign`


An example for using our code:
```python
from simalign import SentenceAligner

# making an instance of our model.
# You can specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

# The source and target sentences should be tokenized to words.
src_sentence = ["This", "is", "a", "test", "."]
trg_sentence = ["Das", "ist", "ein", "Test", "."]

# The output is a dictionary with different matching methods.
# Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)

for matching_method in alignments:
    print(matching_method, ":", alignments[matching_method])

# Expected output:
# mwmf (Match): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# inter (ArgMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# itermax (IterMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
```
For more examples of how to use our code see `example/align_example.py`.

Demo
--------

An online demo is available [here](https://simalign.cis.lmu.de/).


Gold Standards
--------
Links to the gold standars used in the paper are here: 


| Language Pair  | Citation | Type |Link |
| ------------- | ------------- | ------------- | ------------- |
| ENG-CES | Marecek et al. 2008 | Gold Alignment | http://ufal.mff.cuni.cz/czech-english-manual-word-alignment |
| ENG-DEU | EuroParl-based | Gold Alignment | www-i6.informatik.rwth-aachen.de/goldAlignment/ |
| ENG-FAS | Tvakoli et al. 2014 | Gold Alignment | http://eceold.ut.ac.ir/en/node/940 |
| ENG-FRA |  WPT2003, Och et al. 2000,| Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt/ |
| ENG-HIN |   WPT2005 | Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt05/ |
| ENG-RON |  WPT2005 Mihalcea et al. 2003 | Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt05/ |
        
        

Publication
--------

If you use the code, please cite 

```
@inproceedings{jalili-sabet-etal-2020-simalign,
    title = "{S}im{A}lign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings",
    author = {Jalili Sabet, Masoud  and
      Dufter, Philipp  and
      Yvon, Fran{\c{c}}ois  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.147",
    pages = "1627--1643",
}
```

Feedback
--------

Feedback and Contributions more than welcome! Just reach out to @masoudjs or @pdufter. 


FAQ
--------

##### Do I need parallel data to train the system?

No, no parallel training data is required.

##### Which languages can be aligned?

This depends on the underlying pretrained multilingual language model used. For example, if mBERT is used, it covers 104 languages as listed [here](https://github.com/google-research/bert/blob/master/multilingual.md).

##### Do I need GPUs for running this?

Each alignment simply requires a single forward pass in the pretrained language model. While this is certainly 
faster on GPU, it runs fine on CPU. On one GPU (GeForce GTX 1080 Ti) it takes around 15-20 seconds to align 500 parallel sentences.


TODOs
--------

* Add evaluation code
* Add static embedding functionality
* Add wrappers for fast-align, eflomal
* Add data download scripts 



License
-------

Copyright (C) 2020, Masoud Jalili Sabet, Philipp Dufter

Licensed under the terms of the GNU General Public License, version 3. A full copy of the license can be found in LICENSE.
