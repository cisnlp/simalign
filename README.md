SimAlign: Similarity Based Word Aligner
==============

SimAlign is a high-quality word alignment tool that uses static and contextualized embeddings and **does not require parallel training data**.

For more details see the [Paper](https://arxiv.org/pdf/2004.08728.pdf).


Requirements
--------

- Huggingface Transformers 2.3.0. For installation instructions see [their repo](https://github.com/huggingface/transformers#installation).


Usage
--------

For an example how to use our code see `tbd`.


Demo
--------

An online demo is available [here](http://simalign.cis.lmu.de/).


Publications
--------

If you use the code, please cite 

```
@article{sabet2020simalign,
  title={SimAlign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings},
  author={Sabet, Masoud Jalili and Dufter, Philipp and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2004.08728},
  year={2020}
}
```

Feedback
--------

Feedback and Contributions more than welcome! Just reach out to @masoudjs or @pdufter. 


FAQ
--------

##### Do I need parallel data to train the system?

No, no parallal training data is required.


##### Which languages can be aligned?

This depends on the underlying pretrained multilingual language model used. For example, if mBERT is used, it covers 104 language

##### Do I need GPUs for running this?

Each alignment simply requires a single forward pass in the pretrained language model. While this is certainly 
faster on GPU, it runs fine on CPU.


TODOs
--------

* Reduce dependencies and clean requirements.txt
* Add proper logging
* Add usage example
* Make installable
* Add tests


License
-------

Copyright (C) 2020, Masoud Jalili Sabet, Philipp Dufter

Licensed under the terms of the GNU General Public License, version 3. A full copy of the license can be found in LICENSE.
