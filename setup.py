from setuptools import setup


setup(name='simalign',
      version='0.1',
      description='Word Alignments using Pretrained Language Models',
      keywords="NLP deep learning transformer pytorch BERT Word Alignment",
      url='https://github.com/masoudjs/simalign',
      author='Masoud Jalili Sabet, Philipp Dufter',
      author_email='philipp@cis.lmu.de,masoud@cis.lmu.de',
      license='GPL-3.0',
      packages=['simalign'],
      install_requires=[
          "numpy",
          "torch",
          "scipy",
          "transformers",
          "regex",
          "networkx == 2.4",
          "scikit_learn",
      ],
      python_requires=">=3.6.0",
      zip_safe=False)
