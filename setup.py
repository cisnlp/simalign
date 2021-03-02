from setuptools import setup


setup(name='simalign',
      version='0.2',
      description='Word Alignments using Pretrained Language Models',
      keywords="NLP deep learning transformer pytorch BERT Word Alignment",
      url='https://github.com/cisnlp/simalign',
      author='Masoud Jalili Sabet, Philipp Dufter',
      author_email='philipp@cis.lmu.de,masoud@cis.lmu.de',
      license='MIT',
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
