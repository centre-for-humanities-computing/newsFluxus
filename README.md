# NewsFluxus #

Tool for modelling change and persistence in newspaper content. For an exposition of the underlying method see [Persistent News: The Information Dynamics of Nordic Newspapers](https://centre-for-humanities-computing.github.io/Nordic-Digital-Humanities-Laboratory/portfolio/news_c19_method/) and for design see [News-fluxus design specification](https://github.com/centre-for-humanities-computing/newsFluxus).

Publications:

- K. L. Nielbo, R. B. Baglini, P. B. Vahlstrup, K. C. Enevoldsen, A. Bechmann, and A. Roepstorff, “News Information Decoupling: An Information Signature of Catastrophes in Legacy News Media,” arXiv:2101.02956 [cs].

## Prerequisites

For running in virtual environment (recommended) and assuming python3.7+ is installed.

```bash
$ sudo pip3 install virtualenv
$ virtualenv -p /usr/bin/python3.7 venv
$ source venv/bin/activate
```

## Installation

Clone repository and install requirements

```bash
$ git clone https://github.com/centre-for-humanities-computing/newsFluxus.git
$ pip3 install -r requirements.txt
```

### GPU acceleration

Currently the requirements file installs `torch` and `torchvision` without support for GPU acceleration. If you want to use your accelerator(-s) comment out `torch` and `torchvision` in the requirements file, uninstall with pip (if relevant), and run `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html` for your desired CUDA version (in this case 11.0+).

### Install Mallet
Clone and install Mallet (plus dependencies)
```sh
$ sudo apt-get install default-jdk
$ sudo apt-get install ant
$ git clone git@github.com:mimno/Mallet.git
$ cd Mallet/
$ ant
```
Change path the local mallet installation in `src/tekisuto/models/latentsemantics.py`

#### Test Mallet wrapper
```bash
>>> from gensim.test.utils import common_corpus, common_dictionary
>>> from gensim.models.wrappers import LdaMallet

>>> path_to_mallet_binary = "/path/to/mallet/binary"
>>> model = LdaMallet(path_to_mallet_binary, corpus=common_corpus, num_topics=20, id2word=common_dictionary)
```

### Download language resources
```bash
$ python downloader.py --langauge <language-code>
# ex. for Danish langauge resources
$ python downloader.py --language da
```
And you will be prompted for location to store data, just use default. To find language codes see [Stanza](https://stanfordnlp.github.io/stanza/available_models.html)

#### Test Stanza Installation
```
>>> import stanza

>>> nlp = stanza.Pipeline(lang="da")
>>> doc = nlp("Rap! rap! sagde hun, og så rappede de sig alt hvad de kunne, og så til alle sider under de grønne blade, og moderen lod dem se så meget de ville, for det grønne er godt for øjnene.")
>>> doc.sentences[0].print_dependencies()
```

### Train model and extract signal
```bash
$ bash main.sh
```

And individually

```bash
$ python src/bow_mdl.py --dataset <path-to-dataset> --language <language-code> --bytestore <frequency-of-backup> --sourcename <name-of-dataset> --estimate "<start stop step>" --verbose <frequency-of-log>
$ python src/signal_extraction.py --model <path-to-serialized-model>
# ex. for Danish sample
$ python bow_mdl.py --dataset ../dat/sample.ndjson --language da --bytestore 100 --estimate "20 50 10" --sourcename sample --verbose 100
$ python python src/signal_extraction.py --model mdl/da_sample_model.pcl
```

### Research use-case
Requires `matplotlib`
```bash
$ python src/news_uncertainty.py --dataset mdl/da_sample_signal.json --window 7 --figure "fig"
```
resulting visualizations in `fig/`

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :smiling_imp:

## Versioning

| Edition | Date | Comment |
| --- | --- | --- |
| v1.0 | June 04 2020 | Launch |
| v1.1 | January 14 2020 | New NLP pipeline |

## Authors
Kristoffer L. Nielbo

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
[Stopwords ISO](https://github.com/stopwords-iso) for their multilingual collection of stopwords.
