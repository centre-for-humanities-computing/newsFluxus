# NewsFluxus - Demo #

Tool for representing change and persistence in newspapers. For an exposition of the underlying method see [Detection of Persistent Change in Nordic Newspapers](https://centre-for-humanities-computing.github.io/Nordic-Digital-Humanities-Laboratory/portfolio/news_c19_method/).

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

Clone and install Mallet (plus dependencies)
```sh
sudo apt-get install default-jdk
sudo apt-get install ant
git clone git@github.com:mimno/Mallet.git
cd Mallet/
ant
```
Change path the local mallet installation in `src/tekisuto/models/latentsemantics.py`

### GPU acceleration

Currently the requirements file installs `torch` and `torchvision` without support for GPU acceleration. If you want to use your accelerator(-s) comment out `torch` and `torchvision` in the requirements file, uninstall with pip (if relevant), and run `pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html` for your desired CUDA version (in this case 10.1).

#### Test Mallet installation
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
And you will be prompted for location to store data, just use default. To find language codes see [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/models.html#human-languages-supported-by-stanfordnlp)

#### Test StanfordNLP Installation
```bash
>>> import stanfordnlp

>>> nlp = stanfordnlp.Pipeline(lang="da")
>>> doc = nlp("Rap! rap! sagde hun, og så rappede de sig alt hvad de kunne, og så til alle sider under de grønne blade, og moderen lod dem se så meget de ville, for det grønne er godt for øjnene.")
>>> doc.sentences[0].print_dependencies()
```

### Train model and extract signal
```bash
bash main.sh
```

And individually

```bash
$ python src/bow_mdl.py --dataset <path-to-dataset> --language <language-code> --bytestore <frequency-of-backup> --sourcename <name-of-dataset> --estimate "<start stop step>" --verbose <frequency-of-log>
$ python src/signal_extraction.py --model <path-to-serialized-model>
# ex. for Danish sample
$ python bow_mdl.py --dataset ../dat/sample.ndjson --language da --bytestore 100 --estimate "20 50 10" --sourcename sample --verbose 100
$ python python src/signal_extraction.py --model mdl/da_sample_model.pcl
```

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Versioning


## Authors
Kristoffer L. Nielbo

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
[Stopwords ISO](https://github.com/stopwords-iso) for their multilingual collection of stopwords.
