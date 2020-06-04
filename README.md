# NewsFluxus - Demo #

mkvirtualenv nflux -p /usr/bin/python3





















## Dependencies
```sh
$ pip3 install -r requirements.txt
```




## GPU acceleration

Currently the requirements file installs `torch` and `torchvision` without support for GPU acceleration. If you want to use your accelerator(-s) comment out `torch` and `torchvision` in the requirements file, uninstall with pip (if relevant), and run `pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.htm` for your desired CUDA distribution (in this case 10.1).


## Download language resources
```python
$ python downloader.py --langauge <language code>
```
And you will be prompted for location to store data, just use default. To find language codes see [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/models.html#human-languages-supported-by-stanfordnlp)


--dataset ../dat/sample.ndjson --language da --bytestore 100 --sourcename politiken --verbose 100
