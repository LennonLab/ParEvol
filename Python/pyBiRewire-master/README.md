# pyBiRewire
python wrapper BiRewire package

This module is a cython wrapper of <a href="http://bioconductor.jp/packages/3.2/bioc/html/BiRewire.html">BiRewire</a> package pubblished in the bioconductor repository 
implementing the algorithms described in <a href="http://bioinformatics.oxfordjournals.org/content/30/17/i617.full.pdf">Fast randomization of large genomic datasets while preserving alteration counts</a>. 


# How to install

Clone or download the whole progect, in the repo directory run `python3 setup.py build_ext --inplace`, just import the module in python3: `import BiRewire as br` for example. It should works also if compiled with python 2.x (and imported it from python2.x).