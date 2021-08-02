[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3779341.svg)](https://doi.org/10.5281/zenodo.3779341)



# ParEvol

Repository for code associated with the preprint:

Predicting Parallelism and Quantifying Divergence in Experimental Evolution

https://www.biorxiv.org/content/10.1101/2020.05.13.070953v2

This project uses a number of publicly available datasets. Processed data is included in the Zenodo repository. All raw data has been previously published and will have to be accessed as described below.




## Dependencies

An `environment.yml` file is in this repository and can be used to make the conda environment used to perform all analyses. To summarize, the analyses are primarily performed in Python 3.6 and require the following packages: numpy, pandas, matplotlib, scipy, Scikit-learn, Biopython, and networkx.


This repository also requires `asa159.f90` source file which can be obtained from the [FisherExact](https://github.com/maclandrol/FisherExact) repository. The file `asa159.f90` is written on fortran, so make sure you have a fortran compiler (gfortran) installed. See here ==> https://gcc.gnu.org/wiki/GFortranBinaries.

Once `asa159.f90` is in `~/GitHub/ParEvol/Python`, run the command:


```shell
f2py -c -m asa159 asa159.f90
```


## Getting the data

All the processed data is available on zenodo [https://doi.org/10.5281/zenodo.3779341](https://doi.org/10.5281/zenodo.3779341).

Below are instructions for obtaining the original dataset, though it is not necessary to redo the analyses.

### Tenaillon et al.

Data from Tenaillon et al. (2012) can be obtained from their [publication URL](https://science.sciencemag.org/content/335/6067/457). For the mutation data you will need to download **Table S2** as an excel file (`1212986tableS2.xls`) and convert it to CSV format. Move the CSV to `~/GitHub/ParEvol/data/Tenaillon_et_al`. Table S1 in Tenaillon et al. (2019) contains the fitness estimates. It is in the supplement.



### Turner et al.

Data from Turner et al. (2018) can be obtained from the publication's [Dryad repository](https://doi.org/10.5061/dryad.53n0rf5). Move the data in the repository to `~/GitHub/ParEvol/data/Turner_et_al`






## Cleaning the data

Run the following command:

```shell
python clean_data.py
```


## Running the simulations

```shell
python run_simulations.py
```



## Making the figures

```shell
python make_figs.py
```






## References

Olivier Tenaillon, Alejandra Rodraguez-Verdugo, Rebecca L. Gaut, Pamela  McDonald,  Albert F. Bennett, Anthony D. Long, and Brandon S. Gaut. The Molecular Diversity of Adaptive Convergence. Science, 416335(6067): 457–461, 2012.

Caroline B. Turner, Christopher W. Marshall, and Vaughn S. Cooper. Parallel genetic adaptation across environments differing in mode of growth or resource availability. Evolution Letters, 2(4):355–367, August 2018.
