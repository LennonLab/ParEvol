# ParEvol

Project repository for data and Python code associated with the preprint

## Re-running the analyses and re-generating the figures

The code accepts the following arguments from the user.

**Flags**

**`-a`** or **`--analysis`:** Runs the analysis for a specified figure.

**`-f`:** Generates a specified figure.


| Argument |          Figure/ Table         |
|:--------:|:-----------------------:|
|     F1    |         Figure 1        |


### Example

If you want to re-generate the data (warning, this is computationally intensive and will take several days to complete) and the figure for figure one, run the following command.

`python run_analysis.py -a -f F1`


If you only want to generate the figure using the data that has already been generated, run the following command.

	`python run_analysis.py -f F1`




## Dependencies


## Publicaly available data

Mutation data from Wannier et al. (2018) was downloaded in a CSV format from the following links on the [Adaptive Laboratory Evolution database](https://aledb.org/) (Phaneuf et. al., 2018). 

- [C321](https://aledb.org/stats?ale_experiment_id=76)

- [C321.∆A](https://aledb.org/stats?ale_experiment_id=77)

- [C321.∆A-v2	](https://aledb.org/stats?ale_experiment_id=78)

- [ECNR2.1](https://aledb.org/stats?ale_experiment_id=79)





## Attributes


## References

Phaneuf PV, Gosting D, Palsson BO, Feist AM. ALEdb 1.0: a database of mutations from adaptive laboratory evolution experimentation. Nucleic Acids Res. 2018; doi:10.1093/nar/gky983



Running Python on Carbonate

module avail python
module unload python/2.7.13

module unload python/3.6.1
module switch python/2.7.13 python/3.6.1

module load anaconda/python3.6/4.3.1

conda create -n ParEvol python=3.6
