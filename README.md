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


## Attributes




Running Python on Carbonate

module avail python
module unload python/2.7.13

module unload python/3.6.1
module switch python/2.7.13 python/3.6.1

module load anaconda/python3.6/4.3.1

conda create -n ParEvol python=3.6
