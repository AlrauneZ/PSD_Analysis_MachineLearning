# PSD Analysis using Machine Learning to identify hydraulic conductivity

This repository provides routines to perform particle size distribution (PSD) analysis, particularly workflows to estimate hydraulic conductivity with six Machine Learning (ML) algorithms:
- Decision Tree (DT)
- Random Forest (RF)
- XGBoost (XG)
- Linear Regression (LR)
- Support Vector Regression (SVR)
- Artificial Neural Network (ANN)



The package also includes methods for identification of properties, like grain diameter percentiles (d10, d50, d60 etc) and for calculation of hydraulic conductivity through empirical formulas. 

The algorithms are tested on soil sample data from the "TopIntegraal" project provided by TNO. 

## Structure

- `README.md` - description of the project
- `LICENSE` - the default license is MIT
- `data/` - contains the raw and input data:
   + `xyz.xlsx` - raw data from TopIntegraal Database 
   + `xyz.csv`  - required sample data in standard format (columns with sieve sampling values, one column containing K-value and one column with soil class specification)
- `results/` - results of processed data (algorithm performance) and plots used in publication:
   + `01_Fig_xyz.pdf`
- `src/`  - contains all scripts used for data analyses and plotting of results
  + `00_PDS_data_processing.py` - preprocessing of raw data to transform into dataframe stored in csv file with standard format
  + `01_Fig_xzy_.py` - reproducing Figure 1 of the manuscript
 
 
## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages
- `spec-file.txt` - specification file to create the original [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)

## Contact

You can contact us via <a.zech@uu.nl>.


## License

MIT © 2023


