[![DOI](https://zenodo.org/badge/687656693.svg)](https://zenodo.org/doi/10.5281/zenodo.12543936)

# PSD Analysis using Machine Learning to identify hydraulic conductivity

This repository accompanies the manuscript "Predicting Saturated Hydraulic Conductivity from Particle Size Distributions using Machine Learning" (submitted/under revision). 

The repository provides routines to perform particle size distribution (PSD) analysis, particularly workflows to estimate hydraulic conductivity with six Machine Learning (ML) algorithms:
- Decision Tree (DT)
- Random Forest (RF)
- XGBoost (XG)
- Linear Regression (LR)
- Support Vector Regression (SVR)
- Artificial Neural Network (ANN)

The package also includes methods for identification of properties, like grain diameter percentiles (d10, d50, d60 etc) and for calculation of hydraulic conductivity through empirical formulas. 

The algorithms are tested on soil sample data from the "TopIntegraal" project provided by TNO. (data not yet avaialbe due to license issues, planned to be provided soon)

## Structure

- `README.md` - description of the project
- `LICENSE` - the default license is MIT
- `data/` - does not contain the TopIntegral data (PSD) yet (due to license issues):
- `results/` - results of processed data (algorithm performance) and plots used in publication:
    + `Kemp_all.csv`  - estimated Kf values of all samples for 15 empirical methods, inlcuding specification of applicability
    - `Data_analysis/` - results of analysis of PSD for samples:
        + `data_PSD_props.csv`  - results of PSD analysis for all samples (grain diameters, percentage sand/silt/lutum, lithoclass)
        + `data_full_stats.csv` - statistical results (mean, std, percentiles,...) of properties (Kf,  percentage sand/silt/lutum, ...) for all samples
        + `data_sand_stats.csv` - statistical results (mean, std, percentiles,...) of properties (Kf,  percentage sand/silt/lutum, ...) for subset of sand samples
        + `data_silt_stats.csv` - statistical results (mean, std, percentiles,...) of properties (Kf,  percentage sand/silt/lutum, ...) for subset of silt samples
        + `data_clay_stats.csv` - statistical results (mean, std, percentiles,...) of properties (Kf,  percentage sand/silt/lutum, ...) for subset of clay samples
        + `data_por_stats.csv`  - statistical results (mean, std, percentiles,...) of properties (Kf,  percentage sand/silt/lutum, porosity, ...) for subset of samples with porosity
    - `ML_Performance/` - performance measure R^2 or MSE for all 6 ML algorithms on training, testing and all sample for:
        + `Performance_PSD_Kf_topall_r2.csv`  - feature varialble PSD to target variable Kf for data set "Top-All"  (R^2)
        + `Performance_PSD_Kf_topall_mse.csv` - feature varialble PSD to target variable Kf for data set "Top-All" (MSE)
        + `Performance_PSD_Kf_sand_r2.csv` - feature varialble PSD to target variable Kf for data set "Top-Sand" (R^2)
        + `Performance_PSD_Kf_sand_mse.csv` - feature varialble PSD to target variable Kf for data set "Top-Sand" (MSE)
        + `Performance_PSD_Kf_silt_r2.csv` - feature varialble PSD to target variable Kf for data set "Top-Silt" (R^2)
        + `Performance_PSD_Kf_silt_mse.csv` - feature varialble PSD to target variable Kf for data set "Top-Silt" (MSE)
        + `Performance_PSD_Kf_clay_r2.csv` - feature varialble PSD to target variable Kf for data set "Top-Clay" (R^2)
        + `Performance_PSD_Kf_clay_mse.csv` - feature varialble PSD to target variable Kf for data set "Top-Clay" (MSE)
        + `Performance_PSD_Kf_por_r2.csv` - feature varialble PSD to target variable porosity for data set "Top-Por" (R^2)
        + `Performance_PSD_Kf_por_mse.csv` - feature varialble PSD to target variable porosity for data set "Top-Por" (MSE)
        + `Performance_dX_Kf_topall_r2.csv` - feature varialble grain diameters (d_X) to target variable Kf for data set "Top-All" (R^2)
        + `Performance_dX_Kf_topall_mse.csv` - feature varialble grain diameters (d_X) to target variable Kf for data set "Top-All" (MSE)
        + `Performance_dX_Kf_por_r2.csv` - feature varialble grain diameters (d_X) to target variable Kf for data set "Top-Por" (R^2)
        + `Performance_dX_Kf_por_mse.csv` - feature varialble grain diameters (d_X) to target variable Kf for data set "Top-Por" (MSE)
        + `Performance_dX_por_Kf_por_r2.csv` - feature varialble grain diameters (d_X) and porosity to target variable Kf for data set "Top-Por" (R^2)
        + `Performance_dX_por_Kf_por_mse.csv` - feature varialble grain diameters (d_X) and porosity to target variable Kf for data set "Top-Por" (MSE)
        + `Performance_PSD_por_por_r2.csv`  - feature varialble PSD to target variable porosity for data set "Top-Por" (R^2)
        + `Performance_PSD_por_por_mse.csv` - feature varialble PSD to target variable porosity for data set "Top-Por" (MSE)
    - `Figures_paper/` - Figures of results as displayed in the main manuscript of accompanying publication:
        + `Fig01_Bar_NSE_PSD_Kf_topall.pdf`
        + `Fig02_Bar_NSE_PSD_Kf_soiltypes.pdf`
        + `Fig03_Scatter_Measured_topall.pdf`
        + `Fig04_Scatter_RF_Barr.pdf`
        + `Fig05_Feature_importance_RF_topall.pdf`
        + `Fig06_Scatter_Measured_dX.pdf`
        + `Fig07_Bar_NSE_features.pdf`
    - `Figures_SI/` - Figures of results as displayed in the supporting information of accompanying publication:
        + `SI_Fig_Bar_NSE_dX_Kf_por.pdf`
        + `SI_Fig_Bar_NSE_dX_Kf_topall.pdf`
        + `SI_Fig_Bar_NSE_dX_por_Kf_por.pdf`
        + `SI_Fig_Bar_NSE_PSD_Kf_clay.pdf`
        + `SI_Fig_Bar_NSE_PSD_Kf_por.pdf`
        + `SI_Fig_Bar_NSE_PSD_Kf_sand.pdf`
        + `SI_Fig_Bar_NSE_PSD_Kf_silt.pdf`
        + `SI_Fig_Bar_NSE_PSD_por_por.pdf`
        + `SI_Fig_FeatureImportance_RF_soils.pdf`   
        + `SI_Fig_FeatureImportance_topall.pdf`
        + `SI_Fig_Histogram_Kf.pdf`
        + `SI_Fig_Scatter_Kemp.pdf`
        + `SI_Fig_Scatter_Measured_clay.pdf`
        + `SI_Fig_Scatter_Measured_dX_por_Kf.pdf`
        + `SI_Fig_Scatter_Measured_PSD_por.pdf`
        + `SI_Fig_Scatter_Measured_sand.pdf`
        + `SI_Fig_Scatter_Measured_silt.pdf`
- `src/`  - contains all scripts used for data analyses and plotting of results

  + `PSD_Analysis.py` - script containing class "PSD_Analysis" for analysis of PSD (e.g. calculation of dX values, lithoclass)
  + `PSD_K_empirical.py` -  script containing class "PSD_to_K_Empirical" to calculate Kf from 15 different empirical formulas based on PSD information
  + `PSD_2K_ML.py` - script containing class "PSD_2K_ML" to perform machine learning on data set 
  + `data_dictionaries.py` - script containing dictionaries with hyperparameters for the 6 ML algorithms, all feature/target combination, all data(sub)sets

  + `00_data_processing.py` - preprocessing of raw data to transform into dataframe stored in csv file with standard format
  + `01_sample_data_statistics.py` - Script performing data analysis of PSD and derived quantities (e.g. d10, d50, d60 etc) for all sub-datasets
      results are saved to "./results/Data_analysis/"
  + `02_K_empiricial.py` - script calculating Kf from PSD information using empirical formulas implemented in class "PSD_K_empirical" for the Top-Integral data set
  + `03_ML_Hyperparam.py` - Script performing hyperparameter testing for list of algorithms and selected data set (based on soil type)
  + `03_ML_Hyperparam_GridSearch.py` - Script performing hyperparameter testing using GridSearch for a selected algorithm and data set type
  + `03_ML_Hyperparam_skopt.py` - - Script performing hyperparameter testing using SKopt for a selected algorithm and data set type:
  + `04_ML_TrainingPerformance.py` - Script evaluating performance of all six ML algorithms after training
  + `04_ML_TrainingPerformance_all.py` - Script evaluating performance of a selected ML algorithms
  
  + `F01_Bar_NSE_AllAlgorithms_TopAll.py` - reproducing Figure 1 of the manuscript
  + `F01_Bar_NSE_AllAlgorithms_single.py` - reproducing each subplot of Figure 1 of the manuscript
  + `F02_Bar_NSE_AllAlgorithms_soils.py` - reproducing Figure 2 of the manuscript
  + `F03_Scatter_vs_Measured.py` - reproducing Figure 3 of the manuscript
  + `F03_Scatter_vs_Measured_single.py` - reproducing subplots of Figure 3 of the manuscript
  + `F04_Scatter_vs_Empiricial.py` - reproducing Figure 4 of the manuscript
  + `F05_FeatureImportance.py` - reproducing Figure 5 of the manuscript
  + `F06_Scatter_vs_Measured_dX.py` - reproducing Figure 6 of the manuscript
  + `F07_Bar_NSE_AllAlgorithms_features.py` - reproducing Figure 7 of the manuscript
 
  + `SI_Bar_NSE_AllAlgorithms.py` - reproducing figures with barplots of the SI
  + `SI_Fig_FeatureImportance_RF_soils.py` - reproducing figures on feature importance of the SI
  + `SI_Fig_FeatureImportance_topall.py` - reproducing figures on feature importance of the SI
  + `SI_Histogram_Kf_ML.py` - producing figure with histograms of estimated Kf of the SI
  + `SI_Histogram_Measured_soils.py` - reproducing figure of histograms of measured Kf of the SI
  + `SI_plot_PSD.py` - reproducing figure with PSD curves of the SI
  + `SI_Scatter_Kemp.py` - reproducing figure of scatter plots on empirical formulas of the SI
  + `SI_Scatter_vs_Measured.py` - reproducing figures on scatterplots of Kf of the SI
  + `SI_Scatter_vs_Measured_por.py` - reproducing figures on scatterplots of porosity of the SI
 
## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages

## Contact

You can contact us via <a.zech@uu.nl>.


## License

MIT © 2024


