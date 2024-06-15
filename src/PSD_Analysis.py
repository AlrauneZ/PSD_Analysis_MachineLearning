#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script containing class for PSD analysis
Author: A. Zech
"""

import numpy as np
import pandas as pd
import os
import copy

DEF_settings = dict(
        sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2], # in mm
        )    

class PSD_Analysis():
    
    """
    Class for PSD (=particle size distribution) analysis
    Author: A. Zech
    
    PDS data file with extended properties saved to file
    """    
    
    def __init__(
          self,
          data = None,
           **settings_new,
          ):

        self.settings = copy.copy(DEF_settings)
        self.settings.update(**settings_new)

        if data is not None:
            # self.filter_psd_data()
            self.set_data(data)
            
    def read_data(self,
                  filename,
                  **settings,               
                  ): 

        """
        Function to read in data from condensed data file containing PSD, 
        K-values and soil_classes
        
        Input
        -----
            filename (str) - path to file containing data in csv-file format with standard column names
            ** settings - setting update and kwargs-arguments to pass forward to routine "pd.read_csv" for specification of file settings
        Output
        ------
            (self.)data - pd-dataframe containing PSD+Kf data 
        """    
        self.settings.update(**settings)

        if not os.path.isfile(filename):
            raise ValueError("File specified by file name does not exist as path+file location")
            
        self.data = pd.read_csv(filename,**settings)

        ### Identify PSD data and check number of sieve classes
        self.filter_psd_data(**settings)

        return self.data

    def set_data(self,
                data,
                **settings,
                ):
        """
        Function to set data (PSD,K-values and soil_classes) as class object
               
        Input
        -----
            (self.)data - pd-dataframe containing PSD+Kf data with standard column names
            ** settings (kwargs) -  settings update
        Output
        ------
            None
        """    

        self.settings.update(**settings)       
        self.data = data
        self.filter_psd_data(**settings)

    def filter_psd_data(self,
                        sieve_classes = 'standard',
                        **settings):

        """
        function to filter data: 
                - extracting PSD filter classes 
                - setting sieve diameter as "self.sieve_diam"
       
        Input
        -----
            sieve_classes (str) - (default is "standard")
            ** settings (kwargs) -  settings update
        Output
        ------
            (self.)psd - pd-dataframe containing PSD only
       """    

        if sieve_classes == 'standard':
            sieve_classes = self.data.columns[[x.startswith("F") for x in self.data.columns]]        
        elif sieve_classes == 'all_columns':
            sieve_classes = self.data.columns #self.data.columns[1:]
        else:
            sieve_classes = sieve_classes
        
        self.settings.update(**settings)       
        self.sieve_diam = np.array(self.settings['sieve_diam'])

        if len(self.sieve_diam)-1 != len(sieve_classes.values):
            print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")
            # print(len(self.sieve_diam)-1)
            # print(len(sieve_classes.values))
        self.psd = pd.DataFrame(self.data, columns=sieve_classes)
        
        return self.psd

    def calc_psd_diameters(self,
                           diams = [5,10,16,17,20,25,50,60,75,84,90,95],
                           **settings):

        """ Function to calculate d_X (e.g. d10,d50,d60) representing the 
            maximum particle diameter of X% material (by weight) passing.  
            Determined by linear interpolation at log10-scale.
            
            
            Input
            -----
                diam (list/ndarray) - values of percentage to determine 
                grain diameters for
                    default  = [5,10,16,17,20,25,50,60,75,84,90,95], 
                    containing all typical diameters 
                ** settings -  settings update
            Output
            ------
                (self.)psd_properties - pd-dataframe containing d_X values
                             
        """

        self.settings.update(**settings)       
        self.sieve_diam = np.array(self.settings['sieve_diam'])
        
        def interp(x):
            aa = 10**np.interp(np.array(diams),np.cumsum(x),np.log10(self.sieve_diam[1:]))
            return aa

        self.psd_properties = pd.DataFrame(self.psd.apply(interp, axis=1).tolist(),
                                            columns=diams).add_prefix('d')

        return self.psd_properties

    def calc_psd_folk(self,
                      diams = ['d5','d16','d25','d50','d75','d84','d95'],
                      ):

        """ Function to calculate Folk diameters of percentage from PSD: 
            e.g. d10,d50,d60 expressed as log2 values
            
            Input
            -----
                diam (list/ndarray) - values of percentage to determine 
                grain diameters for
                    default  = [5,10,16,17,20,25,50,60,75,84,90,95], 
                    containing all typical diameters 
             Output
            ------
                (self.)psd_properties - pd-dataframe containing d_X values
                             
        """
            
        for index in diams:
            self.psd_properties[index+'f'] = -np.log2(self.psd_properties[index])

        diff_d84f_d16f = self.psd_properties['d84f'] - self.psd_properties['d16f']
        diff_d95f_d5f = self.psd_properties['d95f'] - self.psd_properties['d5f'] 
        self.psd_properties['mean_folk'] = (self.psd_properties['d16f'] + self.psd_properties['d50f'] + self.psd_properties['d84f'])/3.
        self.psd_properties['std_folk'] = - ( 0.25*diff_d84f_d16f + diff_d95f_d5f/6.6)
        self.psd_properties['skewness_folk'] = ( self.psd_properties['d16f'] + self.psd_properties['d84f'] - 2. * self.psd_properties['d50f'] ) / (2.*diff_d84f_d16f) + (self.psd_properties['d5f'] + self.psd_properties['d95f'] - 2.*self.psd_properties['d50f']) / ( 2.*diff_d95f_d5f)
        self.psd_properties['kurtosis_folk'] = diff_d95f_d5f/ (2.44*(self.psd_properties['d75f'] - self.psd_properties['d25f']) )
                    
        return self.psd_properties

    def calc_psd_parameters(self):

        """ Function to calculate additional parameters from PSD diameters:
                - uniformity
                - porosity
                - d_geo
                - d_dom
            
            Input
            -----
                None
    
             Output
            ------
                (self.)psd_properties - pd-dataframe containing additional properties
                             
        """
 
        self.psd_properties['uniformity'] = self.psd_properties['d60'] / self.psd_properties['d10']
        # porosity estimate n:
        self.psd_properties['por'] = 0.255 * ( 1 + 0.83**self.psd_properties['uniformity'] )

        # geometric mean Urumovic & Urumovic
        self.psd_properties['d_geo'] = np.exp(0.01* np.sum(self.psd.values * 0.5 * np.log(self.sieve_diam[1:] * self.sieve_diam[:- 1]),axis=1))
        # find dominant grain size
        self.psd_properties['d_dom'] = self.sieve_diam[np.argmax(self.psd.values,axis=1)]
        # self.psd_properties['d_dom'] = self.sieve_diam[np.argmax(self.psd.values,axis=1)+1]
        
        return self.psd_properties

    def calc_psd_soil_class(self,
            sieve_calc = [0.002,0.008,0.016,0.063,0.075,2], # in mm
            d_lutum = 0.008,  # in mm
            d_silt  = 0.063,  # in mm
            d_sand = 2. ,      # in mm
            classification = True,
            ):

        """
            Function to calculate percentage of soil_types sand, silt and lutum
                - based on standard sieve sizes (sieve_calc)
                - characteristic grain diameters for 
                    lutum/clay (d_lutum)
                    silt (d_silt)
                    sand (d_sand)
                    Calculates d50 of sand: dz50\
                        
            Input
            -----
                sieve_calc (list/ndarray) - values of relevant sieve diameters in [mm]
                    default = [0.002,0.008,0.016,0.063,0.075,2]
                d_lutum (float) - diameter value of upper limit for lutum in [mm]
                    default = 0.008
                d_silt (float) - diameter value of upper limit for silt in [mm]
                    default = 0.063
                d_sand (float) - diameter value of upper limit for sand in [mm]
                    default = 2.    
                classification (Boolean, default True) - perform classification of soil classes
                through class function "calc_NEN5104_classification()"
    
             Output
            ------
                (self.)psd_properties - pd-dataframe containing additional properties

        """
        sieve_calc = np.array(sieve_calc)    
        ### calculate % given a sieve for clay, silt, sand 
        # psd_calc = np.interp(np.log10(sieve_calc),np.log10(self.sieve_diam),self.psd)

        def interp(x):
            aa = np.interp(np.log10(sieve_calc),np.log10(self.sieve_diam[1:]),np.cumsum(x))
            return aa

        psd_calc = pd.DataFrame(self.psd.apply(interp, axis=1).tolist(), 
                                columns=sieve_calc).add_prefix('d')

        # percentages lutum, silt, sand 
        perc_lutum = psd_calc.iloc[:,np.argmin(abs(sieve_calc-d_lutum))]
        perc_silt = psd_calc.iloc[:,np.argmin(abs(sieve_calc-d_silt))] - perc_lutum
        perc_sand = psd_calc.iloc[:,np.argmin(abs(sieve_calc-d_sand))] - perc_lutum - perc_silt

        self.psd_properties['perc_lutum'] = perc_lutum  # percentage lutum/clay
        self.psd_properties['perc_silt'] = perc_silt    # percentage silt
        self.psd_properties['perc_sand'] = perc_sand    # percentage sand
            
        if classification:
            self.calc_NEN5104_classification()
        
        return self.psd_properties


    def calc_NEN5104_classification(self,
                                    treat_peat = False,
                                    write_ext_data = False):

        """
            Function to classify lithology class according to NEN5104 classification
            based on calculated percentages of sand, silt and lutum 
                        
            Input
            -----
                treat_peat (Boolean, default False) - if True includes peat samples
                    by marking them with a litho-class specification "p"
                write_ext_data (str/Boolean, default False) - write resulting 
                    data frame to file specified by name passed to keyword
             Output
            ------
                df - pd-dataframe (Series) containing litho-class specification for each sample
        """
        
        perc_lutum = self.psd_properties['perc_lutum'] # percentage lutum/clay
        perc_silt = self.psd_properties['perc_silt'] #percentage silt
        perc_sand = self.psd_properties['perc_sand'] # percentage sand

        #NEN5104 classification
        df = pd.Series(data = np.zeros(len(self.psd.index)), dtype=str, name='soil_class')
        df.iloc[( 50. <= perc_lutum)] = "ks1"
        df.iloc[(35. <= perc_lutum)*(perc_lutum)<50.] = "ks2"
        df.iloc[(25. < perc_lutum)*(perc_lutum)<35.] = "ks3"

        filter_1 = (perc_lutum <= 25.)*(50. <= perc_sand)
        filter_2 = (perc_lutum <= 25.)*(perc_sand < 50.)
        df.iloc[filter_2] =  "ks4"

        df.iloc[(17.5 <= perc_lutum)*(perc_lutum < 25)*filter_1] = "kz1"
        df.iloc[(12 <= perc_lutum)*(perc_lutum < 17.5)*filter_1] = "kz2"
        df.iloc[(8 <= perc_lutum)*(perc_lutum < 12)*filter_1]    = "kz3"
        df.iloc[(5 <= perc_lutum)*(perc_lutum < 8)*(82.5 <= perc_sand)*filter_1] = "zk"
        df.iloc[(perc_lutum < 8)*(perc_sand < 67.5)*filter_1] = "zs4"
        df.iloc[(perc_lutum < 8)*(67.5<= perc_sand)*(perc_sand < 82.5)*filter_1] = "zs3"
        df.iloc[(perc_lutum < 5)*(82.5<= perc_sand)*(perc_sand < 92)*filter_1] = "zs2"
        df.iloc[(perc_lutum < 5)*(92<= perc_sand)*filter_1] = "zs1"

        filter_3 = ( 1.9367 * perc_lutum + 26.4520 <= perc_silt)
        df.iloc[filter_2*filter_3] =  "lz3"
        df.iloc[filter_2*filter_3*( perc_sand < 15 )] =  "lz1"

        if treat_peat:
            filter_peat = self.data.litho_measured.isin(['V'])
            df.iloc[filter_peat] = "p"

        self.psd_properties['soil_class'] = df.values   
        self.data['soil_class'] = df.values   

        if write_ext_data:
            self.extended_data_to_csv(file_data_ext = write_ext_data)    

        return df

    def filter_litho(self,
                     treat_peat = False,
                     write_ext_data = False,
                     verbose = False,
                     ):

        """ Function to filter samples into the three litho classes (as specified
                for the data set under use) with special adaption to treating peat samples      
                        
            Input
            -----
                treat_peat (Boolean, default False) - if True includes peat samples
                    by marking them with a litho-class specification "p"
                write_ext_data (str/Boolean, default False) - write resulting 
                    data frame to file specified by name passed to keyword
                verbose (Boolean, default False) - if True outputs specifics of procedure
             Output
            ------
                df - pd-dataframe (Series) containing litho-class specification for each sample
        """

        lithoclasses_sand = ['zs1', 'zs2', 'zs3', 'zs4', 'zk']
        lithoclasses_silt = ['lz1','lz3', 'ks4']
        lithoclasses_clay = ['ks1', 'ks2', 'ks3','kz1', 'kz2', 'kz3']

        df = pd.Series(data = np.zeros(len(self.psd.index)), dtype=str, name='litho_main')

        if treat_peat:
            lithoclasses_clay = ['ks1', 'ks2', 'ks3','kz1', 'kz2', 'kz3','p']

        filter_sand = self.data.soil_class.isin(lithoclasses_sand)
        filter_silt = self.data.soil_class.isin(lithoclasses_silt)
        filter_clay = self.data.soil_class.isin(lithoclasses_clay)

        df.iloc[filter_silt] = 'L'           
        df.iloc[filter_sand] = 'Z'
        df.iloc[filter_clay] = 'K'
        
        if verbose:
            print("Number of samples categorized as sand:", np.sum(filter_sand))
            print("Number of samples categorized as silt:", np.sum(filter_silt))
            print("Number of samples categorized as clay:", np.sum(filter_clay))
            if treat_peat:
                print("Number of peat samples within category clay:", np.sum(self.data.soil_class.isin(['p'])))

        self.psd_properties['litho_main'] = df.values   
        self.data['litho_main'] = df.values   
        
        if write_ext_data:
            self.extended_data_to_csv(file_ext_data = write_ext_data)    

        return df


    def extended_data_to_csv(self,
                              file_ext_data = ".data_props.csv"
                              ):
        
        """
            Function to write data with extended information 
                (including litho-classes) to csv-file      
                        
            Input
            -----
                 file_ext_data (str) - name and path of file to write data to
                 
            Output
            ------
                None
        """
        #TODO: add here check on file path 
        self.data.to_csv(file_ext_data,index = False)   
        print("\nPDS data file with extended properties saved to file: ",file_ext_data)


    def psd_properties_to_csv(self,
                              file_psd_props = ".PSD_props.csv"
                              ):
        
        """
            Function to write psd data (including d_X values etc) 
            to csv-file      
                        
            Input
            -----
                 file_psd_props (str) - name and path of file to write data to
                 
            Output
            ------
                None
        """

        #TODO: add here check on file path 
        self.psd_properties.to_csv(file_psd_props,index = False)   
        print("\nPDS Properties saved to file: ",file_psd_props)


    def sub_sample_litho(self,
                        soil_type = 'sand',
                        inplace = False,
                        filter_props = False,
                        verbose = True,
                        ):

        """
            Function to filter data to main lithotype  
                        
            Input
            -----
                soil_type (str) - type of main lithology class filter data,
                    options:
                        - sand
                        - silt
                        - clay
                inplace (Boolean, default False)
                    reduces data in place, i.e. Class DataFrame containing samples
                    is reduced to samples of specified soil_type
                filter_props (Boolean, default False)
                    filter also the data frame containing the pds property 
                    to those samples of the specified soil type
                    (if inplace is True it also applies to the  pds properties)
                verbose (Boolean, default False) - if True outputs specifics of procedure
                
            Output
            ------
                (self.)data_filtered - pd-dataframe containing reduced data set

        """

        self.soil_type = soil_type
        if self.soil_type == 'sand':
            filter_soil_type = self.data.litho_main.isin(['Z'])
        elif self.soil_type == 'clay':
            filter_soil_type = self.data.litho_main.isin(['K'])
        elif self.soil_type == 'silt':
            filter_soil_type = self.data.litho_main.isin(['L'])
        else:
            print("WARNING: soil_type not in the list. \nSelect from: 'sand', 'clay', 'silt'.")

        self.data_filtered = self.data.loc[filter_soil_type]
        if verbose:        
            print("Input data filtered to soil type: {}".format(self.soil_type))
            print("Number of samples in sub-set: {}".format(len(self.data_filtered)))

        if filter_props:
            self.psd_properties_filtered   = self.psd_properties.loc[filter_soil_type] 
        
        if inplace:
            self.data = self.data_filtered
            if filter_props:
                self.psd_properties = self.psd_properties_filtered
            self.psd = self.psd.loc[filter_soil_type]
            #self.filter_psd_data()
        return self.data_filtered

    def sub_sample_por(self,
                       inplace = False,
                       filter_props = False,
                       verbose = True,
                       ):

        """
            Function to filter data to samples including porosity (data set Top-Por)
                        
            Input
            -----
                inplace (Boolean, default False)
                    reduces data in place, i.e. Class DataFrame containing samples
                    is reduced to samples with porosity
                filter_props (Boolean, default False)
                    filter also the data frame containing the pds property 
                    to samples with porosity value
                    (if inplace is True it also applies to the  pds properties)
                verbose (Boolean, default False) - if True outputs specifics of procedure
                
            Output
            ------
                (self.)data_filtered - pd-dataframe containing reduced data set

        """

        filter_por = self.data['porosity'].notna()
        self.data_filtered = self.data.loc[filter_por]
        if verbose:        
            print("Input data filtered to samples with measured porosity")
            print("Number of samples in sub-set: {}".format(len(self.data_filtered)))

        if filter_props:
            self.psd_properties_filtered   = self.psd_properties.loc[filter_por] 
        
        if inplace:
            self.data = self.data_filtered
            if filter_props:
                self.psd_properties = self.psd_properties_filtered
            self.psd = self.psd.loc[filter_por]
        return self.data_filtered

    def stats_data(self, 
                   PSD_stats2save = ['d10','d50','perc_lutum','perc_silt','perc_sand'],
                   other_stats2save = [], #
                   file_data_stats = False, 
                   filter_props = False,
                   verbose = True
                   ):

        """
            Function to write psd data (including d_X values etc) 
            to csv-file      
                        
            Input
            -----
                 file_psd_props (str) - name and path of file to write data to

            Function to filter data to samples including porosity (data set Top-Por)
                        
            Input
            -----
                PSD_stats2save (ist) - selection of column names (=properties) 
                    to perform statistical analysis on
                    default = ['d10','d50','perc_lutum','perc_silt','perc_sand']

                other_stats2save (list) -  additional keys to perform 
                    statistics on and save to result file

                file_data_stats (str/Boolean, default False) - write resulting 
                    statistics to file specified by name passed to keyword
                
                filter_props (Boolean, default False)
                    filter also the data frame containing the pds property 
                    to samples with porosity value
                    (if inplace is True it also applies to the  pds properties)
                
                verbose (Boolean, default True) - outputs specifics of procedure
                
            Output
            ------
                stats - pd-dataframe containing statistics of specified key words
        """


        if filter_props:
            psd_properties = self.psd_properties_filtered
            data = self.data_filtered
        else:
            psd_properties = self.psd_properties
            data = self.data

        stats_data = psd_properties[PSD_stats2save].copy()
        for key in other_stats2save:
            stats_data[key] = data[key]

        stats = stats_data.describe()

        if verbose:        
            print("Statistics of specified data set quantities:")
            print(stats)

        if file_data_stats:
            stats.to_csv(file_data_stats)
            print("Statistics saved to file:", file_data_stats)          

        return stats
          