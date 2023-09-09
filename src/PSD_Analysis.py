#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:24:31 2023

@author: zech0001
"""
import numpy as np
import pandas as pd
import copy

DEF_settings = dict(
        sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2], # in mm
        )    


# def read_data(filename,
#               filter_quality = True,
#               condense = True,
#               psd_only = False,
#               update = True,
#               # verbose = False,                  
#               ): 

#     """
#     Function to read in psd data from data file used in Roy Lin and Valerie de Rijks Thesis:
#     Condensed and modified data from TNO
    
#     """

#     if update:
#         name_quality = 'Kwaliteit_monster_upto2019'
#         name_K = 'tbl_Doorlatendheid_Procedures_M_D60/D10'
#     else:
#         name_quality = 'Kwaliteit_monster'
#         name_K = 'K (m/d 10C)'
        
    
#     data = pd.read_excel(filename)

#     if filter_quality:
#         if update:
#             filter_old = (data[name_quality] == 'ok')
#             filter_q1 = (data['Eindoordeel_from2020onwards'] == 'Niet beoordeeld')
#             filter_q2 = (data['Eindoordeel_from2020onwards'] == 'OK') \
#                         + (data['Eindoordeel_from2020onwards'] == 'OK(G)') \
#                         + (data['Eindoordeel_from2020onwards'] == 'OK(M)') \
#                         + (data['Eindoordeel_from2020onwards'] == 'OK(Z)')
#             filter_q = filter_old*filter_q1 + filter_q2
#             # print(np.sum(filter_old))
#             # print(np.sum(filter_q1))
#             # print(np.sum(filter_old*filter_q1))
#             # print(np.sum(filter_q2))
#             # print(np.sum(filter_q))
#         else:
#             filter_q = (data[name_quality] == 'ok')
#         data = data[filter_q]
#     ### filter psd - NAN values

#     if condense:
#         sieve_classes = data.columns[[x.startswith("F") for x in data.columns]]
#         data_condensed = pd.DataFrame(data, columns=sieve_classes)#.values
#         data_condensed['Kf'] = data[name_K]
#     else:
#         data_condensed = data

#     if psd_only:
#         sieve_classes = data.columns[[x.startswith("F") for x in data.columns]]
#         data_condensed = pd.DataFrame(data, columns=sieve_classes)#.values
    
#     ### filter psd - NAN values
#     data_condensed.dropna(inplace = True)
#     data_condensed.reset_index(drop=True,inplace=True)

#     return data_condensed
   

class PSD_Analysis():
    
    def __init__(
          self,
          data = None,
           **settings_new,
          ):

        self.data = data

        self.settings = copy.copy(DEF_settings)
        self.settings.update(**settings_new)
        
        # self.filter_psd_data()

    def read_data(self,
                  filename,
                  **settings,               
                  ): 

        """
        Function to read in data from condensed data file containing PSD, 
        K-values and soil_classes
        """    
        self.settings.update(**settings)

        self.data = pd.read_csv(filename)

        ### Identify PSD data and check number of sieve classes
        sieve_classes = self.data.columns[[x.startswith("F") for x in self.data.columns]]
        self.sieve_diam = np.array(self.settings['sieve_diam'])
        if len(self.sieve_diam)-1 != len(sieve_classes.values):
            print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")
        self.psd = pd.DataFrame(self.data, columns=sieve_classes)#.values

        return self.data

    def set_psd(self,
                psd,
                sieve_diam,
                ):
        
        self.data = psd
        self.psd = psd
        self.sieve_diam = sieve_diam

    def filter_psd_data(self):

        sieve_classes = self.data.columns[[x.startswith("F") for x in self.data.columns]]
        self.psd = pd.DataFrame(self.data, columns=sieve_classes)#.values
        self.sieve_diam = np.array(self.settings['sieve_diam'])
        if len(self.sieve_diam)-1 != len(sieve_classes.values):
            print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")

    def calc_psd_diameters(self,
                           diams = [5,10,16,17,20,25,50,60,75,84,90,95],
                           ):

        """ calculates percentile diameters from PSD (e.g. d10,d50,d60):
            by linear interpolation
            
            all typical diameters are calculated and saved in a data frame
            
        """
        # self.sieve_diam = np.array(self.settings['sieve_diam'])
        
        def interp(x):
            aa = 10**np.interp(np.array(diams),np.cumsum(x),np.log10(self.sieve_diam[1:]))
            return aa

        self.psd_properties = pd.DataFrame(self.psd.apply(interp, axis=1).tolist(),
                                            columns=diams).add_prefix('d')

        return self.psd_properties

    def calc_psd_folk(self):

        """ identify Folk diameters of percentage from PSD: e.g. d10,d50,d60 
            expressed as log2 values
        """
            
        for index in ['d5','d16','d25','d50','d75','d84','d95']:
            self.psd_properties[index+'f'] = -np.log2(self.psd_properties[index])

        diff_d84f_d16f = self.psd_properties['d84f'] - self.psd_properties['d16f']
        diff_d95f_d5f = self.psd_properties['d95f'] - self.psd_properties['d5f'] 
        self.psd_properties['mean_folk'] = (self.psd_properties['d16f'] + self.psd_properties['d50f'] + self.psd_properties['d84f'])/3.
        self.psd_properties['std_folk'] = - ( 0.25*diff_d84f_d16f + diff_d95f_d5f/6.6)
        self.psd_properties['skewness_folk'] = ( self.psd_properties['d16f'] + self.psd_properties['d84f'] - 2. * self.psd_properties['d50f'] ) / (2.*diff_d84f_d16f) + (self.psd_properties['d5f'] + self.psd_properties['d95f'] - 2.*self.psd_properties['d50f']) / ( 2.*diff_d95f_d5f)
        self.psd_properties['kurtosis_folk'] = diff_d95f_d5f/ (2.44*(self.psd_properties['d75f'] - self.psd_properties['d25f']) )
                    
        return self.psd_properties

    def calc_psd_parameters(self):

        """ calculate parameters from PSD diameters, such as:
                - uniformity
                - porosity
                - d_geo
                - d_dom
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
            # d_calc = [5,10,16,17,20,25,50,60,75,84,90,95],  # in % not fraction
            sieve_calc = [0.002,0.008,0.016,0.063,0.075,2], # in mm
            d_lutum = 0.008,  # in mm
            d_silt  = 0.063,  # in mm
            d_sand = 2. ,      # in mm
            classification = True,
            ):

        """
        Calculate percentage of soil_types sand, silt and lutum
            - based on standard sieve sizes (sieve_calc)
            - characteristic grain diameters for 
                lutum/clay (d_lutum)
                silt (d_silt)
                sand (d_sand)
        Calculates d50 of sand: dz50
        perform classification (classification = True) of soil classes

        """
        sieve_calc = np.array(sieve_calc)    
        ### calculate % given a sieve for clay, silt, sand 
        # psd_calc = np.interp(np.log10(sieve_calc),np.log10(self.sieve_diam),self.psd)

        def interp(x):
            aa = np.interp(np.log10(sieve_calc),np.log10(self.sieve_diam[1:]),np.cumsum(x))
            return aa

        psd_calc = pd.DataFrame(self.psd.apply(interp, axis=1).tolist(), columns=sieve_calc).add_prefix('d')
        self.psd_calc = psd_calc

        # percentages lutum, silt, sand 
        perc_lutum = psd_calc.iloc[:,np.argmin(abs(sieve_calc-d_lutum))]
        perc_silt = psd_calc.iloc[:,np.argmin(abs(sieve_calc-d_silt))] - perc_lutum
        perc_sand = psd_calc.iloc[:,np.argmin(abs(sieve_calc-d_sand))] - perc_lutum - perc_silt

        self.psd_properties['perc_lutum'] = perc_lutum  # percentage lutum/clay
        self.psd_properties['perc_silt'] = perc_silt    # percentage silt
        self.psd_properties['perc_sand'] = perc_sand    # percentage sand

        # # find median sand size
        # # rescale percentages to p(0.063) = 0 and p(2.0) = 100
        # filter_sand = (d_silt<self.sieve_diam[1:])*(self.sieve_diam[1:]<=d_sand)
        # pp_sand = self.psd.cumsum(axis=1).iloc[:,filter_sand]
        # def renormalize(x):
        #     y = (x - (perc_lutum.values + perc_silt.values))* 100/perc_sand.values
        #     return y           
        # psd_sand_renorm = pp_sand.apply(renormalize, axis=0) #.tolist(), columns=sieve_calc).add_prefix('d')
        # sieve_sand = np.compress(filter_sand,self.sieve_diam)
        # def interp(x):
        #     y = 10**np.interp(50,x,np.log10(sieve_sand))
        #     return y
        # # dz50 = np.power(10,np.interp(50,psd_sand_renorm,np.log10(sieve_sand)))
        # self.psd_properties['dz50'] = psd_sand_renorm.apply(interp, axis=1)              # median sand size
            
        if classification:
            self.calc_NEN5104_classification()
        
        return self.psd_properties


    def calc_NEN5104_classification(self):

        """
        Performing classification of soil_type according to NEN5104 classification
        based on calculated percentages of sand, silt and lutum 
        """

        # if not bool(self.psd_properties):
        #     self.calc_psd_soiltype()
        
        perc_lutum = self.psd_properties['perc_lutum'] # percentage lutum/clay
        perc_silt = self.psd_properties['perc_silt'] #percentage silt
        perc_sand = self.psd_properties['perc_sand'] # percentage sand
        # dz50 = self.psd_properties['dz50'] #median sand size

        #NEN5104 classification
        # # <8% C0.002 and >50% sand: sand
        df = pd.Series(data = np.zeros(len(self.psd.index)), dtype=str, name='soil_class')
        df.iloc[( 50 <= perc_lutum)] = "ks1"
        df.iloc[(35 <= perc_lutum)*(perc_lutum)<50] = "ks2"
        df.iloc[(25 <= perc_lutum)*(perc_lutum)<35] = "ks3"

        filter_1 = (perc_lutum < 25)*(50 <= perc_sand)
        filter_2 = (perc_lutum < 25)*(perc_sand < 50)

        df.iloc[(17.5 <= perc_lutum)*(perc_lutum < 25)*filter_1] = "kz1"
        df.iloc[(12 <= perc_lutum)*(perc_lutum < 17.5)*filter_1] = "kz2"
        df.iloc[(8 <= perc_lutum)*(perc_lutum < 12)*filter_1]    = "kz3"
        df.iloc[(5 <= perc_lutum)*(perc_lutum < 8)*(82.5 <= perc_sand)*filter_1] = "zk"
        df.iloc[(perc_lutum < 8)*(perc_sand < 67.5)*filter_1] = "zs4"
        df.iloc[(perc_lutum < 8)*(67.5<= perc_sand)*(perc_sand < 82.5)*filter_1] = "zs3"
        df.iloc[(perc_lutum < 5)*(82.5<= perc_sand)*(perc_sand < 92)*filter_1] = "zs2"
        df.iloc[(perc_lutum < 5)*(92<= perc_sand)*filter_1] = "zs1"

        df.iloc[filter_2] =  "ks4"
        filter_3 = ( 1.9367 * perc_lutum + 26.4520 <= perc_silt)
        df.iloc[filter_2*filter_3] =  "lz3"
        df.iloc[filter_2*filter_3*( perc_sand < 15 )] =  "lz1"

        # if ( 50 <= perc_lutum):
        #     soil = "ks1"
        # elif (35 <= perc_lutum < 50):
        #     soil = "ks2"
        # elif (25 <= perc_lutum < 35):
        #     soil = "ks3"
        # elif ( perc_lutum < 25 and 50 <= perc_sand):
        #     if ( 17.5 <= perc_lutum < 25):
        #         soil = "kz1"
        #     elif ( 12 <= perc_lutum < 17.5):
        #         soil = "kz2"
        #     elif ( 8 <= perc_lutum <12):
        #         soil = "kz3"
        #     elif ( 5 <= perc_lutum < 8 and 82.5 <= perc_sand ):
        #         soil = "zk"
        #     elif ( perc_lutum < 8  and perc_sand < 67.5 ):
        #         soil = "zs4"
        #     elif ( perc_lutum < 8 and 67.5 <= perc_sand < 82.5 ):
        #         soil = "zs3"
        #     elif ( perc_lutum < 5 and 82.5 <= perc_sand < 92 ):
        #         soil = "zs2"
        #     elif ( perc_lutum < 5 and 92 < perc_sand):
        #         soil = "zs1"
        # # sand < 50 and lutum < 25 and silt > 42
        # elif ( perc_lutum < 25 and perc_sand < 50 ):
        #     # Ks4, Lz1 or Lz3. Division line between Ks4 and Lz1/Lz3 extends from lutum=8, silt=42, sand=50 to lutum=25, silt=75, sand=0
        #     # line between Ks4 and Lz1/Lz3 has silt = 1.9367 * lutum + 26.4520
        #     if ( 1.9367 * perc_lutum + 26.4520 <= perc_silt):
        #         if ( perc_sand < 15 ):
        #             soil = "lz1"
        #         else:
        #             soil = "lz3"
        #     else:
        #         soil = "ks4"
    
        # if ( dz50 < .105 ):
        #     sand = "uiterst fijn"
        # if ( dz50 >= .105 and dz50 < .15 ):
        #     sand = "zeer fijn"
        # if ( dz50 >= .15 and dz50 < .21 ):
        #     sand = "matig fijn"
        # if ( dz50 >= .21 and dz50 < .3 ):
        #     sand = "matig grof"
        # if ( dz50 >= .3 and dz50 < .42 ):
        #     sand = "zeer grof"
        # if ( dz50 >= .42 and dz50 < 2 ):
        #     sand = "uiterst grof"

        self.psd_properties['soil_class'] = df.values   
        self.data['soil_class'] = df.values   
        
        # self.psd_properties['sand_median_class'] = sand.values   
        return df

    def sub_sample_soil_type(self,
                             soil_type = 'sand',
                             inplace = False,
                             filter_props = True,
                             verbose = True,
                             ):

        self.soil_type = soil_type
        if self.soil_type == 'sand':
            # soil_classes = ['Zs1', 'Zs2', 'Zs3', 'Zs4', 'Zk','Lz3']
            # soil_classes = ['Zs1', 'Zs2', 'Zs3', 'Zs4', 'Zk']
            soil_classes = ['zs1', 'zs2', 'zs3', 'zs4', 'zk']
        elif self.soil_type == 'clay':
            # soil_classes = ['Ks1', 'Ks2', 'Ks3', 'Ks4']
            soil_classes = ['ks1', 'ks2', 'ks3', 'ks4']
        elif self.soil_type == 'silt':
            # soil_classes = ['Lz1','Lz3', 'Kz1', 'Kz2', 'Kz3']
            soil_classes = ['lz1','lz2','lz3', 'kz1', 'kz2', 'kz3']
        else:
            print("WARNING: soil_type not in the list. \nSelect from: 'sand', 'clay', 'silt'.")

        # filter_soil_type = self.psd_properties.soil_class.isin(soil_classes)
        filter_soil_type = self.data.soil_class.isin(soil_classes)
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
            
        return self.data_filtered




           
