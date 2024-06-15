#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script containing class for calculating hyraulic conductivity from PSD 
information based on empirical formulas.

Todo's: update doc-string of multiple empirical formulas implemented

Author: A. Zech
"""
import numpy as np
# import scipy
import copy
import pandas as pd
from PSD_Analysis import PSD_Analysis

DEF_settings = dict( # HydroGeoSieve parameters
        T = 20, # deg C
        g = 980, # cm/s2 ## gravity constant at earth
        mD2SI = 1e-15 / 1.01325,
        darcy_to_m2 = 9.869233e-13, #darcy to m^2
        millidarcy_to_cm2 = 9.86923e-12, #millidarcy to cm^2
        m_per_day_to_cm_per_s = 100/( 60 * 60 * 24), # m/d to cm/s
        cm_per_s_to_m_per_day = 864, # cm/s to m/d   
        gpd_per_ft2_to_cm_per_s = 4.716e-5,
        sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2], # in mm
        )    

class PSD_to_K_Empirical(PSD_Analysis):
    """
        Class to calculate values of hyraulic conductivity from PSD (= partical                                             
        size distribution) information based on empirical formulas.

        Class inherits from class "PSD_Analysis" for determining specific 
        quantities (e.g. d_10, d_50 etc) from PSD values
                
        Input
        -----
            data - data frame with PSD values (non-cumulative) of samples

    """    
    def __init__(
          self,
          data = None,
          **settings_new,
          ):

        self.settings = copy.copy(DEF_settings)
        self.settings.update(**settings_new)

        if data is not None:
            self.set_data(data,**settings_new) 
            # self.set_input_values()
        
    def set_input_values(self):        
        
        """
            Function to calculate PSD derived values needed for in empirical formulas.
            Routines are inherited from class PSD_Analysis.
        
        Input
        -----
            None.

        Output 
        ------
            None.
        """
        
        self.calc_psd_diameters()
        self.calc_psd_parameters()
        self.calc_parameters()

        self.por = self.psd_properties['por'].values 
        self.d10 = self.psd_properties['d10'].values 
        self.d50 = self.psd_properties['d50'].values
        self.uniformity = self.psd_properties['uniformity'].values
        
        self.K_empirical = pd.DataFrame()

    def calc_parameters(self,
                        **settings):

        """
            Function to calculate setting parameters of the water for conversion 
            of permeability to hydraulic conductivity. Updates parameters saved
            in class dictionary settings
            
            It determines temperature dependend values of
                - water density "rho" [M/V]
                - dynamic viscosity "mu" [M/(L*T)] 
                - tau
    
            And computes the conversion factor rho*g/mu from permeability to hyd. conductivity

        Input
        -----
            None.

        Output 
        ------
            None.

        """


        self.settings.update(**settings)
        T = self.settings["T"]
        rho = (3.1e-8 * T**3 - 7.0e-6 * T**2 + 4.19e-5 * T + 0.99985) # g/cm^3 # density of water #1000.*
        mu = (-7.0e-8 * T**3 + 1.002e-5 * T**2 - 5.7e-4 * T + 0.0178) # g/cm.s # viscosity of water
        tau = 1.093e-4 * T**2 + 2.102e-2 * T + 0.5889 #

        self.rho_g_mu = rho * self.settings["g"] / mu # 1/cm.s
        # print(self.rho_g_mu)
        
        self.settings.update(
            rho = rho,
            mu = mu, 
            tau = tau,
            )

    def write_to_csv(self,
                     filename,
                     add_data = False,
                     ):
        """
            Function to write data frame with data and calculated hydraulic 
            conductivities to a csv file.
                
        Input
        -----
            filename (str) - name and path of file to write data to
            add_data (Boolean, default False) - 
                if True: writes combined PSD data and Kf values to file
                if False: writes only Kf values to file

        Output 
        ------
            None.
        """
       
        
        if add_data:
            df = pd.concat([self.data,self.K_empirical],axis = 1)
        else:
            df = self.K_empirical

        df.to_csv(filename)

    def PSD2K_fullappMethods(self,
                             app=False,
                             **kwargs):

        """
            Function to calculate hydraulic conductivity Kf from PSD 
            with empirical methods which are applicable to 
            all samples (of the "TopIntegraal data set"): 
                - "Barr"
                - "AlyamaniSen"
                - "Shepherd"
                - "vanBaaren"
                - "Bear_KozenyCarman"

        Input
        -----
            app (Boolean, default False) - 
                if True: outputs a column indicating if method is applicable to sample
            **kwargs - settings to be passed to individial methods

        Output 
        ------
            K_empirical - data frame with calculated hydraulic conductivities 
                for all samples in [cm/s], optionally in [m/d]            
        """

        self.K_empirical = pd.DataFrame()

        self.Barr(app=app,**kwargs)
        self.AlyamaniSen(app=app,**kwargs)
        self.Shepherd(app=app,**kwargs)
        self.VanBaaren(app=app,**kwargs)
        self.Bear_KozenyCarman(app=app,**kwargs)
        
        return self.K_empirical


    def PSD2K_allMethods(self,**kwargs):

        """
            Function to calculate hydraulic conductivity Kf from PSD 
            with all implemented empirical methods: 
                - "Barr"
                - "AlyamaniSen"
                - "Shepherd"
                - "vanBaaren"
                - "Bear_KozenyCarman"
                - "Hazen"
                - "Hazen_simplified"
                - "Slichter"
                - "Terzaghi"
                - "Beyer"
                - "Sauerbreij"
                - "Krueger" 
                - "KozenyCarman"
                - "Zunker",
                - "Zamarin"
                - "USBR"
                - "Chapuis"
                - "KrumbeinMonk"

        Input
        -----
            **kwargs - settings to be passed to individial methods

        Output 
        ------
            K_empirical - data frame with calculated hydraulic conductivities 
                for all samples in [cm/s], optionally in [m/d]            
        """

        self.Hazen(**kwargs)
        self.Hazen_simplified(**kwargs)
        self.Slichter(**kwargs)
        self.Terzaghi(**kwargs)
        self.Beyer(**kwargs)
        self.Sauerbreij(**kwargs)
        self.Krueger(**kwargs)
        self.KozenyCarman(**kwargs)
        self.Zunker(**kwargs)
        self.Zamarin(**kwargs)
        self.USBR(**kwargs)
        self.Barr(**kwargs)
        self.AlyamaniSen(**kwargs)
        self.Chapuis(**kwargs)
        self.KrumbeinMonk(**kwargs)
        self.Shepherd(**kwargs)
        self.VanBaaren(**kwargs)
        self.Bear_KozenyCarman(**kwargs)
        
        return self.K_empirical

    def AlyamaniSen(self,   
                    unit = 'cm_s',
                    app = True,
                    **kwargs): 

        """ 
        Function calculating Kf value of a PSD samples based on the method of
        Alyamani and Sen (1993) following the equation

        Kf = 1300 (d_eff)^2 
         
        where Kf is the hydraulic conductivity in m/d
              d_eff = is a method specific effective grain diameter in mm
                       
        Input
        -------
            app (Boolean) - key word specifying if applicability is provided
            unit (str) - specification of unit (default cm/s)
         
        Returns
        -------
            Hydraulic conductivity value in cm/s (or alternatively m/d)
            
        """

        # components of general formula
        phi_n_AlyamaniSen = 1.0
        N_AlyamaniSen= 1300.

        i0 = -0.025*( 10. -  40.*self.d10/( self.d50 - self.d10 ) ) * ( self.d50 - self.d10 ) 
        #x-intercept (grain size) of a percent grain-retention curve plotted on arithmetic axes and focussing on data below 50% retained
        de  = ( i0 + 0.025 * ( self.d50 - self.d10 ) ) # in mm
        
        K_m_d = N_AlyamaniSen * phi_n_AlyamaniSen * de**2 # in m/d 
        K_cm_s = K_m_d * DEF_settings['m_per_day_to_cm_per_s']  #in cm/s     

        if unit == 'cm_s':
            K = K_cm_s
        elif unit == 'm_d':
            K = K_m_d
        else:
            raise ValueError("Specified unit is not implemented.")

        self.K_empirical['K_AlyamaniSen'] =  K

        if app:
            self.K_empirical['app_AlyamaniSen'] =  1
        # self.K_empirical['de_AlyamaniSen'] =  de
        
        return K

    def Barr(self,
             unit = 'cm_s',
             app = True,
             cs = 1.175,
             **kwargs): 
        """ 
        Function calculating Kf value of a PSD samples based on the method of
        Barr (2001) according to Devlin, 2015 following the equation

        Kf = N_Barr (d_10)^2 (n³ / (1- n)²
         
        where Kf is the hydraulic conductivity in cm/s
              d10 = grain size of 10\% weight passing in (cm)
              n = porosity (-)
              N_Barr (int) - method specific constant

        N_Barr is a function of cs with:
              cs2 = 1 for for spherical grains
              cs2 = 1.35  for angular grains
              cs = 1.175 as average between angular and spherical grains --> default
                                      
        Input
        -------
            app (Boolean) - key word specifying if applicability is provided
            unit (str) - specification of unit (default cm/s)
            cs 
         
        Returns
        -------
            Hydraulic conductivity value in cm/s (or alternatively m/d)
            
        """

        # components of general formula
        N_Barr = 1./(36*5*cs**2)
        phi_n = self.por**3 / ( 1 - self.por )**2
        de = 0.1*self.d10 #d10 given in mm --> transform to cm

        K_cm_s = self.rho_g_mu * N_Barr * phi_n * ( de )**2   #in cm/s  
        K_m_d = K_cm_s * DEF_settings['cm_per_s_to_m_per_day']  #in m/d     

        if unit == 'cm_s':
            K = K_cm_s
        elif unit == 'm_d':
            K = K_m_d
        else:
            raise ValueError("Specified unit is not implemented.")

        self.K_empirical['K_Barr'] =  K

        if app:
            self.K_empirical['app_Barr'] =  1
        # self.K_empirical['de_Barr'] =  de

        return K

    def Bear_KozenyCarman(self,
               unit = 'cm_s',
               app = True,
               **kwargs):
        
        """ 
        Function calculating Kf value of a PSD samples based on the 
        Kozeny-Carman method according to Bear, 1972 following the equation
        
            kc = 5.53 (d_10)^2 (n³ / (1- n)²

        where kc is the permeability in (mD = milliDarcy)
              d10 = grain size of 10\% weight passing in (μm)
              n = porosity (-)

        Input
        -------
            app (Boolean) - key word specifying if applicability is provided
            unit (str) - specification of unit (default cm/s)

        Returns
        -------
            Hydraulic conductivity value in cm/s (or alternatively m/d)
            
        """

        # components of general formula
        N_BKC = 5.53 #method specific constant
        phi_n = self.por**3 / ( ( 1 - self.por )**2 )
        de = 1000*self.d10 # d10 given in mm --> transform to micrometer

        # permeability in millidarcy as given in Bear
        kmd  = N_BKC*de**2*phi_n 

        K_cm_s =kmd*self.rho_g_mu*DEF_settings['millidarcy_to_cm2'] # from milliDarcy to cm/s (perm --> cond)
        K_m_d = K_cm_s*DEF_settings['cm_per_s_to_m_per_day'] # from cm/s --> m/d

        # K_cm_s = 0.005458*(0.1*self.d10)**2  *phi_n *self.rho_g_mu

        if unit == 'cm_s':
            K = K_cm_s
        elif unit == 'm_d':
            K = K_m_d
        else:
            raise ValueError("Specified unit is not implemented.")
            
        self.K_empirical['K_Bear_KozenyCarman'] =  K
        if app:
            self.K_empirical['app_Bear_KozenyCarman'] =  1
        # self.K_empirical['de_Kozeny'] =  de

        return K

    def Shepherd(self,
                 sand_type = 'channel',
                 unit = 'cm_s',
                 app = True,
                 **kwargs):

        """ 
        Function calculating Kf value of a PSD samples based on the 
        method described by Shepherd, 1989 following the equation
            K = 3500 d_50^1.65 

        where K is the permeability/hydraulic conductivity in gpd/ft^2
              d50 = grain size of 50\% weight passing in (mm)

        transformed to SI-units:
            K = 0.165 d_50^1.65 

        where K is the hydraulic conductivity in cm/s
              d50 = grain size of 50\% weight passing in (mm)


        Input
        -------
            sand_type (str) - specification of media type, default is 'channel'
            app (Boolean) - key word specifying if applicability is provided
            unit (str) - specification of unit (default cm/s)

        Returns
        -------
            Hydraulic conductivity value in cm/s (or alternatively m/d)
            
        """

        if sand_type == 'channel':
            N = 0.16506 # channel, default
            r = 1.65 # channel, default
            # N =  3500*DEF_settings['gpd_per_ft2_to_cm_per_s'] 
            # N =  142.6*DEF_settings['m_per_d_to_cm_per_s'] 
        elif sand_type == 'beach':
            N = 0.56592 # beach sand
            r = 1.75 # beach
            # N =  12000*DEF_settings['gpd_per_ft2_to_cm_per_s'] 
            # N =  488.95488*DEF_settings['m_per_d_to_cm_per_s'] 

        elif sand_type == 'dune':
            N = 1.8864 # dune sand
            r = 1.85 # dune
            # N =  40000*DEF_settings['gpd_per_ft2_to_cm_per_s'] 
            # N = 1629.8496*DEF_settings['m_per_d_to_cm_per_s'] 

        K_cm_s = N * self.d50**r # cm/s
        K_m_d = K_cm_s*DEF_settings['cm_per_s_to_m_per_day'] # from cm/s --> m/d

        if unit == 'cm_s':
            K = K_cm_s
        elif unit == 'm_d':
            K = K_m_d
        else:
            raise ValueError("Specified unit is not implemented.")
            
        self.K_empirical['K_Shepherd'] =  K
       
        if app:
            de = self.d50**(0.5*r) # de in mm 
            # de = self.d50 # de in mm 
            cond =  (de > 0.0063)*(de < 2)
            # cond = (6.3 < de)*(de < 2000)
            self.K_empirical['app_Shepherd'] = np.where( cond,1,0)

        return K

    def VanBaaren(self,
                  m = 1.5,
                  c = 0.85,
                  unit = 'cm_s',
                  app = True,
                  **kwargs): 

        """ 
        Function calculating Kf value of a PSD samples based on the method of
        Van Baaren (1979) following the equation

             k = 10 d_dom^2 C^{-3.64}  n^{m+3.64}

        where k is the permeability in (mD = milliDarcy)
            d_dom = the dominant grain size in (μm)
            n = porosity (-)         
            c = sorting factor
            m = cementation factor (1.4 for unsonsolidated, 2.0 for very hard sandstone)
                       
        Input
        -------
            m (float) - cementation factor 1.4<m<2 (default 1.5)
            c (float) - sorting factor 0.7<c<1 (default 0.85)
            app (Boolean) - key word specifying if applicability is provided
            unit (str) - specification of unit (default cm/s)
         
        Returns
        -------
            Hydraulic conductivity value in cm/s (or alternatively m/d)
            
        """

        # components of general formula
        if c is None:
            c = 35.93 * ( np.log10(self.psd_properties['d60']) - np.log10(self.d10) ) / ( 60 - 10 ) + 0.63
        N_Baaren = c**(-3.64)*1e7*DEF_settings['millidarcy_to_cm2']  #method specific constant
        phi_n = self.por**(m + 3.64)
        de = self.psd_properties['d_dom']  # d_dom given in mm --> transform to micrometer

        K_cm_s =N_Baaren*de**2*phi_n*self.rho_g_mu # in cm/s (perm --> cond)
        K_m_d = K_cm_s*DEF_settings['cm_per_s_to_m_per_day'] # from cm/s --> m/d


        # N_Baaren_mD = 10 * c**(-3.64) #method specific constant
        # de = 1000*self.psd_properties['d_dom']  # d_dom given in mm --> transform to micrometer
        # permeability in millidarcy
        # kmd  = N_Baaren_mD*de**2*phi_n # mD
        # K_cm_s =kmd*self.rho_g_mu*DEF_settings['millidarcy_to_cm2'] # from milliDarcy to cm/s (perm --> cond)


        if unit == 'cm_s':
            K = K_cm_s
        elif unit == 'm_d':
            K = K_m_d
        else:
            raise ValueError("Specified unit is not implemented.")

        self.K_empirical['K_VanBaaren'] =  K

        if app:
            self.K_empirical['app_VanBaaren'] =  1
             
        return K


    def Beyer(self,
              phi_n_Beyer = 1,
              app = True,
              **kwargs):
        
        # Beyer (1964) 0.0000543
    
        N = 5.2e-4 * np.log10(500./self.uniformity)   
        de = 0.1*self.d10 # convert mm to cm
    
        K = self.rho_g_mu * N * phi_n_Beyer * ( de )**2
        self.K_empirical['K_Beyer'] =  K
    
        # applicability
        if app:
            cond = (de > 0.006)*(de < 0.06)*(self.uniformity > 1)*(self.uniformity < 20 )
            self.K_empirical['app_Beyer'] =  np.where(cond,1,0)
        # self.K_empirical['de_Beyer'] =  de

        return K

    def Chapuis(self,
                N_Chapuis = 1,
                app = True,
                **kwargs): 

        # Chapuis (2004) 0.00000154
        # N = mu / rho * g
        # N taken as constant in Devlin code??
        # N = 1

        void_ratio = self.por / ( 1 - self.por ) # void ratio
        phi_n = 10**(1.291 * void_ratio - 0.6435)
        a = 10 ** (0.5504 - 0.2937 * void_ratio)
        de = self.d10**(0.5*a)
        
        # rho*g/mu factor omitted in Devlin code, deviates from paper
        K = N_Chapuis* phi_n * de**2
        self.K_empirical['K_Chapuis'] =  K

        # applicability
        if app:
            cond = (self.por > 0.3)*(self.por < 0.7)*(self.d10 > 0.1)*(de < 2)*(self.uniformity > 2)*(self.uniformity < 12)*(self.d10/self.psd_properties['d5'] < 1.4)
            self.K_empirical['app_Chapuis'] =  np.where(cond,1,0)
        # self.K_empirical['de_Chapuis'] =  de

        return K


    def Hazen(self,
              N_Hazen = 6e-4,
              app = True,
              **kwargs):
        #  Hazen (1892) = d10^2 (mm)

        """
        Calculation of hydraulic conductivity K from PSD through empirical 
        method of Hazen (1892)
        """

    
        phi_n = ( 1 + 10 * ( self.por - 0.26) )
        de = 0.1*self.d10                       # convert from mm to cm
    
        K = self.rho_g_mu * N_Hazen * phi_n * de**2  
        self.K_empirical['K_Hazen'] =  K

        # applicability
        if app:        
            cond =  (de > 0.01)*(de < 0.3)*(self.uniformity < 5)
            self.K_empirical['app_Hazen'] =  np.where(cond,1,0)
        # self.K_empirical['de_Hazen'] =  de

        return K

    def Hazen_simplified(self,
                         N_Hazensimple = 100,
                         phi_n_Hazensimple = 1,
                         app = True,
                         **kwargs):
        # Hazen simplified (in Freeze & Cherry 1979)   
        
        de = 0.1*self.d10 # convert mm to cm   
        K = N_Hazensimple * phi_n_Hazensimple * de**2
    
        self.K_empirical['K_Hazen_simplified'] =  K

        # applicability 'uniformly graded sand, n = 0.375 (?), T=10 grC
        if app:
            cond = (de > 0.01)*(de < 0.3)*(self.uniformity < 5 )
            self.K_empirical['app_Hazen_simplified'] =  np.where(cond,1,0)
        # self.K_empirical['de_Hazen_simplified'] =  de

        return K

    

    def KozenyCarman(self,
                     N_KoCa = 8.3e-3,
                     app = True,
                     **kwargs):
        # Kozeny-Carman (1953) 0.0000835 (Kozeny) or 0.00161 (geometric)
        
        phi_n = self.por**3 / ( ( 1 - self.por )**2 )
    
        # # simple de from d10
        # de = 0.1*self.d10 # convert from mm to cm here (Devlin code)

    
        # alternative de
#        inv_de = np.sum(np.diff(self.psd)/ 100 * 0.5 *(self.sieve_diam[1:]+self.sieve_diam[:-1])/(self.sieve_diam[1:]*self.sieve_diam[:-1]))
        def inv(x):
            inv_val = np.sum(0.01 * x *0.5  *(self.sieve_diam[1:]+self.sieve_diam[:-1])/(self.sieve_diam[1:]*self.sieve_diam[:-1]))
            return inv_val
        inv_de = self.psd.apply(inv, axis=1).values
        # print(inv_de)

        # this additional part of first factor has to be double checked! (@Hans/Devlin code)
        if ( self.sieve_diam[1] < 0.0025 ):
            # val = 3. / 2. * ( self.psd.iloc[:,2].values ) / 100. / 0.0025
            inv_de += 3. / 2. * ( self.psd.iloc[:,1].values ) / 100. / 0.0025            
 
        de = 0.1/inv_de
        # print(de)
        
        K = self.rho_g_mu * N_KoCa * phi_n * de**2 # de convert mm to cm
        self.K_empirical['K_KozenyCarman'] =  K
   
        # applicability coarse sand
        if app:
            cond = ( self.d50 > 0.5)*(self.d50 < 2)*(self.uniformity > 2 )
            self.K_empirical['app_KozenyCarman'] =  np.where(cond,1,0)
        # self.K_empirical['de_KozenyCarman'] =  de

        return K

    def Krueger(self,
                N_Krueger = 4.35e-4,
                app = True,
                **kwargs):
        # Krüger (1919)

        ### effective radius according to Krueger
        def de_Krueger(x):
            de_val = 0.1/np.sum(.01*x*2/(self.sieve_diam[1:]+self.sieve_diam[:-1]))
            return de_val

        de = self.psd.apply(de_Krueger, axis=1).values
#        de = 0.1/np.sum(np.diff(self.psd)/ 100 * 2/(self.sieve_diam[1:]+self.sieve_diam[:-1]))
   
        phi_n = self.por / ( 1 - self.por)**2
        # de = 0.1*de_Krueger
    
        K = self.rho_g_mu * N_Krueger * phi_n * (de)**2 
        self.K_empirical['K_Krueger'] =  K
   
        # applicability medium sand
        if app:
            cond = (self.d50 > 0.25)*(self.d50 < 0.50)*(self.uniformity > 5)
            self.K_empirical['app_Krueger'] =  np.where(cond,1,0)
        # self.K_empirical['de_Krueger'] =  de

        return K

    def KrumbeinMonk(self,
                     N_KrumbeinMonk = 760,
                     app = True,
                     **kwargs):
        # Krumbein and Monk 0.00109
    
        # equation below has a minus in Devlin publication but a plus in the VB code. Cannot find it in Krumbein 1942...
        # sorting parameter see https://www.geological-digressions.com/analysis-of-sediment-grain-size-distributions/
        # this goes wrong in Devlin's code if small grains have a negative value

        self.calc_psd_folk()           
        phi_n = np.exp(- 1.31 * self.psd_properties['std_folk'] )

        # de = 2**( ( self.psd_properties['folk']['d16f'] + self.psd_properties['folk']['d50f'] + self.psd_properties['folk']['d84f'] ) / 3. )
        # Devlin recommends use of geometric mean suggested by Urumovic & Urumovic (2016)
        de = self.psd_properties['d_geo'] # convert from mm to cm not required here..
    
        K = self.rho_g_mu * N_KrumbeinMonk * phi_n * de**2
        # K = K * self.settings['darcy2m2'] * 10000 # darcy to cm^2 so K in cm/s
        K = K * 0.000000009869233 # darcy to cm^2 so K in cm/s
        self.K_empirical['K_KrumbeinMonk'] =  K

        # applicability
        if app:
            #cond = (63 < self.d50)*(self.d50 < 2000)
            cond = (0.063 < self.d50)*(self.d50 < 2)
            self.K_empirical['app_KrumbeinMonk'] =  np.where(cond,1,0)
            # self.K_empirical['app_KrumbeinMonk'] =  np.where((0.063 < self.d50)*(self.d50 < 2),1,0)
        # self.K_empirical['de_KrumbeinMonk'] =  de
     
        return K

    def Sauerbreij(self,
                   app = True,
                   **kwargs):
        # k Sauerbreij 0.000383
    
        # N in Devlin code 0.00375, in equations 3.75e-5 > factor 100 difference
        N = 3.75e-3 * self.settings['tau']
        phi_n = self.por**3 / ( ( 1 - self.por )**2 )
        de = 0.1*self.psd_properties['d17'] # convert mm to cm
        K = self.rho_g_mu * N * phi_n * ( de )**2 
        self.K_empirical['K_Sauerbreij'] =  K
    
        # applicability sand and sandy clay
        if app:
            cond = (de < 0.005 )
            # cond = (self.psd_properties['d17'] < 0.05 )
            self.K_empirical['app_Sauerbreij'] =  np.where(cond,1,0)
        # self.K_empirical['de_Sauerbreij'] =  de

        return K


    def Slichter(self,
                 N_Slichter = 1e-2,
                 app = True,
                 **kwargs):
    
        # Slichter (1892)
        
        phi_n = self.por**3.287
        de = 0.1*self.d10 # convert mm to cm    
        K = self.rho_g_mu * N_Slichter * phi_n * (de)**2
        self.K_empirical['K_Slichter'] =  K
    
        # applicability
        if app:
            cond = (de > 0.01)*(de < 0.5 )
            self.K_empirical['app_Slichterd'] =  np.where(cond,1,0)
        # self.K_empirical['de_Slichter'] =  de

        return K

    def Terzaghi(self,
                 Ngrains = 'average',
                 app = True,
                 **kwargs):
    
        # Terzaghi (1925) 0.0000161
        if Ngrains == 'average':
            N = ( 6.1e-3 + 10.7e-3 ) / 2 # average grains
        elif Ngrains == 'coarse':
            N = 6.1e-3 # coarse grains
        elif Ngrains == 'smooth':
            N = 10.7e-3 # smooth grains
    
        phi_n = ( ( self.por - 0.13 ) / ( 1 - self.por )**(1/3) ) ** 2
        de = 0.1*self.d10 # convert mm to cm
    
        K = self.rho_g_mu * N * phi_n * ( de )**2
        self.K_empirical['K_Terzaghi'] =  K
    
        # applicality sandy soil, coarse sand
        if app:
            cond = (self.d50 > 0.5)*(self.uniformity > 2)
            self.K_empirical['app_Terzaghi'] =  np.where(cond,1,0)
        # self.K_empirical['de_Terzaghi'] =  de

        return K

    def USBR(self, 
             N_USBR = 4.8e-4 *  10**0.3, 
             phi_n = 1,
             app = True,
             **kwargs):

        # USBR (Bialas 1966)
        de = ( 0.1*self.psd_properties['d20'] )**1.15    
        K = self.rho_g_mu * N_USBR * phi_n * ( de )**2
        self.K_empirical['K_USBR'] =  K
       
        # applicability
        if app:
            cond = (self.d50 > 0.25)*(self.d50 < 5)*(self.uniformity < 5 )
            self.K_empirical['app_USBR'] =  np.where(cond,1,0)
        # self.K_empirical['de_USBR'] =  de

        return K


    def Zamarin(self,
                N_Zamarin = 8.65e-3,
                app = True,
                **kwargs):

        # Zamarin (1928)
        Cn = ( 1.275 - 1.5 * self.por )**2
        phi_n = ( self.por**3 / ( 1 - self.por )**2 ) * Cn

        def inv(x):
            inv_val = np.sum(0.01 * x *np.log(self.sieve_diam[1:]/self.sieve_diam[:-1])/np.diff(self.sieve_diam))
            return inv_val

        inv_de = self.psd.apply(inv, axis=1).values

        if ( self.sieve_diam[1] < 0.0025 ):
            inv_de += 3. / 2. * ( self.psd.iloc[:,1] ) / 100 / 0.0025
        de = 0.1/inv_de

        # inv_de = np.sum(np.diff(self.psd)/ 100 * np.log(self.sieve_diam[1:]/self.sieve_diam[:-1])/np.diff(self.sieve_diam))
        # if ( self.sieve_diam[1] < 0.0025 ):
        #     inv_de += 3. / 2. * ( self.psd[1] - self.psd[2] ) / 100 / 0.0025
        # de = 0.1/inv_de

        K = self.rho_g_mu * N_Zamarin * phi_n * ( de )**2 # de convert mm to cm
        self.K_empirical['K_Zamarin'] =  K
    
        # applicability large grained sand
        if app:
            cond = ( self.d50 > 0.4)*(self.sieve_diam[0] > 0.00025 )
            self.K_empirical['app_Zamarin'] =  np.where(cond,1,0)
        # self.K_empirical['de_Zamarin'] =  de

        return K

    def Zunker(self,
               N_Zunker = 0.00155,
               app = True,
               **kwargs):

        """
        # Zunker (1930)
        N_Zunker = 0.0007 # nonuniform, angular, clayey
        N_Zunker = 0.0012 # nonuniform
        N_Zunker = 0.0014 # uniform coarse
        N_Zunker = 0.00155 # average, default
        N_Zunker = 0.0024 # uniform sand, well rounded
        """
       
        phi_n = (self.por / ( 1 - self.por ))**2

        def inv(x):
            inv_val = np.sum(0.01 * x *np.diff(self.sieve_diam)/(self.sieve_diam[1:]*self.sieve_diam[:-1]*np.log(self.sieve_diam[1:]/self.sieve_diam[:-1])))
            return inv_val

        inv_de = self.psd.apply(inv, axis=1).values

        if ( self.sieve_diam[1] < 0.0025 ):
            inv_de += 3. / 2. * ( self.psd.iloc[:,1] ) / 100 / 0.0025
        de = 0.1/inv_de

        # inv_de = np.sum(np.diff(self.psd)/ 100 * np.diff(self.sieve_diam)/(self.sieve_diam[1:]*self.sieve_diam[:-1]*np.log(self.sieve_diam[1:]/self.sieve_diam[:-1])))
        # if ( self.sieve_diam[1] < 0.0025 ):
        #     inv_de += 3. / 2. * ( self.psd[1] - self.psd[2] ) / 100 / 0.0025
        # de = 0.1/inv_de
    
        K = self.rho_g_mu * N_Zunker * phi_n * ( de )**2 # de convert mm to cm  
        self.K_empirical['K_Zunker'] =  K

        # applicability
        if app:
            cond = ( min(self.sieve_diam) > 0.0025 )
            self.K_empirical['app_Zunker'] =  np.where(cond,1,0)
        # self.K_empirical['de_Zunker'] =  de
       
        return K