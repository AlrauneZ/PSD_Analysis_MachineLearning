#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zech0001
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
        darcy2m2 = 9.869233e-13, 
        sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2], # in mm
        )    

class PSD_to_K_Empirical(PSD_Analysis):
    
    def __init__(
          self,
          data = None,
           **settings_new,
          ):

        self.data = data

        self.settings = copy.copy(DEF_settings)
        self.settings.update(**settings_new)

        if data is not None:
            self.set_data() 
            # self.set_input_values()
        
    def set_input_values(self):        

        self.calc_psd_diameters()
        self.calc_psd_parameters()
        self.calc_parameters()

        self.por = self.psd_properties['por'].values 
        self.d10 = self.psd_properties['d10'].values 
        self.d50 = self.psd_properties['d50'].values
        self.uniformity = self.psd_properties['uniformity'].values
        
        self.K_empirical = pd.DataFrame()

    def calc_parameters(self):
        T = self.settings["T"]

        rho = (3.1e-8 * T**3 - 7.0e-6 * T**2 + 4.19e-5 * T + 0.99985) # g/cm^3 # density of water #1000.*
        mu = (-7.0e-8 * T**3 + 1.002e-5 * T**2 - 5.7e-4 * T + 0.0178) # g/cm.s # viscosity of water
        tau = 1.093e-4 * T**2 + 2.102e-2 * T + 0.5889 #

        self.rho_g_mu = rho * self.settings["g"] / mu

        self.settings.update(
            rho = rho,
            mu = mu, 
            tau = tau,
            )

    def write_to_csv(self,
                     filename,
                     add_data = False,
                     ):
        
        if add_data:
            df = pd.concat([self.data,self.K_empirical],axis = 1)
        else:
            df = self.K_empirical

        df.to_csv(filename)

    def PSD2K_fullappMethods(self,**kwargs):

        """
        Calculation of hydraulic conductivity K from PSD 
        through empirical methods being applicable to all samples: 
            "Barr","AlyamaniSen","Shepherd","vanBaaren","Kozeny"
        """
        self.K_empirical = pd.DataFrame()

        self.Barr(app=False,**kwargs)
        self.AlyamaniSen(app=False,**kwargs)
        self.Shepherd(app=False,**kwargs)
        self.VanBaaren(app=False,**kwargs)
        # self.Kozeny(app=False,**kwargs)
        
        return self.K_empirical


    def PSD2K_allMethods(self,**kwargs):

        """
        Calculation of hydraulic conductivity K from PSD 
        through empirical methods: "Hazen","Hazen_simplified","Slichter",
        "Terzaghi","Beyer","Sauerbreij","Krueger","KozenyCarman","Zunker",
        "Zamarin","USBR","Barr","AlyamaniSen","Chapuis","KrumbeinMonk",
        "Shepherd","vanBaaren"
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
        self.Kozeny(**kwargs)
        
        return self.K_empirical

    def AlyamaniSen(self,
                    phi_n_AlyamaniSen = 1.0,
                    N_AlyamaniSen= 1300.,
                    app = True,
                    **kwargs): 
        # Alyamani and Sen (1993) 0.000124


        Io = -( 10. - ( 40./ ( self.d50 - self.d10 ) ) * self.d10 ) * ( self.d50 - self.d10 ) * 0.025
        de  = ( Io + 0.025 * ( self.d50 - self.d10 ) )

        # rho*g/mu factor omitted in Devlin code
        K = N_AlyamaniSen * phi_n_AlyamaniSen * de**2 # in m/d
        CF = 100. / ( 60 * 60 * 24 ) #mm/s
        K = CF * K      
        self.K_empirical['K_AlyamaniSen'] =  K

        if app:
            self.K_empirical['app_AlyamaniSen'] =  1
        # self.K_empirical['de_AlyamaniSen'] =  de
        
        return K

    def Barr(self,
             N_Barr = 0.00402,
             # cs2 = 1.175,# for average between angular and spherical grains; 
             # cs2 = 1.35, # for angular grains; 
             #cs2 = 1.0, # for spherical grains
             app = True,
             **kwargs): 
        # Barr (2001) 0.0000116

        #  N from Devlin code:
        # Cs2 = 1 # spherical grains
        # Cs2 = 1.35 # angular grains
        # Cs2 = ( 1 + Cs2 ) / 2
        # N = 1 / ( 36 * 5 * Cs2 )

        # N_Barr = 1./(36*5*cs2)

        phi_n = self.por**3 / ( 1 - self.por )**2
        de = 0.1*self.d10
        K = self.rho_g_mu * N_Barr * phi_n * ( de )**2   
        self.K_empirical['K_Barr'] =  K

        if app:
            self.K_empirical['app_Barr'] =  1
        # self.K_empirical['de_Barr'] =  de

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

    def Kozeny(self,
               N_Ko = 5530000,
               app = True,
               **kwargs):
        ##Kozeny
        ### TODO: check units --> here is something off

        phi_n = self.por**3 / ( ( 1 - self.por )**2 )
        de = 0.1*self.d10
        #K  = self.rho_g_mu* N_Ko* de**2 *phi_n * (10000*self.settings['darcy2m2'])*(60*60*24)/100        
    
        Kmd  = 5.53*(1000.*de)**2*phi_n
        K = Kmd * self.settings['darcy2m2']*(Kmd/1000) *10000 * self.rho_g_mu #[m^2 -> cm^2 -> cm^2 * 1/(cm s) --> cm/s]

        K  = self.rho_g_mu* N_Ko* de**2 *phi_n * 10*self.settings['darcy2m2']

        # k = 5.53*(1000*d10)**2 
        # K = (k/1000)*(100*100*darcy2m2)*rho_g_mu
        # CF = (60*60*24)/100
        # K = CF*K
    
        self.K_empirical['K_Kozeny'] =  K

        if app:
            self.K_empirical['app_Kozeny'] =  1
        # self.K_empirical['de_Kozeny'] =  de

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

    def Shepherd(self,
                 phi_n = 1,
                 sand_type = 'channel',
                 app = True,
                 **kwargs):

        # Shepherd (1989) 0.00277

        if sand_type == 'channel':
            N = 142.8 # channel, default
            r = 1.65 # channel, default
        elif sand_type == 'beach':
            N = 489.6 # beach sand
            r = 1.75 # beach
        elif sand_type == 'dune':
            N = 1632 # dune sand
            r = 1.85 # dune

        # devlin code, term containing C should be 142.8 @20 grC
        # N = 142.8  / self.rho_g_mu

        de = self.d50**(0.5*r)
        # K = self.rho_g_mu * N * phi_n * de**2 # m/d
        K = N * phi_n * de**2 # m/d
        K = (100 / ( 60 * 60 * 24 ) ) * K # cm/s
        self.K_empirical['K_Shepherd'] =  K
       
        # applicability      
        if app:
            cond =  (de > 0.0063)*(de < 2)
            # cond = (6.3 < de)*(de < 2000)
            self.K_empirical['app_Shepherd'] = np.where( cond,1,0)
        # self.K_empirical['de_Shepherd'] =  de
        
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

    def VanBaaren(self,
                  m = 1.5,
                  app = True,
                  **kwargs):
        # van Baaren (in mD, grain sizes in mu)

        # correct phi for fine silt percentage to get effective porosity - or do we assume that we already have effective porosity???

        n_cor = self.psd_properties['por'] 
        # cementation factor m 1.4 for unsonsolidated, 2.0 for very hard sandstone
        
        # sorting factor C 1 for poorly sorted, 0.7 for extremely well sorted
        # C = 0.87
        C = 35.93 * ( np.log10(self.psd_properties['d60']) - np.log10(self.psd_properties['d10']) ) / ( 60. - 10. ) + 0.63
    
        #print (n_cor)
        k = 10 * (1000 * self.psd_properties['d_dom'] )**2 * C**(-3.64) * n_cor**(m + 3.64) # mD
        #print ("K mD",K)
        K = 100 * ( k / 1000 ) * self.settings['darcy2m2'] * self.rho_g_mu # in cm/s
        # van Baaren reduction factor for clay
        # f_vb = ( 1 - (1000 * d16 ) / n )**(m + 3.64)
        self.K_empirical['K_VanBaaren'] =  K

        if app:
            self.K_empirical['app_VanBaaren'] =  1
        # de = 0 #0.1*self.d10
        # self.K_empirical['de_VanBaaren'] =  de
             
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