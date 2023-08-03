import math
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pycontrails import MetDataset, MetDataArray, MetVariable
from pycontrails.physics import geo, thermo, units, constants

# Avogadro's number
N_A: float = 6.02214076e23

# This function calculates the thermal rate coeffs for deriv

# INPUTS: Temp, M, H2O
# OUTPUTS: RC(#)

# Simple rate coeffs e.g: KRO2NO  = 2.70D-12*math.exp(360/TEMP)
# Complex rate coeffs e.g: K170 = 5.00D-30*((TEMP/298)**(-1.5))*M
# List of all reactions e.g: RC(#) = 5.60D-34*O2*N2*((TEMP/300)**(-2.6))

def chemco(met, chem):
    temp = met["air_temperature"].data.isel(gm_time=1)
    pressure = met["air_pressure"].data.isel(gm_time=1)
    spec_hum = met["specific_humidity"].data.isel(gm_time=1)

    # Get air density from pycontrails physics.thermo script
    rho_d = thermo.rho_d(temp, pressure)

    # Calculate number density of air (M) to feed into box model calcs
    M = (N_A / constants.M_d) * rho_d * 1e-6 # [molecules / cm^3]

    # Calculate H2O number concentration to feed into box model calcs
    H2O = (spec_hum / constants.M_v) * N_A * rho_d * 1e-6 # [molecules / cm^3]

    # Calculate O2 and N2 number concs based on M
    O2 = 2.079E-01*M
    N2 = 7.809E-01*M

    MOM = np.ones_like(temp) # temporary as need to integrate this into main function every timestep

    # SIMPLE RATE COEFFICIENTS                                                                                     
    KRO2NO  = 2.54e-12*math.exp(360/temp) 
    KAPNO   = 8.10e-12*math.exp(270/temp) 
    KRO2NO3 = 2.50e-12 
    KRO2HO2 = 2.91e-13*math.exp(1300/temp) 
    KAPHO2  = 4.30e-13*math.exp(1040/temp) 
    KNO3AL  = 1.44e-12*math.exp(-1862/temp) 
    KDEC    = 1.0e+06
    KALKOXY = 3.70e-14*math.exp(-460/temp)*O2 
    KALKPXY = 1.80e-14*math.exp(-260/temp)*O2 
    BR01 = (0.156 + 9.77e+08*math.exp(-6415/temp)) 
    KIN = 6.2E-03*MOM
    KOUT2604 = 4.34*math.exp(-7776/(constants.R*temp))
    KOUT4608 = 4.34*math.exp(-9765/(constants.R*temp))
    KOUT2631 = 4.34*math.exp(-14500/(constants.R*temp))
    KOUT2635 = 4.34*math.exp(-12541/(constants.R*temp))
    KOUT4610 = 4.34*math.exp(-10513/(constants.R*temp))
    KOUT2605 = 4.34*math.exp(-8879/(constants.R*temp))
    KOUT2630 = 4.34*math.exp(-12639/(constants.R*temp))
    KOUT2629 = 4.34*math.exp(-4954/(constants.R*temp))
    KOUT2632 = 4.34*math.exp(-3801/(constants.R*temp))
    KOUT2637 = 4.34*math.exp(-16752/(constants.R*temp))
    KOUT3612 = 4.34*math.exp(-8362/(constants.R*temp))
    KOUT3613 = 4.34*math.exp(-11003/(constants.R*temp))
    KOUT3442 = 4.34*math.exp(-12763/(constants.R*temp))

    # COMPLEX RATE COEFFICIENTS                    
                                                                  
    # KFPAN                                                                                                    
    KC0      = 2.70e-28*M*(temp/300)**-7.1 
    KCI      = 1.21e-11*(temp/300)**-0.9    
    KRC      = KC0/KCI    
    FCC      = 0.30       
    FC       = 10**(math.log10(FCC)/(1+(math.log10(KRC))**2)) 
    KFPAN    = (KC0*KCI)*FC/(KC0+KCI) 
                                                           
    # KBPAN                                                   
    KD0      = 4.90e-03*M*math.exp(-12100/temp) 
    KDI      = 3.70e+16*math.exp(-13600/temp)  
    KRD      = KD0/KDI    
    FCD      = 0.30       
    FD       = 10**(math.log10(FCD)/(1+(math.log10(KRD))**2)) 
    KBPAN    = (KD0*KDI)*FD/(KD0+KDI) 
                                                              
    # KMT01                                                   
    K10      = 9.00e-32*M*(temp/300)**-1.5 
    K1I      = 3.00e-11*(temp/300)**0.3    
    KR1      = K10/K1I    
    FC1      = 0.6 
    F1       = 10**(math.log10(FC1)/(1+(math.log10(KR1))**2)) 
    KMT01    = (K10*K1I)*F1/(K10+K1I) 
                                                                         
    # KMT02                                                   
    K20      = 9.00e-32*((temp/300)**-2.0)*M 
    K2I      = 2.20e-11
    KR2      = K20/K2I    
    FC2      = 0.6 
    Fa2      = 10**(math.log10(FC2)/(1+(math.log10(KR2))**2)) 
    KMT02    = (K20*K2I)*Fa2/(K20+K2I) 
                                                                   
    # KMT03  : NO2 + NO3 = N2O5                               
    # IUPAC 2001                                                       
    K30      = 2.70e-30*M*(temp/300)**-3.4 
    K3I      = 2.00e-12*(temp/300)**0.2    
    KR3      = K30/K3I    
    FC3      = (math.exp(-temp/250) + math.exp(-1050/temp)) 
    F3       = 10**(math.log10(FC3)/(1+(math.log10(KR3))**2)) 
    KMT03    = (K30*K3I)*F3/(K30+K3I) 
                                                                 
    # KMT04  : N2O5 = NO2 + NO3                     
    # IUPAC 1997/2001                                                 
    K40      = (2.20e-03*M*(temp/300)**-4.34)*(math.exp(-11080/temp))
    K4I      = (9.70e+14*(temp/300)**0.1)*math.exp(-11080/temp)    
    KR4      = K40/K4I    
    FC4      = (math.exp(-temp/250) + math.exp(-1050/temp))
    Fa4      = 10**(math.log10(FC4)/(1+(math.log10(KR4))**2)) 
    KMT04    = (K40*K4I)*Fa4/(K40+K4I)       

    # KMT05                                                   
    KMT05    =  1 + ((0.6*M)/(2.687e+19*(273/temp))) 

    # KMT06                                                   
    KMT06    =  1 + (1.40e-21*math.exp(2200/temp)*H2O) 

    # KMT07 : OH + NO = HONO                              
    # IUPAC 2001                                                      
    K70      = 7.00e-31*M*(temp/300)**-2.6 
    K7I      = 3.60e-11*(temp/300)**0.1    
    KR7      = K70/K7I    
    FC7      = 0.6  
    F7       = 10**(math.log10(FC7)/(1+(math.log10(KR7))**2)) 
    KMT07    = (K70*K7I)*F7/(K70+K7I) 

    # NASA 2000                                                           

    # KMT08                                                    
    K80      = 2.50e-30*((temp/300)**-4.4)*M 
    K8I      = 1.60e-11 
    KR8      = K80/K8I 
    FC8      = 0.6 
    F8       = 10**(math.log10(FC8)/(1+(math.log10(KR8))**2)) 
    KMT08    = (K80*K8I)*F8/(K80+K8I) 

    # KMT09 : HO2 + NO2 = HO2NO2                            
    # IUPAC 1997/2001                                                 
    K90      = 1.80e-31*M*(temp/300)**-3.2 
    K9I      = 4.70e-12    
    KR9      = K90/K9I    
    FC9      = 0.6 
    F9       = 10**(math.log10(FC9)/(1+(math.log10(KR9))**2)) 
    KMT09    = (K90*K9I)*F9/(K90+K9I) 

    # KMT10 : HO2NO2 = HO2 + NO2                     
    # IUPAC 2001                                                      
    K100     = 4.10e-05*M*math.exp(-10650/temp) 
    K10I     = 5.70e+15*math.exp(-11170/temp)   
    KR10     = K100/K10I    
    FC10     = 0.5 
    F10      = 10**(math.log10(FC10)/(1+(math.log10(KR10))**2)) 
    KMT10    = (K100*K10I)*F10/(K100+K10I) 

    # KMT11 : OH + HNO3 = H2O + NO3                     
    # IUPAC 2001                                                      
    K1       = 7.20e-15*math.exp(785/temp) 
    K3       = 1.90e-33*math.exp(725/temp) 
    K4       = 4.10e-16*math.exp(1440/temp) 
    K2       = (K3*M)/(1+(K3*M/K4)) 
    KMT11    = K1 + K2 

    # KMT12 : OH + SO2 = HSO3                                  
    # IUPAC 2003                                                      
    K0       = 3.0e-31*((temp/300)**-3.3)*M 
    KI       = 1.5e-12 
    KR1      = K0/KI 
    FC       = 0.6 
    F        =10**(math.log10(FC)/(1+(math.log10(KR1))**2)) 
    KMT12    =(K0*KI*F)/(K0+KI) 

    # KMT13 : CH3O2 + NO2 = CH3O2NO2                           
    # IUPAC 2003                                                       
    K130     = 2.50e-30*((temp/300)**-5.5)*M 
    K13I     = 7.50e-12 
    KR13     = K130/K13I 
    FC13     = 0.36 
    F13      = 10**(math.log10(FC13)/(1+(math.log10(KR13))**2)) 
    KMT13    = (K130*K13I)*F13/(K130+K13I) 

    # KMT14 : CH3O2NO2 = CH3O2 + NO2                      
    # IUPAC 2001                                                       
    K140     = 9.00e-05*math.exp(-9690/temp)*M 
    K14I     = 1.10e+16*math.exp(-10560/temp) 
    KR14     = K140/K14I 
    FC14     = 0.36 
    F14      = 10**(math.log10(FC14)/(1+(math.log10(KR14))**2)) 
    KMT14    = (K140*K14I)*F14/(K140+K14I) 

    # KMT15 : OH + C2H4                                      
    # IUPAC 2001                                                      
    K150     = 6.00e-29*((temp/298)**-4.0)*M 
    K15I     = 9.00e-12*((temp/298)**-1.1) 
    KR15     = K150/K15I 
    FC15     = 0.7
    F15      = 10**(math.log10(FC15)/(1+(math.log10(KR15))**2)) 
    KMT15    = (K150*K15I)*F15/(K150+K15I) 

    # KMT16  :  OH  +  C3H6         
    # IUPAC 2003                                                     
    K160     = 3.00e-27*((temp/298)**-3.0)*M 
    K16I     = 2.80e-11*((temp/298)**-1.3) 
    KR16     = K160/K16I 
    FC16     = 0.5 
    F16      = 10**(math.log10(FC16)/(1+(math.log10(KR16))**2)) 
    KMT16    = (K160*K16I)*F16/(K160+K16I) 
                                                                  
    # KMT17                                                   
    K170     = 5.00e-30*((temp/298)**-1.5)*M 
    K17I     = 9.40e-12*math.exp(-700/temp) 
    KR17     = K170/K17I 
    FC17     = (math.exp(-temp/580) + math.exp(-2320/temp)) 
    F17      = 10**(math.log10(FC17)/(1+(math.log10(KR17))**2)) 
    KMT17    = (K170*K17I)*F17/(K170+K17I) 

    # LIST OF ALL REACTIONS
    # Reaction (1) O = O3                                                             
    chem.RC.(1) = 5.60e-34*O2*N2*((temp/300)**-2.6)

    # Reaction (2) O = O3                                                             
    RC(2) = 6.00e-34*O2*O2*((temp/300)**-2.6)

    # Reaction (3) O + O3 =                                                           
    RC(3) = 8.00e-12*math.exp(-2060/temp)         

    # Reaction (4) O + NO = NO2                                                       
    RC(4) = KMT01                            

    # Reaction (5) O + NO2 = NO                                                       
    RC(5) = 5.50e-12*math.exp(188/temp)           

    # Reaction (6) O + NO2 = NO3                                                      
    RC(6) = KMT02                            

    # Reaction (7) O1D = O                                                            
    RC(7) = 3.20e-11*O2*math.exp(67/temp)         

    # Reaction (8) O1D = O                                                            
    RC(8) = 1.80e-11*N2*math.exp(107/temp)        

    # Reaction (9) NO + O3 = NO2                                                      
    RC(9) = 1.40e-12*math.exp(-1310/temp)         

    # Reaction (10) NO2 + O3 = NO3                                                     
    RC(10) = 1.40e-13*math.exp(-2470/temp)         

    # Reaction (11) NO + NO = NO2 + NO2                                                
    RC(11) = 3.30e-39*math.exp(530/temp)*O2        

    # Reaction (12) NO + NO3 = NO2 + NO2                                               
    RC(12) = 1.80e-11*math.exp(110/temp)           

    # Reaction (13) NO2 + NO3 = NO + NO2                                               
    RC(13) = 4.50e-14*math.exp(-1260/temp)         

    # Reaction (14) NO2 + NO3 = N2O5                                                   
    RC(14) = KMT03                            

    # Reaction (15) N2O5 = NO2 + NO3                                                   
    RC(15) = KMT04                            

    # Reaction (16) O1D = OH + OH                                                      
    RC(16) = 2.20e-10                     

    # Reaction (17) OH + O3 = HO2                                                      
    RC(17) = 1.70e-12*math.exp(-940/temp)          

    # Reaction (18) OH + H2 = HO2                                                      
    RC(18) = 7.70e-12*math.exp(-2100/temp)         

    # Reaction (19) OH + CO = HO2                                                      
    RC(19) = 1.30e-13*KMT05                   

    # Reaction (20) OH + H2O2 = HO2                                                    
    RC(20) = 2.90e-12*math.exp(-160/temp)          

    # Reaction (21) HO2 + O3 = OH                                                      
    RC(21) = 2.03e-16*((temp/300)**4.57)*math.exp(693/temp)  

    # Reaction (22) OH + HO2 =                                                         
    RC(22) = 4.80e-11*math.exp(250/temp)           

    # Reaction (23) HO2 + HO2 = H2O2                                                   
    RC(23) = 2.20e-13*KMT06*math.exp(600/temp)     

    # Reaction (24) HO2 + HO2 = H2O2                                                   
    RC(24) = 1.90e-33*M*KMT06*math.exp(980/temp)   

    # Reaction (25) OH + NO = HONO                                                     
    RC(25) = KMT07                            

    # Reaction (26) NO2 = HONO                                                         
    RC(26) = 0.0                          

    # Reaction (27) OH + NO2 = HNO3                                                    
    RC(27) = KMT08                            

    # Reaction (28) OH + NO3 = HO2 + NO2                                               
    RC(28) = 2.00e-11                         

    # Reaction (29) HO2 + NO = OH + NO2                                                
    RC(29) = 3.60e-12*math.exp(270/temp)           

    # Reaction (30) HO2 + NO2 = HO2NO2                                                 
    RC(30) = KMT09                            

    # Reaction (31) HO2NO2 = HO2 + NO2                                                 
    RC(31) = KMT10                            

    # Reaction (32) OH + HO2NO2 = NO2                                                  
    RC(32) = 1.90e-12*math.exp(270/temp)           

    # Reaction (33) HO2 + NO3 = OH + NO2                                               
    RC(33) = 4.00e-12                         

    # Reaction (34) OH + HONO = NO2                                                    
    RC(34) = 2.50e-12*math.exp(-260/temp)          

    # Reaction (35) OH + HNO3 = NO3                                                    
    RC(35) = KMT11                            

    # Reaction (36) O + SO2 = SO3                                                      
    RC(36) = 4.00e-32*math.exp(-1000/temp)*M       

    # Reaction (37) OH + SO2 = HSO3                                                    
    RC(37) = KMT12                            

    # Reaction (38) HSO3 = HO2 + SO3                                                   
    RC(38) = 1.30e-12*math.exp(-330/temp)*O2       

    # Reaction (39) HNO3 = NA                                                          
    RC(39) = 0.0                         

    # Reaction (40) N2O5 = NA + NA                                                     
    RC(40) = 0.0                         

    # Reaction (41) SO3 = SA                                                           
    RC(41) = 1.20e-15*H2O                     

    # Reaction (42) OH + CH4 = CH3O2                                                   
    RC(42) = 9.65e-20*temp**2.58*math.exp(-1082/temp) 

    # Reaction (43) OH + C2H6 = C2H5O2                                                 
    RC(43) = 1.52e-17*temp**2*math.exp(-498/temp) 

    # Reaction (44) OH + C3H8 = IC3H7O2                                                
    RC(44) = 1.55e-17*temp**2*math.exp(-61/temp)*0.736  

    # Reaction (45) OH + C3H8 = RN10O2                                                 
    RC(45) = 1.55e-17*temp**2*math.exp(-61/temp)*0.264  

    # Reaction (46) OH + NC4H10 = RN13O2                                               
    RC(46) = 1.69e-17*temp**2*math.exp(145/temp)  

    # Reaction (47) OH + C2H4 = HOCH2CH2O2                                             
    RC(47) = KMT15                        

    # Reaction (48) OH + C3H6 = RN9O2                                                  
    RC(48) = KMT16                        

    # Reaction (49) OH + TBUT2ENE = RN12O2                                             
    RC(49) = 1.01e-11*math.exp(550/temp)       

    # Reaction (50) NO3 + C2H4 = NRN6O2                                                
    RC(50) = 2.10e-16                     

    # Reaction (51) NO3 + C3H6 = NRN9O2                                                
    RC(51) = 9.40e-15                     

    # Reaction (52) NO3 + TBUT2ENE = NRN12O2                                           
    RC(52) = 3.90e-13                     

    # Reaction (53) O3 + C2H4 = HCHO + CO + HO2 + OH                                   
    RC(53) = 9.14e-15*math.exp(-2580/temp)*0.13  

    # Reaction (54) O3 + C2H4 = HCHO + HCOOH                                           
    RC(54) = 9.14e-15*math.exp(-2580/temp)*0.87  

    # Reaction (55) O3 + C3H6 = HCHO + CO + CH3O2 + OH                                 
    RC(55) = 5.51e-15*math.exp(-1878/temp)*0.36  

    # Reaction (56) O3 + C3H6 = HCHO + CH3CO2H                                         
    RC(56) = 5.51e-15*math.exp(-1878/temp)*0.64  

    # Reaction (57) O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH                           
    RC(57) = 6.64e-15*math.exp(-1059/temp)*0.69 

    # Reaction (58) O3 + TBUT2ENE = CH3CHO + CH3CO2H                                   
    RC(58) = 6.64e-15*math.exp(-1059/temp)*0.31 

    # Reaction (59) OH + C5H8 = RU14O2                                                 
    RC(59) = 2.54e-11*math.exp(410/temp)       

    # Reaction (60) NO3 + C5H8 = NRU14O2                                               
    RC(60) = 3.03e-12*math.exp(-446/temp)      

    # Reaction (61) O3 + C5H8 = UCARB10 + CO + HO2 + OH                                
    RC(61) = 7.86e-15*math.exp(-1913/temp)*0.27 

    # Reaction (62) O3 + C5H8 = UCARB10 + HCOOH                                        
    RC(62) = 7.86e-15*math.exp(-1913/temp)*0.73 

    # Reaction (63) APINENE + OH = RTN28O2                                             
    RC(63) = 1.20e-11*math.exp(444/temp)           

    # Reaction (64) APINENE + NO3 = NRTN28O2                                           
    RC(64) = 1.19e-12*math.exp(490/temp)           

    # Reaction (65) APINENE + O3 = OH + RTN26O2                                        
    RC(65) = 1.01e-15*math.exp(-732/temp)*0.80  

    # Reaction (66) APINENE + O3 = TNCARB26 + H2O2                                     
    RC(66) = 1.01e-15*math.exp(-732/temp)*0.075  

    # Reaction (67) APINENE + O3 = RCOOH25                                             
    RC(67) = 1.01e-15*math.exp(-732/temp)*0.125  

    # Reaction (68) BPINENE + OH = RTX28O2                                             
    RC(68) = 2.38e-11*math.exp(357/temp) 

    # Reaction (69) BPINENE + NO3 = NRTX28O2                                           
    RC(69) = 2.51e-12 

    # Reaction (70) BPINENE + O3 =  RTX24O2 + OH                                       
    RC(70) = 1.50e-17*0.35 

    # Reaction (71) BPINENE + O3 =  HCHO + TXCARB24 + H2O2                             
    RC(71) = 1.50e-17*0.20 

    # Reaction (72) BPINENE + O3 =  HCHO + TXCARB22                                    
    RC(72) = 1.50e-17*0.25 

    # Reaction (73) BPINENE + O3 =  TXCARB24 + CO                                      
    RC(73) = 1.50e-17*0.20 

    # Reaction (74) C2H2 + OH = HCOOH + CO + HO2                                       
    RC(74) = KMT17*0.364 

    # Reaction (75) C2H2 + OH = CARB3 + OH                                             
    RC(75) = KMT17*0.636 

    # Reaction (76) BENZENE + OH = RA13O2                                              
    RC(76) = 2.33e-12*math.exp(-193/temp)*0.47 

    # Reaction (77) BENZENE + OH = AROH14 + HO2                                        
    RC(77) = 2.33e-12*math.exp(-193/temp)*0.53 

    # Reaction (78) TOLUENE + OH = RA16O2                                              
    RC(78) = 1.81e-12*math.exp(338/temp)*0.82 

    # Reaction (79) TOLUENE + OH = AROH17 + HO2                                        
    RC(79) = 1.81e-12*math.exp(338/temp)*0.18 

    # Reaction (80) OXYL + OH = RA19AO2                                                
    RC(80) = 1.36e-11*0.70 

    # Reaction (81) OXYL + OH = RA19CO2                                                
    RC(81) = 1.36e-11*0.30 

    # Reaction (82) OH + HCHO = HO2 + CO                                               
    RC(82) = 1.20e-14*temp*math.exp(287/temp)  

    # Reaction (83) OH + CH3CHO = CH3CO3                                               
    RC(83) = 5.55e-12*math.exp(311/temp)             

    # Reaction (84) OH + C2H5CHO = C2H5CO3                                             
    RC(84) = 1.96e-11                                

    # Reaction (85) NO3 + HCHO = HO2 + CO + HNO3                                       
    RC(85) = 5.80e-16                  

    # Reaction (86) NO3 + CH3CHO = CH3CO3 + HNO3                                       
    RC(86) = KNO3AL                   

    # Reaction (87) NO3 + C2H5CHO = C2H5CO3 + HNO3                                     
    RC(87) = KNO3AL*2.4             

    # Reaction (88) OH + CH3COCH3 = RN8O2                                              
    RC(88) = 5.34e-18*temp**2*math.exp(-230/temp) 

    # Reaction (89) MEK + OH = RN11O2                                                  
    RC(89) = 3.24e-18*temp**2*math.exp(414/temp)

    # Reaction (90) OH + CH3OH = HO2 + HCHO                                            
    RC(90) = 6.01e-18*temp**2*math.exp(170/temp)  

    # Reaction (91) OH + C2H5OH = CH3CHO + HO2                                         
    RC(91) = 6.18e-18*temp**2*math.exp(532/temp)*0.887 

    # Reaction (92) OH + C2H5OH = HOCH2CH2O2                                           
    RC(92) = 6.18e-18*temp**2*math.exp(532/temp)*0.113 

    # Reaction (93) NPROPOL + OH = C2H5CHO + HO2                                       
    RC(93) = 5.53e-12*0.49 

    # Reaction (94) NPROPOL + OH = RN9O2                                               
    RC(94) = 5.53e-12*0.51 

    # Reaction (95) OH + IPROPOL = CH3COCH3 + HO2                                      
    RC(95) = 4.06e-18*temp**2*math.exp(788/temp)*0.86 

    # Reaction (96) OH + IPROPOL = RN9O2                                               
    RC(96) = 4.06e-18*temp**2*math.exp(788/temp)*0.14 

    # Reaction (97) HCOOH + OH = HO2                                                   
    RC(97) = 4.50e-13 

    # Reaction (98) CH3CO2H + OH = CH3O2                                               
    RC(98) = 8.00e-13 

    # Reaction (99) OH + CH3CL = CH3O2                                                 
    RC(99) = 7.33e-18*temp**2*math.exp(-809/temp)   

    # Reaction (100) OH + CH2CL2 = CH3O2                                                
    RC(100) = 6.14e-18*temp**2*math.exp(-389/temp)   

    # Reaction (101) OH + CHCL3 = CH3O2                                                 
    RC(101) = 1.80e-18*temp**2*math.exp(-129/temp)   

    # Reaction (102) OH + CH3CCL3 = C2H5O2                                              
    RC(102) = 2.25e-18*temp**2*math.exp(-910/temp)   

    # Reaction (103) OH + TCE = HOCH2CH2O2                                              
    RC(103) = 9.64e-12*math.exp(-1209/temp)         

    # Reaction (104) OH + TRICLETH = HOCH2CH2O2                                         
    RC(104) = 5.63e-13*math.exp(427/temp)            

    # Reaction (105) OH + CDICLETH = HOCH2CH2O2                                         
    RC(105) = 1.94e-12*math.exp(90/temp)            

    # Reaction (106) OH + TDICLETH = HOCH2CH2O2                                         
    RC(106) = 1.01e-12*math.exp(250/temp)           

    # Reaction (107) CH3O2 + NO = HCHO + HO2 + NO2                                      
    RC(107) = 3.00e-12*math.exp(280/temp)*0.999 

    # Reaction (108) C2H5O2 + NO = CH3CHO + HO2 + NO2                                   
    RC(108) = 2.60e-12*math.exp(365/temp)*0.991 

    # Reaction (109) RN10O2 + NO = C2H5CHO + HO2 + NO2                                  
    RC(109) = 2.80e-12*math.exp(360/temp)*0.980 

    # Reaction (110) IC3H7O2 + NO = CH3COCH3 + HO2 + NO2                                
    RC(110) = 2.70e-12*math.exp(360/temp)*0.958 

    # Reaction (111) RN13O2 + NO = CH3CHO + C2H5O2 + NO2                                
    RC(111) = KRO2NO*0.917*BR01       

    # Reaction (112) RN13O2 + NO = CARB11A + HO2 + NO2                                  
    RC(112) = KRO2NO*0.917*(1-BR01)   

    # Reaction (113) RN16O2 + NO = RN15AO2 + NO2                                        
    RC(113) = KRO2NO*0.877                 

    # Reaction (114) RN19O2 + NO = RN18AO2 + NO2                                        
    RC(114) = KRO2NO*0.788                 

    # Reaction (115) RN13AO2 + NO = RN12O2 + NO2                                        
    RC(115) = KRO2NO                       

    # Reaction (116) RN16AO2 + NO = RN15O2 + NO2                                        
    RC(116) = KRO2NO                       

    # Reaction (117) RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2                          
    RC(117) = KRO2NO*0.918       

    # Reaction (118) RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2                         
    RC(118) = KRO2NO*0.889*0.7 

    # Reaction (119) RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2                          
    RC(119) = KRO2NO*0.889*0.3 

    # Reaction (120) RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2                        
    RC(120) = KRO2NO*0.862       

    # Reaction (121) RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2                         
    RC(121) = KRO2NO*0.862       

    # Reaction (122) HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2                          
    RC(122) = KRO2NO*0.995*0.776  

    # Reaction (123) HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2                             
    RC(123) = KRO2NO*0.995*0.224  

    # Reaction (124) RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2                             
    RC(124) = KRO2NO*0.979     

    # Reaction (125) RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2                          
    RC(125) = KRO2NO*0.959     

    # Reaction (126) RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2                         
    RC(126) = KRO2NO*0.936     

    # Reaction (127) RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2                        
    RC(127) = KRO2NO*0.903     

    # Reaction (128) RN15AO2 + NO = CARB13 + HO2 + NO2                                  
    RC(128) = KRO2NO*0.975     

    # Reaction (129) RN18AO2 + NO = CARB16 + HO2 + NO2                                  
    RC(129) = KRO2NO*0.946     

    # Reaction (130) CH3CO3 + NO = CH3O2 + NO2                                          
    RC(130) = KAPNO                      

    # Reaction (131) C2H5CO3 + NO = C2H5O2 + NO2                                        
    RC(131) = KAPNO                      

    # Reaction (132) HOCH2CO3 + NO = HO2 + HCHO + NO2                                   
    RC(132) = KAPNO                      

    # Reaction (133) RN8O2 + NO = CH3CO3 + HCHO + NO2                                   
    RC(133) = KRO2NO                     

    # Reaction (134) RN11O2 + NO = CH3CO3 + CH3CHO + NO2                                
    RC(134) = KRO2NO                     

    # Reaction (135) RN14O2 + NO = C2H5CO3 + CH3CHO + NO2                               
    RC(135) = KRO2NO                     

    # Reaction (136) RN17O2 + NO = RN16AO2 + NO2                                        
    RC(136) = KRO2NO                     

    # Reaction (137) RU14O2 + NO = UCARB12 + HO2 +  NO2                                 
    RC(137) = KRO2NO*0.900*0.252  

    # Reaction (138) RU14O2 + NO = UCARB10 + HCHO + HO2 + NO2                           
    RC(138) = KRO2NO*0.900*0.748 

    # Reaction (139) RU12O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
    RC(139) = KRO2NO*0.7         

    # Reaction (140) RU12O2 + NO = CARB7 + CO + HO2 + NO2                               
    RC(140) = KRO2NO*0.3         

    # Reaction (141) RU10O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
    RC(141) = KRO2NO*0.5         

    # Reaction (142) RU10O2 + NO = CARB6 + HCHO + HO2 + NO2                             
    RC(142) = KRO2NO*0.3         

    # Reaction (143) RU10O2 + NO = CARB7 + HCHO + HO2 + NO2                             
    RC(143) = KRO2NO*0.2          

    # Reaction (144) NRN6O2 + NO = HCHO + HCHO + NO2 + NO2                              
    RC(144) = KRO2NO                 

    # Reaction (145) NRN9O2 + NO = CH3CHO + HCHO + NO2 + NO2                            
    RC(145) = KRO2NO                 

    # Reaction (146) NRN12O2 + NO = CH3CHO + CH3CHO + NO2 + NO2                         
    RC(146) = KRO2NO                 

    # Reaction (147) NRU14O2 + NO = NUCARB12 + HO2 + NO2                                
    RC(147) = KRO2NO                 

    # Reaction (148) NRU12O2 + NO = NOA + CO + HO2 + NO2                                
    RC(148) = KRO2NO                 

    # Reaction (149) RTN28O2 + NO = TNCARB26 + HO2 + NO2                                
    RC(149) = KRO2NO*0.767*0.915  

    # Reaction (150) RTN28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
    RC(150) = KRO2NO*0.767*0.085  

    # Reaction (151) NRTN28O2 + NO = TNCARB26 + NO2 + NO2                               
    RC(151) = KRO2NO                  

    # Reaction (152) RTN26O2 + NO = RTN25O2 + NO2                                       
    RC(152) = KAPNO                   

    # Reaction (153) RTN25O2 + NO = RTN24O2 + NO2                                       
    RC(153) = KRO2NO*0.840        

    # Reaction (154) RTN24O2 + NO = RTN23O2 + NO2                                       
    RC(154) = KRO2NO                   

    # Reaction (155) RTN23O2 + NO = CH3COCH3 + RTN14O2 + NO2                            
    RC(155) = KRO2NO                  

    # Reaction (156) RTN14O2 + NO = HCHO + TNCARB10 + HO2 + NO2                         
    RC(156) = KRO2NO               

    # Reaction (157) RTN10O2 + NO = RN8O2 + CO + NO2                                    
    RC(157) = KRO2NO               

    # Reaction (158) RTX28O2 + NO = TXCARB24 + HCHO + HO2 + NO2                         
    RC(158) = KRO2NO*0.767*0.915  

    # Reaction (159) RTX28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
    RC(159) = KRO2NO*0.767*0.085  

    # Reaction (160) NRTX28O2 + NO = TXCARB24 + HCHO + NO2 + NO2                        
    RC(160) = KRO2NO            

    # Reaction (161) RTX24O2 + NO = TXCARB22 + HO2 + NO2                                
    RC(161) = KRO2NO*0.843*0.6  

    # Reaction (162) RTX24O2 + NO = CH3COCH3 + RN13AO2 + HCHO + NO2                     
    RC(162) = KRO2NO*0.843*0.4  

    # Reaction (163) RTX22O2 + NO = CH3COCH3 + RN13O2 + NO2                             
    RC(163) = KRO2NO*0.700         

    # Reaction (164) CH3O2 + NO2 = CH3O2NO2                                      
    RC(164) = KMT13         

    # Reaction (165) CH3O2NO2 = CH3O2 + NO2                                 
    RC(165) = KMT14         

    # Reaction (166) CH3O2 + NO = CH3NO3                                                
    RC(166) = 3.00e-12*math.exp(280/temp)*0.001 

    # Reaction (167) C2H5O2 + NO = C2H5NO3                                              
    RC(167) = 2.60e-12*math.exp(365/temp)*0.009 

    # Reaction (168) RN10O2 + NO = RN10NO3                                              
    RC(168) = 2.80e-12*math.exp(360/temp)*0.020 

    # Reaction (169) IC3H7O2 + NO = IC3H7NO3                                            
    RC(169) = 2.70e-12*math.exp(360/temp)*0.042 

    # Reaction (170) RN13O2 + NO = RN13NO3                                              
    RC(170) = KRO2NO*0.083                 

    # Reaction (171) RN16O2 + NO = RN16NO3                                              
    RC(171) = KRO2NO*0.123                 

    # Reaction (172) RN19O2 + NO = RN19NO3                                              
    RC(172) = KRO2NO*0.212                 

    # Reaction (173) HOCH2CH2O2 + NO = HOC2H4NO3                                        
    RC(173) = KRO2NO*0.005                 

    # Reaction (174) RN9O2 + NO = RN9NO3                                                
    RC(174) = KRO2NO*0.021                 

    # Reaction (175) RN12O2 + NO = RN12NO3                                              
    RC(175) = KRO2NO*0.041                 

    # Reaction (176) RN15O2 + NO = RN15NO3                                              
    RC(176) = KRO2NO*0.064                 

    # Reaction (177) RN18O2 + NO = RN18NO3                                              
    RC(177) = KRO2NO*0.097                 

    # Reaction (178) RN15AO2 + NO = RN15NO3                                             
    RC(178) = KRO2NO*0.025                 

    # Reaction (179) RN18AO2 + NO = RN18NO3                                             
    RC(179) = KRO2NO*0.054                 

    # Reaction (180) RU14O2 + NO = RU14NO3                                              
    RC(180) = KRO2NO*0.100                 

    # Reaction (181) RA13O2 + NO = RA13NO3                                              
    RC(181) = KRO2NO*0.082                 

    # Reaction (182) RA16O2 + NO = RA16NO3                                              
    RC(182) = KRO2NO*0.111                 

    # Reaction (183) RA19AO2 + NO = RA19NO3                                             
    RC(183) = KRO2NO*0.138                 

    # Reaction (184) RA19CO2 + NO = RA19NO3                                             
    RC(184) = KRO2NO*0.138                 

    # Reaction (185) RTN28O2 + NO = RTN28NO3                                            
    RC(185) = KRO2NO*0.233        

    # Reaction (186) RTN25O2 + NO = RTN25NO3                                            
    RC(186) = KRO2NO*0.160        

    # Reaction (187) RTX28O2 + NO = RTX28NO3                                            
    RC(187) = KRO2NO*0.233        

    # Reaction (188) RTX24O2 + NO = RTX24NO3                                            
    RC(188) = KRO2NO*0.157        

    # Reaction (189) RTX22O2 + NO = RTX22NO3                                            
    RC(189) = KRO2NO*0.300        

    # Reaction (190) CH3O2 + NO3 = HCHO + HO2 + NO2                                     
    RC(190) = KRO2NO3*0.40          

    # Reaction (191) C2H5O2 + NO3 = CH3CHO + HO2 + NO2                                  
    RC(191) = KRO2NO3               

    # Reaction (192) RN10O2 + NO3 = C2H5CHO + HO2 + NO2                                 
    RC(192) = KRO2NO3               

    # Reaction (193) IC3H7O2 + NO3 = CH3COCH3 + HO2 + NO2                               
    RC(193) = KRO2NO3               

    # Reaction (194) RN13O2 + NO3 = CH3CHO + C2H5O2 + NO2                               
    RC(194) = KRO2NO3*BR01     

    # Reaction (195) RN13O2 + NO3 = CARB11A + HO2 + NO2                                 
    RC(195) = KRO2NO3*(1-BR01) 

    # Reaction (196) RN16O2 + NO3 = RN15AO2 + NO2                                       
    RC(196) = KRO2NO3               

    # Reaction (197) RN19O2 + NO3 = RN18AO2 + NO2                                       
    RC(197) = KRO2NO3               

    # Reaction (198) RN13AO2 + NO3 = RN12O2 + NO2                                       
    RC(198) = KRO2NO3                      

    # Reaction (199) RN16AO2 + NO3 = RN15O2 + NO2                                       
    RC(199) = KRO2NO3                      

    # Reaction (200) RA13O2 + NO3 = CARB3 + UDCARB8 + HO2 + NO2                         
    RC(200) = KRO2NO3            

    # Reaction (201) RA16O2 + NO3 = CARB3 + UDCARB11 + HO2 + NO2                        
    RC(201) = KRO2NO3*0.7     

    # Reaction (202) RA16O2 + NO3 = CARB6 + UDCARB8 + HO2 + NO2                         
    RC(202) = KRO2NO3*0.3     

    # Reaction (203) RA19AO2 + NO3 = CARB3 + UDCARB14 + HO2 + NO2                       
    RC(203) = KRO2NO3           

    # Reaction (204) RA19CO2 + NO3 = CARB9 + UDCARB8 + HO2 + NO2                        
    RC(204) = KRO2NO3           

    # Reaction (205) HOCH2CH2O2 + NO3 = HCHO + HCHO + HO2 + NO2                         
    RC(205) = KRO2NO3*0.776  

    # Reaction (206) HOCH2CH2O2 + NO3 = HOCH2CHO + HO2 + NO2                            
    RC(206) = KRO2NO3*0.224  

    # Reaction (207) RN9O2 + NO3 = CH3CHO + HCHO + HO2 + NO2                            
    RC(207) = KRO2NO3         

    # Reaction (208) RN12O2 + NO3 = CH3CHO + CH3CHO + HO2 + NO2                         
    RC(208) = KRO2NO3         

    # Reaction (209) RN15O2 + NO3 = C2H5CHO + CH3CHO + HO2 + NO2                        
    RC(209) = KRO2NO3         

    # Reaction (210) RN18O2 + NO3 = C2H5CHO + C2H5CHO + HO2 + NO2                       
    RC(210) = KRO2NO3         

    # Reaction (211) RN15AO2 + NO3 = CARB13 + HO2 + NO2                                 
    RC(211) = KRO2NO3         

    # Reaction (212) RN18AO2 + NO3 = CARB16 + HO2 + NO2                                 
    RC(212) = KRO2NO3         

    # Reaction (213) CH3CO3 + NO3 = CH3O2 + NO2                                         
    RC(213) = KRO2NO3*1.60          

    # Reaction (214) C2H5CO3 + NO3 = C2H5O2 + NO2                                       
    RC(214) = KRO2NO3*1.60          

    # Reaction (215) HOCH2CO3 + NO3 = HO2 + HCHO + NO2                                  
    RC(215) = KRO2NO3*1.60         

    # Reaction (216) RN8O2 + NO3 = CH3CO3 + HCHO + NO2                                  
    RC(216) = KRO2NO3               

    # Reaction (217) RN11O2 + NO3 = CH3CO3 + CH3CHO + NO2                               
    RC(217) = KRO2NO3               

    # Reaction (218) RN14O2 + NO3 = C2H5CO3 + CH3CHO + NO2                              
    RC(218) = KRO2NO3               

    # Reaction (219) RN17O2 + NO3 = RN16AO2 + NO2                                       
    RC(219) = KRO2NO3               

    # Reaction (220) RU14O2 + NO3 = UCARB12 + HO2 + NO2                                 
    RC(220) = KRO2NO3*0.252     

    # Reaction (221) RU14O2 + NO3 = UCARB10 + HCHO + HO2 + NO2                          
    RC(221) = KRO2NO3*0.748     

    # Reaction (222) RU12O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
    RC(222) = KRO2NO3*0.7         

    # Reaction (223) RU12O2 + NO3 = CARB7 + CO + HO2 + NO2                              
    RC(223) = KRO2NO3*0.3         

    # Reaction (224) RU10O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
    RC(224) = KRO2NO3*0.5         

    # Reaction (225) RU10O2 + NO3 = CARB6 + HCHO + HO2 + NO2                            
    RC(225) = KRO2NO3*0.3         

    # Reaction (226) RU10O2 + NO3 = CARB7 + HCHO + HO2 + NO2                            
    RC(226) = KRO2NO3*0.2         

    # Reaction (227) NRN6O2 + NO3 = HCHO + HCHO + NO2 + NO2                             
    RC(227) = KRO2NO3               

    # Reaction (228) NRN9O2 + NO3 = CH3CHO + HCHO + NO2 + NO2                           
    RC(228) = KRO2NO3               

    # Reaction (229) NRN12O2 + NO3 = CH3CHO + CH3CHO + NO2 + NO2                        
    RC(229) = KRO2NO3               

    # Reaction (230) NRU14O2 + NO3 = NUCARB12 + HO2 + NO2                               
    RC(230) = KRO2NO3               

    # Reaction (231) NRU12O2 + NO3 = NOA + CO + HO2 + NO2                               
    RC(231) = KRO2NO3               

    # Reaction (232) RTN28O2 + NO3 = TNCARB26 + HO2 + NO2                               
    RC(232) = KRO2NO3                

    # Reaction (233) NRTN28O2 + NO3 = TNCARB26 + NO2 + NO2                              
    RC(233) = KRO2NO3                

    # Reaction (234) RTN26O2 + NO3 = RTN25O2 + NO2                                      
    RC(234) = KRO2NO3*1.60                   

    # Reaction (235) RTN25O2 + NO3 = RTN24O2 + NO2                                      
    RC(235) = KRO2NO3                 

    # Reaction (236) RTN24O2 + NO3 = RTN23O2 + NO2                                      
    RC(236) = KRO2NO3                   

    # Reaction (237) RTN23O2 + NO3 = CH3COCH3 + RTN14O2 + NO2                           
    RC(237) = KRO2NO3                 

    # Reaction (238) RTN14O2 + NO3 = HCHO + TNCARB10 + HO2 + NO2                        
    RC(238) = KRO2NO3             

    # Reaction (239) RTN10O2 + NO3 = RN8O2 + CO + NO2                                   
    RC(239) = KRO2NO3               

    # Reaction (240) RTX28O2 + NO3 = TXCARB24 + HCHO + HO2 + NO2                        
    RC(240) = KRO2NO3             

    # Reaction (241) RTX24O2 + NO3 = TXCARB22 + HO2 + NO2                               
    RC(241) = KRO2NO3             

    # Reaction (242) RTX22O2 + NO3 = CH3COCH3 + RN13O2 + NO2                            
    RC(242) = KRO2NO3             

    # Reaction (243) NRTX28O2 + NO3 = TXCARB24 + HCHO + NO2 + NO2                       
    RC(243) = KRO2NO3            

    # Reaction (244) CH3O2 + HO2 = CH3OOH                                               
    RC(244) = 4.10e-13*math.exp(790/temp)  

    # Reaction (245) C2H5O2 + HO2 = C2H5OOH                                             
    RC(245) = 7.50e-13*math.exp(700/temp)  

    # Reaction (246) RN10O2 + HO2 = RN10OOH                                             
    RC(246) = KRO2HO2*0.520           

    # Reaction (247) IC3H7O2 + HO2 = IC3H7OOH                                           
    RC(247) = KRO2HO2*0.520           

    # Reaction (248) RN13O2 + HO2 = RN13OOH                                             
    RC(248) = KRO2HO2*0.625           

    # Reaction (249) RN16O2 + HO2 = RN16OOH                                             
    RC(249) = KRO2HO2*0.706           

    # Reaction (250) RN19O2 + HO2 = RN19OOH                                             
    RC(250) = KRO2HO2*0.770           

    # Reaction (251) RN13AO2 + HO2 = RN13OOH                                            
    RC(251) = KRO2HO2*0.625           

    # Reaction (252) RN16AO2 + HO2 = RN16OOH                                            
    RC(252) = KRO2HO2*0.706           

    # Reaction (253) RA13O2 + HO2 = RA13OOH                                             
    RC(253) = KRO2HO2*0.770           

    # Reaction (254) RA16O2 + HO2 = RA16OOH                                             
    RC(254) = KRO2HO2*0.820           

    # Reaction (255) RA19AO2 + HO2 = RA19OOH                                            
    RC(255) = KRO2HO2*0.859           

    # Reaction (256) RA19CO2 + HO2 = RA19OOH                                            
    RC(256) = KRO2HO2*0.859           

    # Reaction (257) HOCH2CH2O2 + HO2 = HOC2H4OOH                                       
    RC(257) = 2.03e-13*math.exp(1250/temp) 

    # Reaction (258) RN9O2 + HO2 = RN9OOH                                               
    RC(258) = KRO2HO2*0.520           

    # Reaction (259) RN12O2 + HO2 = RN12OOH                                             
    RC(259) = KRO2HO2*0.625           

    # Reaction (260) RN15O2 + HO2 = RN15OOH                                             
    RC(260) = KRO2HO2*0.706           

    # Reaction (261) RN18O2 + HO2 = RN18OOH                                             
    RC(261) = KRO2HO2*0.770           

    # Reaction (262) RN15AO2 + HO2 = RN15OOH                                            
    RC(262) = KRO2HO2*0.706           

    # Reaction (263) RN18AO2 + HO2 = RN18OOH                                            
    RC(263) = KRO2HO2*0.770           

    # Reaction (264) CH3CO3 + HO2 = CH3CO3H                                             
    RC(264) = KAPHO2                  

    # Reaction (265) C2H5CO3 + HO2 = C2H5CO3H                                           
    RC(265) = KAPHO2                  

    # Reaction (266) HOCH2CO3 + HO2 = HOCH2CO3H                                         
    RC(266) = KAPHO2                  

    # Reaction (267) RN8O2 + HO2 = RN8OOH                                               
    RC(267) = KRO2HO2*0.520           

    # Reaction (268) RN11O2 + HO2 = RN11OOH                                             
    RC(268) = KRO2HO2*0.625           

    # Reaction (269) RN14O2 + HO2 = RN14OOH                                             
    RC(269) = KRO2HO2*0.706           

    # Reaction (270) RN17O2 + HO2 = RN17OOH                                             
    RC(270) = KRO2HO2*0.770           

    # Reaction (271) RU14O2 + HO2 = RU14OOH                                             
    RC(271) = KRO2HO2*0.770           

    # Reaction (272) RU12O2 + HO2 = RU12OOH                                             
    RC(272) = KRO2HO2*0.706           

    # Reaction (273) RU10O2 + HO2 = RU10OOH                                             
    RC(273) = KRO2HO2*0.625           

    # Reaction (274) NRN6O2 + HO2 = NRN6OOH                                             
    RC(274) = KRO2HO2*0.387         

    # Reaction (275) NRN9O2 + HO2 = NRN9OOH                                             
    RC(275) = KRO2HO2*0.520         

    # Reaction (276) NRN12O2 + HO2 = NRN12OOH                                           
    RC(276) = KRO2HO2*0.625         

    # Reaction (277) NRU14O2 + HO2 = NRU14OOH                                           
    RC(277) = KRO2HO2*0.770         

    # Reaction (278) NRU12O2 + HO2 = NRU12OOH                                           
    RC(278) = KRO2HO2*0.625         

    # Reaction (279) RTN28O2 + HO2 = RTN28OOH                                           
    RC(279) = KRO2HO2*0.914         

    # Reaction (280) NRTN28O2 + HO2 = NRTN28OOH                                         
    RC(280) = KRO2HO2*0.914         

    # Reaction (281) RTN26O2 + HO2 = RTN26OOH                                           
    RC(281) = KAPHO2                     

    # Reaction (282) RTN25O2 + HO2 = RTN25OOH                                           
    RC(282) = KRO2HO2*0.890       

    # Reaction (283) RTN24O2 + HO2 = RTN24OOH                                           
    RC(283) = KRO2HO2*0.890       

    # Reaction (284) RTN23O2 + HO2 = RTN23OOH                                           
    RC(284) = KRO2HO2*0.890       

    # Reaction (285) RTN14O2 + HO2 = RTN14OOH                                           
    RC(285) = KRO2HO2*0.770       

    # Reaction (286) RTN10O2 + HO2 = RTN10OOH                                           
    RC(286) = KRO2HO2*0.706       

    # Reaction (287) RTX28O2 + HO2 = RTX28OOH                                           
    RC(287) = KRO2HO2*0.914       

    # Reaction (288) RTX24O2 + HO2 = RTX24OOH                                           
    RC(288) = KRO2HO2*0.890       

    # Reaction (289) RTX22O2 + HO2 = RTX22OOH                                           
    RC(289) = KRO2HO2*0.890       

    # Reaction (290) NRTX28O2 + HO2 = NRTX28OOH                                         
    RC(290) = KRO2HO2*0.914       

    # Reaction (291) CH3O2 = HCHO + HO2                                                 
    RC(291) = 1.82e-13*math.exp(416/temp)*0.33*RO2  

    # Reaction (292) CH3O2 = HCHO                                                       
    RC(292) = 1.82e-13*math.exp(416/temp)*0.335*RO2 

    # Reaction (293) CH3O2 = CH3OH                                                      
    RC(293) = 1.82e-13*math.exp(416/temp)*0.335*RO2 

    # Reaction (294) C2H5O2 = CH3CHO + HO2                                              
    RC(294) = 3.10e-13*0.6*RO2             

    # Reaction (295) C2H5O2 = CH3CHO                                                    
    RC(295) = 3.10e-13*0.2*RO2             

    # Reaction (296) C2H5O2 = C2H5OH                                                    
    RC(296) = 3.10e-13*0.2*RO2             

    # Reaction (297) RN10O2 = C2H5CHO + HO2                                             
    RC(297) = 6.00e-13*0.6*RO2             

    # Reaction (298) RN10O2 = C2H5CHO                                                   
    RC(298) = 6.00e-13*0.2*RO2             

    # Reaction (299) RN10O2 = NPROPOL                                                   
    RC(299) = 6.00e-13*0.2*RO2             

    # Reaction (300) IC3H7O2 = CH3COCH3 + HO2                                           
    RC(300) = 4.00e-14*0.6*RO2             

    # Reaction (301) IC3H7O2 = CH3COCH3                                                 
    RC(301) = 4.00e-14*0.2*RO2             

    # Reaction (302) IC3H7O2 = IPROPOL                                                  
    RC(302) = 4.00e-14*0.2*RO2             

    # Reaction (303) RN13O2 = CH3CHO + C2H5O2                                           
    RC(303) = 2.50e-13*RO2*BR01       

    # Reaction (304) RN13O2 = CARB11A + HO2                                             
    RC(304) = 2.50e-13*RO2*(1-BR01)   

    # Reaction (305) RN13AO2 = RN12O2                                                   
    RC(305) = 8.80e-13*RO2                 

    # Reaction (306) RN16AO2 = RN15O2                                                   
    RC(306) = 8.80e-13*RO2                 

    # Reaction (307) RA13O2 = CARB3 + UDCARB8 + HO2                                     
    RC(307) = 8.80e-13*RO2                 

    # Reaction (308) RA16O2 = CARB3 + UDCARB11 + HO2                                    
    RC(308) = 8.80e-13*RO2*0.7          

    # Reaction (309) RA16O2 = CARB6 + UDCARB8 + HO2                                     
    RC(309) = 8.80e-13*RO2*0.3          

    # Reaction (310) RA19AO2 = CARB3 + UDCARB14 + HO2                                   
    RC(310) = 8.80e-13*RO2                 

    # Reaction (311) RA19CO2 = CARB3 + UDCARB14 + HO2                                   
    RC(311) = 8.80e-13*RO2                 

    # Reaction (312) RN16O2 = RN15AO2                                                   
    RC(312) = 2.50e-13*RO2                 

    # Reaction (313) RN19O2 = RN18AO2                                                   
    RC(313) = 2.50e-13*RO2                 

    # Reaction (314) HOCH2CH2O2 = HCHO + HCHO + HO2                                     
    RC(314) = 2.00e-12*RO2*0.776       

    # Reaction (315) HOCH2CH2O2 = HOCH2CHO + HO2                                        
    RC(315) = 2.00e-12*RO2*0.224       

    # Reaction (316) RN9O2 = CH3CHO + HCHO + HO2                                        
    RC(316) = 8.80e-13*RO2                 

    # Reaction (317) RN12O2 = CH3CHO + CH3CHO + HO2                                     
    RC(317) = 8.80e-13*RO2                 

    # Reaction (318) RN15O2 = C2H5CHO + CH3CHO + HO2                                    
    RC(318) = 8.80e-13*RO2                 

    # Reaction (319) RN18O2 = C2H5CHO + C2H5CHO + HO2                                   
    RC(319) = 8.80e-13*RO2                 

    # Reaction (320) RN15AO2 = CARB13 + HO2                                             
    RC(320) = 8.80e-13*RO2                 

    # Reaction (321) RN18AO2 = CARB16 + HO2                                             
    RC(321) = 8.80e-13*RO2                 

    # Reaction (322) CH3CO3 = CH3O2                                                     
    RC(322) = 1.00e-11*RO2                 

    # Reaction (323) C2H5CO3 = C2H5O2                                                   
    RC(323) = 1.00e-11*RO2                 

    # Reaction (324) HOCH2CO3 = HCHO + HO2                                              
    RC(324) = 1.00e-11*RO2                 

    # Reaction (325) RN8O2 = CH3CO3 + HCHO                                              
    RC(325) = 1.40e-12*RO2                 

    # Reaction (326) RN11O2 = CH3CO3 + CH3CHO                                           
    RC(326) = 1.40e-12*RO2                 

    # Reaction (327) RN14O2 = C2H5CO3 + CH3CHO                                          
    RC(327) = 1.40e-12*RO2                 

    # Reaction (328) RN17O2 = RN16AO2                                                   
    RC(328) = 1.40e-12*RO2                 

    # Reaction (329) RU14O2 = UCARB12 + HO2                                             
    RC(329) = 1.71e-12*RO2*0.252        

    # Reaction (330) RU14O2 = UCARB10 + HCHO + HO2                                      
    RC(330) = 1.71e-12*RO2*0.748        

    # Reaction (331) RU12O2 = CH3CO3 + HOCH2CHO                                         
    RC(331) = 2.00e-12*RO2*0.7            

    # Reaction (332) RU12O2 = CARB7 + HOCH2CHO + HO2                                    
    RC(332) = 2.00e-12*RO2*0.3            

    # Reaction (333) RU10O2 = CH3CO3 + HOCH2CHO                                         
    RC(333) = 2.00e-12*RO2*0.5            

    # Reaction (334) RU10O2 = CARB6 + HCHO + HO2                                        
    RC(334) = 2.00e-12*RO2*0.3            

    # Reaction (335) RU10O2 = CARB7 + HCHO + HO2                                        
    RC(335) = 2.00e-12*RO2*0.2            

    # Reaction (336) NRN6O2 = HCHO + HCHO + NO2                                         
    RC(336) = 6.00e-13*RO2                 

    # Reaction (337) NRN9O2 = CH3CHO + HCHO + NO2                                       
    RC(337) = 2.30e-13*RO2                 

    # Reaction (338) NRN12O2 = CH3CHO + CH3CHO + NO2                                    
    RC(338) = 2.50e-13*RO2                 

    # Reaction (339) NRU14O2 = NUCARB12 + HO2                                           
    RC(339) = 1.30e-12*RO2                 

    # Reaction (340) NRU12O2 = NOA + CO + HO2                                           
    RC(340) = 9.60e-13*RO2                 

    # Reaction (341) RTN28O2 = TNCARB26 + HO2                                           
    RC(341) = 2.85e-13*RO2                 

    # Reaction (342) NRTN28O2 = TNCARB26 + NO2                                          
    RC(342) = 1.00e-13*RO2                 

    # Reaction (343) RTN26O2 = RTN25O2                                                  
    RC(343) = 1.00e-11*RO2                   

    # Reaction (344) RTN25O2 = RTN24O2                                                  
    RC(344) = 1.30e-12*RO2           

    # Reaction (345) RTN24O2 = RTN23O2                                                  
    RC(345) = 6.70e-15*RO2             

    # Reaction (346) RTN23O2 = CH3COCH3 + RTN14O2                                       
    RC(346) = 6.70e-15*RO2            

    # Reaction (347) RTN14O2 = HCHO + TNCARB10 + HO2                                    
    RC(347) = 8.80e-13*RO2        

    # Reaction (348) RTN10O2 = RN8O2 + CO                                               
    RC(348) = 2.00e-12*RO2        

    # Reaction (349) RTX28O2 = TXCARB24 + HCHO + HO2                                    
    RC(349) = 2.00e-12*RO2       

    # Reaction (350) RTX24O2 = TXCARB22 + HO2                                           
    RC(350) = 2.50e-13*RO2       

    # Reaction (351) RTX22O2 = CH3COCH3 + RN13O2                                        
    RC(351) = 2.50e-13*RO2       

    # Reaction (352) NRTX28O2 = TXCARB24 + HCHO + NO2                                   
    RC(352) = 9.20e-14*RO2       

    # Reaction (353) OH + CARB14 = RN14O2                                               
    RC(353) = 1.87e-11       

    # Reaction (354) OH + CARB17 = RN17O2                                               
    RC(354) = 4.36e-12       

    # Reaction (355) OH + CARB11A = RN11O2                                              
    RC(355) = 3.24e-18*temp**2*math.exp(414/temp)

    # Reaction (356) OH + CARB7 = CARB6 + HO2                                           
    RC(356) = 3.00e-12       

    # Reaction (357) OH + CARB10 = CARB9 + HO2                                          
    RC(357) = 5.86e-12       

    # Reaction (358) OH + CARB13 = RN13O2                                               
    RC(358) = 1.65e-11       

    # Reaction (359) OH + CARB16 = RN16O2                                               
    RC(359) = 1.25e-11       

    # Reaction (360) OH + UCARB10 = RU10O2                                              
    RC(360) = 2.50e-11       

    # Reaction (361) NO3 + UCARB10 = RU10O2 + HNO3                                      
    RC(361) = KNO3AL       

    # Reaction (362) O3 + UCARB10 = HCHO + CH3CO3 + CO + OH                             
    RC(362) = 2.85e-18*0.59       

    # Reaction (363) O3 + UCARB10 = HCHO + CARB6 + H2O2                                 
    RC(363) = 2.85e-18*0.41       

    # Reaction (364) OH + HOCH2CHO = HOCH2CO3                                           
    RC(364) = 1.00e-11       

    # Reaction (365) NO3 + HOCH2CHO = HOCH2CO3 + HNO3                                   
    RC(365) = KNO3AL        

    # Reaction (366) OH + CARB3 = CO + CO + HO2                                         
    RC(366) = 1.14e-11       

    # Reaction (367) OH + CARB6 = CH3CO3 + CO                                           
    RC(367) = 1.72e-11       

    # Reaction (368) OH + CARB9 = RN9O2                                                 
    RC(368) = 2.40e-13       

    # Reaction (369) OH + CARB12 = RN12O2                                               
    RC(369) = 1.38e-12       

    # Reaction (370) OH + CARB15 = RN15O2                                               
    RC(370) = 4.81e-12       

    # Reaction (371) OH + CCARB12 = RN12O2                                              
    RC(371) = 4.79e-12       

    # Reaction (372) OH + UCARB12 = RU12O2                                              
    RC(372) = 4.52e-11            

    # Reaction (373) NO3 + UCARB12 = RU12O2 + HNO3                                      
    RC(373) = KNO3AL*4.25    

    # Reaction (374) O3 + UCARB12 = HOCH2CHO + CH3CO3 + CO + OH                         
    RC(374) = 2.40e-17*0.89   

    # Reaction (375) O3 + UCARB12 = HOCH2CHO + CARB6 + H2O2                             
    RC(375) = 2.40e-17*0.11   

    # Reaction (376) OH + NUCARB12 = NRU12O2                                            
    RC(376) = 4.16e-11            

    # Reaction (377) OH + NOA = CARB6 + NO2                                             
    RC(377) = 1.30e-13            

    # Reaction (378) OH + UDCARB8 = C2H5O2                                              
    RC(378) = 5.20e-11*0.50        

    # Reaction (379) OH + UDCARB8 = ANHY + HO2                                          
    RC(379) = 5.20e-11*0.50        

    # Reaction (380) OH + UDCARB11 = RN10O2                                             
    RC(380) = 5.58e-11*0.55     

    # Reaction (381) OH + UDCARB11 = ANHY + CH3O2                                       
    RC(381) = 5.58e-11*0.45     

    # Reaction (382) OH + UDCARB14 = RN13O2                                             
    RC(382) = 7.00e-11*0.55     

    # Reaction (383) OH + UDCARB14 = ANHY + C2H5O2                                      
    RC(383) = 7.00e-11*0.45     

    # Reaction (384) OH + TNCARB26 = RTN26O2                                            
    RC(384) = 4.20e-11           

    # Reaction (385) OH + TNCARB15 = RN15AO2                                            
    RC(385) = 1.00e-12           

    # Reaction (386) OH + TNCARB10 = RTN10O2                                            
    RC(386) = 1.00e-10           

    # Reaction (387) NO3 + TNCARB26 = RTN26O2 + HNO3                                    
    RC(387) = 3.80e-14            

    # Reaction (388) NO3 + TNCARB10 = RTN10O2 + HNO3                                    
    RC(388) = KNO3AL*5.5      

    # Reaction (389) OH + RCOOH25 = RTN25O2                                             
    RC(389) = 6.65e-12            

    # Reaction (390) OH + TXCARB24 = RTX24O2                                            
    RC(390) = 1.55e-11           

    # Reaction (391) OH + TXCARB22 = RTX22O2                                            
    RC(391) = 4.55e-12           

    # Reaction (392) OH + CH3NO3 = HCHO + NO2                                           
    RC(392) = 1.00e-14*math.exp(1060/temp)      

    # Reaction (393) OH + C2H5NO3 = CH3CHO + NO2                                        
    RC(393) = 4.40e-14*math.exp(720/temp)       

    # Reaction (394) OH + RN10NO3 = C2H5CHO + NO2                                       
    RC(394) = 7.30e-13                     

    # Reaction (395) OH + IC3H7NO3 = CH3COCH3 + NO2                                     
    RC(395) = 4.90e-13                     

    # Reaction (396) OH + RN13NO3 = CARB11A + NO2                                       
    RC(396) = 9.20e-13                     

    # Reaction (397) OH + RN16NO3 = CARB14 + NO2                                        
    RC(397) = 1.85e-12                     

    # Reaction (398) OH + RN19NO3 = CARB17 + NO2                                        
    RC(398) = 3.02e-12                     

    # Reaction (399) OH + HOC2H4NO3 = HOCH2CHO + NO2                                    
    RC(399) = 1.09e-12               

    # Reaction (400) OH + RN9NO3 = CARB7 + NO2                                          
    RC(400) = 1.31e-12               

    # Reaction (401) OH + RN12NO3 = CARB10 + NO2                                        
    RC(401) = 1.79e-12               

    # Reaction (402) OH + RN15NO3 = CARB13 + NO2                                        
    RC(402) = 1.03e-11               

    # Reaction (403) OH + RN18NO3 = CARB16 + NO2                                        
    RC(403) = 1.34e-11               

    # Reaction (404) OH + RU14NO3 = UCARB12 + NO2                                       
    RC(404) = 5.55e-11               

    # Reaction (405) OH + RA13NO3 = CARB3 + UDCARB8 + NO2                               
    RC(405) = 7.30e-11               

    # Reaction (406) OH + RA16NO3 = CARB3 + UDCARB11 + NO2                              
    RC(406) = 7.16e-11               

    # Reaction (407) OH + RA19NO3 = CARB6 + UDCARB11 + NO2                              
    RC(407) = 8.31e-11               

    # Reaction (408) OH + RTN28NO3 = TNCARB26 + NO2                                     
    RC(408) = 4.35e-12               

    # Reaction (409) OH + RTN25NO3 = CH3COCH3 + TNCARB15 + NO2                          
    RC(409) = 2.88e-12               

    # Reaction (410) OH + RTX28NO3 = TXCARB24 + HCHO + NO2                              
    RC(410) = 3.53e-12                  

    # Reaction (411) OH + RTX24NO3 = TXCARB22 + NO2                                     
    RC(411) = 6.48e-12                  

    # Reaction (412) OH + RTX22NO3 = CH3COCH3 + CCARB12 + NO2                           
    RC(412) = 4.74e-12                  

    # Reaction (413) OH + AROH14 = RAROH14                                              
    RC(413) = 2.63e-11             

    # Reaction (414) NO3 + AROH14 = RAROH14 + HNO3                                      
    RC(414) = 3.78e-12               

    # Reaction (415) RAROH14 + NO2 = ARNOH14                                            
    RC(415) = 2.08e-12               

    # Reaction (416) OH + ARNOH14 = CARB13 + NO2                                        
    RC(416) = 9.00e-13               

    # Reaction (417) NO3 + ARNOH14 = CARB13 + NO2 + HNO3                                
    RC(417) = 9.00e-14               

    # Reaction (418) OH + AROH17 = RAROH17                                              
    RC(418) = 4.65e-11               

    # Reaction (419) NO3 + AROH17 = RAROH17 + HNO3                                      
    RC(419) = 1.25e-11               

    # Reaction (420) RAROH17 + NO2 = ARNOH17                                            
    RC(420) = 2.08e-12               

    # Reaction (421) OH + ARNOH17 = CARB16 + NO2                                        
    RC(421) = 1.53e-12               

    # Reaction (422) NO3 + ARNOH17 = CARB16 + NO2 + HNO3                                
    RC(422) = 3.13e-13               

    # Reaction (423) OH + CH3OOH = CH3O2                                                
    RC(423) = 1.90e-11*math.exp(190/temp)       

    # Reaction (424) OH + CH3OOH = HCHO + OH                                            
    RC(424) = 1.00e-11*math.exp(190/temp)       

    # Reaction (425) OH + C2H5OOH = CH3CHO + OH                                         
    RC(425) = 1.36e-11               

    # Reaction (426) OH + RN10OOH = C2H5CHO + OH                                        
    RC(426) = 1.89e-11               

    # Reaction (427) OH + IC3H7OOH = CH3COCH3 + OH                                      
    RC(427) = 2.78e-11               

    # Reaction (428) OH + RN13OOH = CARB11A + OH                                        
    RC(428) = 3.57e-11               

    # Reaction (429) OH + RN16OOH = CARB14 + OH                                         
    RC(429) = 4.21e-11               

    # Reaction (430) OH + RN19OOH = CARB17 + OH                                         
    RC(430) = 4.71e-11               

    # Reaction (431) OH + CH3CO3H = CH3CO3                                              
    RC(431) = 3.70e-12                     

    # Reaction (432) OH + C2H5CO3H = C2H5CO3                                            
    RC(432) = 4.42e-12                     

    # Reaction (433) OH + HOCH2CO3H = HOCH2CO3                                          
    RC(433) = 6.19e-12                     

    # Reaction (434) OH + RN8OOH = CARB6 + OH                                           
    RC(434) = 4.42e-12                     

    # Reaction (435) OH + RN11OOH = CARB9 + OH                                          
    RC(435) = 2.50e-11                     

    # Reaction (436) OH + RN14OOH = CARB12 + OH                                         
    RC(436) = 3.20e-11                     

    # Reaction (437) OH + RN17OOH = CARB15 + OH                                         
    RC(437) = 3.35e-11                     

    # Reaction (438) OH + RU14OOH = UCARB12 + OH                                        
    RC(438) = 7.51e-11                     

    # Reaction (439) OH + RU12OOH = RU12O2                                              
    RC(439) = 3.00e-11                     

    # Reaction (440) OH + RU10OOH = RU10O2                                              
    RC(440) = 3.00e-11                     

    # Reaction (441) OH + NRU14OOH = NUCARB12 + OH                                      
    RC(441) = 1.03e-10                     

    # Reaction (442) OH + NRU12OOH = NOA + CO + OH                                      
    RC(442) = 2.65e-11                     

    # Reaction (443) OH + HOC2H4OOH = HOCH2CHO + OH                                     
    RC(443) = 2.13e-11               

    # Reaction (444) OH + RN9OOH = CARB7 + OH                                           
    RC(444) = 2.50e-11               

    # Reaction (445) OH + RN12OOH = CARB10 + OH                                         
    RC(445) = 3.25e-11               

    # Reaction (446) OH + RN15OOH = CARB13 + OH                                         
    RC(446) = 3.74e-11               

    # Reaction (447) OH + RN18OOH = CARB16 + OH                                         
    RC(447) = 3.83e-11               

    # Reaction (448) OH + NRN6OOH = HCHO + HCHO + NO2 + OH                              
    RC(448) = 5.22e-12               

    # Reaction (449) OH + NRN9OOH = CH3CHO + HCHO + NO2 + OH                            
    RC(449) = 6.50e-12               

    # Reaction (450) OH + NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                         
    RC(450) = 7.15e-12               

    # Reaction (451) OH + RA13OOH = CARB3 + UDCARB8 + OH                                
    RC(451) = 9.77e-11               

    # Reaction (452) OH + RA16OOH = CARB3 + UDCARB11 + OH                               
    RC(452) = 9.64e-11               

    # Reaction (453) OH + RA19OOH = CARB6 + UDCARB11 + OH                               
    RC(453) = 1.12e-10               

    # Reaction (454) OH + RTN28OOH = TNCARB26 + OH                                      
    RC(454) = 2.38e-11               

    # Reaction (455) OH + RTN26OOH = RTN26O2                                            
    RC(455) = 1.20e-11               

    # Reaction (456) OH + NRTN28OOH = TNCARB26 + NO2 + OH                               
    RC(456) = 9.50e-12               

    # Reaction (457) OH + RTN25OOH = RTN25O2                                            
    RC(457) = 1.66e-11               

    # Reaction (458) OH + RTN24OOH = RTN24O2                                            
    RC(458) = 1.05e-11               

    # Reaction (459) OH + RTN23OOH = RTN23O2                                            
    RC(459) = 2.05e-11               

    # Reaction (460) OH + RTN14OOH = RTN14O2                                            
    RC(460) = 8.69e-11               

    # Reaction (461) OH + RTN10OOH = RTN10O2                                            
    RC(461) = 4.23e-12               

    # Reaction (462) OH + RTX28OOH = RTX28O2                                            
    RC(462) = 2.00e-11               

    # Reaction (463) OH + RTX24OOH = TXCARB22 + OH                                      
    RC(463) = 8.59e-11               

    # Reaction (464) OH + RTX22OOH = CH3COCH3 + CCARB12 + OH                            
    RC(464) = 7.50e-11               

    # Reaction (465) OH + NRTX28OOH = NRTX28O2                                          
    RC(465) = 9.58e-12               

    # Reaction (466) OH + ANHY = HOCH2CH2O2                                             
    RC(466) = 1.50e-12        

    # Reaction (467) CH3CO3 + NO2 = PAN                                                 
    RC(467) = KFPAN                        

    # Reaction (468) PAN = CH3CO3 + NO2                                                 
    RC(468) = KBPAN                        

    # Reaction (469) C2H5CO3 + NO2 = PPN                                                
    RC(469) = KFPAN                        

    # Reaction (470) PPN = C2H5CO3 + NO2                                                
    RC(470) = KBPAN                        

    # Reaction (471) HOCH2CO3 + NO2 = PHAN                                              
    RC(471) = KFPAN                        

    # Reaction (472) PHAN = HOCH2CO3 + NO2                                              
    RC(472) = KBPAN                        

    # Reaction (473) OH + PAN = HCHO + CO + NO2                                         
    RC(473) = 9.50e-13*math.exp(-650/temp)      

    # Reaction (474) OH + PPN = CH3CHO + CO + NO2                                       
    RC(474) = 1.27e-12                       

    # Reaction (475) OH + PHAN = HCHO + CO + NO2                                        
    RC(475) = 1.12e-12                       

    # Reaction (476) RU12O2 + NO2 = RU12PAN                                             
    RC(476) = KFPAN*0.061             

    # Reaction (477) RU12PAN = RU12O2 + NO2                                             
    RC(477) = KBPAN                   

    # Reaction (478) RU10O2 + NO2 = MPAN                                                
    RC(478) = KFPAN*0.041             

    # Reaction (479) MPAN = RU10O2 + NO2                                                
    RC(479) = KBPAN                  

    # Reaction (480) OH + MPAN = CARB7 + CO + NO2                                       
    RC(480) = 3.60e-12 

    # Reaction (481) OH + RU12PAN = UCARB10 + NO2                                       
    RC(481) = 2.52e-11 

    # Reaction (482) RTN26O2 + NO2 = RTN26PAN                                           
    RC(482) = KFPAN*0.722      

    # Reaction (483) RTN26PAN = RTN26O2 + NO2                                           
    RC(483) = KBPAN                   

    # Reaction (484) OH + RTN26PAN = CH3COCH3 + CARB16 + NO2                            
    RC(484) = 3.66e-12  

    # Reaction (485) RTN28NO3 = P2604                                                   
    RC(485) = KIN  		

    # Reaction (486) P2604 = RTN28NO3                                                   
    RC(486) = KOUT2604 	

    # Reaction (487) RTX28NO3 = P4608                                                   
    RC(487) = KIN 		

    # Reaction (488) P4608 = RTX28NO3                                                   
    RC(488) = KOUT4608 	

    # Reaction (489) RCOOH25 = P2631                                                    
    RC(489) = KIN  		

    # Reaction (490) P2631 = RCOOH25                                                    
    RC(490) = KOUT2631 	

    # Reaction (491) RTN24OOH = P2635                                                   
    RC(491) = KIN  		

    # Reaction (492) P2635 = RTN24OOH                                                   
    RC(492) = KOUT2635 	

    # Reaction (493) RTX28OOH = P4610                                                   
    RC(493) = KIN  		

    # Reaction (494) P4610 = RTX28OOH                                                   
    RC(494) = KOUT4610 	

    # Reaction (495) RTN28OOH = P2605                                                   
    RC(495) = KIN  		

    # Reaction (496) P2605 = RTN28OOH                                                   
    RC(496) = KOUT2605 	

    # Reaction (497) RTN26OOH = P2630                                                   
    RC(497) = KIN  		

    # Reaction (498) P2630 = RTN26OOH                                                   
    RC(498) = KOUT2630 	

    # Reaction (499) RTN26PAN = P2629                                                   
    RC(499) = KIN  		

    # Reaction (500) P2629 = RTN26PAN                                                   
    RC(500) = KOUT2629 	

    # Reaction (501) RTN25OOH = P2632                                                   
    RC(501) = KIN 		

    # Reaction (502) P2632 = RTN25OOH                                                   
    RC(502) = KOUT2632 	

    # Reaction (503) RTN23OOH = P2637                                                   
    RC(503) = KIN  		

    # Reaction (504) P2637 = RTN23OOH                                                   
    RC(504) = KOUT2637 	

    # Reaction (505) ARNOH14 = P3612                                                    
    RC(505) = KIN  		

    # Reaction (506) P3612 = ARNOH14                                                    
    RC(506) = KOUT3612 	

    # Reaction (507) ARNOH17 = P3613                                                    
    RC(507) = KIN 		

    # Reaction (508) P3613 = ARNOH17                                                    
    RC(508) = KOUT3613 	

    # Reaction (509) ANHY = P3442                                                       
    RC(509) = KIN  		

    # Reaction (510) P3442 = ANHY                                                       
    RC(510) = KOUT3442 	



    return chem, rho_d, M, H2O, O2, N2

#       SUBROUTINE CHEMCO(RC,TEMP,M,O2,H2O,RO2,MOM,BR01)
# C----------------------------------------------------------------------
# C-
# C-   Purpose and Methods : CALCULATES RATE COEFFICIENTS
# C-
# C-   Inputs  : TEMP,M,H2O
# C-   Outputs : RC,DJ
# C-   Controls:

	
#       INTEGER I
#       N2       = 7.809E-01*M 
#       O2       = 2.079E-01*M 
#       R = 8.314
#       DO 317 I=1,510
#         RC(I)=0.0
#   317 CONTINUE

# C    SIMPLE RATE COEFFICIENTS                     
# C                                                                     
#       KRO2NO  = 2.54D-12*math.exp(360/TEMP) 
#       KAPNO   = 8.10D-12*math.exp(270/TEMP) 
#       KRO2NO3 = 2.50D-12 
#       KRO2HO2 = 2.91D-13*math.exp(1300/TEMP) 
#       KAPHO2  = 4.30D-13*math.exp(1040/TEMP) 
#       KNO3AL  = 1.44D-12*math.exp(-1862/TEMP) 
#       KDEC    = 1.0D+06
#       KALKOXY = 3.70D-14*math.exp(-460/TEMP)*O2 
#       KALKPXY = 1.80D-14*math.exp(-260/TEMP)*O2 
#       BR01 = (0.156 + 9.77D+08*math.exp(-6415/TEMP)) 
# C
#       KIN = 6.2E-03*MOM
#       KOUT2604 = 4.34*math.exp(-7776/(R*TEMP))
#       KOUT4608 = 4.34*math.exp(-9765/(R*TEMP))
#       KOUT2631 = 4.34*math.exp(-14500/(R*TEMP))
#       KOUT2635 = 4.34*math.exp(-12541/(R*TEMP))
#       KOUT4610 = 4.34*math.exp(-10513/(R*TEMP))
#       KOUT2605 = 4.34*math.exp(-8879/(R*TEMP))
#       KOUT2630 = 4.34*math.exp(-12639/(R*TEMP))
#       KOUT2629 = 4.34*math.exp(-4954/(R*TEMP))
#       KOUT2632 = 4.34*math.exp(-3801/(R*TEMP))
#       KOUT2637 = 4.34*math.exp(-16752/(R*TEMP))
#       KOUT3612 = 4.34*math.exp(-8362/(R*TEMP))
#       KOUT3613 = 4.34*math.exp(-11003/(R*TEMP))
#       KOUT3442 = 4.34*math.exp(-12763/(R*TEMP))
# C
# C    COMPLEX RATE COEFFICIENTS                    
# C                                                                     
# C    KFPAN                                                   
# C                                                                     
#       KC0     = 2.70D-28*M*(TEMP/300)**-7.1 
#       KCI     = 1.21D-11*(TEMP/300)**-0.9    
#       KRC     = KC0/KCI    
#       FCC     = 0.30       
#       FC      = 10**(LOG10(FCC)/(1+(LOG10(KRC))**2)) 
#       KFPAN   = (KC0*KCI)*FC/(KC0+KCI) 
# C                                                                   
# C    KBPAN                                                   
#       KD0     = 4.90D-03*M*math.exp(-12100/TEMP) 
#       KDI     = 3.70D+16*math.exp(-13600/TEMP)  
#       KRD     = KD0/KDI    
#       FCD     = 0.30       
#       FD      = 10**(LOG10(FCD)/(1+(LOG10(KRD))**2)) 
#       KBPAN   = (KD0*KDI)*FD/(KD0+KDI) 
# C                                                                     
# C     KMT01                                                   
#       K10     = 9.00D-32*M*(TEMP/300)**-1.5 
#       K1I     = 3.00D-11*(TEMP/300)**0.3    
#       KR1     = K10/K1I    
#       FC1     = 0.6 
#       F1      = 10**(LOG10(FC1)/(1+(LOG10(KR1))**2)) 
#       KMT01   = (K10*K1I)*F1/(K10+K1I) 
# C                                                                     
# C     KMT02                                                   
#       K20 = 9.00D-32*((temp/300)**-2.0)*M 
#       K2I = 2.20D-11
#       KR2     = K20/K2I    
#       FC2 = 0.6 
#       Fa2      = 10**(LOG10(FC2)/(1+(LOG10(KR2))**2)) 
#       KMT02   = (K20*K2I)*Fa2/(K20+K2I) 
# C                                                                     
# C      KMT03  : NO2      + NO3     = N2O5                               
# C    IUPAC 2001                                                       
#       K30     = 2.70D-30*M*(TEMP/300)**-3.4 
#       K3I     = 2.00D-12*(TEMP/300)**0.2    
#       KR3     = K30/K3I    
#       FC3     = (math.exp(-TEMP/250) + math.exp(-1050/TEMP)) 
#       F3      = 10**(LOG10(FC3)/(1+(LOG10(KR3))**2)) 
#       KMT03   = (K30*K3I)*F3/(K30+K3I) 
# C                                                                     
# C     KMT04  : N2O5               = NO2     + NO3                     
# C IUPAC 1997/2001                                                 
#       K40     = (2.20D-03*M*(TEMP/300)**-4.34)*(math.exp(-11080/TEMP))
#       K4I     = (9.70D+14*(TEMP/300)**0.1)*math.exp(-11080/TEMP)    
#       KR4     = K40/K4I    
#       FC4     = (math.exp(-TEMP/250) + math.exp(-1050/TEMP))
#       Fa4      = 10**(LOG10(FC4)/(1+(LOG10(KR4))**2)) 
#       KMT04   = (K40*K4I)*Fa4/(K40+K4I)       
# C	WRITE(6,*)'KMTO4=',KMT04                                                               
# C    KMT05                                                   
#       KMT05  =  1 + ((0.6*M)/(2.687D+19*(273/TEMP))) 
# C                                                                     
# C    KMT06                                                   
#       KMT06  =  1 + (1.40D-21*math.exp(2200/TEMP)*H2O) 
# C                                                                     
# C    KMT07  : OH       + NO      = HONO                              
# C    IUPAC 2001                                                      
#       K70     = 7.00D-31*M*(TEMP/300)**-2.6 
#       K7I     = 3.60D-11*(TEMP/300)**0.1    
#       KR7     = K70/K7I    
#       FC7     = 0.6  
#       F7      = 10**(LOG10(FC7)/(1+(LOG10(KR7))**2)) 
#       KMT07   = (K70*K7I)*F7/(K70+K7I) 
# C                                                                     
# C NASA 2000                                                           
  
# C    KMT08                                                    
#       K80 = 2.50D-30*((temp/300)**-4.4)*M 
#       K8I = 1.60D-11 
#       KR8 = K80/K8I 
#       FC8 = 0.6 
#       F8      = 10**(LOG10(FC8)/(1+(LOG10(KR8))**2)) 
#       KMT08   = (K80*K8I)*F8/(K80+K8I) 
# C                                                                     
# C    KMT09  : HO2      + NO2     = HO2NO2                            
# C IUPAC 1997/2001                                                 
#       K90     = 1.80D-31*M*(TEMP/300)**-3.2 
#       K9I     = 4.70D-12    
#       KR9     = K90/K9I    
#       FC9     = 0.6 
#       F9      = 10**(LOG10(FC9)/(1+(LOG10(KR9))**2)) 
#       KMT09   = (K90*K9I)*F9/(K90+K9I) 
# C                                                                     
# C KMT10  : HO2NO2             = HO2     + NO2                     
# C IUPAC 2001                                                      
# C
#       K100     = 4.10D-05*M*math.exp(-10650/TEMP) 
#       K10I     = 5.70D+15*math.exp(-11170/TEMP)   
#       KR10     = K100/K10I    
#       FC10     = 0.5 
#       F10      = 10**(LOG10(FC10)/(1+(LOG10(KR10))**2)) 
#       KMT10    = (K100*K10I)*F10/(K100+K10I) 
# C                                                                     
# C   KMT11  : OH       + HNO3    = H2O     + NO3                     
# C   IUPAC 2001                                                      
#       K1     = 7.20D-15*math.exp(785/TEMP) 
#       K3     = 1.90D-33*math.exp(725/TEMP) 
#       K4     = 4.10D-16*math.exp(1440/TEMP) 
#       K2     = (K3*M)/(1+(K3*M/K4)) 
#       KMT11  = K1 + K2 
# C                                                                     
# C KMT12 : OH    +   SO2  =  HSO3                                  
# C IUPAC 2003                                                      
#       K0 = 3.0D-31*((TEMP/300)**-3.3)*M 
#       KI = 1.5D-12 
#       KR1 = K0/KI 
#       FC = 0.6 
#       F=10**(LOG10(FC)/(1+(LOG10(KR1))**2)) 
#       KMT12=(K0*KI*F)/(K0+KI) 
# C                                                                     
# C KMT13  : CH3O2    + NO2     = CH3O2NO2                           
# C IUPAC 2003                                                       
#       K130     = 2.50D-30*((TEMP/300)**-5.5)*M 
#       K13I     = 7.50D-12 
#       KR13     = K130/K13I 
#       FC13     = 0.36 
#       F13      = 10**(LOG10(FC13)/(1+(LOG10(KR13))**2)) 
#       KMT13    = (K130*K13I)*F13/(K130+K13I) 
# C                                                                     
# C  KMT14  : CH3O2NO2           = CH3O2   + NO2                      
# C  IUPAC 2001                                                       
#       K140     = 9.00D-05*math.exp(-9690/TEMP)*M 
#       K14I     = 1.10D+16*math.exp(-10560/TEMP) 
#       KR14     = K140/K14I 
#       FC14     = 0.36 
#       F14      = 10**(LOG10(FC14)/(1+(LOG10(KR14))**2)) 
#       KMT14    = (K140*K14I)*F14/(K140+K14I) 
# C                                                                   
# C KMT15  :    OH  +  C2H4  =                                       
# C IUPAC 2001                                                      
#       K150 = 6.00D-29*((TEMP/298)**-4.0)*M 
#       K15I = 9.00D-12*((TEMP/298)**-1.1) 
#       KR15 = K150/K15I 
#       FC15 = 0.7
#       F15      = 10**(LOG10(FC15)/(1+(LOG10(KR15))**2)) 
#       KMT15    = (K150*K15I)*F15/(K150+K15I) 
# C                                                                    
# C KMT16  :  OH  +  C3H6         
# C IUPAC 2003                                                     
#       K160     = 3.00D-27*((TEMP/298)**-3.0)*M 
#       K16I     = 2.80D-11*((TEMP/298)**-1.3) 
#       KR16     = K160/K16I 
#       FC16     = 0.5 
#       F16      = 10**(LOG10(FC16)/(1+(LOG10(KR16))**2)) 
#       KMT16    = (K160*K16I)*F16/(K160+K16I) 
# C                                                                     
# C    KMT17                                                   
#       K170 = 5.00D-30*((TEMP/298)**-1.5)*M 
#       K17I = 9.40D-12*math.exp(-700/TEMP) 
#       KR17     = K170/K17I 
#       FC17 = (math.exp(-TEMP/580) + math.exp(-2320/TEMP)) 
#       F17      = 10**(LOG10(FC17)/(1+(LOG10(KR17))**2)) 
#       KMT17    = (K170*K17I)*F17/(K170+K17I) 
# C
# C  LIST OF ALL REACTIONS 
# C
# C     Reaction (1) O = O3                                                             
#          RC(1) = 5.60D-34*O2*N2*((TEMP/300)**-2.6)
# C
# C     Reaction (2) O = O3                                                             
#          RC(2) = 6.00D-34*O2*O2*((TEMP/300)**-2.6)
# C
# C     Reaction (3) O + O3 =                                                           
#          RC(3) = 8.00D-12*math.exp(-2060/TEMP)         
# C
# C     Reaction (4) O + NO = NO2                                                       
#          RC(4) = KMT01                            
# C
# C     Reaction (5) O + NO2 = NO                                                       
#          RC(5) = 5.50D-12*math.exp(188/TEMP)           
# C
# C     Reaction (6) O + NO2 = NO3                                                      
#          RC(6) = KMT02                            
# C
# C     Reaction (7) O1D = O                                                            
#          RC(7) = 3.20D-11*O2*math.exp(67/TEMP)         
# C
# C     Reaction (8) O1D = O                                                            
#          RC(8) = 1.80D-11*N2*math.exp(107/TEMP)        
# C
# C     Reaction (9) NO + O3 = NO2                                                      
#          RC(9) = 1.40D-12*math.exp(-1310/TEMP)         
# C
# C     Reaction (10) NO2 + O3 = NO3                                                     
#          RC(10) = 1.40D-13*math.exp(-2470/TEMP)         
# C
# C     Reaction (11) NO + NO = NO2 + NO2                                                
#          RC(11) = 3.30D-39*math.exp(530/TEMP)*O2        
# C
# C     Reaction (12) NO + NO3 = NO2 + NO2                                               
#          RC(12) = 1.80D-11*math.exp(110/TEMP)           
# C
# C     Reaction (13) NO2 + NO3 = NO + NO2                                               
#          RC(13) = 4.50D-14*math.exp(-1260/TEMP)         
# C
# C     Reaction (14) NO2 + NO3 = N2O5                                                   
#          RC(14) = KMT03                            
# C
# C     Reaction (15) N2O5 = NO2 + NO3                                                   
#          RC(15) = KMT04                            
# C
# C     Reaction (16) O1D = OH + OH                                                      
#          RC(16) = 2.20D-10                     
# C
# C     Reaction (17) OH + O3 = HO2                                                      
#          RC(17) = 1.70D-12*math.exp(-940/TEMP)          
# C
# C     Reaction (18) OH + H2 = HO2                                                      
#          RC(18) = 7.70D-12*math.exp(-2100/TEMP)         
# C
# C     Reaction (19) OH + CO = HO2                                                      
#          RC(19) = 1.30D-13*KMT05                   
# C
# C     Reaction (20) OH + H2O2 = HO2                                                    
#          RC(20) = 2.90D-12*math.exp(-160/TEMP)          
# C
# C     Reaction (21) HO2 + O3 = OH                                                      
#          RC(21) = 2.03D-16*((TEMP/300)**4.57)*math.exp(693/TEMP)  
# C
# C     Reaction (22) OH + HO2 =                                                         
#          RC(22) = 4.80D-11*math.exp(250/TEMP)           
# C
# C     Reaction (23) HO2 + HO2 = H2O2                                                   
#          RC(23) = 2.20D-13*KMT06*math.exp(600/TEMP)     
# C
# C     Reaction (24) HO2 + HO2 = H2O2                                                   
#          RC(24) = 1.90D-33*M*KMT06*math.exp(980/TEMP)   
# C
# C     Reaction (25) OH + NO = HONO                                                     
#          RC(25) = KMT07                            
# C
# C     Reaction (26) NO2 = HONO                                                         
#          RC(26) = 0.0                          
# C
# C     Reaction (27) OH + NO2 = HNO3                                                    
#          RC(27) = KMT08                            
# C
# C     Reaction (28) OH + NO3 = HO2 + NO2                                               
#          RC(28) = 2.00D-11                         
# C
# C     Reaction (29) HO2 + NO = OH + NO2                                                
#          RC(29) = 3.60D-12*math.exp(270/TEMP)           
# C
# C     Reaction (30) HO2 + NO2 = HO2NO2                                                 
#          RC(30) = KMT09                            
# C
# C     Reaction (31) HO2NO2 = HO2 + NO2                                                 
#          RC(31) = KMT10                            
# C
# C     Reaction (32) OH + HO2NO2 = NO2                                                  
#          RC(32) = 1.90D-12*math.exp(270/TEMP)           
# C
# C     Reaction (33) HO2 + NO3 = OH + NO2                                               
#          RC(33) = 4.00D-12                         
# C
# C     Reaction (34) OH + HONO = NO2                                                    
#          RC(34) = 2.50D-12*math.exp(-260/TEMP)          
# C
# C     Reaction (35) OH + HNO3 = NO3                                                    
#          RC(35) = KMT11                            
# C
# C     Reaction (36) O + SO2 = SO3                                                      
#          RC(36) = 4.00D-32*math.exp(-1000/TEMP)*M       
# C
# C     Reaction (37) OH + SO2 = HSO3                                                    
#          RC(37) = KMT12                            
# C
# C     Reaction (38) HSO3 = HO2 + SO3                                                   
#          RC(38) = 1.30D-12*math.exp(-330/TEMP)*O2       
# C
# C     Reaction (39) HNO3 = NA                                                          
#          RC(39) = 0.0                         
# C
# C     Reaction (40) N2O5 = NA + NA                                                     
#          RC(40) = 0.0                         
# C
# C     Reaction (41) SO3 = SA                                                           
#          RC(41) = 1.20D-15*H2O                     
# C
# C     Reaction (42) OH + CH4 = CH3O2                                                   
#          RC(42) = 9.65D-20*TEMP**2.58*math.exp(-1082/TEMP) 
# C
# C     Reaction (43) OH + C2H6 = C2H5O2                                                 
#          RC(43) = 1.52D-17*TEMP**2*math.exp(-498/TEMP) 
# C
# C     Reaction (44) OH + C3H8 = IC3H7O2                                                
#          RC(44) = 1.55D-17*TEMP**2*math.exp(-61/TEMP)*0.736  
# C
# C     Reaction (45) OH + C3H8 = RN10O2                                                 
#          RC(45) = 1.55D-17*TEMP**2*math.exp(-61/TEMP)*0.264  
# C
# C     Reaction (46) OH + NC4H10 = RN13O2                                               
#          RC(46) = 1.69D-17*TEMP**2*math.exp(145/TEMP)  
# C
# C     Reaction (47) OH + C2H4 = HOCH2CH2O2                                             
#          RC(47) = KMT15                        
# C
# C     Reaction (48) OH + C3H6 = RN9O2                                                  
#          RC(48) = KMT16                        
# C
# C     Reaction (49) OH + TBUT2ENE = RN12O2                                             
#          RC(49) = 1.01D-11*math.exp(550/TEMP)       
# C
# C     Reaction (50) NO3 + C2H4 = NRN6O2                                                
#          RC(50) = 2.10D-16                     
# C
# C     Reaction (51) NO3 + C3H6 = NRN9O2                                                
#          RC(51) = 9.40D-15                     
# C
# C     Reaction (52) NO3 + TBUT2ENE = NRN12O2                                           
#          RC(52) = 3.90D-13                     
# C
# C     Reaction (53) O3 + C2H4 = HCHO + CO + HO2 + OH                                   
#          RC(53) = 9.14D-15*math.exp(-2580/TEMP)*0.13  
# C
# C     Reaction (54) O3 + C2H4 = HCHO + HCOOH                                           
#          RC(54) = 9.14D-15*math.exp(-2580/TEMP)*0.87  
# C
# C     Reaction (55) O3 + C3H6 = HCHO + CO + CH3O2 + OH                                 
#          RC(55) = 5.51D-15*math.exp(-1878/TEMP)*0.36  
# C
# C     Reaction (56) O3 + C3H6 = HCHO + CH3CO2H                                         
#          RC(56) = 5.51D-15*math.exp(-1878/TEMP)*0.64  
# C
# C     Reaction (57) O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH                           
#          RC(57) = 6.64D-15*math.exp(-1059/TEMP)*0.69 
# C
# C     Reaction (58) O3 + TBUT2ENE = CH3CHO + CH3CO2H                                   
#          RC(58) = 6.64D-15*math.exp(-1059/TEMP)*0.31 
# C
# C     Reaction (59) OH + C5H8 = RU14O2                                                 
#          RC(59) = 2.54D-11*math.exp(410/TEMP)       
# C
# C     Reaction (60) NO3 + C5H8 = NRU14O2                                               
#          RC(60) = 3.03D-12*math.exp(-446/TEMP)      
# C
# C     Reaction (61) O3 + C5H8 = UCARB10 + CO + HO2 + OH                                
#          RC(61) = 7.86D-15*math.exp(-1913/TEMP)*0.27 
# C
# C     Reaction (62) O3 + C5H8 = UCARB10 + HCOOH                                        
#          RC(62) = 7.86D-15*math.exp(-1913/TEMP)*0.73 
# C
# C     Reaction (63) APINENE + OH = RTN28O2                                             
#          RC(63) = 1.20D-11*math.exp(444/TEMP)           
# C
# C     Reaction (64) APINENE + NO3 = NRTN28O2                                           
#          RC(64) = 1.19D-12*math.exp(490/TEMP)           
# C
# C     Reaction (65) APINENE + O3 = OH + RTN26O2                                        
#          RC(65) = 1.01D-15*math.exp(-732/TEMP)*0.80  
# C
# C     Reaction (66) APINENE + O3 = TNCARB26 + H2O2                                     
#          RC(66) = 1.01D-15*math.exp(-732/TEMP)*0.075  
# C
# C     Reaction (67) APINENE + O3 = RCOOH25                                             
#          RC(67) = 1.01D-15*math.exp(-732/TEMP)*0.125  
# C
# C     Reaction (68) BPINENE + OH = RTX28O2                                             
#          RC(68) = 2.38D-11*math.exp(357/TEMP) 
# C
# C     Reaction (69) BPINENE + NO3 = NRTX28O2                                           
#          RC(69) = 2.51D-12 
# C
# C     Reaction (70) BPINENE + O3 =  RTX24O2 + OH                                       
#          RC(70) = 1.50D-17*0.35 
# C
# C     Reaction (71) BPINENE + O3 =  HCHO + TXCARB24 + H2O2                             
#          RC(71) = 1.50D-17*0.20 
# C
# C     Reaction (72) BPINENE + O3 =  HCHO + TXCARB22                                    
#          RC(72) = 1.50D-17*0.25 
# C
# C     Reaction (73) BPINENE + O3 =  TXCARB24 + CO                                      
#          RC(73) = 1.50D-17*0.20 
# C
# C     Reaction (74) C2H2 + OH = HCOOH + CO + HO2                                       
#          RC(74) = KMT17*0.364 
# C
# C     Reaction (75) C2H2 + OH = CARB3 + OH                                             
#          RC(75) = KMT17*0.636 
# C
# C     Reaction (76) BENZENE + OH = RA13O2                                              
#          RC(76) = 2.33D-12*math.exp(-193/TEMP)*0.47 
# C
# C     Reaction (77) BENZENE + OH = AROH14 + HO2                                        
#          RC(77) = 2.33D-12*math.exp(-193/TEMP)*0.53 
# C
# C     Reaction (78) TOLUENE + OH = RA16O2                                              
#          RC(78) = 1.81D-12*math.exp(338/TEMP)*0.82 
# C
# C     Reaction (79) TOLUENE + OH = AROH17 + HO2                                        
#          RC(79) = 1.81D-12*math.exp(338/TEMP)*0.18 
# C
# C     Reaction (80) OXYL + OH = RA19AO2                                                
#          RC(80) = 1.36D-11*0.70 
# C
# C     Reaction (81) OXYL + OH = RA19CO2                                                
#          RC(81) = 1.36D-11*0.30 
# C
# C     Reaction (82) OH + HCHO = HO2 + CO                                               
#          RC(82) = 1.20D-14*TEMP*math.exp(287/TEMP)  
# C
# C     Reaction (83) OH + CH3CHO = CH3CO3                                               
#          RC(83) = 5.55D-12*math.exp(311/TEMP)             
# C
# C     Reaction (84) OH + C2H5CHO = C2H5CO3                                             
#          RC(84) = 1.96D-11                                
# C
# C     Reaction (85) NO3 + HCHO = HO2 + CO + HNO3                                       
#          RC(85) = 5.80D-16                  
# C
# C     Reaction (86) NO3 + CH3CHO = CH3CO3 + HNO3                                       
#          RC(86) = KNO3AL                   
# C
# C     Reaction (87) NO3 + C2H5CHO = C2H5CO3 + HNO3                                     
#          RC(87) = KNO3AL*2.4             
# C
# C     Reaction (88) OH + CH3COCH3 = RN8O2                                              
#          RC(88) = 5.34D-18*TEMP**2*math.exp(-230/TEMP) 
# C
# C     Reaction (89) MEK + OH = RN11O2                                                  
#          RC(89) = 3.24D-18*TEMP**2*math.exp(414/TEMP)
# C
# C     Reaction (90) OH + CH3OH = HO2 + HCHO                                            
#          RC(90) = 6.01D-18*TEMP**2*math.exp(170/TEMP)  
# C
# C     Reaction (91) OH + C2H5OH = CH3CHO + HO2                                         
#          RC(91) = 6.18D-18*TEMP**2*math.exp(532/TEMP)*0.887 
# C
# C     Reaction (92) OH + C2H5OH = HOCH2CH2O2                                           
#          RC(92) = 6.18D-18*TEMP**2*math.exp(532/TEMP)*0.113 
# C
# C     Reaction (93) NPROPOL + OH = C2H5CHO + HO2                                       
#          RC(93) = 5.53D-12*0.49 
# C
# C     Reaction (94) NPROPOL + OH = RN9O2                                               
#          RC(94) = 5.53D-12*0.51 
# C
# C     Reaction (95) OH + IPROPOL = CH3COCH3 + HO2                                      
#          RC(95) = 4.06D-18*TEMP**2*math.exp(788/TEMP)*0.86 
# C
# C     Reaction (96) OH + IPROPOL = RN9O2                                               
#          RC(96) = 4.06D-18*TEMP**2*math.exp(788/TEMP)*0.14 
# C
# C     Reaction (97) HCOOH + OH = HO2                                                   
#          RC(97) = 4.50D-13 
# C
# C     Reaction (98) CH3CO2H + OH = CH3O2                                               
#          RC(98) = 8.00D-13 
# C
# C     Reaction (99) OH + CH3CL = CH3O2                                                 
#          RC(99) = 7.33D-18*TEMP**2*math.exp(-809/TEMP)   
# C
# C     Reaction (100) OH + CH2CL2 = CH3O2                                                
#          RC(100) = 6.14D-18*TEMP**2*math.exp(-389/TEMP)   
# C
# C     Reaction (101) OH + CHCL3 = CH3O2                                                 
#          RC(101) = 1.80D-18*TEMP**2*math.exp(-129/TEMP)   
# C
# C     Reaction (102) OH + CH3CCL3 = C2H5O2                                              
#          RC(102) = 2.25D-18*TEMP**2*math.exp(-910/TEMP)   
# C
# C     Reaction (103) OH + TCE = HOCH2CH2O2                                              
#          RC(103) = 9.64D-12*math.exp(-1209/TEMP)         
# C
# C     Reaction (104) OH + TRICLETH = HOCH2CH2O2                                         
#          RC(104) = 5.63D-13*math.exp(427/TEMP)            
# C
# C     Reaction (105) OH + CDICLETH = HOCH2CH2O2                                         
#          RC(105) = 1.94D-12*math.exp(90/TEMP)            
# C
# C     Reaction (106) OH + TDICLETH = HOCH2CH2O2                                         
#          RC(106) = 1.01D-12*math.exp(250/TEMP)           
# C
# C     Reaction (107) CH3O2 + NO = HCHO + HO2 + NO2                                      
#          RC(107) = 3.00D-12*math.exp(280/TEMP)*0.999 
# C
# C     Reaction (108) C2H5O2 + NO = CH3CHO + HO2 + NO2                                   
#          RC(108) = 2.60D-12*math.exp(365/TEMP)*0.991 
# C
# C     Reaction (109) RN10O2 + NO = C2H5CHO + HO2 + NO2                                  
#          RC(109) = 2.80D-12*math.exp(360/TEMP)*0.980 
# C
# C     Reaction (110) IC3H7O2 + NO = CH3COCH3 + HO2 + NO2                                
#          RC(110) = 2.70D-12*math.exp(360/TEMP)*0.958 
# C
# C     Reaction (111) RN13O2 + NO = CH3CHO + C2H5O2 + NO2                                
#          RC(111) = KRO2NO*0.917*BR01       
# C
# C     Reaction (112) RN13O2 + NO = CARB11A + HO2 + NO2                                  
#          RC(112) = KRO2NO*0.917*(1-BR01)   
# C
# C     Reaction (113) RN16O2 + NO = RN15AO2 + NO2                                        
#          RC(113) = KRO2NO*0.877                 
# C
# C     Reaction (114) RN19O2 + NO = RN18AO2 + NO2                                        
#          RC(114) = KRO2NO*0.788                 
# C
# C     Reaction (115) RN13AO2 + NO = RN12O2 + NO2                                        
#          RC(115) = KRO2NO                       
# C
# C     Reaction (116) RN16AO2 + NO = RN15O2 + NO2                                        
#          RC(116) = KRO2NO                       
# C
# C     Reaction (117) RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2                          
#          RC(117) = KRO2NO*0.918       
# C
# C     Reaction (118) RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2                         
#          RC(118) = KRO2NO*0.889*0.7 
# C
# C     Reaction (119) RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2                          
#          RC(119) = KRO2NO*0.889*0.3 
# C
# C     Reaction (120) RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2                        
#          RC(120) = KRO2NO*0.862       
# C
# C     Reaction (121) RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2                         
#          RC(121) = KRO2NO*0.862       
# C
# C     Reaction (122) HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2                          
#          RC(122) = KRO2NO*0.995*0.776  
# C
# C     Reaction (123) HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2                             
#          RC(123) = KRO2NO*0.995*0.224  
# C
# C     Reaction (124) RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2                             
#          RC(124) = KRO2NO*0.979     
# C
# C     Reaction (125) RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2                          
#          RC(125) = KRO2NO*0.959     
# C
# C     Reaction (126) RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2                         
#          RC(126) = KRO2NO*0.936     
# C
# C     Reaction (127) RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2                        
#          RC(127) = KRO2NO*0.903     
# C
# C     Reaction (128) RN15AO2 + NO = CARB13 + HO2 + NO2                                  
#          RC(128) = KRO2NO*0.975     
# C
# C     Reaction (129) RN18AO2 + NO = CARB16 + HO2 + NO2                                  
#          RC(129) = KRO2NO*0.946     
# C
# C     Reaction (130) CH3CO3 + NO = CH3O2 + NO2                                          
#          RC(130) = KAPNO                      
# C
# C     Reaction (131) C2H5CO3 + NO = C2H5O2 + NO2                                        
#          RC(131) = KAPNO                      
# C
# C     Reaction (132) HOCH2CO3 + NO = HO2 + HCHO + NO2                                   
#          RC(132) = KAPNO                      
# C
# C     Reaction (133) RN8O2 + NO = CH3CO3 + HCHO + NO2                                   
#          RC(133) = KRO2NO                     
# C
# C     Reaction (134) RN11O2 + NO = CH3CO3 + CH3CHO + NO2                                
#          RC(134) = KRO2NO                     
# C
# C     Reaction (135) RN14O2 + NO = C2H5CO3 + CH3CHO + NO2                               
#          RC(135) = KRO2NO                     
# C
# C     Reaction (136) RN17O2 + NO = RN16AO2 + NO2                                        
#          RC(136) = KRO2NO                     
# C
# C     Reaction (137) RU14O2 + NO = UCARB12 + HO2 +  NO2                                 
#          RC(137) = KRO2NO*0.900*0.252  
# C
# C     Reaction (138) RU14O2 + NO = UCARB10 + HCHO + HO2 + NO2                           
#          RC(138) = KRO2NO*0.900*0.748 
# C
# C     Reaction (139) RU12O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
#          RC(139) = KRO2NO*0.7         
# C
# C     Reaction (140) RU12O2 + NO = CARB7 + CO + HO2 + NO2                               
#          RC(140) = KRO2NO*0.3         
# C
# C     Reaction (141) RU10O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
#          RC(141) = KRO2NO*0.5         
# C
# C     Reaction (142) RU10O2 + NO = CARB6 + HCHO + HO2 + NO2                             
#          RC(142) = KRO2NO*0.3         
# C
# C     Reaction (143) RU10O2 + NO = CARB7 + HCHO + HO2 + NO2                             
#          RC(143) = KRO2NO*0.2          
# C
# C     Reaction (144) NRN6O2 + NO = HCHO + HCHO + NO2 + NO2                              
#          RC(144) = KRO2NO                 
# C
# C     Reaction (145) NRN9O2 + NO = CH3CHO + HCHO + NO2 + NO2                            
#          RC(145) = KRO2NO                 
# C
# C     Reaction (146) NRN12O2 + NO = CH3CHO + CH3CHO + NO2 + NO2                         
#          RC(146) = KRO2NO                 
# C
# C     Reaction (147) NRU14O2 + NO = NUCARB12 + HO2 + NO2                                
#          RC(147) = KRO2NO                 
# C
# C     Reaction (148) NRU12O2 + NO = NOA + CO + HO2 + NO2                                
#          RC(148) = KRO2NO                 
# C
# C     Reaction (149) RTN28O2 + NO = TNCARB26 + HO2 + NO2                                
#          RC(149) = KRO2NO*0.767*0.915  
# C
# C     Reaction (150) RTN28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
#          RC(150) = KRO2NO*0.767*0.085  
# C
# C     Reaction (151) NRTN28O2 + NO = TNCARB26 + NO2 + NO2                               
#          RC(151) = KRO2NO                  
# C
# C     Reaction (152) RTN26O2 + NO = RTN25O2 + NO2                                       
#          RC(152) = KAPNO                   
# C
# C     Reaction (153) RTN25O2 + NO = RTN24O2 + NO2                                       
#          RC(153) = KRO2NO*0.840        
# C
# C     Reaction (154) RTN24O2 + NO = RTN23O2 + NO2                                       
#          RC(154) = KRO2NO                   
# C
# C     Reaction (155) RTN23O2 + NO = CH3COCH3 + RTN14O2 + NO2                            
#          RC(155) = KRO2NO                  
# C
# C     Reaction (156) RTN14O2 + NO = HCHO + TNCARB10 + HO2 + NO2                         
#          RC(156) = KRO2NO               
# C
# C     Reaction (157) RTN10O2 + NO = RN8O2 + CO + NO2                                    
#          RC(157) = KRO2NO               
# C
# C     Reaction (158) RTX28O2 + NO = TXCARB24 + HCHO + HO2 + NO2                         
#          RC(158) = KRO2NO*0.767*0.915  
# C
# C     Reaction (159) RTX28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
#          RC(159) = KRO2NO*0.767*0.085  
# C
# C     Reaction (160) NRTX28O2 + NO = TXCARB24 + HCHO + NO2 + NO2                        
#          RC(160) = KRO2NO            
# C
# C     Reaction (161) RTX24O2 + NO = TXCARB22 + HO2 + NO2                                
#          RC(161) = KRO2NO*0.843*0.6  
# C
# C     Reaction (162) RTX24O2 + NO = CH3COCH3 + RN13AO2 + HCHO + NO2                     
#          RC(162) = KRO2NO*0.843*0.4  
# C
# C     Reaction (163) RTX22O2 + NO = CH3COCH3 + RN13O2 + NO2                             
#          RC(163) = KRO2NO*0.700         
# C
# C     Reaction (164) CH3O2    + NO2     = CH3O2NO2                                      
#          RC(164) = KMT13         
# C
# C     Reaction (165) CH3O2NO2           = CH3O2   + NO2                                 
#          RC(165) = KMT14         
# C
# C     Reaction (166) CH3O2 + NO = CH3NO3                                                
#          RC(166) = 3.00D-12*math.exp(280/TEMP)*0.001 
# C
# C     Reaction (167) C2H5O2 + NO = C2H5NO3                                              
#          RC(167) = 2.60D-12*math.exp(365/TEMP)*0.009 
# C
# C     Reaction (168) RN10O2 + NO = RN10NO3                                              
#          RC(168) = 2.80D-12*math.exp(360/TEMP)*0.020 
# C
# C     Reaction (169) IC3H7O2 + NO = IC3H7NO3                                            
#          RC(169) = 2.70D-12*math.exp(360/TEMP)*0.042 
# C
# C     Reaction (170) RN13O2 + NO = RN13NO3                                              
#          RC(170) = KRO2NO*0.083                 
# C
# C     Reaction (171) RN16O2 + NO = RN16NO3                                              
#          RC(171) = KRO2NO*0.123                 
# C
# C     Reaction (172) RN19O2 + NO = RN19NO3                                              
#          RC(172) = KRO2NO*0.212                 
# C
# C     Reaction (173) HOCH2CH2O2 + NO = HOC2H4NO3                                        
#          RC(173) = KRO2NO*0.005                 
# C
# C     Reaction (174) RN9O2 + NO = RN9NO3                                                
#          RC(174) = KRO2NO*0.021                 
# C
# C     Reaction (175) RN12O2 + NO = RN12NO3                                              
#          RC(175) = KRO2NO*0.041                 
# C
# C     Reaction (176) RN15O2 + NO = RN15NO3                                              
#          RC(176) = KRO2NO*0.064                 
# C
# C     Reaction (177) RN18O2 + NO = RN18NO3                                              
#          RC(177) = KRO2NO*0.097                 
# C
# C     Reaction (178) RN15AO2 + NO = RN15NO3                                             
#          RC(178) = KRO2NO*0.025                 
# C
# C     Reaction (179) RN18AO2 + NO = RN18NO3                                             
#          RC(179) = KRO2NO*0.054                 
# C
# C     Reaction (180) RU14O2 + NO = RU14NO3                                              
#          RC(180) = KRO2NO*0.100                 
# C
# C     Reaction (181) RA13O2 + NO = RA13NO3                                              
#          RC(181) = KRO2NO*0.082                 
# C
# C     Reaction (182) RA16O2 + NO = RA16NO3                                              
#          RC(182) = KRO2NO*0.111                 
# C
# C     Reaction (183) RA19AO2 + NO = RA19NO3                                             
#          RC(183) = KRO2NO*0.138                 
# C
# C     Reaction (184) RA19CO2 + NO = RA19NO3                                             
#          RC(184) = KRO2NO*0.138                 
# C
# C     Reaction (185) RTN28O2 + NO = RTN28NO3                                            
#          RC(185) = KRO2NO*0.233        
# C
# C     Reaction (186) RTN25O2 + NO = RTN25NO3                                            
#          RC(186) = KRO2NO*0.160        
# C
# C     Reaction (187) RTX28O2 + NO = RTX28NO3                                            
#          RC(187) = KRO2NO*0.233        
# C
# C     Reaction (188) RTX24O2 + NO = RTX24NO3                                            
#          RC(188) = KRO2NO*0.157        
# C
# C     Reaction (189) RTX22O2 + NO = RTX22NO3                                            
#          RC(189) = KRO2NO*0.300        
# C
# C     Reaction (190) CH3O2 + NO3 = HCHO + HO2 + NO2                                     
#          RC(190) = KRO2NO3*0.40          
# C
# C     Reaction (191) C2H5O2 + NO3 = CH3CHO + HO2 + NO2                                  
#          RC(191) = KRO2NO3               
# C
# C     Reaction (192) RN10O2 + NO3 = C2H5CHO + HO2 + NO2                                 
#          RC(192) = KRO2NO3               
# C
# C     Reaction (193) IC3H7O2 + NO3 = CH3COCH3 + HO2 + NO2                               
#          RC(193) = KRO2NO3               
# C
# C     Reaction (194) RN13O2 + NO3 = CH3CHO + C2H5O2 + NO2                               
#          RC(194) = KRO2NO3*BR01     
# C
# C     Reaction (195) RN13O2 + NO3 = CARB11A + HO2 + NO2                                 
#          RC(195) = KRO2NO3*(1-BR01) 
# C
# C     Reaction (196) RN16O2 + NO3 = RN15AO2 + NO2                                       
#          RC(196) = KRO2NO3               
# C
# C     Reaction (197) RN19O2 + NO3 = RN18AO2 + NO2                                       
#          RC(197) = KRO2NO3               
# C
# C     Reaction (198) RN13AO2 + NO3 = RN12O2 + NO2                                       
#          RC(198) = KRO2NO3                      
# C
# C     Reaction (199) RN16AO2 + NO3 = RN15O2 + NO2                                       
#          RC(199) = KRO2NO3                      
# C
# C     Reaction (200) RA13O2 + NO3 = CARB3 + UDCARB8 + HO2 + NO2                         
#          RC(200) = KRO2NO3            
# C
# C     Reaction (201) RA16O2 + NO3 = CARB3 + UDCARB11 + HO2 + NO2                        
#          RC(201) = KRO2NO3*0.7     
# C
# C     Reaction (202) RA16O2 + NO3 = CARB6 + UDCARB8 + HO2 + NO2                         
#          RC(202) = KRO2NO3*0.3     
# C
# C     Reaction (203) RA19AO2 + NO3 = CARB3 + UDCARB14 + HO2 + NO2                       
#          RC(203) = KRO2NO3           
# C
# C     Reaction (204) RA19CO2 + NO3 = CARB9 + UDCARB8 + HO2 + NO2                        
#          RC(204) = KRO2NO3           
# C
# C     Reaction (205) HOCH2CH2O2 + NO3 = HCHO + HCHO + HO2 + NO2                         
#          RC(205) = KRO2NO3*0.776  
# C
# C     Reaction (206) HOCH2CH2O2 + NO3 = HOCH2CHO + HO2 + NO2                            
#          RC(206) = KRO2NO3*0.224  
# C
# C     Reaction (207) RN9O2 + NO3 = CH3CHO + HCHO + HO2 + NO2                            
#          RC(207) = KRO2NO3         
# C
# C     Reaction (208) RN12O2 + NO3 = CH3CHO + CH3CHO + HO2 + NO2                         
#          RC(208) = KRO2NO3         
# C
# C     Reaction (209) RN15O2 + NO3 = C2H5CHO + CH3CHO + HO2 + NO2                        
#          RC(209) = KRO2NO3         
# C
# C     Reaction (210) RN18O2 + NO3 = C2H5CHO + C2H5CHO + HO2 + NO2                       
#          RC(210) = KRO2NO3         
# C
# C     Reaction (211) RN15AO2 + NO3 = CARB13 + HO2 + NO2                                 
#          RC(211) = KRO2NO3         
# C
# C     Reaction (212) RN18AO2 + NO3 = CARB16 + HO2 + NO2                                 
#          RC(212) = KRO2NO3         
# C
# C     Reaction (213) CH3CO3 + NO3 = CH3O2 + NO2                                         
#          RC(213) = KRO2NO3*1.60          
# C
# C     Reaction (214) C2H5CO3 + NO3 = C2H5O2 + NO2                                       
#          RC(214) = KRO2NO3*1.60          
# C
# C     Reaction (215) HOCH2CO3 + NO3 = HO2 + HCHO + NO2                                  
#          RC(215) = KRO2NO3*1.60         
# C
# C     Reaction (216) RN8O2 + NO3 = CH3CO3 + HCHO + NO2                                  
#          RC(216) = KRO2NO3               
# C
# C     Reaction (217) RN11O2 + NO3 = CH3CO3 + CH3CHO + NO2                               
#          RC(217) = KRO2NO3               
# C
# C     Reaction (218) RN14O2 + NO3 = C2H5CO3 + CH3CHO + NO2                              
#          RC(218) = KRO2NO3               
# C
# C     Reaction (219) RN17O2 + NO3 = RN16AO2 + NO2                                       
#          RC(219) = KRO2NO3               
# C
# C     Reaction (220) RU14O2 + NO3 = UCARB12 + HO2 + NO2                                 
#          RC(220) = KRO2NO3*0.252     
# C
# C     Reaction (221) RU14O2 + NO3 = UCARB10 + HCHO + HO2 + NO2                          
#          RC(221) = KRO2NO3*0.748     
# C
# C     Reaction (222) RU12O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
#          RC(222) = KRO2NO3*0.7         
# C
# C     Reaction (223) RU12O2 + NO3 = CARB7 + CO + HO2 + NO2                              
#          RC(223) = KRO2NO3*0.3         
# C
# C     Reaction (224) RU10O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
#          RC(224) = KRO2NO3*0.5         
# C
# C     Reaction (225) RU10O2 + NO3 = CARB6 + HCHO + HO2 + NO2                            
#          RC(225) = KRO2NO3*0.3         
# C
# C     Reaction (226) RU10O2 + NO3 = CARB7 + HCHO + HO2 + NO2                            
#          RC(226) = KRO2NO3*0.2         
# C
# C     Reaction (227) NRN6O2 + NO3 = HCHO + HCHO + NO2 + NO2                             
#          RC(227) = KRO2NO3               
# C
# C     Reaction (228) NRN9O2 + NO3 = CH3CHO + HCHO + NO2 + NO2                           
#          RC(228) = KRO2NO3               
# C
# C     Reaction (229) NRN12O2 + NO3 = CH3CHO + CH3CHO + NO2 + NO2                        
#          RC(229) = KRO2NO3               
# C
# C     Reaction (230) NRU14O2 + NO3 = NUCARB12 + HO2 + NO2                               
#          RC(230) = KRO2NO3               
# C
# C     Reaction (231) NRU12O2 + NO3 = NOA + CO + HO2 + NO2                               
#          RC(231) = KRO2NO3               
# C
# C     Reaction (232) RTN28O2 + NO3 = TNCARB26 + HO2 + NO2                               
#          RC(232) = KRO2NO3                
# C
# C     Reaction (233) NRTN28O2 + NO3 = TNCARB26 + NO2 + NO2                              
#          RC(233) = KRO2NO3                
# C
# C     Reaction (234) RTN26O2 + NO3 = RTN25O2 + NO2                                      
#          RC(234) = KRO2NO3*1.60                   
# C
# C     Reaction (235) RTN25O2 + NO3 = RTN24O2 + NO2                                      
#          RC(235) = KRO2NO3                 
# C
# C     Reaction (236) RTN24O2 + NO3 = RTN23O2 + NO2                                      
#          RC(236) = KRO2NO3                   
# C
# C     Reaction (237) RTN23O2 + NO3 = CH3COCH3 + RTN14O2 + NO2                           
#          RC(237) = KRO2NO3                 
# C
# C     Reaction (238) RTN14O2 + NO3 = HCHO + TNCARB10 + HO2 + NO2                        
#          RC(238) = KRO2NO3             
# C
# C     Reaction (239) RTN10O2 + NO3 = RN8O2 + CO + NO2                                   
#          RC(239) = KRO2NO3               
# C
# C     Reaction (240) RTX28O2 + NO3 = TXCARB24 + HCHO + HO2 + NO2                        
#          RC(240) = KRO2NO3             
# C
# C     Reaction (241) RTX24O2 + NO3 = TXCARB22 + HO2 + NO2                               
#          RC(241) = KRO2NO3             
# C
# C     Reaction (242) RTX22O2 + NO3 = CH3COCH3 + RN13O2 + NO2                            
#          RC(242) = KRO2NO3             
# C
# C     Reaction (243) NRTX28O2 + NO3 = TXCARB24 + HCHO + NO2 + NO2                       
#          RC(243) = KRO2NO3            
# C
# C     Reaction (244) CH3O2 + HO2 = CH3OOH                                               
#          RC(244) = 4.10D-13*math.exp(790/TEMP)  
# C
# C     Reaction (245) C2H5O2 + HO2 = C2H5OOH                                             
#          RC(245) = 7.50D-13*math.exp(700/TEMP)  
# C
# C     Reaction (246) RN10O2 + HO2 = RN10OOH                                             
#          RC(246) = KRO2HO2*0.520           
# C
# C     Reaction (247) IC3H7O2 + HO2 = IC3H7OOH                                           
#          RC(247) = KRO2HO2*0.520           
# C
# C     Reaction (248) RN13O2 + HO2 = RN13OOH                                             
#          RC(248) = KRO2HO2*0.625           
# C
# C     Reaction (249) RN16O2 + HO2 = RN16OOH                                             
#          RC(249) = KRO2HO2*0.706           
# C
# C     Reaction (250) RN19O2 + HO2 = RN19OOH                                             
#          RC(250) = KRO2HO2*0.770           
# C
# C     Reaction (251) RN13AO2 + HO2 = RN13OOH                                            
#          RC(251) = KRO2HO2*0.625           
# C
# C     Reaction (252) RN16AO2 + HO2 = RN16OOH                                            
#          RC(252) = KRO2HO2*0.706           
# C
# C     Reaction (253) RA13O2 + HO2 = RA13OOH                                             
#          RC(253) = KRO2HO2*0.770           
# C
# C     Reaction (254) RA16O2 + HO2 = RA16OOH                                             
#          RC(254) = KRO2HO2*0.820           
# C
# C     Reaction (255) RA19AO2 + HO2 = RA19OOH                                            
#          RC(255) = KRO2HO2*0.859           
# C
# C     Reaction (256) RA19CO2 + HO2 = RA19OOH                                            
#          RC(256) = KRO2HO2*0.859           
# C
# C     Reaction (257) HOCH2CH2O2 + HO2 = HOC2H4OOH                                       
#          RC(257) = 2.03D-13*math.exp(1250/TEMP) 
# C
# C     Reaction (258) RN9O2 + HO2 = RN9OOH                                               
#          RC(258) = KRO2HO2*0.520           
# C
# C     Reaction (259) RN12O2 + HO2 = RN12OOH                                             
#          RC(259) = KRO2HO2*0.625           
# C
# C     Reaction (260) RN15O2 + HO2 = RN15OOH                                             
#          RC(260) = KRO2HO2*0.706           
# C
# C     Reaction (261) RN18O2 + HO2 = RN18OOH                                             
#          RC(261) = KRO2HO2*0.770           
# C
# C     Reaction (262) RN15AO2 + HO2 = RN15OOH                                            
#          RC(262) = KRO2HO2*0.706           
# C
# C     Reaction (263) RN18AO2 + HO2 = RN18OOH                                            
#          RC(263) = KRO2HO2*0.770           
# C
# C     Reaction (264) CH3CO3 + HO2 = CH3CO3H                                             
#          RC(264) = KAPHO2                  
# C
# C     Reaction (265) C2H5CO3 + HO2 = C2H5CO3H                                           
#          RC(265) = KAPHO2                  
# C
# C     Reaction (266) HOCH2CO3 + HO2 = HOCH2CO3H                                         
#          RC(266) = KAPHO2                  
# C
# C     Reaction (267) RN8O2 + HO2 = RN8OOH                                               
#          RC(267) = KRO2HO2*0.520           
# C
# C     Reaction (268) RN11O2 + HO2 = RN11OOH                                             
#          RC(268) = KRO2HO2*0.625           
# C
# C     Reaction (269) RN14O2 + HO2 = RN14OOH                                             
#          RC(269) = KRO2HO2*0.706           
# C
# C     Reaction (270) RN17O2 + HO2 = RN17OOH                                             
#          RC(270) = KRO2HO2*0.770           
# C
# C     Reaction (271) RU14O2 + HO2 = RU14OOH                                             
#          RC(271) = KRO2HO2*0.770           
# C
# C     Reaction (272) RU12O2 + HO2 = RU12OOH                                             
#          RC(272) = KRO2HO2*0.706           
# C
# C     Reaction (273) RU10O2 + HO2 = RU10OOH                                             
#          RC(273) = KRO2HO2*0.625           
# C
# C     Reaction (274) NRN6O2 + HO2 = NRN6OOH                                             
#          RC(274) = KRO2HO2*0.387         
# C
# C     Reaction (275) NRN9O2 + HO2 = NRN9OOH                                             
#          RC(275) = KRO2HO2*0.520         
# C
# C     Reaction (276) NRN12O2 + HO2 = NRN12OOH                                           
#          RC(276) = KRO2HO2*0.625         
# C
# C     Reaction (277) NRU14O2 + HO2 = NRU14OOH                                           
#          RC(277) = KRO2HO2*0.770         
# C
# C     Reaction (278) NRU12O2 + HO2 = NRU12OOH                                           
#          RC(278) = KRO2HO2*0.625         
# C
# C     Reaction (279) RTN28O2 + HO2 = RTN28OOH                                           
#          RC(279) = KRO2HO2*0.914         
# C
# C     Reaction (280) NRTN28O2 + HO2 = NRTN28OOH                                         
#          RC(280) = KRO2HO2*0.914         
# C
# C     Reaction (281) RTN26O2 + HO2 = RTN26OOH                                           
#          RC(281) = KAPHO2                     
# C
# C     Reaction (282) RTN25O2 + HO2 = RTN25OOH                                           
#          RC(282) = KRO2HO2*0.890       
# C
# C     Reaction (283) RTN24O2 + HO2 = RTN24OOH                                           
#          RC(283) = KRO2HO2*0.890       
# C
# C     Reaction (284) RTN23O2 + HO2 = RTN23OOH                                           
#          RC(284) = KRO2HO2*0.890       
# C
# C     Reaction (285) RTN14O2 + HO2 = RTN14OOH                                           
#          RC(285) = KRO2HO2*0.770       
# C
# C     Reaction (286) RTN10O2 + HO2 = RTN10OOH                                           
#          RC(286) = KRO2HO2*0.706       
# C
# C     Reaction (287) RTX28O2 + HO2 = RTX28OOH                                           
#          RC(287) = KRO2HO2*0.914       
# C
# C     Reaction (288) RTX24O2 + HO2 = RTX24OOH                                           
#          RC(288) = KRO2HO2*0.890       
# C
# C     Reaction (289) RTX22O2 + HO2 = RTX22OOH                                           
#          RC(289) = KRO2HO2*0.890       
# C
# C     Reaction (290) NRTX28O2 + HO2 = NRTX28OOH                                         
#          RC(290) = KRO2HO2*0.914       
# C
# C     Reaction (291) CH3O2 = HCHO + HO2                                                 
#          RC(291) = 1.82D-13*math.exp(416/TEMP)*0.33*RO2  
# C
# C     Reaction (292) CH3O2 = HCHO                                                       
#          RC(292) = 1.82D-13*math.exp(416/TEMP)*0.335*RO2 
# C
# C     Reaction (293) CH3O2 = CH3OH                                                      
#          RC(293) = 1.82D-13*math.exp(416/TEMP)*0.335*RO2 
# C
# C     Reaction (294) C2H5O2 = CH3CHO + HO2                                              
#          RC(294) = 3.10D-13*0.6*RO2             
# C
# C     Reaction (295) C2H5O2 = CH3CHO                                                    
#          RC(295) = 3.10D-13*0.2*RO2             
# C
# C     Reaction (296) C2H5O2 = C2H5OH                                                    
#          RC(296) = 3.10D-13*0.2*RO2             
# C
# C     Reaction (297) RN10O2 = C2H5CHO + HO2                                             
#          RC(297) = 6.00D-13*0.6*RO2             
# C
# C     Reaction (298) RN10O2 = C2H5CHO                                                   
#          RC(298) = 6.00D-13*0.2*RO2             
# C
# C     Reaction (299) RN10O2 = NPROPOL                                                   
#          RC(299) = 6.00D-13*0.2*RO2             
# C
# C     Reaction (300) IC3H7O2 = CH3COCH3 + HO2                                           
#          RC(300) = 4.00D-14*0.6*RO2             
# C
# C     Reaction (301) IC3H7O2 = CH3COCH3                                                 
#          RC(301) = 4.00D-14*0.2*RO2             
# C
# C     Reaction (302) IC3H7O2 = IPROPOL                                                  
#          RC(302) = 4.00D-14*0.2*RO2             
# C
# C     Reaction (303) RN13O2 = CH3CHO + C2H5O2                                           
#          RC(303) = 2.50D-13*RO2*BR01       
# C
# C     Reaction (304) RN13O2 = CARB11A + HO2                                             
#          RC(304) = 2.50D-13*RO2*(1-BR01)   
# C
# C     Reaction (305) RN13AO2 = RN12O2                                                   
#          RC(305) = 8.80D-13*RO2                 
# C
# C     Reaction (306) RN16AO2 = RN15O2                                                   
#          RC(306) = 8.80D-13*RO2                 
# C
# C     Reaction (307) RA13O2 = CARB3 + UDCARB8 + HO2                                     
#          RC(307) = 8.80D-13*RO2                 
# C
# C     Reaction (308) RA16O2 = CARB3 + UDCARB11 + HO2                                    
#          RC(308) = 8.80D-13*RO2*0.7          
# C
# C     Reaction (309) RA16O2 = CARB6 + UDCARB8 + HO2                                     
#          RC(309) = 8.80D-13*RO2*0.3          
# C
# C     Reaction (310) RA19AO2 = CARB3 + UDCARB14 + HO2                                   
#          RC(310) = 8.80D-13*RO2                 
# C
# C     Reaction (311) RA19CO2 = CARB3 + UDCARB14 + HO2                                   
#          RC(311) = 8.80D-13*RO2                 
# C
# C     Reaction (312) RN16O2 = RN15AO2                                                   
#          RC(312) = 2.50D-13*RO2                 
# C
# C     Reaction (313) RN19O2 = RN18AO2                                                   
#          RC(313) = 2.50D-13*RO2                 
# C
# C     Reaction (314) HOCH2CH2O2 = HCHO + HCHO + HO2                                     
#          RC(314) = 2.00D-12*RO2*0.776       
# C
# C     Reaction (315) HOCH2CH2O2 = HOCH2CHO + HO2                                        
#          RC(315) = 2.00D-12*RO2*0.224       
# C
# C     Reaction (316) RN9O2 = CH3CHO + HCHO + HO2                                        
#          RC(316) = 8.80D-13*RO2                 
# C
# C     Reaction (317) RN12O2 = CH3CHO + CH3CHO + HO2                                     
#          RC(317) = 8.80D-13*RO2                 
# C
# C     Reaction (318) RN15O2 = C2H5CHO + CH3CHO + HO2                                    
#          RC(318) = 8.80D-13*RO2                 
# C
# C     Reaction (319) RN18O2 = C2H5CHO + C2H5CHO + HO2                                   
#          RC(319) = 8.80D-13*RO2                 
# C
# C     Reaction (320) RN15AO2 = CARB13 + HO2                                             
#          RC(320) = 8.80D-13*RO2                 
# C
# C     Reaction (321) RN18AO2 = CARB16 + HO2                                             
#          RC(321) = 8.80D-13*RO2                 
# C
# C     Reaction (322) CH3CO3 = CH3O2                                                     
#          RC(322) = 1.00D-11*RO2                 
# C
# C     Reaction (323) C2H5CO3 = C2H5O2                                                   
#          RC(323) = 1.00D-11*RO2                 
# C
# C     Reaction (324) HOCH2CO3 = HCHO + HO2                                              
#          RC(324) = 1.00D-11*RO2                 
# C
# C     Reaction (325) RN8O2 = CH3CO3 + HCHO                                              
#          RC(325) = 1.40D-12*RO2                 
# C
# C     Reaction (326) RN11O2 = CH3CO3 + CH3CHO                                           
#          RC(326) = 1.40D-12*RO2                 
# C
# C     Reaction (327) RN14O2 = C2H5CO3 + CH3CHO                                          
#          RC(327) = 1.40D-12*RO2                 
# C
# C     Reaction (328) RN17O2 = RN16AO2                                                   
#          RC(328) = 1.40D-12*RO2                 
# C
# C     Reaction (329) RU14O2 = UCARB12 + HO2                                             
#          RC(329) = 1.71D-12*RO2*0.252        
# C
# C     Reaction (330) RU14O2 = UCARB10 + HCHO + HO2                                      
#          RC(330) = 1.71D-12*RO2*0.748        
# C
# C     Reaction (331) RU12O2 = CH3CO3 + HOCH2CHO                                         
#          RC(331) = 2.00D-12*RO2*0.7            
# C
# C     Reaction (332) RU12O2 = CARB7 + HOCH2CHO + HO2                                    
#          RC(332) = 2.00D-12*RO2*0.3            
# C
# C     Reaction (333) RU10O2 = CH3CO3 + HOCH2CHO                                         
#          RC(333) = 2.00D-12*RO2*0.5            
# C
# C     Reaction (334) RU10O2 = CARB6 + HCHO + HO2                                        
#          RC(334) = 2.00D-12*RO2*0.3            
# C
# C     Reaction (335) RU10O2 = CARB7 + HCHO + HO2                                        
#          RC(335) = 2.00D-12*RO2*0.2            
# C
# C     Reaction (336) NRN6O2 = HCHO + HCHO + NO2                                         
#          RC(336) = 6.00D-13*RO2                 
# C
# C     Reaction (337) NRN9O2 = CH3CHO + HCHO + NO2                                       
#          RC(337) = 2.30D-13*RO2                 
# C
# C     Reaction (338) NRN12O2 = CH3CHO + CH3CHO + NO2                                    
#          RC(338) = 2.50D-13*RO2                 
# C
# C     Reaction (339) NRU14O2 = NUCARB12 + HO2                                           
#          RC(339) = 1.30D-12*RO2                 
# C
# C     Reaction (340) NRU12O2 = NOA + CO + HO2                                           
#          RC(340) = 9.60D-13*RO2                 
# C
# C     Reaction (341) RTN28O2 = TNCARB26 + HO2                                           
#          RC(341) = 2.85D-13*RO2                 
# C
# C     Reaction (342) NRTN28O2 = TNCARB26 + NO2                                          
#          RC(342) = 1.00D-13*RO2                 
# C
# C     Reaction (343) RTN26O2 = RTN25O2                                                  
#          RC(343) = 1.00D-11*RO2                   
# C
# C     Reaction (344) RTN25O2 = RTN24O2                                                  
#          RC(344) = 1.30D-12*RO2           
# C
# C     Reaction (345) RTN24O2 = RTN23O2                                                  
#          RC(345) = 6.70D-15*RO2             
# C
# C     Reaction (346) RTN23O2 = CH3COCH3 + RTN14O2                                       
#          RC(346) = 6.70D-15*RO2            
# C
# C     Reaction (347) RTN14O2 = HCHO + TNCARB10 + HO2                                    
#          RC(347) = 8.80D-13*RO2        
# C
# C     Reaction (348) RTN10O2 = RN8O2 + CO                                               
#          RC(348) = 2.00D-12*RO2        
# C
# C     Reaction (349) RTX28O2 = TXCARB24 + HCHO + HO2                                    
#          RC(349) = 2.00D-12*RO2       
# C
# C     Reaction (350) RTX24O2 = TXCARB22 + HO2                                           
#          RC(350) = 2.50D-13*RO2       
# C
# C     Reaction (351) RTX22O2 = CH3COCH3 + RN13O2                                        
#          RC(351) = 2.50D-13*RO2       
# C
# C     Reaction (352) NRTX28O2 = TXCARB24 + HCHO + NO2                                   
#          RC(352) = 9.20D-14*RO2       
# C
# C     Reaction (353) OH + CARB14 = RN14O2                                               
#          RC(353) = 1.87D-11       
# C
# C     Reaction (354) OH + CARB17 = RN17O2                                               
#          RC(354) = 4.36D-12       
# C
# C     Reaction (355) OH + CARB11A = RN11O2                                              
#          RC(355) = 3.24D-18*TEMP**2*math.exp(414/TEMP)
# C
# C     Reaction (356) OH + CARB7 = CARB6 + HO2                                           
#          RC(356) = 3.00D-12       
# C
# C     Reaction (357) OH + CARB10 = CARB9 + HO2                                          
#          RC(357) = 5.86D-12       
# C
# C     Reaction (358) OH + CARB13 = RN13O2                                               
#          RC(358) = 1.65D-11       
# C
# C     Reaction (359) OH + CARB16 = RN16O2                                               
#          RC(359) = 1.25D-11       
# C
# C     Reaction (360) OH + UCARB10 = RU10O2                                              
#          RC(360) = 2.50D-11       
# C
# C     Reaction (361) NO3 + UCARB10 = RU10O2 + HNO3                                      
#          RC(361) = KNO3AL       
# C
# C     Reaction (362) O3 + UCARB10 = HCHO + CH3CO3 + CO + OH                             
#          RC(362) = 2.85D-18*0.59       
# C
# C     Reaction (363) O3 + UCARB10 = HCHO + CARB6 + H2O2                                 
#          RC(363) = 2.85D-18*0.41       
# C
# C     Reaction (364) OH + HOCH2CHO = HOCH2CO3                                           
#          RC(364) = 1.00D-11       
# C
# C     Reaction (365) NO3 + HOCH2CHO = HOCH2CO3 + HNO3                                   
#          RC(365) = KNO3AL        
# C
# C     Reaction (366) OH + CARB3 = CO + CO + HO2                                         
#          RC(366) = 1.14D-11       
# C
# C     Reaction (367) OH + CARB6 = CH3CO3 + CO                                           
#          RC(367) = 1.72D-11       
# C
# C     Reaction (368) OH + CARB9 = RN9O2                                                 
#          RC(368) = 2.40D-13       
# C
# C     Reaction (369) OH + CARB12 = RN12O2                                               
#          RC(369) = 1.38D-12       
# C
# C     Reaction (370) OH + CARB15 = RN15O2                                               
#          RC(370) = 4.81D-12       
# C
# C     Reaction (371) OH + CCARB12 = RN12O2                                              
#          RC(371) = 4.79D-12       
# C
# C     Reaction (372) OH + UCARB12 = RU12O2                                              
#          RC(372) = 4.52D-11            
# C
# C     Reaction (373) NO3 + UCARB12 = RU12O2 + HNO3                                      
#          RC(373) = KNO3AL*4.25    
# C
# C     Reaction (374) O3 + UCARB12 = HOCH2CHO + CH3CO3 + CO + OH                         
#          RC(374) = 2.40D-17*0.89   
# C
# C     Reaction (375) O3 + UCARB12 = HOCH2CHO + CARB6 + H2O2                             
#          RC(375) = 2.40D-17*0.11   
# C
# C     Reaction (376) OH + NUCARB12 = NRU12O2                                            
#          RC(376) = 4.16D-11            
# C
# C     Reaction (377) OH + NOA = CARB6 + NO2                                             
#          RC(377) = 1.30D-13            
# C
# C     Reaction (378) OH + UDCARB8 = C2H5O2                                              
#          RC(378) = 5.20D-11*0.50        
# C
# C     Reaction (379) OH + UDCARB8 = ANHY + HO2                                          
#          RC(379) = 5.20D-11*0.50        
# C
# C     Reaction (380) OH + UDCARB11 = RN10O2                                             
#          RC(380) = 5.58D-11*0.55     
# C
# C     Reaction (381) OH + UDCARB11 = ANHY + CH3O2                                       
#          RC(381) = 5.58D-11*0.45     
# C
# C     Reaction (382) OH + UDCARB14 = RN13O2                                             
#          RC(382) = 7.00D-11*0.55     
# C
# C     Reaction (383) OH + UDCARB14 = ANHY + C2H5O2                                      
#          RC(383) = 7.00D-11*0.45     
# C
# C     Reaction (384) OH + TNCARB26 = RTN26O2                                            
#          RC(384) = 4.20D-11           
# C
# C     Reaction (385) OH + TNCARB15 = RN15AO2                                            
#          RC(385) = 1.00D-12           
# C
# C     Reaction (386) OH + TNCARB10 = RTN10O2                                            
#          RC(386) = 1.00D-10           
# C
# C     Reaction (387) NO3 + TNCARB26 = RTN26O2 + HNO3                                    
#          RC(387) = 3.80D-14            
# C
# C     Reaction (388) NO3 + TNCARB10 = RTN10O2 + HNO3                                    
#          RC(388) = KNO3AL*5.5      
# C
# C     Reaction (389) OH + RCOOH25 = RTN25O2                                             
#          RC(389) = 6.65D-12            
# C
# C     Reaction (390) OH + TXCARB24 = RTX24O2                                            
#          RC(390) = 1.55D-11           
# C
# C     Reaction (391) OH + TXCARB22 = RTX22O2                                            
#          RC(391) = 4.55D-12           
# C
# C     Reaction (392) OH + CH3NO3 = HCHO + NO2                                           
#          RC(392) = 1.00D-14*math.exp(1060/TEMP)      
# C
# C     Reaction (393) OH + C2H5NO3 = CH3CHO + NO2                                        
#          RC(393) = 4.40D-14*math.exp(720/TEMP)       
# C
# C     Reaction (394) OH + RN10NO3 = C2H5CHO + NO2                                       
#          RC(394) = 7.30D-13                     
# C
# C     Reaction (395) OH + IC3H7NO3 = CH3COCH3 + NO2                                     
#          RC(395) = 4.90D-13                     
# C
# C     Reaction (396) OH + RN13NO3 = CARB11A + NO2                                       
#          RC(396) = 9.20D-13                     
# C
# C     Reaction (397) OH + RN16NO3 = CARB14 + NO2                                        
#          RC(397) = 1.85D-12                     
# C
# C     Reaction (398) OH + RN19NO3 = CARB17 + NO2                                        
#          RC(398) = 3.02D-12                     
# C
# C     Reaction (399) OH + HOC2H4NO3 = HOCH2CHO + NO2                                    
#          RC(399) = 1.09D-12               
# C
# C     Reaction (400) OH + RN9NO3 = CARB7 + NO2                                          
#          RC(400) = 1.31D-12               
# C
# C     Reaction (401) OH + RN12NO3 = CARB10 + NO2                                        
#          RC(401) = 1.79D-12               
# C
# C     Reaction (402) OH + RN15NO3 = CARB13 + NO2                                        
#          RC(402) = 1.03D-11               
# C
# C     Reaction (403) OH + RN18NO3 = CARB16 + NO2                                        
#          RC(403) = 1.34D-11               
# C
# C     Reaction (404) OH + RU14NO3 = UCARB12 + NO2                                       
#          RC(404) = 5.55D-11               
# C
# C     Reaction (405) OH + RA13NO3 = CARB3 + UDCARB8 + NO2                               
#          RC(405) = 7.30D-11               
# C
# C     Reaction (406) OH + RA16NO3 = CARB3 + UDCARB11 + NO2                              
#          RC(406) = 7.16D-11               
# C
# C     Reaction (407) OH + RA19NO3 = CARB6 + UDCARB11 + NO2                              
#          RC(407) = 8.31D-11               
# C
# C     Reaction (408) OH + RTN28NO3 = TNCARB26 + NO2                                     
#          RC(408) = 4.35D-12               
# C
# C     Reaction (409) OH + RTN25NO3 = CH3COCH3 + TNCARB15 + NO2                          
#          RC(409) = 2.88D-12               
# C
# C     Reaction (410) OH + RTX28NO3 = TXCARB24 + HCHO + NO2                              
#          RC(410) = 3.53D-12                  
# C
# C     Reaction (411) OH + RTX24NO3 = TXCARB22 + NO2                                     
#          RC(411) = 6.48D-12                  
# C
# C     Reaction (412) OH + RTX22NO3 = CH3COCH3 + CCARB12 + NO2                           
#          RC(412) = 4.74D-12                  
# C
# C     Reaction (413) OH + AROH14 = RAROH14                                              
#          RC(413) = 2.63D-11             
# C
# C     Reaction (414) NO3 + AROH14 = RAROH14 + HNO3                                      
#          RC(414) = 3.78D-12               
# C
# C     Reaction (415) RAROH14 + NO2 = ARNOH14                                            
#          RC(415) = 2.08D-12               
# C
# C     Reaction (416) OH + ARNOH14 = CARB13 + NO2                                        
#          RC(416) = 9.00D-13               
# C
# C     Reaction (417) NO3 + ARNOH14 = CARB13 + NO2 + HNO3                                
#          RC(417) = 9.00D-14               
# C
# C     Reaction (418) OH + AROH17 = RAROH17                                              
#          RC(418) = 4.65D-11               
# C
# C     Reaction (419) NO3 + AROH17 = RAROH17 + HNO3                                      
#          RC(419) = 1.25D-11               
# C
# C     Reaction (420) RAROH17 + NO2 = ARNOH17                                            
#          RC(420) = 2.08D-12               
# C
# C     Reaction (421) OH + ARNOH17 = CARB16 + NO2                                        
#          RC(421) = 1.53D-12               
# C
# C     Reaction (422) NO3 + ARNOH17 = CARB16 + NO2 + HNO3                                
#          RC(422) = 3.13D-13               
# C
# C     Reaction (423) OH + CH3OOH = CH3O2                                                
#          RC(423) = 1.90D-11*math.exp(190/TEMP)       
# C
# C     Reaction (424) OH + CH3OOH = HCHO + OH                                            
#          RC(424) = 1.00D-11*math.exp(190/TEMP)       
# C
# C     Reaction (425) OH + C2H5OOH = CH3CHO + OH                                         
#          RC(425) = 1.36D-11               
# C
# C     Reaction (426) OH + RN10OOH = C2H5CHO + OH                                        
#          RC(426) = 1.89D-11               
# C
# C     Reaction (427) OH + IC3H7OOH = CH3COCH3 + OH                                      
#          RC(427) = 2.78D-11               
# C
# C     Reaction (428) OH + RN13OOH = CARB11A + OH                                        
#          RC(428) = 3.57D-11               
# C
# C     Reaction (429) OH + RN16OOH = CARB14 + OH                                         
#          RC(429) = 4.21D-11               
# C
# C     Reaction (430) OH + RN19OOH = CARB17 + OH                                         
#          RC(430) = 4.71D-11               
# C
# C     Reaction (431) OH + CH3CO3H = CH3CO3                                              
#          RC(431) = 3.70D-12                     
# C
# C     Reaction (432) OH + C2H5CO3H = C2H5CO3                                            
#          RC(432) = 4.42D-12                     
# C
# C     Reaction (433) OH + HOCH2CO3H = HOCH2CO3                                          
#          RC(433) = 6.19D-12                     
# C
# C     Reaction (434) OH + RN8OOH = CARB6 + OH                                           
#          RC(434) = 4.42D-12                     
# C
# C     Reaction (435) OH + RN11OOH = CARB9 + OH                                          
#          RC(435) = 2.50D-11                     
# C
# C     Reaction (436) OH + RN14OOH = CARB12 + OH                                         
#          RC(436) = 3.20D-11                     
# C
# C     Reaction (437) OH + RN17OOH = CARB15 + OH                                         
#          RC(437) = 3.35D-11                     
# C
# C     Reaction (438) OH + RU14OOH = UCARB12 + OH                                        
#          RC(438) = 7.51D-11                     
# C
# C     Reaction (439) OH + RU12OOH = RU12O2                                              
#          RC(439) = 3.00D-11                     
# C
# C     Reaction (440) OH + RU10OOH = RU10O2                                              
#          RC(440) = 3.00D-11                     
# C
# C     Reaction (441) OH + NRU14OOH = NUCARB12 + OH                                      
#          RC(441) = 1.03D-10                     
# C
# C     Reaction (442) OH + NRU12OOH = NOA + CO + OH                                      
#          RC(442) = 2.65D-11                     
# C
# C     Reaction (443) OH + HOC2H4OOH = HOCH2CHO + OH                                     
#          RC(443) = 2.13D-11               
# C
# C     Reaction (444) OH + RN9OOH = CARB7 + OH                                           
#          RC(444) = 2.50D-11               
# C
# C     Reaction (445) OH + RN12OOH = CARB10 + OH                                         
#          RC(445) = 3.25D-11               
# C
# C     Reaction (446) OH + RN15OOH = CARB13 + OH                                         
#          RC(446) = 3.74D-11               
# C
# C     Reaction (447) OH + RN18OOH = CARB16 + OH                                         
#          RC(447) = 3.83D-11               
# C
# C     Reaction (448) OH + NRN6OOH = HCHO + HCHO + NO2 + OH                              
#          RC(448) = 5.22D-12               
# C
# C     Reaction (449) OH + NRN9OOH = CH3CHO + HCHO + NO2 + OH                            
#          RC(449) = 6.50D-12               
# C
# C     Reaction (450) OH + NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                         
#          RC(450) = 7.15D-12               
# C
# C     Reaction (451) OH + RA13OOH = CARB3 + UDCARB8 + OH                                
#          RC(451) = 9.77D-11               
# C
# C     Reaction (452) OH + RA16OOH = CARB3 + UDCARB11 + OH                               
#          RC(452) = 9.64D-11               
# C
# C     Reaction (453) OH + RA19OOH = CARB6 + UDCARB11 + OH                               
#          RC(453) = 1.12D-10               
# C
# C     Reaction (454) OH + RTN28OOH = TNCARB26 + OH                                      
#          RC(454) = 2.38D-11               
# C
# C     Reaction (455) OH + RTN26OOH = RTN26O2                                            
#          RC(455) = 1.20D-11               
# C
# C     Reaction (456) OH + NRTN28OOH = TNCARB26 + NO2 + OH                               
#          RC(456) = 9.50D-12               
# C
# C     Reaction (457) OH + RTN25OOH = RTN25O2                                            
#          RC(457) = 1.66D-11               
# C
# C     Reaction (458) OH + RTN24OOH = RTN24O2                                            
#          RC(458) = 1.05D-11               
# C
# C     Reaction (459) OH + RTN23OOH = RTN23O2                                            
#          RC(459) = 2.05D-11               
# C
# C     Reaction (460) OH + RTN14OOH = RTN14O2                                            
#          RC(460) = 8.69D-11               
# C
# C     Reaction (461) OH + RTN10OOH = RTN10O2                                            
#          RC(461) = 4.23D-12               
# C
# C     Reaction (462) OH + RTX28OOH = RTX28O2                                            
#          RC(462) = 2.00D-11               
# C
# C     Reaction (463) OH + RTX24OOH = TXCARB22 + OH                                      
#          RC(463) = 8.59D-11               
# C
# C     Reaction (464) OH + RTX22OOH = CH3COCH3 + CCARB12 + OH                            
#          RC(464) = 7.50D-11               
# C
# C     Reaction (465) OH + NRTX28OOH = NRTX28O2                                          
#          RC(465) = 9.58D-12               
# C
# C     Reaction (466) OH + ANHY = HOCH2CH2O2                                             
#          RC(466) = 1.50D-12        
# C
# C     Reaction (467) CH3CO3 + NO2 = PAN                                                 
#          RC(467) = KFPAN                        
# C
# C     Reaction (468) PAN = CH3CO3 + NO2                                                 
#          RC(468) = KBPAN                        
# C
# C     Reaction (469) C2H5CO3 + NO2 = PPN                                                
#          RC(469) = KFPAN                        
# C
# C     Reaction (470) PPN = C2H5CO3 + NO2                                                
#          RC(470) = KBPAN                        
# C
# C     Reaction (471) HOCH2CO3 + NO2 = PHAN                                              
#          RC(471) = KFPAN                        
# C
# C     Reaction (472) PHAN = HOCH2CO3 + NO2                                              
#          RC(472) = KBPAN                        
# C
# C     Reaction (473) OH + PAN = HCHO + CO + NO2                                         
#          RC(473) = 9.50D-13*math.exp(-650/TEMP)      
# C
# C     Reaction (474) OH + PPN = CH3CHO + CO + NO2                                       
#          RC(474) = 1.27D-12                       
# C
# C     Reaction (475) OH + PHAN = HCHO + CO + NO2                                        
#          RC(475) = 1.12D-12                       
# C
# C     Reaction (476) RU12O2 + NO2 = RU12PAN                                             
#          RC(476) = KFPAN*0.061             
# C
# C     Reaction (477) RU12PAN = RU12O2 + NO2                                             
#          RC(477) = KBPAN                   
# C
# C     Reaction (478) RU10O2 + NO2 = MPAN                                                
#          RC(478) = KFPAN*0.041             
# C
# C     Reaction (479) MPAN = RU10O2 + NO2                                                
#          RC(479) = KBPAN                  
# C
# C     Reaction (480) OH + MPAN = CARB7 + CO + NO2                                       
#          RC(480) = 3.60D-12 
# C
# C     Reaction (481) OH + RU12PAN = UCARB10 + NO2                                       
#          RC(481) = 2.52D-11 
# C
# C     Reaction (482) RTN26O2 + NO2 = RTN26PAN                                           
#          RC(482) = KFPAN*0.722      
# C
# C     Reaction (483) RTN26PAN = RTN26O2 + NO2                                           
#          RC(483) = KBPAN                   
# C
# C     Reaction (484) OH + RTN26PAN = CH3COCH3 + CARB16 + NO2                            
#          RC(484) = 3.66D-12  
# C
# C     Reaction (485) RTN28NO3 = P2604                                                   
#          RC(485) = KIN  		
# C
# C     Reaction (486) P2604 = RTN28NO3                                                   
#          RC(486) = KOUT2604 	
# C
# C     Reaction (487) RTX28NO3 = P4608                                                   
#          RC(487) = KIN 		
# C
# C     Reaction (488) P4608 = RTX28NO3                                                   
#          RC(488) = KOUT4608 	
# C
# C     Reaction (489) RCOOH25 = P2631                                                    
#          RC(489) = KIN  		
# C
# C     Reaction (490) P2631 = RCOOH25                                                    
#          RC(490) = KOUT2631 	
# C
# C     Reaction (491) RTN24OOH = P2635                                                   
#          RC(491) = KIN  		
# C
# C     Reaction (492) P2635 = RTN24OOH                                                   
#          RC(492) = KOUT2635 	
# C
# C     Reaction (493) RTX28OOH = P4610                                                   
#          RC(493) = KIN  		
# C
# C     Reaction (494) P4610 = RTX28OOH                                                   
#          RC(494) = KOUT4610 	
# C
# C     Reaction (495) RTN28OOH = P2605                                                   
#          RC(495) = KIN  		
# C
# C     Reaction (496) P2605 = RTN28OOH                                                   
#          RC(496) = KOUT2605 	
# C
# C     Reaction (497) RTN26OOH = P2630                                                   
#          RC(497) = KIN  		
# C
# C     Reaction (498) P2630 = RTN26OOH                                                   
#          RC(498) = KOUT2630 	
# C
# C     Reaction (499) RTN26PAN = P2629                                                   
#          RC(499) = KIN  		
# C
# C     Reaction (500) P2629 = RTN26PAN                                                   
#          RC(500) = KOUT2629 	
# C
# C     Reaction (501) RTN25OOH = P2632                                                   
#          RC(501) = KIN 		
# C
# C     Reaction (502) P2632 = RTN25OOH                                                   
#          RC(502) = KOUT2632 	
# C
# C     Reaction (503) RTN23OOH = P2637                                                   
#          RC(503) = KIN  		
# C
# C     Reaction (504) P2637 = RTN23OOH                                                   
#          RC(504) = KOUT2637 	
# C
# C     Reaction (505) ARNOH14 = P3612                                                    
#          RC(505) = KIN  		
# C
# C     Reaction (506) P3612 = ARNOH14                                                    
#          RC(506) = KOUT3612 	
# C
# C     Reaction (507) ARNOH17 = P3613                                                    
#          RC(507) = KIN 		
# C
# C     Reaction (508) P3613 = ARNOH17                                                    
#          RC(508) = KOUT3613 	
# C
# C     Reaction (509) ANHY = P3442                                                       
#          RC(509) = KIN  		
# C
# C     Reaction (510) P3442 = ANHY                                                       
#          RC(510) = KOUT3442 	
# C

# C
#   999 RETURN
#       END

