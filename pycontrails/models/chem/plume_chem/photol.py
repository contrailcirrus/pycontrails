import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pycontrails import MetDataset, MetDataArray, MetVariable
from pycontrails.physics import geo, thermo, units

# Calculate photolysis rate coeffs for deriv

# INPUTS: J(#), BR01
# OUTPUTS: DJ(#)

# e.g: DJ(46) = J(53)*(1-BR01)

def photol(chem: MetDataset):
    # PHOTOLYSIS PARAMETERS IN FORMAT J = L*COSX**M*EXP(-N*SECX)  
    consts = pd.read_pickle('J.pkl')
    
    # Extract the constants
    idx, L, M, N = np.array(consts).T


    # Reshape sza to have a third dimension
    sza_J = chem.sza.expand_dims(dim={'photol_params': idx}, axis=3)
    print(sza_J[:, :, 0, 1])
    # If sza is greater than 90 degrees, set J to zero
    condition = (np.radians(sza_J) < np.pi / 2)
    J = np.where(condition, L * (np.cos(np.radians(sza_J))**M) * np.exp(-N / np.cos(np.radians(sza_J))), 0)
   
    
    # Calculate photol rate coeffs DJ
    
    

    # #    Photol Reaction (1) O3 = O1D                                                           
    # DJ[1] = J[1]                             

    # #    Photol Reaction (2) O3 = O                                                             
    # DJ[2] = J[2]                             

    # #    Photol Reaction (3) H2O2 = OH + OH                                                     
    # DJ[3] = J[3]                             
     
    # #    Photol Reaction (4) NO2 = NO + O                                                       
    # DJ[4] = J[4]                             
     
    # #    Photol Reaction (5) NO3 = NO                                                           
    # DJ[5] = J[5]                             
     
    # #    Photol Reaction (6) NO3 = NO2 + O                                                      
    # DJ[6] = J[6]                             
     
    # #    Photol Reaction (7) HONO = OH + NO                                                     
    # DJ[7] = J[7]                             
     
    # #    Photol Reaction (8) HNO3 = OH + NO2                                                    
    # DJ[8] = J[8]                             
     
    # #    Photol Reaction (9) HCHO = CO + HO2 + HO2                                              
    # DJ[9] = J[11]                        
     
    # #    Photol Reaction (10) HCHO = H2 + CO                                                     
    # DJ[10] = J[12]                        
     
    # #    Photol Reaction (11) CH3CHO = CH3O2 + HO2 + CO                                          
    # DJ[11] = J[13]                        
     
    # #    Photol Reaction (12) C2H5CHO = C2H5O2 + CO + HO2                                        
    # DJ[12] = J[14]                        
     
    # #    Photol Reaction (13) CH3COCH3 = CH3CO3 + CH3O2                                          
    # DJ[13] = J[21]                        
     
    # #    Photol Reaction (14) MEK = CH3CO3 + C2H5O2                                              
    # DJ[14] = J[22]                        
     
    # #    Photol Reaction (15) CARB14 = CH3CO3 + RN10O2                                           
    # DJ[15] = J[22]*4.74               
     
    # #    Photol Reaction (16) CARB17 = RN8O2 + RN10O2                                            
    # DJ[16] = J[22]*1.33               
     
    # #    Photol Reaction (17) CARB11A = CH3CO3 + C2H5O2                                          
    # DJ[17] = J[22]                        
     
    # #    Photol Reaction (18) CARB7 = CH3CO3 + HCHO + HO2                                        
    # DJ[18] = J[22]                        
     
    # #    Photol Reaction (19) CARB10 = CH3CO3 + CH3CHO + HO2                                     
    # DJ[19] = J[22]                        
     
    # #    Photol Reaction (20) CARB13 = RN8O2 + CH3CHO + HO2                                      
    # DJ[20] = J[22]*3.00               
     
    # #    Photol Reaction (21) CARB16 = RN8O2 + C2H5CHO + HO2                                     
    # DJ[21] = J[22]*3.35               
     
    # #    Photol Reaction (22) HOCH2CHO = HCHO + CO + HO2 + HO2                                   
    # DJ[22] = J[15]                        
     
    # #    Photol Reaction (23) UCARB10 = CH3CO3 + HCHO + HO2                                      
    # DJ[23] = J[18]*2                       
     
    # #    Photol Reaction (24) CARB3 = CO + CO + HO2 + HO2                                        
    # DJ[24] = J[33]                        
     
    # #    Photol Reaction (25) CARB6 = CH3CO3 + CO + HO2                                          
    # DJ[25] = J[34]                        
     
    # #    Photol Reaction (26) CARB9 = CH3CO3 + CH3CO3                                            
    # DJ[26] = J[35]                        
     
    # #    Photol Reaction (27) CARB12 = CH3CO3 + RN8O2                                            
    # DJ[27] = J[35]                        
     
    # #    Photol Reaction (28) CARB15 = RN8O2 + RN8O2                                             
    # DJ[28] = J[35]                        
     
    # #    Photol Reaction (29) UCARB12 = CH3CO3 + HOCH2CHO + CO + HO2                             
    # DJ[29] = J[18]*2           
     
    # #    Photol Reaction (30) NUCARB12 = NOA + CO + CO + HO2 + HO2                               
    # DJ[30] = J[18]             
     
    # #    Photol Reaction (31) NOA = CH3CO3 + HCHO + NO2                                          
    # DJ[31] = J[56]             
     
    # #    Photol Reaction (32) NOA = CH3CO3 + HCHO + NO2                                          
    # DJ[32] = J[57]             
     
    # #    Photol Reaction (33) UDCARB8 = C2H5O2 + HO2                                             
    # DJ[33] = J[4]*0.02*0.64   
     
    # #    Photol Reaction (34) UDCARB8 = ANHY + HO2 + HO2                                         
    # DJ[34] = J[4]*0.02*0.36   
     
    # #    Photol Reaction (35) UDCARB11 = RN10O2 + HO2                                            
    # DJ[35] = J[4]*0.02*0.55   
     
    # #    Photol Reaction (36) UDCARB11 = ANHY + HO2 + CH3O2                                      
    # DJ[36] = J[4]*0.02*0.45   
     
    # #    Photol Reaction (37) UDCARB14 = RN13O2 + HO2                                            
    # DJ[37] = J[4]*0.02*0.55   
     
    # #    Photol Reaction (38) UDCARB14 = ANHY + HO2 + C2H5O2                                     
    # DJ[38] = J[4]*0.02*0.45   
     
    # #    Photol Reaction (39) TNCARB26 = RTN26O2 + HO2                                           
    # DJ[39] = J[15]             
     
    # #    Photol Reaction (40) TNCARB10 = CH3CO3 + CH3CO3 + CO                                    
    # DJ[40] = J[35]*0.5        
     
    # #    Photol Reaction (41) CH3NO3 = HCHO + HO2 + NO2                                          
    # DJ[41] = J[51]                        
     
    # #    Photol Reaction (42) C2H5NO3 = CH3CHO + HO2 + NO2                                       
    # DJ[42] = J[52]                        
     
    # #    Photol Reaction (43) RN10NO3 = C2H5CHO + HO2 + NO2                                      
    # DJ[43] = J[53]                        
     
    # #    Photol Reaction (44) IC3H7NO3 = CH3COCH3 + HO2 + NO2                                    
    # DJ[44] = J[54]                        
     
    # #    Photol Reaction (45) RN13NO3 =  CH3CHO + C2H5O2 + NO2                                   
    # DJ[45] = J[53]*BR01               
     
    # #    Photol Reaction (46) RN13NO3 =  CARB11A + HO2 + NO2                                     
    # DJ[46] = J[53]*(1-BR01)           
     
    # #    Photol Reaction (47) RN16NO3 = RN15O2 + NO2                                             
    # DJ[47] = J[53]                        
     
    # #    Photol Reaction (48) RN19NO3 = RN18O2 + NO2                                             
    # DJ[48] = J[53]                        
     
    # #    Photol Reaction (49) RA13NO3 = CARB3 + UDCARB8 + HO2 + NO2                              
    # DJ[49] = J[54]                    
     
    # #    Photol Reaction (50) RA16NO3 = CARB3 + UDCARB11 + HO2 + NO2                             
    # DJ[50] = J[54]                    
     
    # #    Photol Reaction (51) RA19NO3 = CARB6 + UDCARB11 + HO2 + NO2                             
    # DJ[51] = J[54]                    
     
    # #    Photol Reaction (52) RTX24NO3 = TXCARB22 + HO2 + NO2                                    
    # DJ[52] = J[54]                    
     
    # #    Photol Reaction (53) CH3OOH = HCHO + HO2 + OH                                           
    # DJ[53] = J[41]                        
     
    # #    Photol Reaction (54) C2H5OOH = CH3CHO + HO2 + OH                                        
    # DJ[54] = J[41]                        
     
    # #    Photol Reaction (55) RN10OOH = C2H5CHO + HO2 + OH                                       
    # DJ[55] = J[41]                        
     
    # #    Photol Reaction (56) IC3H7OOH = CH3COCH3 + HO2 + OH                                     
    # DJ[56] = J[41]                        
     
    # #    Photol Reaction (57) RN13OOH =  CH3CHO + C2H5O2 + OH                                    
    # DJ[57] = J[41]*BR01         
     
    # #    Photol Reaction (58) RN13OOH =  CARB11A + HO2 + OH                                      
    # DJ[58] = J[41]*(1-BR01)     
     
    # #    Photol Reaction (59) RN16OOH = RN15AO2 + OH                                             
    # DJ[59] = J[41]                        
     
    # #    Photol Reaction (60) RN19OOH = RN18AO2 + OH                                             
    # DJ[60] = J[41]                        
     
    # #    Photol Reaction (61) CH3CO3H = CH3O2 + OH                                               
    # DJ[61] = J[41]                        
     
    # #    Photol Reaction (62) C2H5CO3H = C2H5O2 + OH                                             
    # DJ[62] = J[41]                        
     
    # #    Photol Reaction (63) HOCH2CO3H = HCHO + HO2 + OH                                        
    # DJ[63] = J[41]                        
     
    # #    Photol Reaction (64) RN8OOH = C2H5O2 + OH                                               
    # DJ[64] = J[41]                        
     
    # #    Photol Reaction (65) RN11OOH = RN10O2 + OH                                              
    # DJ[65] = J[41]                        
     
    # #    Photol Reaction (66) RN14OOH = RN13O2 + OH                                              
    # DJ[66] = J[41]                        
     
    # #    Photol Reaction (67) RN17OOH = RN16O2 + OH                                              
    # DJ[67] = J[41]                        
     
    # #    Photol Reaction (68) RU14OOH = UCARB12 + HO2 + OH                                       
    # DJ[68] = J[41]*0.252              
     
    # #    Photol Reaction (69) RU14OOH = UCARB10 + HCHO + HO2 + OH                                
    # DJ[69] = J[41]*0.748              
     
    # #    Photol Reaction (70) RU12OOH = CARB6 + HOCH2CHO + HO2 + OH                              
    # DJ[70] = J[41]                   
     
    # #    Photol Reaction (71) RU10OOH = CH3CO3 + HOCH2CHO + OH                                   
    # DJ[71] = J[41]                   
     
    # #    Photol Reaction (72) NRU14OOH = NUCARB12 + HO2 + OH                                     
    # DJ[72] = J[41]                   
     
    # #    Photol Reaction (73) NRU12OOH = NOA + CO + HO2 + OH                                     
    # DJ[73] = J[41]                   
     
    # #    Photol Reaction (74) HOC2H4OOH = HCHO + HCHO + HO2 + OH                                 
    # DJ[74] = J[41]                  
     
    # #    Photol Reaction (75) RN9OOH = CH3CHO + HCHO + HO2 + OH                                  
    # DJ[75] = J[41]                  
     
    # #    Photol Reaction (76) RN12OOH = CH3CHO + CH3CHO + HO2 + OH                               
    # DJ[76] = J[41]                  
     
    # #    Photol Reaction (77) RN15OOH = C2H5CHO + CH3CHO + HO2 + OH                              
    # DJ[77] = J[41]                  
     
    # #    Photol Reaction (78) RN18OOH = C2H5CHO + C2H5CHO + HO2 + OH                             
    # DJ[78] = J[41]                 
     
    # #    Photol Reaction (79) NRN6OOH = HCHO + HCHO + NO2 + OH                                   
    # DJ[79] = J[41]                  
     
    # #    Photol Reaction (80) NRN9OOH = CH3CHO + HCHO + NO2 + OH                                 
    # DJ[80] = J[41]                  
     
    # #    Photol Reaction (81) NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                              
    # DJ[81] = J[41]                  
     
    # #    Photol Reaction (82) RA13OOH = CARB3 + UDCARB8 + HO2 + OH                               
    # DJ[82] = J[41]                  
     
    # #    Photol Reaction (83) RA16OOH = CARB3 + UDCARB11 + HO2 + OH                              
    # DJ[83] = J[41]                  
     
    # #    Photol Reaction (84) RA19OOH = CARB6 + UDCARB11 + HO2 + OH                              
    # DJ[84] = J[41]                  
     
    # #    Photol Reaction (85) RTN28OOH = TNCARB26 + HO2 + OH                                     
    # DJ[85] = J[41]                  
     
    # #    Photol Reaction (86) NRTN28OOH = TNCARB26 + NO2 + OH                                    
    # DJ[86] = J[41]                  
     
    # #    Photol Reaction (87) RTN26OOH = RTN25O2 + OH                                            
    # DJ[87] = J[41]             
     
    # #    Photol Reaction (88) RTN25OOH = RTN24O2 + OH                                            
    # DJ[88] = J[41]             
     
    # #    Photol Reaction (89) RTN24OOH = RTN23O2 + OH                                            
    # DJ[89] = J[41]             
     
    # #    Photol Reaction (90) RTN23OOH = CH3COCH3 + RTN14O2 + OH                                 
    # DJ[90] = J[41]             
     
    # #    Photol Reaction (91) RTN14OOH = TNCARB10 + HCHO + HO2 + OH                              
    # DJ[91] = J[41]             
     
    # #    Photol Reaction (92) RTN10OOH = RN8O2 + CO + OH                                         
    # DJ[92] = J[41]             
     
    # #    Photol Reaction (93) RTX28OOH = TXCARB24 + HCHO + HO2 + OH                              
    # DJ[93] = J[41]                  
     
    # #    Photol Reaction (94) RTX24OOH = TXCARB22 + HO2 + OH                                     
    # DJ[94] = J[41]                  
     
    # #    Photol Reaction (95) RTX22OOH = CH3COCH3 + RN13O2 + OH                                  
    # DJ[95] = J[41]                  
     
    # #    Photol Reaction (96) NRTX28OOH = TXCARB24 + HCHO + NO2 + OH                             
    # DJ[96] = J[41]   

    return J

