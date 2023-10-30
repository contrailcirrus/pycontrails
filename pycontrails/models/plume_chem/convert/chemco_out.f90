      SUBROUTINE CHEM(:,:,:,TS)CO(RC,TEM(:,:,:,TS)P(:,:,:,TS),M(:,:,:,TS),O2(:,:,:,TS),H2O(:,:,:,TS),RO2(:,:,:,TS),M(:,:,:,TS)OM(:,:,:,TS),BR01)
C----------------------------------------------------------------------
C-
C-   Purpose and M(:,:,:,TS)ethods : CALCULATES RATE COEFFICIENTS
C-
C-   Inputs  : TEM(:,:,:,TS)P(:,:,:,TS),M(:,:,:,TS),H2O(:,:,:,TS)
C-   Outputs : RC,DJ
C-   Controls:

C----------------------------------------------------------------------
      IM(:,:,:,TS)PLICIT NONE
C----------------------------------------------------------------------
    DOUBLE PRECISION RC(510),TEM(:,:,:,TS)P(:,:,:,TS),M(:,:,:,TS),RKLOW,RKHIGH,FC,BRN,FAC1,FAC2,FAC3
    DOUBLE PRECISION DJ(96),J(70)
    DOUBLE PRECISION RK0,RK2,RK3
    DOUBLE PRECISION H2O(:,:,:,TS),O2(:,:,:,TS),RO2(:,:,:,TS)
    DOUBLE PRECISION K0,KI,F
    DOUBLE PRECISION KRO2(:,:,:,TS)NO,KRO2(:,:,:,TS)HO2(:,:,:,TS),KRO2(:,:,:,TS)NO3,KNO3AL,KDEC
    DOUBLE PRECISION KAPHO2(:,:,:,TS),KFPAN,KBPAN,KAPNO
    DOUBLE PRECISION KC0,KCI,KRC,FCC
    DOUBLE PRECISION KD0,KDI,KRD,FCD,FD,K10,K1I,KR1,FC1,F1
    DOUBLE PRECISION K20,K2I,KR2,FC2,Fa2,K30,K3I,KR3,FC3,F3
    DOUBLE PRECISION K40,K4I,KR4,FC4,Fa4,K70,K7I,KR7,FC7,F7
    DOUBLE PRECISION K80,K8I,KR8,FC8,F8,K90,K9I,KR9,FC9,F9
    DOUBLE PRECISION K100,K10I,KR10,FC10,F10,K130,K13I,KR13,FC13,F13
    DOUBLE PRECISION K140,K14I,KR14,FC14,F14,K160,K16I,KR16,FC16,F16
    DOUBLE PRECISION K1,K2,K3,K4,KM(:,:,:,TS)T01,KM(:,:,:,TS)T02,KM(:,:,:,TS)T03,KM(:,:,:,TS)T04,KM(:,:,:,TS)T05
    DOUBLE PRECISION KM(:,:,:,TS)T06,KM(:,:,:,TS)T07,KM(:,:,:,TS)T08,KM(:,:,:,TS)T09,KM(:,:,:,TS)T10,KM(:,:,:,TS)T11
    DOUBLE PRECISION KM(:,:,:,TS)T12,KM(:,:,:,TS)T13,KM(:,:,:,TS)T14,KM(:,:,:,TS)T15,KM(:,:,:,TS)T16,KM(:,:,:,TS)T17 
    DOUBLE PRECISION KROPRIM(:,:,:,TS),KROSEC,PANTOT,KDEC1,KTOT1,BR01 
    DOUBLE PRECISION FAC4,SOA,SOAM(:,:,:,TS),YY,SC,OM(:,:,:,TS),M(:,:,:,TS)OM(:,:,:,TS),KIN,KOUT2604,KOUT4608
	DOUBLE PRECISION KOUT2631,KOUT2635,KOUT2641
    DOUBLE PRECISION KOUT4610,KOUT2605,KOUT4830,KOUT4829,KOUT3442
	DOUBLE PRECISION KOUT2630,KOUT2671,KOUT4834,KOUT5276,KOUT2617 
	DOUBLE PRECISION KOUT5236,KOUT4552,KOUT2703,KOUT2629
    DOUBLE PRECISION KOUT2669,KOUT3613,KOUT3612,KOUT2637,KOUT2632
    DOUBLE PRECISION R, BGOAM(:,:,:,TS),KALKOXY,KALKPXY,K150,K15I,KR15,FC15,F15
	DOUBLE PRECISION K170, K17I, KR17,FC17,F17
	DOUBLE PRECISION N2(:,:,:,TS)
	
      INTEGER I
      N2(:,:,:,TS)       = 7.809E-01*M(:,:,:,TS) 
      O2(:,:,:,TS)       = 2.079E-01*M(:,:,:,TS) 
      R = 8.314
      DO 317 I=1,510
        RC(I)=0.0
  317 CONTINUE

C    SIM(:,:,:,TS)PLE RATE COEFFICIENTS                     
C                                                                     
      KRO2(:,:,:,TS)NO(:,:,:)  = 2.54D-12*EXP(360/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KAPNO(:,:,:)   = 8.10D-12*EXP(270/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KRO2(:,:,:,TS)NO3(:,:,:) = 2.50D-12 
      KRO2(:,:,:,TS)HO2(:,:,:,TS)(:,:,:) = 2.91D-13*EXP(1300/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KAPHO2(:,:,:,TS)(:,:,:)  = 4.30D-13*EXP(1040/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KNO3AL(:,:,:)  = 1.44D-12*EXP(-1862/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KDEC(:,:,:)    = 1.0D+06
      KALKOXY(:,:,:) = 3.70D-14*EXP(-460/TEM(:,:,:,TS)P(:,:,:,TS))*O2(:,:,:,TS) 
      KALKPXY(:,:,:) = 1.80D-14*EXP(-260/TEM(:,:,:,TS)P(:,:,:,TS))*O2(:,:,:,TS) 
      BR01(:,:,:) = (0.156 + 9.77D+08*EXP(-6415/TEM(:,:,:,TS)P(:,:,:,TS))) 
C
      KIN(:,:,:) = 6.2E-03*M(:,:,:,TS)OM(:,:,:,TS)
      KOUT2604(:,:,:) = 4.34*EXP(-7776/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT4608(:,:,:) = 4.34*EXP(-9765/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2631(:,:,:) = 4.34*EXP(-14500/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2635(:,:,:) = 4.34*EXP(-12541/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT4610(:,:,:) = 4.34*EXP(-10513/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2605(:,:,:) = 4.34*EXP(-8879/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2630(:,:,:) = 4.34*EXP(-12639/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2629(:,:,:) = 4.34*EXP(-4954/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2632(:,:,:) = 4.34*EXP(-3801/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT2637(:,:,:) = 4.34*EXP(-16752/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT3612(:,:,:) = 4.34*EXP(-8362/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT3613(:,:,:) = 4.34*EXP(-11003/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
      KOUT3442(:,:,:) = 4.34*EXP(-12763/(R*TEM(:,:,:,TS)P(:,:,:,TS)))
C
C    COM(:,:,:,TS)PLEX RATE COEFFICIENTS                    
C                                                                     
C    KFPAN                                                   
C                                                                     
      KC0     = 2.70D-28*M(:,:,:,TS)*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-7.1 
      KCI     = 1.21D-11*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-0.9    
      KRC     = KC0/KCI    
      FCC     = 0.30       
      FC      = 10**(LOG10(FCC)/(1+(LOG10(KRC))**2)) 
      KFPAN   = (KC0*KCI)*FC/(KC0+KCI) 
C                                                                   
C    KBPAN                                                   
      KD0     = 4.90D-03*M(:,:,:,TS)*EXP(-12100/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KDI     = 3.70D+16*EXP(-13600/TEM(:,:,:,TS)P(:,:,:,TS))  
      KRD     = KD0/KDI    
      FCD     = 0.30       
      FD      = 10**(LOG10(FCD)/(1+(LOG10(KRD))**2)) 
      KBPAN   = (KD0*KDI)*FD/(KD0+KDI) 
C                                                                     
C     KM(:,:,:,TS)T01                                                   
      K10     = 9.00D-32*M(:,:,:,TS)*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-1.5 
      K1I     = 3.00D-11*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**0.3    
      KR1     = K10/K1I    
      FC1     = 0.6 
      F1      = 10**(LOG10(FC1)/(1+(LOG10(KR1))**2)) 
      KM(:,:,:,TS)T01   = (K10*K1I)*F1/(K10+K1I) 
C                                                                     
C     KM(:,:,:,TS)T02                                                   
      K20(:,:,:)s= 9.00D-32*((temp/300)**-2.0)*M(:,:,:,TS) 
      K2I(:,:,:)s= 2.20D-11
      KR2     = K20/K2I    
      FC2(:,:,:)s= 0.6 
      Fa2      = 10**(LOG10(FC2)/(1+(LOG10(KR2))**2)) 
      KM(:,:,:,TS)T02   = (K20*K2I)*Fa2/(K20+K2I) 
C                                                                     
C      KM(:,:,:,TS)T03  : NO2(:,:,:,TS)      + NO3     = N2(:,:,:,TS)O5                               
C    IUPAC 2001                                                       
      K30     = 2.70D-30*M(:,:,:,TS)*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-3.4 
      K3I     = 2.00D-12*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**0.2    
      KR3     = K30/K3I    
      FC3     = (EXP(-TEM(:,:,:,TS)P(:,:,:,TS)/250) + EXP(-1050/TEM(:,:,:,TS)P(:,:,:,TS))) 
      F3      = 10**(LOG10(FC3)/(1+(LOG10(KR3))**2)) 
      KM(:,:,:,TS)T03   = (K30*K3I)*F3/(K30+K3I) 
C                                                                     
C     KM(:,:,:,TS)T04  : N2(:,:,:,TS)O5               = NO2(:,:,:,TS)     + NO3                     
C IUPAC 1997/2001                                                 
      K40     = (2.20D-03*M(:,:,:,TS)*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-4.34)*(EXP(-11080/TEM(:,:,:,TS)P(:,:,:,TS)))
      K4I     = (9.70D+14*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**0.1)*EXP(-11080/TEM(:,:,:,TS)P(:,:,:,TS))    
      KR4     = K40/K4I    
      FC4     = (EXP(-TEM(:,:,:,TS)P(:,:,:,TS)/250) + EXP(-1050/TEM(:,:,:,TS)P(:,:,:,TS)))
      Fa4      = 10**(LOG10(FC4)/(1+(LOG10(KR4))**2)) 
      KM(:,:,:,TS)T04   = (K40*K4I)*Fa4/(K40+K4I)       
C	WRITE(6,*)'KM(:,:,:,TS)TO4=',KM(:,:,:,TS)T04                                                               
C    KM(:,:,:,TS)T05                                                   
      KM(:,:,:,TS)T05  =  1 + ((0.6*M(:,:,:,TS))/(2.687D+19*(273/TEM(:,:,:,TS)P(:,:,:,TS)))) 
C                                                                     
C    KM(:,:,:,TS)T06                                                   
      KM(:,:,:,TS)T06  =  1 + (1.40D-21*EXP(2200/TEM(:,:,:,TS)P(:,:,:,TS))*H2O(:,:,:,TS)) 
C                                                                     
C    KM(:,:,:,TS)T07  : OH       + NO      = HONO                              
C    IUPAC 2001                                                      
      K70     = 7.00D-31*M(:,:,:,TS)*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-2.6 
      K7I     = 3.60D-11*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**0.1    
      KR7     = K70/K7I    
      FC7     = 0.6  
      F7      = 10**(LOG10(FC7)/(1+(LOG10(KR7))**2)) 
      KM(:,:,:,TS)T07   = (K70*K7I)*F7/(K70+K7I) 
C                                                                     
C NASA 2000                                                           
  
C    KM(:,:,:,TS)T08                                                    
      K80(:,:,:)s= 2.50D-30*((temp/300)**-4.4)*M(:,:,:,TS) 
      K8I(:,:,:)s= 1.60D-11 
      KR8(:,:,:)s= K80/K8I 
      FC8(:,:,:)s= 0.6 
      F8      = 10**(LOG10(FC8)/(1+(LOG10(KR8))**2)) 
      KM(:,:,:,TS)T08   = (K80*K8I)*F8/(K80+K8I) 
C                                                                     
C    KM(:,:,:,TS)T09  : HO2(:,:,:,TS)      + NO2(:,:,:,TS)     = HO2(:,:,:,TS)NO2(:,:,:,TS)                            
C IUPAC 1997/2001                                                 
      K90     = 1.80D-31*M(:,:,:,TS)*(TEM(:,:,:,TS)P(:,:,:,TS)/300)**-3.2 
      K9I     = 4.70D-12    
      KR9     = K90/K9I    
      FC9     = 0.6 
      F9      = 10**(LOG10(FC9)/(1+(LOG10(KR9))**2)) 
      KM(:,:,:,TS)T09   = (K90*K9I)*F9/(K90+K9I) 
C                                                                     
C KM(:,:,:,TS)T10  : HO2(:,:,:,TS)NO2(:,:,:,TS)             = HO2(:,:,:,TS)     + NO2(:,:,:,TS)                     
C IUPAC 2001                                                      
C
      K100     = 4.10D-05*M(:,:,:,TS)*EXP(-10650/TEM(:,:,:,TS)P(:,:,:,TS)) 
      K10I     = 5.70D+15*EXP(-11170/TEM(:,:,:,TS)P(:,:,:,TS))   
      KR10     = K100/K10I    
      FC10     = 0.5 
      F10      = 10**(LOG10(FC10)/(1+(LOG10(KR10))**2)) 
      KM(:,:,:,TS)T10    = (K100*K10I)*F10/(K100+K10I) 
C                                                                     
C   KM(:,:,:,TS)T11  : OH       + HNO3    = H2O(:,:,:,TS)     + NO3                     
C   IUPAC 2001                                                      
      K1     = 7.20D-15*EXP(785/TEM(:,:,:,TS)P(:,:,:,TS)) 
      K3     = 1.90D-33*EXP(725/TEM(:,:,:,TS)P(:,:,:,TS)) 
      K4     = 4.10D-16*EXP(1440/TEM(:,:,:,TS)P(:,:,:,TS)) 
      K2     = (K3*M(:,:,:,TS))/(1+(K3*M(:,:,:,TS)/K4)) 
      KM(:,:,:,TS)T11  = K1 + K2 
C                                                                     
C KM(:,:,:,TS)T12 : OH    +   SO2(:,:,:,TS)  =  HSO3                                  
C IUPAC 2003                                                      
      K0(:,:,:)s= 3.0D-31*((TEM(:,:,:,TS)P(:,:,:,TS)/300)**-3.3)*M(:,:,:,TS) 
      KI(:,:,:)s= 1.5D-12 
      KR1(:,:,:)s= K0/KI 
      FC(:,:,:)s= 0.6 
      F=10**(LOG10(FC)/(1+(LOG10(KR1))**2)) 
      KM(:,:,:,TS)T12=(K0*KI*F)/(K0+KI) 
C                                                                     
C KM(:,:,:,TS)T13  : CH3O2(:,:,:,TS)    + NO2(:,:,:,TS)     = CH3O2(:,:,:,TS)NO2(:,:,:,TS)                           
C IUPAC 2003                                                       
      K130     = 2.50D-30*((TEM(:,:,:,TS)P(:,:,:,TS)/300)**-5.5)*M(:,:,:,TS) 
      K13I     = 7.50D-12 
      KR13     = K130/K13I 
      FC13     = 0.36 
      F13      = 10**(LOG10(FC13)/(1+(LOG10(KR13))**2)) 
      KM(:,:,:,TS)T13    = (K130*K13I)*F13/(K130+K13I) 
C                                                                     
C  KM(:,:,:,TS)T14  : CH3O2(:,:,:,TS)NO2(:,:,:,TS)           = CH3O2(:,:,:,TS)   + NO2(:,:,:,TS)                      
C  IUPAC 2001                                                       
      K140     = 9.00D-05*EXP(-9690/TEM(:,:,:,TS)P(:,:,:,TS))*M(:,:,:,TS) 
      K14I     = 1.10D+16*EXP(-10560/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KR14     = K140/K14I 
      FC14     = 0.36 
      F14      = 10**(LOG10(FC14)/(1+(LOG10(KR14))**2)) 
      KM(:,:,:,TS)T14    = (K140*K14I)*F14/(K140+K14I) 
C                                                                   
C KM(:,:,:,TS)T15  :    OH  +  C2H4  =                                       
C IUPAC 2001                                                      
      K150(:,:,:)s= 6.00D-29*((TEM(:,:,:,TS)P(:,:,:,TS)/298)**-4.0)*M(:,:,:,TS) 
      K15I(:,:,:)s= 9.00D-12*((TEM(:,:,:,TS)P(:,:,:,TS)/298)**-1.1) 
      KR15(:,:,:)s= K150/K15I 
      FC15(:,:,:)s= 0.7
      F15      = 10**(LOG10(FC15)/(1+(LOG10(KR15))**2)) 
      KM(:,:,:,TS)T15    = (K150*K15I)*F15/(K150+K15I) 
C                                                                    
C KM(:,:,:,TS)T16  :  OH  +  C3H6         
C IUPAC 2003                                                     
      K160     = 3.00D-27*((TEM(:,:,:,TS)P(:,:,:,TS)/298)**-3.0)*M(:,:,:,TS) 
      K16I     = 2.80D-11*((TEM(:,:,:,TS)P(:,:,:,TS)/298)**-1.3) 
      KR16     = K160/K16I 
      FC16     = 0.5 
      F16      = 10**(LOG10(FC16)/(1+(LOG10(KR16))**2)) 
      KM(:,:,:,TS)T16    = (K160*K16I)*F16/(K160+K16I) 
C                                                                     
C    KM(:,:,:,TS)T17                                                   
      K170(:,:,:)s= 5.00D-30*((TEM(:,:,:,TS)P(:,:,:,TS)/298)**-1.5)*M(:,:,:,TS) 
      K17I(:,:,:)s= 9.40D-12*EXP(-700/TEM(:,:,:,TS)P(:,:,:,TS)) 
      KR17     = K170/K17I 
      FC17(:,:,:)s= (EXP(-TEM(:,:,:,TS)P(:,:,:,TS)/580) + EXP(-2320/TEM(:,:,:,TS)P(:,:,:,TS))) 
      F17      = 10**(LOG10(FC17)/(1+(LOG10(KR17))**2)) 
      KM(:,:,:,TS)T17    = (K170*K17I)*F17/(K170+K17I) 
C
C  LIST OF ALL REACTIONS 
C
C     Reaction (1) O = O3                                                             
         RC(:,:,:,TS,1) = 5.60D-34*O2(:,:,:,TS)*N2(:,:,:,TS)*((TEM(:,:,:,TS)P(:,:,:,TS)/300)**-2.6)
C
C     Reaction (2) O = O3                                                             
         RC(:,:,:,TS,2) = 6.00D-34*O2(:,:,:,TS)*O2(:,:,:,TS)*((TEM(:,:,:,TS)P(:,:,:,TS)/300)**-2.6)
C
C     Reaction (3) O + O3 =                                                           
         RC(:,:,:,TS,3) = 8.00D-12*EXP(-2060/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (4) O + NO = NO2(:,:,:,TS)                                                       
         RC(:,:,:,TS,4)s=sKM(:,:,:,TS)T01(:,:,:)                            
C
C     Reaction (5) O + NO2(:,:,:,TS) = NO                                                       
         RC(:,:,:,TS,5) = 5.50D-12*EXP(188/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (6) O + NO2(:,:,:,TS) = NO3                                                      
         RC(:,:,:,TS,6)s=sKM(:,:,:,TS)T02(:,:,:)                            
C
C     Reaction (7) O1D = O                                                            
         RC(:,:,:,TS,7) = 3.20D-11*O2(:,:,:,TS)*EXP(67/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (8) O1D = O                                                            
         RC(:,:,:,TS,8) = 1.80D-11*N2(:,:,:,TS)*EXP(107/TEM(:,:,:,TS)P(:,:,:,TS))        
C
C     Reaction (9) NO + O3 = NO2(:,:,:,TS)                                                      
         RC(:,:,:,TS,9) = 1.40D-12*EXP(-1310/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (10) NO2(:,:,:,TS) + O3 = NO3                                                     
         RC(:,:,:,TS,10) = 1.40D-13*EXP(-2470/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (11) NO + NO = NO2(:,:,:,TS) + NO2(:,:,:,TS)                                                
         RC(:,:,:,TS,11) = 3.30D-39*EXP(530/TEM(:,:,:,TS)P(:,:,:,TS))*O2(:,:,:,TS)        
C
C     Reaction (12) NO + NO3 = NO2(:,:,:,TS) + NO2(:,:,:,TS)                                               
         RC(:,:,:,TS,12) = 1.80D-11*EXP(110/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (13) NO2(:,:,:,TS) + NO3 = NO + NO2(:,:,:,TS)                                               
         RC(:,:,:,TS,13) = 4.50D-14*EXP(-1260/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (14) NO2(:,:,:,TS) + NO3 = N2(:,:,:,TS)O5                                                   
         RC(:,:,:,TS,14)s=sKM(:,:,:,TS)T03(:,:,:)                            
C
C     Reaction (15) N2(:,:,:,TS)O5 = NO2(:,:,:,TS) + NO3                                                   
         RC(:,:,:,TS,15)s=sKM(:,:,:,TS)T04(:,:,:)                            
C
C     Reaction (16) O1D = OH + OH                                                      
         RC(:,:,:,TS,16) = 2.20D-10                     
C
C     Reaction (17) OH + O3 = HO2(:,:,:,TS)                                                      
         RC(:,:,:,TS,17) = 1.70D-12*EXP(-940/TEM(:,:,:,TS)P(:,:,:,TS))          
C
C     Reaction (18) OH + H2 = HO2(:,:,:,TS)                                                      
         RC(:,:,:,TS,18) = 7.70D-12*EXP(-2100/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (19) OH + CO = HO2(:,:,:,TS)                                                      
         RC(:,:,:,TS,19) = 1.30D-13*KM(:,:,:,TS)T05                   
C
C     Reaction (20) OH + H2O(:,:,:,TS)2 = HO2(:,:,:,TS)                                                    
         RC(:,:,:,TS,20) = 2.90D-12*EXP(-160/TEM(:,:,:,TS)P(:,:,:,TS))          
C
C     Reaction (21) HO2(:,:,:,TS) + O3 = OH                                                      
         RC(:,:,:,TS,21) = 2.03D-16*((TEM(:,:,:,TS)P(:,:,:,TS)/300)**4.57)*EXP(693/TEM(:,:,:,TS)P(:,:,:,TS))  
C
C     Reaction (22) OH + HO2(:,:,:,TS) =                                                         
         RC(:,:,:,TS,22) = 4.80D-11*EXP(250/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (23) HO2(:,:,:,TS) + HO2(:,:,:,TS) = H2O(:,:,:,TS)2                                                   
         RC(:,:,:,TS,23) = 2.20D-13*KM(:,:,:,TS)T06*EXP(600/TEM(:,:,:,TS)P(:,:,:,TS))     
C
C     Reaction (24) HO2(:,:,:,TS) + HO2(:,:,:,TS) = H2O(:,:,:,TS)2                                                   
         RC(:,:,:,TS,24) = 1.90D-33*M(:,:,:,TS)*KM(:,:,:,TS)T06*EXP(980/TEM(:,:,:,TS)P(:,:,:,TS))   
C
C     Reaction (25) OH + NO = HONO                                                     
         RC(:,:,:,TS,25)s=sKM(:,:,:,TS)T07(:,:,:)                            
C
C     Reaction (26) NO2(:,:,:,TS) = HONO                                                         
         RC(:,:,:,TS,26) = 0.0                          
C
C     Reaction (27) OH + NO2(:,:,:,TS) = HNO3                                                    
         RC(:,:,:,TS,27)s=sKM(:,:,:,TS)T08(:,:,:)                            
C
C     Reaction (28) OH + NO3 = HO2(:,:,:,TS) + NO2(:,:,:,TS)                                               
         RC(:,:,:,TS,28) = 2.00D-11                         
C
C     Reaction (29) HO2(:,:,:,TS) + NO = OH + NO2(:,:,:,TS)                                                
         RC(:,:,:,TS,29) = 3.60D-12*EXP(270/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (30) HO2(:,:,:,TS) + NO2(:,:,:,TS) = HO2(:,:,:,TS)NO2(:,:,:,TS)                                                 
         RC(:,:,:,TS,30)s=sKM(:,:,:,TS)T09(:,:,:)                            
C
C     Reaction (31) HO2(:,:,:,TS)NO2(:,:,:,TS) = HO2(:,:,:,TS) + NO2(:,:,:,TS)                                                 
         RC(:,:,:,TS,31)s=sKM(:,:,:,TS)T10(:,:,:)                            
C
C     Reaction (32) OH + HO2(:,:,:,TS)NO2(:,:,:,TS) = NO2(:,:,:,TS)                                                  
         RC(:,:,:,TS,32) = 1.90D-12*EXP(270/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (33) HO2(:,:,:,TS) + NO3 = OH + NO2(:,:,:,TS)                                               
         RC(:,:,:,TS,33) = 4.00D-12                         
C
C     Reaction (34) OH + HONO = NO2(:,:,:,TS)                                                    
         RC(:,:,:,TS,34) = 2.50D-12*EXP(-260/TEM(:,:,:,TS)P(:,:,:,TS))          
C
C     Reaction (35) OH + HNO3 = NO3                                                    
         RC(:,:,:,TS,35)s=sKM(:,:,:,TS)T11(:,:,:)                            
C
C     Reaction (36) O + SO2(:,:,:,TS) = SO3                                                      
         RC(:,:,:,TS,36) = 4.00D-32*EXP(-1000/TEM(:,:,:,TS)P(:,:,:,TS))*M(:,:,:,TS)       
C
C     Reaction (37) OH + SO2(:,:,:,TS) = HSO3                                                    
         RC(:,:,:,TS,37)s=sKM(:,:,:,TS)T12(:,:,:)                            
C
C     Reaction (38) HSO3 = HO2(:,:,:,TS) + SO3                                                   
         RC(:,:,:,TS,38) = 1.30D-12*EXP(-330/TEM(:,:,:,TS)P(:,:,:,TS))*O2(:,:,:,TS)       
C
C     Reaction (39) HNO3 = NA                                                          
         RC(:,:,:,TS,39) = 0.0                         
C
C     Reaction (40) N2(:,:,:,TS)O5 = NA + NA                                                     
         RC(:,:,:,TS,40) = 0.0                         
C
C     Reaction (41) SO3 = SA                                                           
         RC(:,:,:,TS,41) = 1.20D-15*H2O(:,:,:,TS)                     
C
C     Reaction (42) OH + CH4 = CH3O2(:,:,:,TS)                                                   
         RC(:,:,:,TS,42) = 9.65D-20*TEM(:,:,:,TS)P(:,:,:,TS)**2.58*EXP(-1082/TEM(:,:,:,TS)P(:,:,:,TS)) 
C
C     Reaction (43) OH + C2H6 = C2H5O2(:,:,:,TS)                                                 
         RC(:,:,:,TS,43) = 1.52D-17*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-498/TEM(:,:,:,TS)P(:,:,:,TS)) 
C
C     Reaction (44) OH + C3H8 = IC3H7O2(:,:,:,TS)                                                
         RC(:,:,:,TS,44) = 1.55D-17*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-61/TEM(:,:,:,TS)P(:,:,:,TS))*0.736  
C
C     Reaction (45) OH + C3H8 = RN10O2(:,:,:,TS)                                                 
         RC(:,:,:,TS,45) = 1.55D-17*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-61/TEM(:,:,:,TS)P(:,:,:,TS))*0.264  
C
C     Reaction (46) OH + NC4H10 = RN13O2(:,:,:,TS)                                               
         RC(:,:,:,TS,46) = 1.69D-17*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(145/TEM(:,:,:,TS)P(:,:,:,TS))  
C
C     Reaction (47) OH + C2H4 = HOCH2CH2O(:,:,:,TS)2                                             
         RC(:,:,:,TS,47)s=sKM(:,:,:,TS)T15(:,:,:)                        
C
C     Reaction (48) OH + C3H6 = RN9O2(:,:,:,TS)                                                  
         RC(:,:,:,TS,48)s=sKM(:,:,:,TS)T16(:,:,:)                        
C
C     Reaction (49) OH + TBUT2ENE = RN12O2(:,:,:,TS)                                             
         RC(:,:,:,TS,49) = 1.01D-11*EXP(550/TEM(:,:,:,TS)P(:,:,:,TS))       
C
C     Reaction (50) NO3 + C2H4 = NRN6O2(:,:,:,TS)                                                
         RC(:,:,:,TS,50) = 2.10D-16                     
C
C     Reaction (51) NO3 + C3H6 = NRN9O2(:,:,:,TS)                                                
         RC(:,:,:,TS,51) = 9.40D-15                     
C
C     Reaction (52) NO3 + TBUT2ENE = NRN12O2(:,:,:,TS)                                           
         RC(:,:,:,TS,52) = 3.90D-13                     
C
C     Reaction (53) O3 + C2H4 = HCHO + CO + HO2(:,:,:,TS) + OH                                   
         RC(:,:,:,TS,53) = 9.14D-15*EXP(-2580/TEM(:,:,:,TS)P(:,:,:,TS))*0.13  
C
C     Reaction (54) O3 + C2H4 = HCHO + HCOOH                                           
         RC(:,:,:,TS,54) = 9.14D-15*EXP(-2580/TEM(:,:,:,TS)P(:,:,:,TS))*0.87  
C
C     Reaction (55) O3 + C3H6 = HCHO + CO + CH3O2(:,:,:,TS) + OH                                 
         RC(:,:,:,TS,55) = 5.51D-15*EXP(-1878/TEM(:,:,:,TS)P(:,:,:,TS))*0.36  
C
C     Reaction (56) O3 + C3H6 = HCHO + CH3CO2(:,:,:,TS)H                                         
         RC(:,:,:,TS,56) = 5.51D-15*EXP(-1878/TEM(:,:,:,TS)P(:,:,:,TS))*0.64  
C
C     Reaction (57) O3 + TBUT2ENE = CH3CHO + CO + CH3O2(:,:,:,TS) + OH                           
         RC(:,:,:,TS,57) = 6.64D-15*EXP(-1059/TEM(:,:,:,TS)P(:,:,:,TS))*0.69 
C
C     Reaction (58) O3 + TBUT2ENE = CH3CHO + CH3CO2(:,:,:,TS)H                                   
         RC(:,:,:,TS,58) = 6.64D-15*EXP(-1059/TEM(:,:,:,TS)P(:,:,:,TS))*0.31 
C
C     Reaction (59) OH + C5H8 = RU14O2(:,:,:,TS)                                                 
         RC(:,:,:,TS,59) = 2.54D-11*EXP(410/TEM(:,:,:,TS)P(:,:,:,TS))       
C
C     Reaction (60) NO3 + C5H8 = NRU14O2(:,:,:,TS)                                               
         RC(:,:,:,TS,60) = 3.03D-12*EXP(-446/TEM(:,:,:,TS)P(:,:,:,TS))      
C
C     Reaction (61) O3 + C5H8 = UCARB10 + CO + HO2(:,:,:,TS) + OH                                
         RC(:,:,:,TS,61) = 7.86D-15*EXP(-1913/TEM(:,:,:,TS)P(:,:,:,TS))*0.27 
C
C     Reaction (62) O3 + C5H8 = UCARB10 + HCOOH                                        
         RC(:,:,:,TS,62) = 7.86D-15*EXP(-1913/TEM(:,:,:,TS)P(:,:,:,TS))*0.73 
C
C     Reaction (63) APINENE + OH = RTN2(:,:,:,TS)8O2(:,:,:,TS)                                             
         RC(:,:,:,TS,63) = 1.20D-11*EXP(444/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (64) APINENE + NO3 = NRTN2(:,:,:,TS)8O2(:,:,:,TS)                                           
         RC(:,:,:,TS,64) = 1.19D-12*EXP(490/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (65) APINENE + O3 = OH + RTN2(:,:,:,TS)6O2(:,:,:,TS)                                        
         RC(:,:,:,TS,65) = 1.01D-15*EXP(-732/TEM(:,:,:,TS)P(:,:,:,TS))*0.80  
C
C     Reaction (66) APINENE + O3 = TNCARB26 + H2O(:,:,:,TS)2                                     
         RC(:,:,:,TS,66) = 1.01D-15*EXP(-732/TEM(:,:,:,TS)P(:,:,:,TS))*0.075  
C
C     Reaction (67) APINENE + O3 = RCOOH25                                             
         RC(:,:,:,TS,67) = 1.01D-15*EXP(-732/TEM(:,:,:,TS)P(:,:,:,TS))*0.125  
C
C     Reaction (68) BPINENE + OH = RTX28O2(:,:,:,TS)                                             
         RC(:,:,:,TS,68) = 2.38D-11*EXP(357/TEM(:,:,:,TS)P(:,:,:,TS)) 
C
C     Reaction (69) BPINENE + NO3 = NRTX28O2(:,:,:,TS)                                           
         RC(:,:,:,TS,69) = 2.51D-12 
C
C     Reaction (70) BPINENE + O3 =  RTX24O2(:,:,:,TS) + OH                                       
         RC(:,:,:,TS,70) = 1.50D-17*0.35 
C
C     Reaction (71) BPINENE + O3 =  HCHO + TXCARB24 + H2O(:,:,:,TS)2                             
         RC(:,:,:,TS,71) = 1.50D-17*0.20 
C
C     Reaction (72) BPINENE + O3 =  HCHO + TXCARB22                                    
         RC(:,:,:,TS,72) = 1.50D-17*0.25 
C
C     Reaction (73) BPINENE + O3 =  TXCARB24 + CO                                      
         RC(:,:,:,TS,73) = 1.50D-17*0.20 
C
C     Reaction (74) C2H2 + OH = HCOOH + CO + HO2(:,:,:,TS)                                       
         RC(:,:,:,TS,74)s=sKM(:,:,:,TS)T17(:,:,:)*0.364 
C
C     Reaction (75) C2H2 + OH = CARB3 + OH                                             
         RC(:,:,:,TS,75)s=sKM(:,:,:,TS)T17(:,:,:)*0.636 
C
C     Reaction (76) BENZENE + OH = RA13O2(:,:,:,TS)                                              
         RC(:,:,:,TS,76) = 2.33D-12*EXP(-193/TEM(:,:,:,TS)P(:,:,:,TS))*0.47 
C
C     Reaction (77) BENZENE + OH = AROH14 + HO2(:,:,:,TS)                                        
         RC(:,:,:,TS,77) = 2.33D-12*EXP(-193/TEM(:,:,:,TS)P(:,:,:,TS))*0.53 
C
C     Reaction (78) TOLUENE + OH = RA16O2(:,:,:,TS)                                              
         RC(:,:,:,TS,78) = 1.81D-12*EXP(338/TEM(:,:,:,TS)P(:,:,:,TS))*0.82 
C
C     Reaction (79) TOLUENE + OH = AROH17 + HO2(:,:,:,TS)                                        
         RC(:,:,:,TS,79) = 1.81D-12*EXP(338/TEM(:,:,:,TS)P(:,:,:,TS))*0.18 
C
C     Reaction (80) OXYL + OH = RA19AO2(:,:,:,TS)                                                
         RC(:,:,:,TS,80) = 1.36D-11*0.70 
C
C     Reaction (81) OXYL + OH = RA19CO2(:,:,:,TS)                                                
         RC(:,:,:,TS,81) = 1.36D-11*0.30 
C
C     Reaction (82) OH + HCHO = HO2(:,:,:,TS) + CO                                               
         RC(:,:,:,TS,82) = 1.20D-14*TEM(:,:,:,TS)P(:,:,:,TS)*EXP(287/TEM(:,:,:,TS)P(:,:,:,TS))  
C
C     Reaction (83) OH + CH3CHO = CH3CO3                                               
         RC(:,:,:,TS,83) = 5.55D-12*EXP(311/TEM(:,:,:,TS)P(:,:,:,TS))             
C
C     Reaction (84) OH + C2H5CHO = C2H5CO3                                             
         RC(:,:,:,TS,84) = 1.96D-11                                
C
C     Reaction (85) NO3 + HCHO = HO2(:,:,:,TS) + CO + HNO3                                       
         RC(:,:,:,TS,85) = 5.80D-16                  
C
C     Reaction (86) NO3 + CH3CHO = CH3CO3 + HNO3                                       
         RC(:,:,:,TS,86)s=sKNO3AL(:,:,:)                   
C
C     Reaction (87) NO3 + C2H5CHO = C2H5CO3 + HNO3                                     
         RC(:,:,:,TS,87)s=sKNO3AL(:,:,:)*2.4             
C
C     Reaction (88) OH + CH3COCH3 = RN8O2(:,:,:,TS)                                              
         RC(:,:,:,TS,88) = 5.34D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-230/TEM(:,:,:,TS)P(:,:,:,TS)) 
C
C     Reaction (89) M(:,:,:,TS)EK + OH = RN11O2(:,:,:,TS)                                                  
         RC(:,:,:,TS,89) = 3.24D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(414/TEM(:,:,:,TS)P(:,:,:,TS))
C
C     Reaction (90) OH + CH3OH = HO2(:,:,:,TS) + HCHO                                            
         RC(:,:,:,TS,90) = 6.01D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(170/TEM(:,:,:,TS)P(:,:,:,TS))  
C
C     Reaction (91) OH + C2H5OH = CH3CHO + HO2(:,:,:,TS)                                         
         RC(:,:,:,TS,91) = 6.18D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(532/TEM(:,:,:,TS)P(:,:,:,TS))*0.887 
C
C     Reaction (92) OH + C2H5OH = HOCH2CH2O(:,:,:,TS)2                                           
         RC(:,:,:,TS,92) = 6.18D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(532/TEM(:,:,:,TS)P(:,:,:,TS))*0.113 
C
C     Reaction (93) NPROPOL + OH = C2H5CHO + HO2(:,:,:,TS)                                       
         RC(:,:,:,TS,93) = 5.53D-12*0.49 
C
C     Reaction (94) NPROPOL + OH = RN9O2(:,:,:,TS)                                               
         RC(:,:,:,TS,94) = 5.53D-12*0.51 
C
C     Reaction (95) OH + IPROPOL = CH3COCH3 + HO2(:,:,:,TS)                                      
         RC(:,:,:,TS,95) = 4.06D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(788/TEM(:,:,:,TS)P(:,:,:,TS))*0.86 
C
C     Reaction (96) OH + IPROPOL = RN9O2(:,:,:,TS)                                               
         RC(:,:,:,TS,96) = 4.06D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(788/TEM(:,:,:,TS)P(:,:,:,TS))*0.14 
C
C     Reaction (97) HCOOH + OH = HO2(:,:,:,TS)                                                   
         RC(:,:,:,TS,97) = 4.50D-13 
C
C     Reaction (98) CH3CO2(:,:,:,TS)H + OH = CH3O2(:,:,:,TS)                                               
         RC(:,:,:,TS,98) = 8.00D-13 
C
C     Reaction (99) OH + CH3CL = CH3O2(:,:,:,TS)                                                 
         RC(:,:,:,TS,99) = 7.33D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-809/TEM(:,:,:,TS)P(:,:,:,TS))   
C
C     Reaction (100) OH + CH2CL2 = CH3O2(:,:,:,TS)                                                
         RC(:,:,:,TS,100) = 6.14D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-389/TEM(:,:,:,TS)P(:,:,:,TS))   
C
C     Reaction (101) OH + CHCL3 = CH3O2(:,:,:,TS)                                                 
         RC(:,:,:,TS,101) = 1.80D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-129/TEM(:,:,:,TS)P(:,:,:,TS))   
C
C     Reaction (102) OH + CH3CCL3 = C2H5O2(:,:,:,TS)                                              
         RC(:,:,:,TS,102) = 2.25D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(-910/TEM(:,:,:,TS)P(:,:,:,TS))   
C
C     Reaction (103) OH + TCE = HOCH2CH2O(:,:,:,TS)2                                              
         RC(:,:,:,TS,103) = 9.64D-12*EXP(-1209/TEM(:,:,:,TS)P(:,:,:,TS))         
C
C     Reaction (104) OH + TRICLETH = HOCH2CH2O(:,:,:,TS)2                                         
         RC(:,:,:,TS,104) = 5.63D-13*EXP(427/TEM(:,:,:,TS)P(:,:,:,TS))            
C
C     Reaction (105) OH + CDICLETH = HOCH2CH2O(:,:,:,TS)2                                         
         RC(:,:,:,TS,105) = 1.94D-12*EXP(90/TEM(:,:,:,TS)P(:,:,:,TS))            
C
C     Reaction (106) OH + TDICLETH = HOCH2CH2O(:,:,:,TS)2                                         
         RC(:,:,:,TS,106) = 1.01D-12*EXP(250/TEM(:,:,:,TS)P(:,:,:,TS))           
C
C     Reaction (107) CH3O2(:,:,:,TS) + NO = HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                      
         RC(:,:,:,TS,107) = 3.00D-12*EXP(280/TEM(:,:,:,TS)P(:,:,:,TS))*0.999 
C
C     Reaction (108) C2H5O2(:,:,:,TS) + NO = CH3CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                   
         RC(:,:,:,TS,108) = 2.60D-12*EXP(365/TEM(:,:,:,TS)P(:,:,:,TS))*0.991 
C
C     Reaction (109) RN10O2(:,:,:,TS) + NO = C2H5CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,109) = 2.80D-12*EXP(360/TEM(:,:,:,TS)P(:,:,:,TS))*0.980 
C
C     Reaction (110) IC3H7O2(:,:,:,TS) + NO = CH3COCH3 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,110) = 2.70D-12*EXP(360/TEM(:,:,:,TS)P(:,:,:,TS))*0.958 
C
C     Reaction (111) RN13O2(:,:,:,TS) + NO = CH3CHO + C2H5O2(:,:,:,TS) + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,111)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.917*BR01       
C
C     Reaction (112) RN13O2(:,:,:,TS) + NO = CARB11A + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,112)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.917*(1-BR01)   
C
C     Reaction (113) RN16O2(:,:,:,TS) + NO = RN15AO2(:,:,:,TS) + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,113)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.877                 
C
C     Reaction (114) RN19O2(:,:,:,TS) + NO = RN18AO2(:,:,:,TS) + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,114)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.788                 
C
C     Reaction (115) RN13AO2(:,:,:,TS) + NO = RN12O2(:,:,:,TS) + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,115)s=sKRO2(:,:,:,TS)NO(:,:,:)                       
C
C     Reaction (116) RN16AO2(:,:,:,TS) + NO = RN15O2(:,:,:,TS) + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,116)s=sKRO2(:,:,:,TS)NO(:,:,:)                       
C
C     Reaction (117) RA13O2(:,:,:,TS) + NO = CARB3 + UDCARB8 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                          
         RC(:,:,:,TS,117)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.918       
C
C     Reaction (118) RA16O2(:,:,:,TS) + NO = CARB3 + UDCARB11 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,118)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.889*0.7 
C
C     Reaction (119) RA16O2(:,:,:,TS) + NO = CARB6 + UDCARB8 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                          
         RC(:,:,:,TS,119)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.889*0.3 
C
C     Reaction (120) RA19AO2(:,:,:,TS) + NO = CARB3 + UDCARB14 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,120)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.862       
C
C     Reaction (121) RA19CO2(:,:,:,TS) + NO = CARB9 + UDCARB8 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,121)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.862       
C
C     Reaction (122) HOCH2CH2O(:,:,:,TS)2 + NO = HCHO + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                          
         RC(:,:,:,TS,122)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.995*0.776  
C
C     Reaction (123) HOCH2CH2O(:,:,:,TS)2 + NO = HOCH2CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,123)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.995*0.224  
C
C     Reaction (124) RN9O2(:,:,:,TS) + NO = CH3CHO + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,124)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.979     
C
C     Reaction (125) RN12O2(:,:,:,TS) + NO = CH3CHO + CH3CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                          
         RC(:,:,:,TS,125)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.959     
C
C     Reaction (126) RN15O2(:,:,:,TS) + NO = C2H5CHO + CH3CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,126)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.936     
C
C     Reaction (127) RN18O2(:,:,:,TS) + NO = C2H5CHO + C2H5CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,127)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.903     
C
C     Reaction (128) RN15AO2(:,:,:,TS) + NO = CARB13 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,128)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.975     
C
C     Reaction (129) RN18AO2(:,:,:,TS) + NO = CARB16 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,129)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.946     
C
C     Reaction (130) CH3CO3 + NO = CH3O2(:,:,:,TS) + NO2(:,:,:,TS)                                          
         RC(:,:,:,TS,130)s=sKAPNO(:,:,:)                      
C
C     Reaction (131) C2H5CO3 + NO = C2H5O2(:,:,:,TS) + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,131)s=sKAPNO(:,:,:)                      
C
C     Reaction (132) HOCH2CO3 + NO = HO2(:,:,:,TS) + HCHO + NO2(:,:,:,TS)                                   
         RC(:,:,:,TS,132)s=sKAPNO(:,:,:)                      
C
C     Reaction (133) RN8O2(:,:,:,TS) + NO = CH3CO3 + HCHO + NO2(:,:,:,TS)                                   
         RC(:,:,:,TS,133)s=sKRO2(:,:,:,TS)NO(:,:,:)                     
C
C     Reaction (134) RN11O2(:,:,:,TS) + NO = CH3CO3 + CH3CHO + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,134)s=sKRO2(:,:,:,TS)NO(:,:,:)                     
C
C     Reaction (135) RN14O2(:,:,:,TS) + NO = C2H5CO3 + CH3CHO + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,135)s=sKRO2(:,:,:,TS)NO(:,:,:)                     
C
C     Reaction (136) RN17O2(:,:,:,TS) + NO = RN16AO2(:,:,:,TS) + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,136)s=sKRO2(:,:,:,TS)NO(:,:,:)                     
C
C     Reaction (137) RU14O2(:,:,:,TS) + NO = UCARB12 + HO2(:,:,:,TS) +  NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,137)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.900*0.252  
C
C     Reaction (138) RU14O2(:,:,:,TS) + NO = UCARB10 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                           
         RC(:,:,:,TS,138)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.900*0.748 
C
C     Reaction (139) RU12O2(:,:,:,TS) + NO = CH3CO3 + HOCH2CHO + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,139)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.7         
C
C     Reaction (140) RU12O2(:,:,:,TS) + NO = CARB7 + CO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,140)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.3         
C
C     Reaction (141) RU10O2(:,:,:,TS) + NO = CH3CO3 + HOCH2CHO + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,141)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.5         
C
C     Reaction (142) RU10O2(:,:,:,TS) + NO = CARB6 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,142)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.3         
C
C     Reaction (143) RU10O2(:,:,:,TS) + NO = CARB7 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,143)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.2          
C
C     Reaction (144) NRN6O2(:,:,:,TS) + NO = HCHO + HCHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,144)s=sKRO2(:,:,:,TS)NO(:,:,:)                 
C
C     Reaction (145) NRN9O2(:,:,:,TS) + NO = CH3CHO + HCHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,145)s=sKRO2(:,:,:,TS)NO(:,:,:)                 
C
C     Reaction (146) NRN12O2(:,:,:,TS) + NO = CH3CHO + CH3CHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,146)s=sKRO2(:,:,:,TS)NO(:,:,:)                 
C
C     Reaction (147) NRU14O2(:,:,:,TS) + NO = NUCARB12 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,147)s=sKRO2(:,:,:,TS)NO(:,:,:)                 
C
C     Reaction (148) NRU12O2(:,:,:,TS) + NO = NOA + CO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,148)s=sKRO2(:,:,:,TS)NO(:,:,:)                 
C
C     Reaction (149) RTN2(:,:,:,TS)8O2(:,:,:,TS) + NO = TNCARB26 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,149)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.767*0.915  
C
C     Reaction (150) RTN2(:,:,:,TS)8O2(:,:,:,TS) + NO = CH3COCH3 + RN19O2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,150)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.767*0.085  
C
C     Reaction (151) NRTN2(:,:,:,TS)8O2(:,:,:,TS) + NO = TNCARB26 + NO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,151)s=sKRO2(:,:,:,TS)NO(:,:,:)                  
C
C     Reaction (152) RTN2(:,:,:,TS)6O2(:,:,:,TS) + NO = RTN2(:,:,:,TS)5O2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,152)s=sKAPNO(:,:,:)                   
C
C     Reaction (153) RTN2(:,:,:,TS)5O2(:,:,:,TS) + NO = RTN2(:,:,:,TS)4O2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,153)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.840        
C
C     Reaction (154) RTN2(:,:,:,TS)4O2(:,:,:,TS) + NO = RTN2(:,:,:,TS)3O2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,154)s=sKRO2(:,:,:,TS)NO(:,:,:)                   
C
C     Reaction (155) RTN2(:,:,:,TS)3O2(:,:,:,TS) + NO = CH3COCH3 + RTN14O2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,155)s=sKRO2(:,:,:,TS)NO(:,:,:)                  
C
C     Reaction (156) RTN14O2(:,:,:,TS) + NO = HCHO + TNCARB10 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,156)s=sKRO2(:,:,:,TS)NO(:,:,:)               
C
C     Reaction (157) RTN10O2(:,:,:,TS) + NO = RN8O2(:,:,:,TS) + CO + NO2(:,:,:,TS)                                    
         RC(:,:,:,TS,157)s=sKRO2(:,:,:,TS)NO(:,:,:)               
C
C     Reaction (158) RTX28O2(:,:,:,TS) + NO = TXCARB24 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,158)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.767*0.915  
C
C     Reaction (159) RTX28O2(:,:,:,TS) + NO = CH3COCH3 + RN19O2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,159)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.767*0.085  
C
C     Reaction (160) NRTX28O2(:,:,:,TS) + NO = TXCARB24 + HCHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,160)s=sKRO2(:,:,:,TS)NO(:,:,:)            
C
C     Reaction (161) RTX24O2(:,:,:,TS) + NO = TXCARB22 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                
         RC(:,:,:,TS,161)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.843*0.6  
C
C     Reaction (162) RTX24O2(:,:,:,TS) + NO = CH3COCH3 + RN13AO2(:,:,:,TS) + HCHO + NO2(:,:,:,TS)                     
         RC(:,:,:,TS,162)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.843*0.4  
C
C     Reaction (163) RTX22O2(:,:,:,TS) + NO = CH3COCH3 + RN13O2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,163)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.700         
C
C     Reaction (164) CH3O2(:,:,:,TS)    + NO2(:,:,:,TS)     = CH3O2(:,:,:,TS)NO2(:,:,:,TS)                                      
         RC(:,:,:,TS,164)s=sKM(:,:,:,TS)T13(:,:,:)         
C
C     Reaction (165) CH3O2(:,:,:,TS)NO2(:,:,:,TS)           = CH3O2(:,:,:,TS)   + NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,165)s=sKM(:,:,:,TS)T14(:,:,:)         
C
C     Reaction (166) CH3O2(:,:,:,TS) + NO = CH3NO3                                                
         RC(:,:,:,TS,166) = 3.00D-12*EXP(280/TEM(:,:,:,TS)P(:,:,:,TS))*0.001 
C
C     Reaction (167) C2H5O2(:,:,:,TS) + NO = C2H5NO3                                              
         RC(:,:,:,TS,167) = 2.60D-12*EXP(365/TEM(:,:,:,TS)P(:,:,:,TS))*0.009 
C
C     Reaction (168) RN10O2(:,:,:,TS) + NO = RN10NO3                                              
         RC(:,:,:,TS,168) = 2.80D-12*EXP(360/TEM(:,:,:,TS)P(:,:,:,TS))*0.020 
C
C     Reaction (169) IC3H7O2(:,:,:,TS) + NO = IC3H7NO3                                            
         RC(:,:,:,TS,169) = 2.70D-12*EXP(360/TEM(:,:,:,TS)P(:,:,:,TS))*0.042 
C
C     Reaction (170) RN13O2(:,:,:,TS) + NO = RN13NO3                                              
         RC(:,:,:,TS,170)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.083                 
C
C     Reaction (171) RN16O2(:,:,:,TS) + NO = RN16NO3                                              
         RC(:,:,:,TS,171)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.123                 
C
C     Reaction (172) RN19O2(:,:,:,TS) + NO = RN19NO3                                              
         RC(:,:,:,TS,172)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.212                 
C
C     Reaction (173) HOCH2CH2O(:,:,:,TS)2 + NO = HOC2H4NO3                                        
         RC(:,:,:,TS,173)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.005                 
C
C     Reaction (174) RN9O2(:,:,:,TS) + NO = RN9NO3                                                
         RC(:,:,:,TS,174)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.021                 
C
C     Reaction (175) RN12O2(:,:,:,TS) + NO = RN12NO3                                              
         RC(:,:,:,TS,175)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.041                 
C
C     Reaction (176) RN15O2(:,:,:,TS) + NO = RN15NO3                                              
         RC(:,:,:,TS,176)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.064                 
C
C     Reaction (177) RN18O2(:,:,:,TS) + NO = RN18NO3                                              
         RC(:,:,:,TS,177)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.097                 
C
C     Reaction (178) RN15AO2(:,:,:,TS) + NO = RN15NO3                                             
         RC(:,:,:,TS,178)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.025                 
C
C     Reaction (179) RN18AO2(:,:,:,TS) + NO = RN18NO3                                             
         RC(:,:,:,TS,179)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.054                 
C
C     Reaction (180) RU14O2(:,:,:,TS) + NO = RU14NO3                                              
         RC(:,:,:,TS,180)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.100                 
C
C     Reaction (181) RA13O2(:,:,:,TS) + NO = RA13NO3                                              
         RC(:,:,:,TS,181)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.082                 
C
C     Reaction (182) RA16O2(:,:,:,TS) + NO = RA16NO3                                              
         RC(:,:,:,TS,182)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.111                 
C
C     Reaction (183) RA19AO2(:,:,:,TS) + NO = RA19NO3                                             
         RC(:,:,:,TS,183)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.138                 
C
C     Reaction (184) RA19CO2(:,:,:,TS) + NO = RA19NO3                                             
         RC(:,:,:,TS,184)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.138                 
C
C     Reaction (185) RTN2(:,:,:,TS)8O2(:,:,:,TS) + NO = RTN2(:,:,:,TS)8NO3                                            
         RC(:,:,:,TS,185)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.233        
C
C     Reaction (186) RTN2(:,:,:,TS)5O2(:,:,:,TS) + NO = RTN2(:,:,:,TS)5NO3                                            
         RC(:,:,:,TS,186)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.160        
C
C     Reaction (187) RTX28O2(:,:,:,TS) + NO = RTX28NO3                                            
         RC(:,:,:,TS,187)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.233        
C
C     Reaction (188) RTX24O2(:,:,:,TS) + NO = RTX24NO3                                            
         RC(:,:,:,TS,188)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.157        
C
C     Reaction (189) RTX22O2(:,:,:,TS) + NO = RTX22NO3                                            
         RC(:,:,:,TS,189)s=sKRO2(:,:,:,TS)NO(:,:,:)*0.300        
C
C     Reaction (190) CH3O2(:,:,:,TS) + NO3 = HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                     
         RC(:,:,:,TS,190)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.40          
C
C     Reaction (191) C2H5O2(:,:,:,TS) + NO3 = CH3CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,191)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (192) RN10O2(:,:,:,TS) + NO3 = C2H5CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,192)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (193) IC3H7O2(:,:,:,TS) + NO3 = CH3COCH3 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,193)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (194) RN13O2(:,:,:,TS) + NO3 = CH3CHO + C2H5O2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,194)s=sKRO2(:,:,:,TS)NO(:,:,:)3*BR01     
C
C     Reaction (195) RN13O2(:,:,:,TS) + NO3 = CARB11A + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,195)s=sKRO2(:,:,:,TS)NO(:,:,:)3*(1-BR01) 
C
C     Reaction (196) RN16O2(:,:,:,TS) + NO3 = RN15AO2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,196)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (197) RN19O2(:,:,:,TS) + NO3 = RN18AO2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,197)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (198) RN13AO2(:,:,:,TS) + NO3 = RN12O2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,198)s=sKRO2(:,:,:,TS)NO(:,:,:)3                      
C
C     Reaction (199) RN16AO2(:,:,:,TS) + NO3 = RN15O2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,199)s=sKRO2(:,:,:,TS)NO(:,:,:)3                      
C
C     Reaction (200) RA13O2(:,:,:,TS) + NO3 = CARB3 + UDCARB8 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,200)s=sKRO2(:,:,:,TS)NO(:,:,:)3            
C
C     Reaction (201) RA16O2(:,:,:,TS) + NO3 = CARB3 + UDCARB11 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,201)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.7     
C
C     Reaction (202) RA16O2(:,:,:,TS) + NO3 = CARB6 + UDCARB8 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,202)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.3     
C
C     Reaction (203) RA19AO2(:,:,:,TS) + NO3 = CARB3 + UDCARB14 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                       
         RC(:,:,:,TS,203)s=sKRO2(:,:,:,TS)NO(:,:,:)3           
C
C     Reaction (204) RA19CO2(:,:,:,TS) + NO3 = CARB9 + UDCARB8 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,204)s=sKRO2(:,:,:,TS)NO(:,:,:)3           
C
C     Reaction (205) HOCH2CH2O(:,:,:,TS)2 + NO3 = HCHO + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,205)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.776  
C
C     Reaction (206) HOCH2CH2O(:,:,:,TS)2 + NO3 = HOCH2CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,206)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.224  
C
C     Reaction (207) RN9O2(:,:,:,TS) + NO3 = CH3CHO + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,207)s=sKRO2(:,:,:,TS)NO(:,:,:)3         
C
C     Reaction (208) RN12O2(:,:,:,TS) + NO3 = CH3CHO + CH3CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                         
         RC(:,:,:,TS,208)s=sKRO2(:,:,:,TS)NO(:,:,:)3         
C
C     Reaction (209) RN15O2(:,:,:,TS) + NO3 = C2H5CHO + CH3CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,209)s=sKRO2(:,:,:,TS)NO(:,:,:)3         
C
C     Reaction (210) RN18O2(:,:,:,TS) + NO3 = C2H5CHO + C2H5CHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                       
         RC(:,:,:,TS,210)s=sKRO2(:,:,:,TS)NO(:,:,:)3         
C
C     Reaction (211) RN15AO2(:,:,:,TS) + NO3 = CARB13 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,211)s=sKRO2(:,:,:,TS)NO(:,:,:)3         
C
C     Reaction (212) RN18AO2(:,:,:,TS) + NO3 = CARB16 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,212)s=sKRO2(:,:,:,TS)NO(:,:,:)3         
C
C     Reaction (213) CH3CO3 + NO3 = CH3O2(:,:,:,TS) + NO2(:,:,:,TS)                                         
         RC(:,:,:,TS,213)s=sKRO2(:,:,:,TS)NO(:,:,:)3*1.60          
C
C     Reaction (214) C2H5CO3 + NO3 = C2H5O2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,214)s=sKRO2(:,:,:,TS)NO(:,:,:)3*1.60          
C
C     Reaction (215) HOCH2CO3 + NO3 = HO2(:,:,:,TS) + HCHO + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,215)s=sKRO2(:,:,:,TS)NO(:,:,:)3*1.60         
C
C     Reaction (216) RN8O2(:,:,:,TS) + NO3 = CH3CO3 + HCHO + NO2(:,:,:,TS)                                  
         RC(:,:,:,TS,216)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (217) RN11O2(:,:,:,TS) + NO3 = CH3CO3 + CH3CHO + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,217)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (218) RN14O2(:,:,:,TS) + NO3 = C2H5CO3 + CH3CHO + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,218)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (219) RN17O2(:,:,:,TS) + NO3 = RN16AO2(:,:,:,TS) + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,219)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (220) RU14O2(:,:,:,TS) + NO3 = UCARB12 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                                 
         RC(:,:,:,TS,220)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.252     
C
C     Reaction (221) RU14O2(:,:,:,TS) + NO3 = UCARB10 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                          
         RC(:,:,:,TS,221)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.748     
C
C     Reaction (222) RU12O2(:,:,:,TS) + NO3 = CH3CO3 + HOCH2CHO + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,222)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.7         
C
C     Reaction (223) RU12O2(:,:,:,TS) + NO3 = CARB7 + CO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,223)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.3         
C
C     Reaction (224) RU10O2(:,:,:,TS) + NO3 = CH3CO3 + HOCH2CHO + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,224)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.5         
C
C     Reaction (225) RU10O2(:,:,:,TS) + NO3 = CARB6 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,225)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.3         
C
C     Reaction (226) RU10O2(:,:,:,TS) + NO3 = CARB7 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,226)s=sKRO2(:,:,:,TS)NO(:,:,:)3*0.2         
C
C     Reaction (227) NRN6O2(:,:,:,TS) + NO3 = HCHO + HCHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                             
         RC(:,:,:,TS,227)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (228) NRN9O2(:,:,:,TS) + NO3 = CH3CHO + HCHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                           
         RC(:,:,:,TS,228)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (229) NRN12O2(:,:,:,TS) + NO3 = CH3CHO + CH3CHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,229)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (230) NRU14O2(:,:,:,TS) + NO3 = NUCARB12 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,230)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (231) NRU12O2(:,:,:,TS) + NO3 = NOA + CO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,231)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (232) RTN2(:,:,:,TS)8O2(:,:,:,TS) + NO3 = TNCARB26 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,232)s=sKRO2(:,:,:,TS)NO(:,:,:)3                
C
C     Reaction (233) NRTN2(:,:,:,TS)8O2(:,:,:,TS) + NO3 = TNCARB26 + NO2(:,:,:,TS) + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,233)s=sKRO2(:,:,:,TS)NO(:,:,:)3                
C
C     Reaction (234) RTN2(:,:,:,TS)6O2(:,:,:,TS) + NO3 = RTN2(:,:,:,TS)5O2(:,:,:,TS) + NO2(:,:,:,TS)                                      
         RC(:,:,:,TS,234)s=sKRO2(:,:,:,TS)NO(:,:,:)3*1.60                   
C
C     Reaction (235) RTN2(:,:,:,TS)5O2(:,:,:,TS) + NO3 = RTN2(:,:,:,TS)4O2(:,:,:,TS) + NO2(:,:,:,TS)                                      
         RC(:,:,:,TS,235)s=sKRO2(:,:,:,TS)NO(:,:,:)3                 
C
C     Reaction (236) RTN2(:,:,:,TS)4O2(:,:,:,TS) + NO3 = RTN2(:,:,:,TS)3O2(:,:,:,TS) + NO2(:,:,:,TS)                                      
         RC(:,:,:,TS,236)s=sKRO2(:,:,:,TS)NO(:,:,:)3                   
C
C     Reaction (237) RTN2(:,:,:,TS)3O2(:,:,:,TS) + NO3 = CH3COCH3 + RTN14O2(:,:,:,TS) + NO2(:,:,:,TS)                           
         RC(:,:,:,TS,237)s=sKRO2(:,:,:,TS)NO(:,:,:)3                 
C
C     Reaction (238) RTN14O2(:,:,:,TS) + NO3 = HCHO + TNCARB10 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,238)s=sKRO2(:,:,:,TS)NO(:,:,:)3             
C
C     Reaction (239) RTN10O2(:,:,:,TS) + NO3 = RN8O2(:,:,:,TS) + CO + NO2(:,:,:,TS)                                   
         RC(:,:,:,TS,239)s=sKRO2(:,:,:,TS)NO(:,:,:)3               
C
C     Reaction (240) RTX28O2(:,:,:,TS) + NO3 = TXCARB24 + HCHO + HO2(:,:,:,TS) + NO2(:,:,:,TS)                        
         RC(:,:,:,TS,240)s=sKRO2(:,:,:,TS)NO(:,:,:)3             
C
C     Reaction (241) RTX24O2(:,:,:,TS) + NO3 = TXCARB22 + HO2(:,:,:,TS) + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,241)s=sKRO2(:,:,:,TS)NO(:,:,:)3             
C
C     Reaction (242) RTX22O2(:,:,:,TS) + NO3 = CH3COCH3 + RN13O2(:,:,:,TS) + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,242)s=sKRO2(:,:,:,TS)NO(:,:,:)3             
C
C     Reaction (243) NRTX28O2(:,:,:,TS) + NO3 = TXCARB24 + HCHO + NO2(:,:,:,TS) + NO2(:,:,:,TS)                       
         RC(:,:,:,TS,243)s=sKRO2(:,:,:,TS)NO(:,:,:)3            
C
C     Reaction (244) CH3O2(:,:,:,TS) + HO2(:,:,:,TS) = CH3OOH                                               
         RC(:,:,:,TS,244) = 4.10D-13*EXP(790/TEM(:,:,:,TS)P(:,:,:,TS))  
C
C     Reaction (245) C2H5O2(:,:,:,TS) + HO2(:,:,:,TS) = C2H5OOH                                             
         RC(:,:,:,TS,245) = 7.50D-13*EXP(700/TEM(:,:,:,TS)P(:,:,:,TS))  
C
C     Reaction (246) RN10O2(:,:,:,TS) + HO2(:,:,:,TS) = RN10OOH                                             
         RC(:,:,:,TS,246)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.520           
C
C     Reaction (247) IC3H7O2(:,:,:,TS) + HO2(:,:,:,TS) = IC3H7OOH                                           
         RC(:,:,:,TS,247)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.520           
C
C     Reaction (248) RN13O2(:,:,:,TS) + HO2(:,:,:,TS) = RN13OOH                                             
         RC(:,:,:,TS,248)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625           
C
C     Reaction (249) RN16O2(:,:,:,TS) + HO2(:,:,:,TS) = RN16OOH                                             
         RC(:,:,:,TS,249)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706           
C
C     Reaction (250) RN19O2(:,:,:,TS) + HO2(:,:,:,TS) = RN19OOH                                             
         RC(:,:,:,TS,250)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770           
C
C     Reaction (251) RN13AO2(:,:,:,TS) + HO2(:,:,:,TS) = RN13OOH                                            
         RC(:,:,:,TS,251)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625           
C
C     Reaction (252) RN16AO2(:,:,:,TS) + HO2(:,:,:,TS) = RN16OOH                                            
         RC(:,:,:,TS,252)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706           
C
C     Reaction (253) RA13O2(:,:,:,TS) + HO2(:,:,:,TS) = RA13OOH                                             
         RC(:,:,:,TS,253)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770           
C
C     Reaction (254) RA16O2(:,:,:,TS) + HO2(:,:,:,TS) = RA16OOH                                             
         RC(:,:,:,TS,254)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.820           
C
C     Reaction (255) RA19AO2(:,:,:,TS) + HO2(:,:,:,TS) = RA19OOH                                            
         RC(:,:,:,TS,255)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.859           
C
C     Reaction (256) RA19CO2(:,:,:,TS) + HO2(:,:,:,TS) = RA19OOH                                            
         RC(:,:,:,TS,256)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.859           
C
C     Reaction (257) HOCH2CH2O(:,:,:,TS)2 + HO2(:,:,:,TS) = HOC2H4OOH                                       
         RC(:,:,:,TS,257) = 2.03D-13*EXP(1250/TEM(:,:,:,TS)P(:,:,:,TS)) 
C
C     Reaction (258) RN9O2(:,:,:,TS) + HO2(:,:,:,TS) = RN9OOH                                               
         RC(:,:,:,TS,258)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.520           
C
C     Reaction (259) RN12O2(:,:,:,TS) + HO2(:,:,:,TS) = RN12OOH                                             
         RC(:,:,:,TS,259)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625           
C
C     Reaction (260) RN15O2(:,:,:,TS) + HO2(:,:,:,TS) = RN15OOH                                             
         RC(:,:,:,TS,260)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706           
C
C     Reaction (261) RN18O2(:,:,:,TS) + HO2(:,:,:,TS) = RN18OOH                                             
         RC(:,:,:,TS,261)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770           
C
C     Reaction (262) RN15AO2(:,:,:,TS) + HO2(:,:,:,TS) = RN15OOH                                            
         RC(:,:,:,TS,262)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706           
C
C     Reaction (263) RN18AO2(:,:,:,TS) + HO2(:,:,:,TS) = RN18OOH                                            
         RC(:,:,:,TS,263)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770           
C
C     Reaction (264) CH3CO3 + HO2(:,:,:,TS) = CH3CO3H                                             
         RC(:,:,:,TS,264)s=sKAPHO2(:,:,:,TS)(:,:,:)                  
C
C     Reaction (265) C2H5CO3 + HO2(:,:,:,TS) = C2H5CO3H                                           
         RC(:,:,:,TS,265)s=sKAPHO2(:,:,:,TS)(:,:,:)                  
C
C     Reaction (266) HOCH2CO3 + HO2(:,:,:,TS) = HOCH2CO3H                                         
         RC(:,:,:,TS,266)s=sKAPHO2(:,:,:,TS)(:,:,:)                  
C
C     Reaction (267) RN8O2(:,:,:,TS) + HO2(:,:,:,TS) = RN8OOH                                               
         RC(:,:,:,TS,267)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.520           
C
C     Reaction (268) RN11O2(:,:,:,TS) + HO2(:,:,:,TS) = RN11OOH                                             
         RC(:,:,:,TS,268)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625           
C
C     Reaction (269) RN14O2(:,:,:,TS) + HO2(:,:,:,TS) = RN14OOH                                             
         RC(:,:,:,TS,269)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706           
C
C     Reaction (270) RN17O2(:,:,:,TS) + HO2(:,:,:,TS) = RN17OOH                                             
         RC(:,:,:,TS,270)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770           
C
C     Reaction (271) RU14O2(:,:,:,TS) + HO2(:,:,:,TS) = RU14OOH                                             
         RC(:,:,:,TS,271)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770           
C
C     Reaction (272) RU12O2(:,:,:,TS) + HO2(:,:,:,TS) = RU12OOH                                             
         RC(:,:,:,TS,272)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706           
C
C     Reaction (273) RU10O2(:,:,:,TS) + HO2(:,:,:,TS) = RU10OOH                                             
         RC(:,:,:,TS,273)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625           
C
C     Reaction (274) NRN6O2(:,:,:,TS) + HO2(:,:,:,TS) = NRN6OOH                                             
         RC(:,:,:,TS,274)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.387         
C
C     Reaction (275) NRN9O2(:,:,:,TS) + HO2(:,:,:,TS) = NRN9OOH                                             
         RC(:,:,:,TS,275)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.520         
C
C     Reaction (276) NRN12O2(:,:,:,TS) + HO2(:,:,:,TS) = NRN12OOH                                           
         RC(:,:,:,TS,276)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625         
C
C     Reaction (277) NRU14O2(:,:,:,TS) + HO2(:,:,:,TS) = NRU14OOH                                           
         RC(:,:,:,TS,277)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770         
C
C     Reaction (278) NRU12O2(:,:,:,TS) + HO2(:,:,:,TS) = NRU12OOH                                           
         RC(:,:,:,TS,278)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.625         
C
C     Reaction (279) RTN2(:,:,:,TS)8O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN2(:,:,:,TS)8OOH                                           
         RC(:,:,:,TS,279)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.914         
C
C     Reaction (280) NRTN2(:,:,:,TS)8O2(:,:,:,TS) + HO2(:,:,:,TS) = NRTN2(:,:,:,TS)8OOH                                         
         RC(:,:,:,TS,280)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.914         
C
C     Reaction (281) RTN2(:,:,:,TS)6O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN2(:,:,:,TS)6OOH                                           
         RC(:,:,:,TS,281)s=sKAPHO2(:,:,:,TS)(:,:,:)                     
C
C     Reaction (282) RTN2(:,:,:,TS)5O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN2(:,:,:,TS)5OOH                                           
         RC(:,:,:,TS,282)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.890       
C
C     Reaction (283) RTN2(:,:,:,TS)4O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN2(:,:,:,TS)4OOH                                           
         RC(:,:,:,TS,283)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.890       
C
C     Reaction (284) RTN2(:,:,:,TS)3O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN2(:,:,:,TS)3OOH                                           
         RC(:,:,:,TS,284)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.890       
C
C     Reaction (285) RTN14O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN14OOH                                           
         RC(:,:,:,TS,285)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.770       
C
C     Reaction (286) RTN10O2(:,:,:,TS) + HO2(:,:,:,TS) = RTN10OOH                                           
         RC(:,:,:,TS,286)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.706       
C
C     Reaction (287) RTX28O2(:,:,:,TS) + HO2(:,:,:,TS) = RTX28OOH                                           
         RC(:,:,:,TS,287)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.914       
C
C     Reaction (288) RTX24O2(:,:,:,TS) + HO2(:,:,:,TS) = RTX24OOH                                           
         RC(:,:,:,TS,288)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.890       
C
C     Reaction (289) RTX22O2(:,:,:,TS) + HO2(:,:,:,TS) = RTX22OOH                                           
         RC(:,:,:,TS,289)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.890       
C
C     Reaction (290) NRTX28O2(:,:,:,TS) + HO2(:,:,:,TS) = NRTX28OOH                                         
         RC(:,:,:,TS,290)s=sKRO2(:,:,:,TS)HO(:,:,:)2*0.914       
C
C     Reaction (291) CH3O2(:,:,:,TS) = HCHO + HO2(:,:,:,TS)                                                 
         RC(:,:,:,TS,291) = 1.82D-13*EXP(416/TEM(:,:,:,TS)P(:,:,:,TS))*0.33*RO2(:,:,:,TS)  
C
C     Reaction (292) CH3O2(:,:,:,TS) = HCHO                                                       
         RC(:,:,:,TS,292) = 1.82D-13*EXP(416/TEM(:,:,:,TS)P(:,:,:,TS))*0.335*RO2(:,:,:,TS) 
C
C     Reaction (293) CH3O2(:,:,:,TS) = CH3OH                                                      
         RC(:,:,:,TS,293) = 1.82D-13*EXP(416/TEM(:,:,:,TS)P(:,:,:,TS))*0.335*RO2(:,:,:,TS) 
C
C     Reaction (294) C2H5O2(:,:,:,TS) = CH3CHO + HO2(:,:,:,TS)                                              
         RC(:,:,:,TS,294) = 3.10D-13*0.6*RO2(:,:,:,TS)             
C
C     Reaction (295) C2H5O2(:,:,:,TS) = CH3CHO                                                    
         RC(:,:,:,TS,295) = 3.10D-13*0.2*RO2(:,:,:,TS)             
C
C     Reaction (296) C2H5O2(:,:,:,TS) = C2H5OH                                                    
         RC(:,:,:,TS,296) = 3.10D-13*0.2*RO2(:,:,:,TS)             
C
C     Reaction (297) RN10O2(:,:,:,TS) = C2H5CHO + HO2(:,:,:,TS)                                             
         RC(:,:,:,TS,297) = 6.00D-13*0.6*RO2(:,:,:,TS)             
C
C     Reaction (298) RN10O2(:,:,:,TS) = C2H5CHO                                                   
         RC(:,:,:,TS,298) = 6.00D-13*0.2*RO2(:,:,:,TS)             
C
C     Reaction (299) RN10O2(:,:,:,TS) = NPROPOL                                                   
         RC(:,:,:,TS,299) = 6.00D-13*0.2*RO2(:,:,:,TS)             
C
C     Reaction (300) IC3H7O2(:,:,:,TS) = CH3COCH3 + HO2(:,:,:,TS)                                           
         RC(:,:,:,TS,300) = 4.00D-14*0.6*RO2(:,:,:,TS)             
C
C     Reaction (301) IC3H7O2(:,:,:,TS) = CH3COCH3                                                 
         RC(:,:,:,TS,301) = 4.00D-14*0.2*RO2(:,:,:,TS)             
C
C     Reaction (302) IC3H7O2(:,:,:,TS) = IPROPOL                                                  
         RC(:,:,:,TS,302) = 4.00D-14*0.2*RO2(:,:,:,TS)             
C
C     Reaction (303) RN13O2(:,:,:,TS) = CH3CHO + C2H5O2(:,:,:,TS)                                           
         RC(:,:,:,TS,303) = 2.50D-13*RO2(:,:,:,TS)*BR01       
C
C     Reaction (304) RN13O2(:,:,:,TS) = CARB11A + HO2(:,:,:,TS)                                             
         RC(:,:,:,TS,304) = 2.50D-13*RO2(:,:,:,TS)*(1-BR01)   
C
C     Reaction (305) RN13AO2(:,:,:,TS) = RN12O2(:,:,:,TS)                                                   
         RC(:,:,:,TS,305) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (306) RN16AO2(:,:,:,TS) = RN15O2(:,:,:,TS)                                                   
         RC(:,:,:,TS,306) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (307) RA13O2(:,:,:,TS) = CARB3 + UDCARB8 + HO2(:,:,:,TS)                                     
         RC(:,:,:,TS,307) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (308) RA16O2(:,:,:,TS) = CARB3 + UDCARB11 + HO2(:,:,:,TS)                                    
         RC(:,:,:,TS,308) = 8.80D-13*RO2(:,:,:,TS)*0.7          
C
C     Reaction (309) RA16O2(:,:,:,TS) = CARB6 + UDCARB8 + HO2(:,:,:,TS)                                     
         RC(:,:,:,TS,309) = 8.80D-13*RO2(:,:,:,TS)*0.3          
C
C     Reaction (310) RA19AO2(:,:,:,TS) = CARB3 + UDCARB14 + HO2(:,:,:,TS)                                   
         RC(:,:,:,TS,310) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (311) RA19CO2(:,:,:,TS) = CARB3 + UDCARB14 + HO2(:,:,:,TS)                                   
         RC(:,:,:,TS,311) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (312) RN16O2(:,:,:,TS) = RN15AO2(:,:,:,TS)                                                   
         RC(:,:,:,TS,312) = 2.50D-13*RO2(:,:,:,TS)                 
C
C     Reaction (313) RN19O2(:,:,:,TS) = RN18AO2(:,:,:,TS)                                                   
         RC(:,:,:,TS,313) = 2.50D-13*RO2(:,:,:,TS)                 
C
C     Reaction (314) HOCH2CH2O(:,:,:,TS)2 = HCHO + HCHO + HO2(:,:,:,TS)                                     
         RC(:,:,:,TS,314) = 2.00D-12*RO2(:,:,:,TS)*0.776       
C
C     Reaction (315) HOCH2CH2O(:,:,:,TS)2 = HOCH2CHO + HO2(:,:,:,TS)                                        
         RC(:,:,:,TS,315) = 2.00D-12*RO2(:,:,:,TS)*0.224       
C
C     Reaction (316) RN9O2(:,:,:,TS) = CH3CHO + HCHO + HO2(:,:,:,TS)                                        
         RC(:,:,:,TS,316) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (317) RN12O2(:,:,:,TS) = CH3CHO + CH3CHO + HO2(:,:,:,TS)                                     
         RC(:,:,:,TS,317) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (318) RN15O2(:,:,:,TS) = C2H5CHO + CH3CHO + HO2(:,:,:,TS)                                    
         RC(:,:,:,TS,318) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (319) RN18O2(:,:,:,TS) = C2H5CHO + C2H5CHO + HO2(:,:,:,TS)                                   
         RC(:,:,:,TS,319) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (320) RN15AO2(:,:,:,TS) = CARB13 + HO2(:,:,:,TS)                                             
         RC(:,:,:,TS,320) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (321) RN18AO2(:,:,:,TS) = CARB16 + HO2(:,:,:,TS)                                             
         RC(:,:,:,TS,321) = 8.80D-13*RO2(:,:,:,TS)                 
C
C     Reaction (322) CH3CO3 = CH3O2(:,:,:,TS)                                                     
         RC(:,:,:,TS,322) = 1.00D-11*RO2(:,:,:,TS)                 
C
C     Reaction (323) C2H5CO3 = C2H5O2(:,:,:,TS)                                                   
         RC(:,:,:,TS,323) = 1.00D-11*RO2(:,:,:,TS)                 
C
C     Reaction (324) HOCH2CO3 = HCHO + HO2(:,:,:,TS)                                              
         RC(:,:,:,TS,324) = 1.00D-11*RO2(:,:,:,TS)                 
C
C     Reaction (325) RN8O2(:,:,:,TS) = CH3CO3 + HCHO                                              
         RC(:,:,:,TS,325) = 1.40D-12*RO2(:,:,:,TS)                 
C
C     Reaction (326) RN11O2(:,:,:,TS) = CH3CO3 + CH3CHO                                           
         RC(:,:,:,TS,326) = 1.40D-12*RO2(:,:,:,TS)                 
C
C     Reaction (327) RN14O2(:,:,:,TS) = C2H5CO3 + CH3CHO                                          
         RC(:,:,:,TS,327) = 1.40D-12*RO2(:,:,:,TS)                 
C
C     Reaction (328) RN17O2(:,:,:,TS) = RN16AO2(:,:,:,TS)                                                   
         RC(:,:,:,TS,328) = 1.40D-12*RO2(:,:,:,TS)                 
C
C     Reaction (329) RU14O2(:,:,:,TS) = UCARB12 + HO2(:,:,:,TS)                                             
         RC(:,:,:,TS,329) = 1.71D-12*RO2(:,:,:,TS)*0.252        
C
C     Reaction (330) RU14O2(:,:,:,TS) = UCARB10 + HCHO + HO2(:,:,:,TS)                                      
         RC(:,:,:,TS,330) = 1.71D-12*RO2(:,:,:,TS)*0.748        
C
C     Reaction (331) RU12O2(:,:,:,TS) = CH3CO3 + HOCH2CHO                                         
         RC(:,:,:,TS,331) = 2.00D-12*RO2(:,:,:,TS)*0.7            
C
C     Reaction (332) RU12O2(:,:,:,TS) = CARB7 + HOCH2CHO + HO2(:,:,:,TS)                                    
         RC(:,:,:,TS,332) = 2.00D-12*RO2(:,:,:,TS)*0.3            
C
C     Reaction (333) RU10O2(:,:,:,TS) = CH3CO3 + HOCH2CHO                                         
         RC(:,:,:,TS,333) = 2.00D-12*RO2(:,:,:,TS)*0.5            
C
C     Reaction (334) RU10O2(:,:,:,TS) = CARB6 + HCHO + HO2(:,:,:,TS)                                        
         RC(:,:,:,TS,334) = 2.00D-12*RO2(:,:,:,TS)*0.3            
C
C     Reaction (335) RU10O2(:,:,:,TS) = CARB7 + HCHO + HO2(:,:,:,TS)                                        
         RC(:,:,:,TS,335) = 2.00D-12*RO2(:,:,:,TS)*0.2            
C
C     Reaction (336) NRN6O2(:,:,:,TS) = HCHO + HCHO + NO2(:,:,:,TS)                                         
         RC(:,:,:,TS,336) = 6.00D-13*RO2(:,:,:,TS)                 
C
C     Reaction (337) NRN9O2(:,:,:,TS) = CH3CHO + HCHO + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,337) = 2.30D-13*RO2(:,:,:,TS)                 
C
C     Reaction (338) NRN12O2(:,:,:,TS) = CH3CHO + CH3CHO + NO2(:,:,:,TS)                                    
         RC(:,:,:,TS,338) = 2.50D-13*RO2(:,:,:,TS)                 
C
C     Reaction (339) NRU14O2(:,:,:,TS) = NUCARB12 + HO2(:,:,:,TS)                                           
         RC(:,:,:,TS,339) = 1.30D-12*RO2(:,:,:,TS)                 
C
C     Reaction (340) NRU12O2(:,:,:,TS) = NOA + CO + HO2(:,:,:,TS)                                           
         RC(:,:,:,TS,340) = 9.60D-13*RO2(:,:,:,TS)                 
C
C     Reaction (341) RTN2(:,:,:,TS)8O2(:,:,:,TS) = TNCARB26 + HO2(:,:,:,TS)                                           
         RC(:,:,:,TS,341) = 2.85D-13*RO2(:,:,:,TS)                 
C
C     Reaction (342) NRTN2(:,:,:,TS)8O2(:,:,:,TS) = TNCARB26 + NO2(:,:,:,TS)                                          
         RC(:,:,:,TS,342) = 1.00D-13*RO2(:,:,:,TS)                 
C
C     Reaction (343) RTN2(:,:,:,TS)6O2(:,:,:,TS) = RTN2(:,:,:,TS)5O2(:,:,:,TS)                                                  
         RC(:,:,:,TS,343) = 1.00D-11*RO2(:,:,:,TS)                   
C
C     Reaction (344) RTN2(:,:,:,TS)5O2(:,:,:,TS) = RTN2(:,:,:,TS)4O2(:,:,:,TS)                                                  
         RC(:,:,:,TS,344) = 1.30D-12*RO2(:,:,:,TS)           
C
C     Reaction (345) RTN2(:,:,:,TS)4O2(:,:,:,TS) = RTN2(:,:,:,TS)3O2(:,:,:,TS)                                                  
         RC(:,:,:,TS,345) = 6.70D-15*RO2(:,:,:,TS)             
C
C     Reaction (346) RTN2(:,:,:,TS)3O2(:,:,:,TS) = CH3COCH3 + RTN14O2(:,:,:,TS)                                       
         RC(:,:,:,TS,346) = 6.70D-15*RO2(:,:,:,TS)            
C
C     Reaction (347) RTN14O2(:,:,:,TS) = HCHO + TNCARB10 + HO2(:,:,:,TS)                                    
         RC(:,:,:,TS,347) = 8.80D-13*RO2(:,:,:,TS)        
C
C     Reaction (348) RTN10O2(:,:,:,TS) = RN8O2(:,:,:,TS) + CO                                               
         RC(:,:,:,TS,348) = 2.00D-12*RO2(:,:,:,TS)        
C
C     Reaction (349) RTX28O2(:,:,:,TS) = TXCARB24 + HCHO + HO2(:,:,:,TS)                                    
         RC(:,:,:,TS,349) = 2.00D-12*RO2(:,:,:,TS)       
C
C     Reaction (350) RTX24O2(:,:,:,TS) = TXCARB22 + HO2(:,:,:,TS)                                           
         RC(:,:,:,TS,350) = 2.50D-13*RO2(:,:,:,TS)       
C
C     Reaction (351) RTX22O2(:,:,:,TS) = CH3COCH3 + RN13O2(:,:,:,TS)                                        
         RC(:,:,:,TS,351) = 2.50D-13*RO2(:,:,:,TS)       
C
C     Reaction (352) NRTX28O2(:,:,:,TS) = TXCARB24 + HCHO + NO2(:,:,:,TS)                                   
         RC(:,:,:,TS,352) = 9.20D-14*RO2(:,:,:,TS)       
C
C     Reaction (353) OH + CARB14 = RN14O2(:,:,:,TS)                                               
         RC(:,:,:,TS,353) = 1.87D-11       
C
C     Reaction (354) OH + CARB17 = RN17O2(:,:,:,TS)                                               
         RC(:,:,:,TS,354) = 4.36D-12       
C
C     Reaction (355) OH + CARB11A = RN11O2(:,:,:,TS)                                              
         RC(:,:,:,TS,355) = 3.24D-18*TEM(:,:,:,TS)P(:,:,:,TS)**2*EXP(414/TEM(:,:,:,TS)P(:,:,:,TS))
C
C     Reaction (356) OH + CARB7 = CARB6 + HO2(:,:,:,TS)                                           
         RC(:,:,:,TS,356) = 3.00D-12       
C
C     Reaction (357) OH + CARB10 = CARB9 + HO2(:,:,:,TS)                                          
         RC(:,:,:,TS,357) = 5.86D-12       
C
C     Reaction (358) OH + CARB13 = RN13O2(:,:,:,TS)                                               
         RC(:,:,:,TS,358) = 1.65D-11       
C
C     Reaction (359) OH + CARB16 = RN16O2(:,:,:,TS)                                               
         RC(:,:,:,TS,359) = 1.25D-11       
C
C     Reaction (360) OH + UCARB10 = RU10O2(:,:,:,TS)                                              
         RC(:,:,:,TS,360) = 2.50D-11       
C
C     Reaction (361) NO3 + UCARB10 = RU10O2(:,:,:,TS) + HNO3                                      
         RC(:,:,:,TS,361)s=sKNO3AL(:,:,:)       
C
C     Reaction (362) O3 + UCARB10 = HCHO + CH3CO3 + CO + OH                             
         RC(:,:,:,TS,362) = 2.85D-18*0.59       
C
C     Reaction (363) O3 + UCARB10 = HCHO + CARB6 + H2O(:,:,:,TS)2                                 
         RC(:,:,:,TS,363) = 2.85D-18*0.41       
C
C     Reaction (364) OH + HOCH2CHO = HOCH2CO3                                           
         RC(:,:,:,TS,364) = 1.00D-11       
C
C     Reaction (365) NO3 + HOCH2CHO = HOCH2CO3 + HNO3                                   
         RC(:,:,:,TS,365)s=sKNO3AL(:,:,:)        
C
C     Reaction (366) OH + CARB3 = CO + CO + HO2(:,:,:,TS)                                         
         RC(:,:,:,TS,366) = 1.14D-11       
C
C     Reaction (367) OH + CARB6 = CH3CO3 + CO                                           
         RC(:,:,:,TS,367) = 1.72D-11       
C
C     Reaction (368) OH + CARB9 = RN9O2(:,:,:,TS)                                                 
         RC(:,:,:,TS,368) = 2.40D-13       
C
C     Reaction (369) OH + CARB12 = RN12O2(:,:,:,TS)                                               
         RC(:,:,:,TS,369) = 1.38D-12       
C
C     Reaction (370) OH + CARB15 = RN15O2(:,:,:,TS)                                               
         RC(:,:,:,TS,370) = 4.81D-12       
C
C     Reaction (371) OH + CCARB12 = RN12O2(:,:,:,TS)                                              
         RC(:,:,:,TS,371) = 4.79D-12       
C
C     Reaction (372) OH + UCARB12 = RU12O2(:,:,:,TS)                                              
         RC(:,:,:,TS,372) = 4.52D-11            
C
C     Reaction (373) NO3 + UCARB12 = RU12O2(:,:,:,TS) + HNO3                                      
         RC(:,:,:,TS,373)s=sKNO3AL(:,:,:)*4.25    
C
C     Reaction (374) O3 + UCARB12 = HOCH2CHO + CH3CO3 + CO + OH                         
         RC(:,:,:,TS,374) = 2.40D-17*0.89   
C
C     Reaction (375) O3 + UCARB12 = HOCH2CHO + CARB6 + H2O(:,:,:,TS)2                             
         RC(:,:,:,TS,375) = 2.40D-17*0.11   
C
C     Reaction (376) OH + NUCARB12 = NRU12O2(:,:,:,TS)                                            
         RC(:,:,:,TS,376) = 4.16D-11            
C
C     Reaction (377) OH + NOA = CARB6 + NO2(:,:,:,TS)                                             
         RC(:,:,:,TS,377) = 1.30D-13            
C
C     Reaction (378) OH + UDCARB8 = C2H5O2(:,:,:,TS)                                              
         RC(:,:,:,TS,378) = 5.20D-11*0.50        
C
C     Reaction (379) OH + UDCARB8 = ANHY + HO2(:,:,:,TS)                                          
         RC(:,:,:,TS,379) = 5.20D-11*0.50        
C
C     Reaction (380) OH + UDCARB11 = RN10O2(:,:,:,TS)                                             
         RC(:,:,:,TS,380) = 5.58D-11*0.55     
C
C     Reaction (381) OH + UDCARB11 = ANHY + CH3O2(:,:,:,TS)                                       
         RC(:,:,:,TS,381) = 5.58D-11*0.45     
C
C     Reaction (382) OH + UDCARB14 = RN13O2(:,:,:,TS)                                             
         RC(:,:,:,TS,382) = 7.00D-11*0.55     
C
C     Reaction (383) OH + UDCARB14 = ANHY + C2H5O2(:,:,:,TS)                                      
         RC(:,:,:,TS,383) = 7.00D-11*0.45     
C
C     Reaction (384) OH + TNCARB26 = RTN2(:,:,:,TS)6O2(:,:,:,TS)                                            
         RC(:,:,:,TS,384) = 4.20D-11           
C
C     Reaction (385) OH + TNCARB15 = RN15AO2(:,:,:,TS)                                            
         RC(:,:,:,TS,385) = 1.00D-12           
C
C     Reaction (386) OH + TNCARB10 = RTN10O2(:,:,:,TS)                                            
         RC(:,:,:,TS,386) = 1.00D-10           
C
C     Reaction (387) NO3 + TNCARB26 = RTN2(:,:,:,TS)6O2(:,:,:,TS) + HNO3                                    
         RC(:,:,:,TS,387) = 3.80D-14            
C
C     Reaction (388) NO3 + TNCARB10 = RTN10O2(:,:,:,TS) + HNO3                                    
         RC(:,:,:,TS,388)s=sKNO3AL(:,:,:)*5.5      
C
C     Reaction (389) OH + RCOOH25 = RTN2(:,:,:,TS)5O2(:,:,:,TS)                                             
         RC(:,:,:,TS,389) = 6.65D-12            
C
C     Reaction (390) OH + TXCARB24 = RTX24O2(:,:,:,TS)                                            
         RC(:,:,:,TS,390) = 1.55D-11           
C
C     Reaction (391) OH + TXCARB22 = RTX22O2(:,:,:,TS)                                            
         RC(:,:,:,TS,391) = 4.55D-12           
C
C     Reaction (392) OH + CH3NO3 = HCHO + NO2(:,:,:,TS)                                           
         RC(:,:,:,TS,392) = 1.00D-14*EXP(1060/TEM(:,:,:,TS)P(:,:,:,TS))      
C
C     Reaction (393) OH + C2H5NO3 = CH3CHO + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,393) = 4.40D-14*EXP(720/TEM(:,:,:,TS)P(:,:,:,TS))       
C
C     Reaction (394) OH + RN10NO3 = C2H5CHO + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,394) = 7.30D-13                     
C
C     Reaction (395) OH + IC3H7NO3 = CH3COCH3 + NO2(:,:,:,TS)                                     
         RC(:,:,:,TS,395) = 4.90D-13                     
C
C     Reaction (396) OH + RN13NO3 = CARB11A + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,396) = 9.20D-13                     
C
C     Reaction (397) OH + RN16NO3 = CARB14 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,397) = 1.85D-12                     
C
C     Reaction (398) OH + RN19NO3 = CARB17 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,398) = 3.02D-12                     
C
C     Reaction (399) OH + HOC2H4NO3 = HOCH2CHO + NO2(:,:,:,TS)                                    
         RC(:,:,:,TS,399) = 1.09D-12               
C
C     Reaction (400) OH + RN9NO3 = CARB7 + NO2(:,:,:,TS)                                          
         RC(:,:,:,TS,400) = 1.31D-12               
C
C     Reaction (401) OH + RN12NO3 = CARB10 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,401) = 1.79D-12               
C
C     Reaction (402) OH + RN15NO3 = CARB13 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,402) = 1.03D-11               
C
C     Reaction (403) OH + RN18NO3 = CARB16 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,403) = 1.34D-11               
C
C     Reaction (404) OH + RU14NO3 = UCARB12 + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,404) = 5.55D-11               
C
C     Reaction (405) OH + RA13NO3 = CARB3 + UDCARB8 + NO2(:,:,:,TS)                               
         RC(:,:,:,TS,405) = 7.30D-11               
C
C     Reaction (406) OH + RA16NO3 = CARB3 + UDCARB11 + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,406) = 7.16D-11               
C
C     Reaction (407) OH + RA19NO3 = CARB6 + UDCARB11 + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,407) = 8.31D-11               
C
C     Reaction (408) OH + RTN2(:,:,:,TS)8NO3 = TNCARB26 + NO2(:,:,:,TS)                                     
         RC(:,:,:,TS,408) = 4.35D-12               
C
C     Reaction (409) OH + RTN2(:,:,:,TS)5NO3 = CH3COCH3 + TNCARB15 + NO2(:,:,:,TS)                          
         RC(:,:,:,TS,409) = 2.88D-12               
C
C     Reaction (410) OH + RTX28NO3 = TXCARB24 + HCHO + NO2(:,:,:,TS)                              
         RC(:,:,:,TS,410) = 3.53D-12                  
C
C     Reaction (411) OH + RTX24NO3 = TXCARB22 + NO2(:,:,:,TS)                                     
         RC(:,:,:,TS,411) = 6.48D-12                  
C
C     Reaction (412) OH + RTX22NO3 = CH3COCH3 + CCARB12 + NO2(:,:,:,TS)                           
         RC(:,:,:,TS,412) = 4.74D-12                  
C
C     Reaction (413) OH + AROH14 = RAROH14                                              
         RC(:,:,:,TS,413) = 2.63D-11             
C
C     Reaction (414) NO3 + AROH14 = RAROH14 + HNO3                                      
         RC(:,:,:,TS,414) = 3.78D-12               
C
C     Reaction (415) RAROH14 + NO2(:,:,:,TS) = ARNOH14                                            
         RC(:,:,:,TS,415) = 2.08D-12               
C
C     Reaction (416) OH + ARNOH14 = CARB13 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,416) = 9.00D-13               
C
C     Reaction (417) NO3 + ARNOH14 = CARB13 + NO2(:,:,:,TS) + HNO3                                
         RC(:,:,:,TS,417) = 9.00D-14               
C
C     Reaction (418) OH + AROH17 = RAROH17                                              
         RC(:,:,:,TS,418) = 4.65D-11               
C
C     Reaction (419) NO3 + AROH17 = RAROH17 + HNO3                                      
         RC(:,:,:,TS,419) = 1.25D-11               
C
C     Reaction (420) RAROH17 + NO2(:,:,:,TS) = ARNOH17                                            
         RC(:,:,:,TS,420) = 2.08D-12               
C
C     Reaction (421) OH + ARNOH17 = CARB16 + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,421) = 1.53D-12               
C
C     Reaction (422) NO3 + ARNOH17 = CARB16 + NO2(:,:,:,TS) + HNO3                                
         RC(:,:,:,TS,422) = 3.13D-13               
C
C     Reaction (423) OH + CH3OOH = CH3O2(:,:,:,TS)                                                
         RC(:,:,:,TS,423) = 1.90D-11*EXP(190/TEM(:,:,:,TS)P(:,:,:,TS))       
C
C     Reaction (424) OH + CH3OOH = HCHO + OH                                            
         RC(:,:,:,TS,424) = 1.00D-11*EXP(190/TEM(:,:,:,TS)P(:,:,:,TS))       
C
C     Reaction (425) OH + C2H5OOH = CH3CHO + OH                                         
         RC(:,:,:,TS,425) = 1.36D-11               
C
C     Reaction (426) OH + RN10OOH = C2H5CHO + OH                                        
         RC(:,:,:,TS,426) = 1.89D-11               
C
C     Reaction (427) OH + IC3H7OOH = CH3COCH3 + OH                                      
         RC(:,:,:,TS,427) = 2.78D-11               
C
C     Reaction (428) OH + RN13OOH = CARB11A + OH                                        
         RC(:,:,:,TS,428) = 3.57D-11               
C
C     Reaction (429) OH + RN16OOH = CARB14 + OH                                         
         RC(:,:,:,TS,429) = 4.21D-11               
C
C     Reaction (430) OH + RN19OOH = CARB17 + OH                                         
         RC(:,:,:,TS,430) = 4.71D-11               
C
C     Reaction (431) OH + CH3CO3H = CH3CO3                                              
         RC(:,:,:,TS,431) = 3.70D-12                     
C
C     Reaction (432) OH + C2H5CO3H = C2H5CO3                                            
         RC(:,:,:,TS,432) = 4.42D-12                     
C
C     Reaction (433) OH + HOCH2CO3H = HOCH2CO3                                          
         RC(:,:,:,TS,433) = 6.19D-12                     
C
C     Reaction (434) OH + RN8OOH = CARB6 + OH                                           
         RC(:,:,:,TS,434) = 4.42D-12                     
C
C     Reaction (435) OH + RN11OOH = CARB9 + OH                                          
         RC(:,:,:,TS,435) = 2.50D-11                     
C
C     Reaction (436) OH + RN14OOH = CARB12 + OH                                         
         RC(:,:,:,TS,436) = 3.20D-11                     
C
C     Reaction (437) OH + RN17OOH = CARB15 + OH                                         
         RC(:,:,:,TS,437) = 3.35D-11                     
C
C     Reaction (438) OH + RU14OOH = UCARB12 + OH                                        
         RC(:,:,:,TS,438) = 7.51D-11                     
C
C     Reaction (439) OH + RU12OOH = RU12O2(:,:,:,TS)                                              
         RC(:,:,:,TS,439) = 3.00D-11                     
C
C     Reaction (440) OH + RU10OOH = RU10O2(:,:,:,TS)                                              
         RC(:,:,:,TS,440) = 3.00D-11                     
C
C     Reaction (441) OH + NRU14OOH = NUCARB12 + OH                                      
         RC(:,:,:,TS,441) = 1.03D-10                     
C
C     Reaction (442) OH + NRU12OOH = NOA + CO + OH                                      
         RC(:,:,:,TS,442) = 2.65D-11                     
C
C     Reaction (443) OH + HOC2H4OOH = HOCH2CHO + OH                                     
         RC(:,:,:,TS,443) = 2.13D-11               
C
C     Reaction (444) OH + RN9OOH = CARB7 + OH                                           
         RC(:,:,:,TS,444) = 2.50D-11               
C
C     Reaction (445) OH + RN12OOH = CARB10 + OH                                         
         RC(:,:,:,TS,445) = 3.25D-11               
C
C     Reaction (446) OH + RN15OOH = CARB13 + OH                                         
         RC(:,:,:,TS,446) = 3.74D-11               
C
C     Reaction (447) OH + RN18OOH = CARB16 + OH                                         
         RC(:,:,:,TS,447) = 3.83D-11               
C
C     Reaction (448) OH + NRN6OOH = HCHO + HCHO + NO2(:,:,:,TS) + OH                              
         RC(:,:,:,TS,448) = 5.22D-12               
C
C     Reaction (449) OH + NRN9OOH = CH3CHO + HCHO + NO2(:,:,:,TS) + OH                            
         RC(:,:,:,TS,449) = 6.50D-12               
C
C     Reaction (450) OH + NRN12OOH = CH3CHO + CH3CHO + NO2(:,:,:,TS) + OH                         
         RC(:,:,:,TS,450) = 7.15D-12               
C
C     Reaction (451) OH + RA13OOH = CARB3 + UDCARB8 + OH                                
         RC(:,:,:,TS,451) = 9.77D-11               
C
C     Reaction (452) OH + RA16OOH = CARB3 + UDCARB11 + OH                               
         RC(:,:,:,TS,452) = 9.64D-11               
C
C     Reaction (453) OH + RA19OOH = CARB6 + UDCARB11 + OH                               
         RC(:,:,:,TS,453) = 1.12D-10               
C
C     Reaction (454) OH + RTN2(:,:,:,TS)8OOH = TNCARB26 + OH                                      
         RC(:,:,:,TS,454) = 2.38D-11               
C
C     Reaction (455) OH + RTN2(:,:,:,TS)6OOH = RTN2(:,:,:,TS)6O2(:,:,:,TS)                                            
         RC(:,:,:,TS,455) = 1.20D-11               
C
C     Reaction (456) OH + NRTN2(:,:,:,TS)8OOH = TNCARB26 + NO2(:,:,:,TS) + OH                               
         RC(:,:,:,TS,456) = 9.50D-12               
C
C     Reaction (457) OH + RTN2(:,:,:,TS)5OOH = RTN2(:,:,:,TS)5O2(:,:,:,TS)                                            
         RC(:,:,:,TS,457) = 1.66D-11               
C
C     Reaction (458) OH + RTN2(:,:,:,TS)4OOH = RTN2(:,:,:,TS)4O2(:,:,:,TS)                                            
         RC(:,:,:,TS,458) = 1.05D-11               
C
C     Reaction (459) OH + RTN2(:,:,:,TS)3OOH = RTN2(:,:,:,TS)3O2(:,:,:,TS)                                            
         RC(:,:,:,TS,459) = 2.05D-11               
C
C     Reaction (460) OH + RTN14OOH = RTN14O2(:,:,:,TS)                                            
         RC(:,:,:,TS,460) = 8.69D-11               
C
C     Reaction (461) OH + RTN10OOH = RTN10O2(:,:,:,TS)                                            
         RC(:,:,:,TS,461) = 4.23D-12               
C
C     Reaction (462) OH + RTX28OOH = RTX28O2(:,:,:,TS)                                            
         RC(:,:,:,TS,462) = 2.00D-11               
C
C     Reaction (463) OH + RTX24OOH = TXCARB22 + OH                                      
         RC(:,:,:,TS,463) = 8.59D-11               
C
C     Reaction (464) OH + RTX22OOH = CH3COCH3 + CCARB12 + OH                            
         RC(:,:,:,TS,464) = 7.50D-11               
C
C     Reaction (465) OH + NRTX28OOH = NRTX28O2(:,:,:,TS)                                          
         RC(:,:,:,TS,465) = 9.58D-12               
C
C     Reaction (466) OH + ANHY = HOCH2CH2O(:,:,:,TS)2                                             
         RC(:,:,:,TS,466) = 1.50D-12        
C
C     Reaction (467) CH3CO3 + NO2(:,:,:,TS) = PAN                                                 
         RC(:,:,:,TS,467)s=sKFPAN(:,:,:)                        
C
C     Reaction (468) PAN = CH3CO3 + NO2(:,:,:,TS)                                                 
         RC(:,:,:,TS,468)s=sKBPAN(:,:,:)                        
C
C     Reaction (469) C2H5CO3 + NO2(:,:,:,TS) = PPN                                                
         RC(:,:,:,TS,469)s=sKFPAN(:,:,:)                        
C
C     Reaction (470) PPN = C2H5CO3 + NO2(:,:,:,TS)                                                
         RC(:,:,:,TS,470)s=sKBPAN(:,:,:)                        
C
C     Reaction (471) HOCH2CO3 + NO2(:,:,:,TS) = PHAN                                              
         RC(:,:,:,TS,471)s=sKFPAN(:,:,:)                        
C
C     Reaction (472) PHAN = HOCH2CO3 + NO2(:,:,:,TS)                                              
         RC(:,:,:,TS,472)s=sKBPAN(:,:,:)                        
C
C     Reaction (473) OH + PAN = HCHO + CO + NO2(:,:,:,TS)                                         
         RC(:,:,:,TS,473) = 9.50D-13*EXP(-650/TEM(:,:,:,TS)P(:,:,:,TS))      
C
C     Reaction (474) OH + PPN = CH3CHO + CO + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,474) = 1.27D-12                       
C
C     Reaction (475) OH + PHAN = HCHO + CO + NO2(:,:,:,TS)                                        
         RC(:,:,:,TS,475) = 1.12D-12                       
C
C     Reaction (476) RU12O2(:,:,:,TS) + NO2(:,:,:,TS) = RU12PAN                                             
         RC(:,:,:,TS,476)s=sKFPAN(:,:,:)*0.061             
C
C     Reaction (477) RU12PAN = RU12O2(:,:,:,TS) + NO2(:,:,:,TS)                                             
         RC(:,:,:,TS,477)s=sKBPAN(:,:,:)                   
C
C     Reaction (478) RU10O2(:,:,:,TS) + NO2(:,:,:,TS) = M(:,:,:,TS)PAN                                                
         RC(:,:,:,TS,478)s=sKFPAN(:,:,:)*0.041             
C
C     Reaction (479) M(:,:,:,TS)PAN = RU10O2(:,:,:,TS) + NO2(:,:,:,TS)                                                
         RC(:,:,:,TS,479)s=sKBPAN(:,:,:)                  
C
C     Reaction (480) OH + M(:,:,:,TS)PAN = CARB7 + CO + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,480) = 3.60D-12 
C
C     Reaction (481) OH + RU12PAN = UCARB10 + NO2(:,:,:,TS)                                       
         RC(:,:,:,TS,481) = 2.52D-11 
C
C     Reaction (482) RTN2(:,:,:,TS)6O2(:,:,:,TS) + NO2(:,:,:,TS) = RTN2(:,:,:,TS)6PAN                                           
         RC(:,:,:,TS,482)s=sKFPAN(:,:,:)*0.722      
C
C     Reaction (483) RTN2(:,:,:,TS)6PAN = RTN2(:,:,:,TS)6O2(:,:,:,TS) + NO2(:,:,:,TS)                                           
         RC(:,:,:,TS,483)s=sKBPAN(:,:,:)                   
C
C     Reaction (484) OH + RTN2(:,:,:,TS)6PAN = CH3COCH3 + CARB16 + NO2(:,:,:,TS)                            
         RC(:,:,:,TS,484) = 3.66D-12  
C
C     Reaction (485) RTN2(:,:,:,TS)8NO3 = P2604                                                   
         RC(:,:,:,TS,485)s=sKIN(:,:,:)  		
C
C     Reaction (486) P2604 = RTN2(:,:,:,TS)8NO3                                                   
         RC(:,:,:,TS,486)s=sKOUT2604(:,:,:) 	
C
C     Reaction (487) RTX28NO3 = P4608                                                   
         RC(:,:,:,TS,487)s=sKIN(:,:,:) 		
C
C     Reaction (488) P4608 = RTX28NO3                                                   
         RC(:,:,:,TS,488)s=sKOUT4608(:,:,:) 	
C
C     Reaction (489) RCOOH25 = P2631                                                    
         RC(:,:,:,TS,489)s=sKIN(:,:,:)  		
C
C     Reaction (490) P2631 = RCOOH25                                                    
         RC(:,:,:,TS,490)s=sKOUT2631(:,:,:) 	
C
C     Reaction (491) RTN2(:,:,:,TS)4OOH = P2635                                                   
         RC(:,:,:,TS,491)s=sKIN(:,:,:)  		
C
C     Reaction (492) P2635 = RTN2(:,:,:,TS)4OOH                                                   
         RC(:,:,:,TS,492)s=sKOUT2635(:,:,:) 	
C
C     Reaction (493) RTX28OOH = P4610                                                   
         RC(:,:,:,TS,493)s=sKIN(:,:,:)  		
C
C     Reaction (494) P4610 = RTX28OOH                                                   
         RC(:,:,:,TS,494)s=sKOUT4610(:,:,:) 	
C
C     Reaction (495) RTN2(:,:,:,TS)8OOH = P2605                                                   
         RC(:,:,:,TS,495)s=sKIN(:,:,:)  		
C
C     Reaction (496) P2605 = RTN2(:,:,:,TS)8OOH                                                   
         RC(:,:,:,TS,496)s=sKOUT2605(:,:,:) 	
C
C     Reaction (497) RTN2(:,:,:,TS)6OOH = P2630                                                   
         RC(:,:,:,TS,497)s=sKIN(:,:,:)  		
C
C     Reaction (498) P2630 = RTN2(:,:,:,TS)6OOH                                                   
         RC(:,:,:,TS,498)s=sKOUT2630(:,:,:) 	
C
C     Reaction (499) RTN2(:,:,:,TS)6PAN = P2629                                                   
         RC(:,:,:,TS,499)s=sKIN(:,:,:)  		
C
C     Reaction (500) P2629 = RTN2(:,:,:,TS)6PAN                                                   
         RC(:,:,:,TS,500)s=sKOUT2629(:,:,:) 	
C
C     Reaction (501) RTN2(:,:,:,TS)5OOH = P2632                                                   
         RC(:,:,:,TS,501)s=sKIN(:,:,:) 		
C
C     Reaction (502) P2632 = RTN2(:,:,:,TS)5OOH                                                   
         RC(:,:,:,TS,502)s=sKOUT2632(:,:,:) 	
C
C     Reaction (503) RTN2(:,:,:,TS)3OOH = P2637                                                   
         RC(:,:,:,TS,503)s=sKIN(:,:,:)  		
C
C     Reaction (504) P2637 = RTN2(:,:,:,TS)3OOH                                                   
         RC(:,:,:,TS,504)s=sKOUT2637(:,:,:) 	
C
C     Reaction (505) ARNOH14 = P3612                                                    
         RC(:,:,:,TS,505)s=sKIN(:,:,:)  		
C
C     Reaction (506) P3612 = ARNOH14                                                    
         RC(:,:,:,TS,506)s=sKOUT3612(:,:,:) 	
C
C     Reaction (507) ARNOH17 = P3613                                                    
         RC(:,:,:,TS,507)s=sKIN(:,:,:) 		
C
C     Reaction (508) P3613 = ARNOH17                                                    
         RC(:,:,:,TS,508)s=sKOUT3613(:,:,:) 	
C
C     Reaction (509) ANHY = P3442                                                       
         RC(:,:,:,TS,509)s=sKIN(:,:,:)  		
C
C     Reaction (510) P3442 = ANHY                                                       
         RC(:,:,:,TS,510)s=sKOUT3442(:,:,:)