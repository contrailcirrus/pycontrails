      PROGRAM BOXMODEL
C-----------------------------------------------------------------------
C      BACKIT.F   -  BACKWARD EULER ITERATIVE INTEGRATION ROUTINE
C                    FOR CHEMICAL EQUATIONS.
C
C                    VERSION INCORPORATING CRI-MECHANISM
C                    CODED UP FROM STEVEN UTEMBE STOCHEM CODE
C                    LINE-BY-LINE CHECK OCTOBER 2014 DICK DERWENT
C
C     VERSION SET UP FOR ANWAR KHAN - UNIVERSITY OF BRISTOL BY DICK DERWENT
C-----------------------------------------------------------------------
      IMPLICIT NONE
C-----------------------------------------------------------------------
      INTEGER NC,NR,NE,NSTEP,I,URBFAC
      PARAMETER(NC=220,NR=606,NE=14)
      DOUBLE PRECISION PI, R
      DOUBLE PRECISION ENDTIME,TSTART,TSTORE
      DOUBLE PRECISION Y(NC),YP(NC),EM(NC),D(NR),E(NE),FL(NR)
      DOUBLE PRECISION RC(512),DJ(96),TC,M,H2O,DTS,O2,FI,
     &                 A(17),B(17),DECL,XLHA,SEC,TIME,XZ,TV,PPB
      DOUBLE PRECISION EB,EA,HFRAC(22),MOLWTHC(22),EMI(NC),DIL
      DOUBLE PRECISION MICROGSA, MICROGNA, BGOAM, TEMP, NAV,N2
      DOUBLE PRECISION ECO0,ESO20,EPM0,EPM,EPOA,NAM,SAM
      DOUBLE PRECISION ENOX,EAVOC0,EAVOC,EBVOC0,EBVOC,EAFAC
      DOUBLE PRECISION ECO, ESO2, ENOX0,RINJECT(25),SECYEAR
      DOUBLE PRECISION THETA, SECX, COSX, POAM, OM, MOM
      DOUBLE PRECISION PRESSURE,SOA,XYEAR,TIME1,Z(NC),ZP(NC),ZRO2
      DOUBLE PRECISION L(70),MM(70),NN(70),J(70),K,JS1,KZ
      DOUBLE PRECISION RO2, BR01, RADIAN,LONGRAD,COSZEN,ZENNOW
      INTEGER MONTHDAY(12),IDAY, TOTALDAY,IMONTH
        CHARACTER*20  CNAMES(NC)
      DOUBLE PRECISION LAT, LONG, LATRAD,YEAR,FYEAR,EMTOT
      DOUBLE PRECISION KMT03, KMT04,K40
      DATA MONTHDAY / 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31  /
C
      DATA (CNAMES(I), I=1,52) /
     & 'O1D         ','O           ','OH          ', 'NO2         ',
     & 'NO3         ', 'O3          ','N2O5        ','NO          ',
     & 'HO2         ','H2          ', 'CO          ', 'H2O2        ',
     & 'HONO        ', 'HNO3        ', 'HO2NO2      ', 'SO2         ',
     & 'SO3         ', 'HSO3        ', 'NA          ', 'SA          ',
     & 'CH4         ', 'CH3O2       ', 'C2H6        ', 'C2H5O2      ',
     & 'C3H8        ', 'IC3H7O2     ', 'RN10O2      ', 'NC4H10      ',
     & 'RN13O2      ', 'C2H4        ', 'HOCH2CH2O2  ', 'C3H6        ',
     & 'RN9O2       ', 'TBUT2ENE    ', 'RN12O2      ', 'NRN6O2      ',
     & 'NRN9O2      ', 'NRN12O2     ', 'HCHO        ', 'HCOOH       ',
     & 'CH3CO2H     ', 'CH3CHO      ', 'C5H8        ', 'RU14O2      ',
     & 'NRU14O2     ', 'UCARB10     ', 'APINENE     ', 'RTN28O2     ',
     & 'NRTN28O2    ', 'RTN26O2     ', 'TNCARB26    ', 'RCOOH25    '/

      DATA (CNAMES(I), I=53,101) /
     & 'BPINENE     ', 'RTX28O2     ', 'NRTX28O2    ', 'RTX24O2     ',
     & 'TXCARB24    ', 'TXCARB22    ', 'C2H2        ', 'CARB3       ',
     & 'BENZENE     ', 'RA13O2      ', 'AROH14      ', 'TOLUENE     ',
     & 'RA16O2      ', 'AROH17      ', 'OXYL        ', 'RA19AO2     ',
     & 'RA19CO2     ', 'CH3CO3      ', 'C2H5CHO     ', 'C2H5CO3     ',
     & 'CH3COCH3    ', 'RN8O2       ', 'RN11O2      ', 'CH3OH       ',
     & 'C2H5OH      ', 'NPROPOL     ', 'IPROPOL     ', 'CH3CL       ',
     & 'CH2CL2      ', 'CHCL3       ', 'CH3CCL3     ', 'TCE         ',
     & 'TRICLETH    ', 'CDICLETH    ', 'TDICLETH    ', 'CARB11A     ',
     & 'RN16O2      ', 'RN15AO2     ', 'RN19O2      ', 'RN18AO2     ',
     & 'RN13AO2     ', 'RN16AO2     ', 'RN15O2      ', 'UDCARB8     ',
     & 'UDCARB11    ', 'CARB6       ', 'UDCARB14    ', 'CARB9       ',
     & 'MEK        '/

      DATA (CNAMES(I), I=102,152) /
     & 'HOCH2CHO    ', 'RN18O2      ', 'CARB13      ', 'CARB16      ',
     & 'HOCH2CO3    ', 'RN14O2      ', 'RN17O2      ', 'UCARB12     ',
     & 'RU12O2      ', 'CARB7       ', 'RU10O2      ', 'NUCARB12    ',
     & 'NRU12O2     ', 'NOA         ', 'RTN25O2     ', 'RTN24O2     ',
     & 'RTN23O2     ', 'RTN14O2     ', 'TNCARB10    ', 'RTN10O2     ',
     & 'RTX22O2     ', 'CH3NO3      ', 'C2H5NO3     ', 'RN10NO3     ',
     & 'IC3H7NO3    ', 'RN13NO3     ', 'RN16NO3     ', 'RN19NO3     ',
     & 'HOC2H4NO3   ', 'RN9NO3      ', 'RN12NO3     ', 'RN15NO3     ',
     & 'RN18NO3     ', 'RU14NO3     ', 'RA13NO3     ', 'RA16NO3     ',
     & 'RA19NO3     ', 'RTN28NO3    ', 'RTN25NO3    ', 'RTX28NO3    ',
     & 'RTX24NO3    ', 'RTX22NO3    ', 'CH3OOH      ', 'C2H5OOH     ',
     & 'RN10OOH     ', 'IC3H7OOH    ', 'RN13OOH     ', 'RN16OOH     ',
     & 'RN19OOH     ', 'RA13OOH     ', 'RA16OOH    '/

      DATA (CNAMES(I), I=153,220) /
     & 'RA19OOH     ', 'HOC2H4OOH   ', 'RN9OOH      ', 'RN12OOH     ',
     & 'RN15OOH     ', 'RN18OOH     ', 'CH3CO3H     ', 'C2H5CO3H    ',
     & 'HOCH2CO3H   ', 'RN8OOH      ', 'RN11OOH     ', 'RN14OOH     ',
     & 'RN17OOH     ', 'RU14OOH     ', 'RU12OOH     ', 'RU10OOH     ',
     & 'NRN6OOH     ', 'NRN9OOH     ', 'NRN12OOH    ', 'NRU14OOH    ',
     & 'NRU12OOH    ', 'RTN28OOH    ', 'NRTN28OOH   ', 'RTN26OOH    ',
     & 'RTN25OOH    ', 'RTN24OOH    ', 'RTN23OOH    ', 'RTN14OOH    ',
     & 'RTN10OOH    ', 'RTX28OOH    ', 'RTX24OOH    ', 'RTX22OOH    ',
     & 'NRTX28OOH   ', 'CARB14      ', 'CARB17      ', 'CARB10      ',
     & 'CARB12      ', 'CARB15      ', 'CCARB12     ', 'ANHY        ',
     & 'TNCARB15    ', 'RAROH14     ', 'ARNOH14     ', 'RAROH17     ',
     & 'ARNOH17     ', 'PAN         ', 'PPN         ', 'PHAN        ',
     & 'RU12PAN     ', 'MPAN        ', 'RTN26PAN    ', 'P2604       ',
     & 'P4608       ', 'P2631       ', 'P2635       ', 'P4610       ',
     & 'P2605       ', 'P2630       ', 'P2629       ', 'P2632       ',
     & 'P2637       ', 'P3612       ', 'P3613       ', 'P3442       ',
     & 'CH3O2NO2    ', 'EMPOA       ', 'P2007       ', 'DUMMY       '/
C
C
      DATA  (MOLWTHC(I), I=1,22) /
     &  30.0,	
     &  44.0,	
     &  58.0,	
     &  28.0,	
     &  42.0,	
     &  56.0,	
     &  26.0,	
     &  30.0,	
     &  44.0,	
     &  58.0,	
     &  58.0,	
     &  72.0,	
     &  32.0,	
     &  46.0,	
     &  46.0,	
     &  60.0,	
     &  78.0,	
     &  92.0,	
     &  106.0,	
     &  68.0, 
     &  136.0, 
     &  136.0 /
C
      INTEGER:: inDAY, inMONTH, inYEAR, inLEVEL, inLONG,
     &inLAT

      REAL :: inRUNTIME, inM, inP, inH2O, inTEMP, inNO2, inNO, inO3, 
     &inCO, inCH4, inHCHO, inCH3CHO, inCH3COCH3, inC2H6, inC2H4,
     &inC3H8, inC3H6, inC2H2, inNC4H10, inTBUT2ENE, inBENZENE,
     &inTOLUENE, inOXYL, inC5H8, inH2O2, inHNO3, inC2H5CHO, 
     &inCH3OH, inMEK, inCH3OOH, inPAN, inMPAN, inEMI_NO2, inEMI_CO, 
     &inEMI_HCHO, inEMI_CH3CHO, inEMI_CH3COCH3, inEMI_C2H6, inEMI_C2H4, 
     &inEMI_C3H8, inEMI_C3H6, inEMI_C2H2, inEMI_BENZENE, inEMI_TOLUENE, 
     &inEMI_C2H5CHO, inFF, inT0, inA0, inV0, inKZ  
C
      OPEN(7, FILE='BACKITNE.OUT',STATUS='UNKNOWN')
      OPEN(8, FILE ='Y.OUT', STATUS = 'UNKNOWN') 
      OPEN(9, FILE ='ZEN.OUT', STATUS = 'UNKNOWN')
      OPEN(10, FILE ='J.OUT', STATUS = 'UNKNOWN')
      OPEN(11, FILE ='DJ.OUT', STATUS = 'UNKNOWN')
      OPEN(12, FILE ='RC.OUT', STATUS = 'UNKNOWN') 
      OPEN(13, FILE = 'newBOXMODEL.IN', STATUS = 'UNKNOWN')
C
      READ(13, *) inDAY, inMONTH, inYEAR, inLEVEL, inLONG,
     &inLAT, inRUNTIME, inM, inP, inH2O, inTEMP, inNO2, inNO, 
     &inO3, inCO, inCH4, inHCHO, inCH3CHO, inCH3COCH3, inC2H6, inC2H4, 
     &inC3H8, inC3H6,inC2H2, inNC4H10, inTBUT2ENE, inBENZENE, inTOLUENE,
     &inOXYL, inC5H8, inH2O2, inHNO3, inC2H5CHO, inCH3OH, inMEK, 
     &inCH3OOH, inPAN, inMPAN
C
C
      WRITE(8,519)'TIME',' NO2',' NO',' OH',' HO2',' O3',' CO',' CH4',
     &' HCHO',' HNO3',' PAN'
C
      WRITE(9,520)'TIME',' ZEN'
C
      WRITE(10,519)'TIME',' J1',' J2',' J3',' J4',' J5',' J6',' J7',
     &' J8',' J9',' J10'
C
      WRITE(11,519)'TIME',' DJ1',' DJ2',' DJ3',' DJ4',' DJ5',' DJ6',
     &' DJ7',' DJ8',' DJ9',' DJ10'
C
      WRITE(12,519)'TIME',' RC1',' RC2',' RC3',' RC4',' RC5',' RC6',
     &' RC7',' RC8',' RC9',' RC10'
C
C
C      WRITE(9,520)'TIME',' NO2',' NO',' O3',' CO',' CH4',' HCHO',
C     &' HNO3',' PAN'
C
C      WRITE(*, *) inDAY, inMONTH, inYEAR, inLEVEL, inLONG,
C     &inLAT, inRUNTIME,inM, inH2O, inTEMP, inNO2,inNO, inO3, inCO, 
C     &inCH4, inHCHO, inCH3CHO, inCH3COCH3, inC2H6, inC2H4,inC3H8,
C     &inC3H6, inC2H2, inNC4H10, inTBUT2ENE, inBENZENE,
C     &inTOLUENE, inOXYL, inC5H8, inH2O2, inHNO3, inC2H5CHO,
C     &inCH3OH, inMEK, inCH3OOH, inPAN, inMPAN     
      EAFAC = 1.0
      TSTART = 3600*12.0
      ENDTIME = 86400.0*inRUNTIME + 3600*18.0
      SECYEAR = 3.6525E+02*2.40E+01*3.60E+03
C      SET TIMESTEP
      DTS = 20.0
C     INITIALISATION                                                                          
      DO 203 I=1,512
        RC(I)=0.0
	  FL(I)=0.0
  203 CONTINUE
      DO 202 I=1,NC
        EMI(I)=0.0
  202 CONTINUE
C
C
C      SETTING EMISSION INDICES IN g/kg fuel burnt from Anwar Khan + literature
C
      EMI(4)  = 20.8            ! NO2
      EMI(11) =  1.2      	! CO
      EMI(39) =  0.0139  	! HCHO
      EMI(42) =  0.0049 	! CH3CHO
      EMI(73) =  0.0037 	! CH3COCH3
      EMI(23) =  0.0010 	! C2H6
      EMI(30) =  0.0112 	! C2H4
      EMI(25) =  0.0008  	! C3H8
      EMI(32) =  0.0023 	! C3H6
      EMI(59) =  0.0024 	! C2H2
      EMI(61) =  0.0013 	! BENZENE
      EMI(64) =  0.0011 	! TOLUENE
      EMI(71) =  0.0011 	! C2H5CHO


C      ESTABLISHMENT OF INITIAL CONDITIONS:                                      
C
      PI = 4.00E+00*ATAN(1.00E+00) 
      RADIAN  = 1.80E+02/PI 
      TEMP     = inTEMP ! 
      PRESSURE = inP 
      R        = 8.314
      NAV      = 6.022E+23 
      M        = inM 
      H2O      = inH2O
      PPB      = 1.00E-09*M 
      N2       = 7.809E-01*M 
      O2       = 2.079E-01*M 
      MICROGSA = 6.022E+11/96 
      MICROGNA = 6.022E+11/62 
      BGOAM = 0.7 
      LAT    =  (inLAT-1)*5-87.5
      RO2 = 0.0
      LATRAD  = LAT/RADIAN 
      LONG    = (inLONG-1)*5-177.5 
      LONGRAD = LONG/RADIAN 
      IDAY         = inDAY 
      IMONTH       = inMONTH
      YEAR        = inYEAR                                                                   
      TOTALDAY = 0
C
C     Dilution factor set for 75 seconds after emission based on Schumann
C     (1997). Atmos. Environ. 31, 1723-1733 and Tremmel et al. (1998).
C     JGR 103, 10,803-10,816.
C
C      DIL = ((R * inFF * inT0) / (inP * inA0 * inV0)) * (inLPATH / inV0)
C
C      Exchange between plume and background set at 10 days
C
C      KZ = inKZ
C
      DO 69 I = 1,IMONTH-1                                                    
        TOTALDAY = TOTALDAY + MONTHDAY(I)                                                    
   69 CONTINUE 
      TOTALDAY = TOTALDAY + IDAY - 1 
      FYEAR     = REAL(TOTALDAY)/3.6525E+02 
C
      DO 205 I=1,NC
        Y(I)=0.0
        Z(I)=0.0
  205 CONTINUE 
C
      Y(4)  = inNO2*PPB      ! NO2
      Y(8)  = inNO*PPB 	! NO
      Y(6)  = inO3*PPB    	! O3
      Y(11) = inCO*PPB 	! CO
      Y(21) = inCH4*PPB       ! CH4
      Y(39) = inHCHO*PPB  	! HCHO
      Y(42) = inCH3CHO*PPB 	! CH3CHO
      Y(73) = inCH3COCH3*PPB 	! CH3COCH3
      Y(23) = inC2H6*PPB 	! C2H6
      Y(30) = inC2H4*PPB 	! C2H4
      Y(25) = inC3H8*PPB  	! C3H8
      Y(32) = inC3H6*PPB 	! C3H6
      Y(59) = inC2H2*PPB 	! C2H2
      Y(28) = inNC4H10*PPB 	! NC4H10
      Y(34) = inTBUT2ENE*PPB 	! TBUT2ENE
      Y(61) = inBENZENE*PPB 	! BENZENE
      Y(64) = inTOLUENE*PPB 	! TOLUENE
      Y(67) = inOXYL*PPB 	! OXYL
      Y(43) = inC5H8*PPB      ! C5H8
      Y(12) = inH2O2*PPB 	! H2O2
      Y(14) = inHNO3*PPB 	! HNO3
      Y(71) = inC2H5CHO*PPB 	! C2H5CHO
      Y(76) = inCH3OH*PPB 	! CH3OH
      Y(101)= inMEK*PPB      ! MEK
      Y(144)= inCH3OOH*PPB 	! CH3OOH
      Y(198)= inPAN*PPB   	! PAN
      Y(202)= inMPAN*PPB 	! MPAN
c Z(4)  = inNO2*PPB 	                      ! NO2
c Z(6)  = inO3*PPB    	                      ! O3
c Z(8)  = (inNO+((EMI(4)*DIL)/46.0))*PPB      ! NO
c Z(11) = (inCO+((EMI(11)*DIL)/28.0))*PPB        ! CO
c Z(21) = inCH4*PPB                             ! CH4
c Z(23) = (inC2H6+((EMI(23)*DIL)/MOLWTHC(1)))*PPB 	! C2H6
c Z(25) = (inC3H8+((EMI(25)*DIL)/MOLWTHC(2)))*PPB  	! C3H8
c Z(30) = (inC2H4+((EMI(30)*DIL)/MOLWTHC(4)))*PPB 	! C2H4
c Z(32) = (inC3H6+((EMI(32)*DIL)/MOLWTHC(5)))*PPB 	! C3H6
c Z(59) = (inC2H2+((EMI(59)*DIL)/MOLWTHC(7)))*PPB 	! C2H2
c Z(39) = (inHCHO+((EMI(39)*DIL)/MOLWTHC(8)))*PPB  	! HCHO
c Z(42) = (inCH3CHO+((EMI(42)*DIL)/MOLWTHC(9)))*PPB 	! CH3CHO
c Z(71) = (inC2H5CHO+((EMI(71)*DIL)/MOLWTHC(10)))*PPB 	! C2H5CHO
c Z(73) = (inCH3COCH3+((EMI(73)*DIL)/MOLWTHC(11)))*PPB 	! CH3COCH3
c Z(61) = (inBENZENE+((EMI(61)*DIL)/MOLWTHC(17)))*PPB 	! BENZENE
c Z(64) = (inTOLUENE+((EMI(64)*DIL)/MOLWTHC(18)))*PPB 	! TOLUENE
c Z(28) = inNC4H10*PPB 	! NC4H10
c Z(34) = inTBUT2ENE*PPB 	! TBUT2ENE
c Z(67) = inOXYL*PPB 	! OXYL
c Z(43) = inC5H8*PPB      ! C5H8
c Z(12) = inH2O2*PPB 	! H2O2
c Z(14) = inHNO3*PPB 	! HNO3
c Z(76) = inCH3OH*PPB 	! CH3OH
c Z(101)= inMEK*PPB      ! MEK
c Z(144)= inCH3OOH*PPB 	! CH3OOH
c Z(198)= inPAN*PPB   	! PAN
c Z(202)= inMPAN*PPB 	! MPAN
C
      TIME = TSTART
C
C     EMISSIONS 
C
      DO 113 I=1,NC
        EM(I)=0.0
  113 CONTINUE
C                                       
C      START INTEGRATION
C
      NSTEP=0
  35  CONTINUE                                                     
C
      TIME1 = TIME - TSTART
	CALL ZENITH(COSZEN,TIME,FYEAR,SECYEAR,ZENNOW,
     & LONGRAD,LATRAD,XYEAR) 

      WRITE(9,521) TIME1,ZENNOW,TEMP,O2,N2                      
C
C Mass of organic particulate material             
C -------------------------------------------------------------------
C     
	SOA = Y(204)*3.574E-10 + Y(205)*3.574E-10 + 
     & Y(206)*3.059E-10 + Y(207)*3.126E-10 + Y(208)*3.093E-10 + 
     & Y(209)*3.093E-10 + Y(210)*3.325E-10 + Y(211)*4.072E-10 + 
     & Y(212)*2.860E-10 + Y(213)*3.391E-10 + Y(214)*2.310E-10 + 
     & Y(215)*2.543E-10 + Y(216)*1.628E-10 + Y(219)*2.493E-10
C
      MOM = Y(218) + BGOAM + SOA 
C
C     CALCULATE RATE COEFFICIENTS
C
      CALL CHEMCO(RC,TEMP,M,O2,H2O,RO2,MOM,BR01)     

      WRITE(12,517) TIME1,RC(1),RC(2),RC(3),RC(4),RC(5),RC(6),RC(7),
     &RC(8),RC(9),RC(10)
C
C      CALCULATE PHOTOLYSIS RATES
C
	IF((ZENNOW-8.999E+01).LT.0) THEN
C **** PHOTOLYSIS PARAMETERS IN FORMAT J = L*COSX**M*EXP(-N*SECX)      
C                                                                     
      COSX = COSZEN 
      SECX = 1.00E+00/COSZEN                                                              
       J(1)=6.073D-05*(COSX**(1.743))*EXP(-0.474*SECX) 
       J(2)=4.775D-04*(COSX**(0.298))*EXP(-0.080*SECX) 
       J(3)=1.041D-05*(COSX**(0.723))*EXP(-0.279*SECX) 
       J(4)=1.165D-02*(COSX**(0.244))*EXP(-0.267*SECX) 
       J(5)=2.485D-02*(COSX**(0.168))*EXP(-0.108*SECX) 
       J(6)=1.747D-01*(COSX**(0.155))*EXP(-0.125*SECX) 
       J(7)=2.644D-03*(COSX**(0.261))*EXP(-0.288*SECX) 
       J(8)=9.312D-07*(COSX**(1.230))*EXP(-0.307*SECX) 
       J(11)=4.642D-05*(COSX**(0.762))*EXP(-0.353*SECX)
       J(12)=6.853D-05*(COSX**(0.477))*EXP(-0.323*SECX) 
       J(13)=7.344D-06*(COSX**(1.202))*EXP(-0.417*SECX) 
       J(14)=2.879D-05*(COSX**(1.067))*EXP(-0.358*SECX) 
       J(15)=2.792D-05*(COSX**(0.805))*EXP(-0.338*SECX) 
       J(16)=1.675D-05*(COSX**(0.805))*EXP(-0.338*SECX) 
       J(17)=7.914D-05*(COSX**(0.764))*EXP(-0.364*SECX) 
       J(18)=1.140D-05*(COSX**(0.396))*EXP(-0.298*SECX) 
       J(19)=1.140D-05*(COSX**(0.396))*EXP(-0.298*SECX) 
       J(21)=7.992D-07*(COSX**(1.578))*EXP(-0.271*SECX) 
       J(22)=5.804D-06*(COSX**(1.092))*EXP(-0.377*SECX) 
       J(23)=1.836D-05*(COSX**(0.395))*EXP(-0.296*SECX) 
       J(24)=1.836D-05*(COSX**(0.395))*EXP(-0.296*SECX) 
       J(31)=6.845D-05*(COSX**(0.130))*EXP(-0.201*SECX) 
       J(32)=1.032D-05*(COSX**(0.130))*EXP(-0.201*SECX) 
       J(33)=3.802D-05*(COSX**(0.644))*EXP(-0.312*SECX) 
       J(34)=1.537D-04*(COSX**(0.170))*EXP(-0.208*SECX) 
       J(35)=3.326D-04*(COSX**(0.148))*EXP(-0.215*SECX) 
       J(41)=7.649D-06*(COSX**(0.682))*EXP(-0.279*SECX) 
       J(51)=1.588D-06*(COSX**(1.154))*EXP(-0.318*SECX) 
       J(52)=1.907D-06*(COSX**(1.244))*EXP(-0.335*SECX) 
       J(53)=2.485D-06*(COSX**(1.196))*EXP(-0.328*SECX) 
       J(54)=4.095D-06*(COSX**(1.111))*EXP(-0.316*SECX) 
       J(55)=1.135D-05*(COSX**(0.974))*EXP(-0.309*SECX) 
       J(56)=7.549D-06*(COSX**(1.015))*EXP(-0.324*SECX) 
       J(57)=3.363D-06*(COSX**(1.296))*EXP(-0.322*SECX) 
       JS1 = J(1) 
      ELSE
	DO I=1,70
        J(I)=1.0E-30 
	END DO
      END IF
C                
      
C
C      reset previous concentrations at current value
      DO 10  I = 1,NC
        YP(I)=Y(I)
        ZP(I)=Z(I)
  10  CONTINUE
C
      IF (TSTORE.LT.1.0.OR.(TIME-TSTORE).EQ.3600) THEN
      WRITE(6,*)TIME1

      WRITE(10,517) TIME1,J(1),J(2),J(3),J(4),J(5),J(6),J(7),J(8),
     &J(9),J(10)
C
      WRITE(8,517) TIME1,Y(8)/M*1.0E9,Y(4)/M*1.0E9,
     &  Y(3)/M*1.0E9,Y(9)/M*1.0E9,Y(6)/M*1.0E9,Y(11)/M*1.0E9,
     &  Y(21)/M*1.0E9,Y(39)/M*1.0E9,Y(14)/M*1.0E9,Y(198)/M*1.0E9

c       WRITE(9, 518) TIME1,Z(8)/M*1.0E9,Z(4)/M*1.0E9,
c     &  Z(6)/M*1.0E9,Z(11)/M*1.0E9,Z(21)/M*1.0E9,Z(39)/M*1.0E9,
c     &  Z(14)/M*1.0E9,Z(198)/M*1.0E9


c       WRITE(9,518) TIME1,Z(4)/M*1.0E9,Z(8)/M*1.0E9,Z(9),
c     &	Z(3),Z(14)/M*1.0E9,Z(198)/M*1.0E9
C	
	TSTORE=TIME
	CALL PHOTOL(J,DJ,BR01)
      
C
	WRITE(11,517) TIME1,DJ(1),DJ(2),DJ(3),DJ(4),DJ(5),DJ(6),DJ(7),
     &DJ(8),DJ(9),DJ(10)
C	
      ENDIF	
C
      CALL DERIV(RC,FL,E,DJ,H2O,M,O2,YP,Y,DTS,TIME1,RO2,ZP,Z,KZ)
C
C
C
  30  CONTINUE
C      end of iteration
C
      TIME=TIME+DTS
      NSTEP=NSTEP+1
C
      IF (TIME.LE.ENDTIME) GOTO 35
C
C      WRITE OUT THE RESULTS
C
      WRITE(7,501) TIME, NSTEP
C
      DO 300 I =1,NC
        WRITE(7,500) CNAMES(I), Y(I), Y(I)/PPB
 300  CONTINUE
C
      STOP
C
  500 FORMAT(A10,1P2E12.5)
  501 FORMAT('TIME = ',E12.5,' NSTEP = ',I12//
     &       ' SPECIES ','  MOLECULES/CM^3 ',' PPB')
  502 FORMAT('TIME = ',F8.0,' O3: ',E10.4,' Y2: ',E10.4)
  503 FORMAT(1X,1P6E12.5) 
  504 FORMAT(1X,8A12)
  510 FORMAT(//'FLUX THROUGH EQUATIONS (molecules/cm^3)'/)
  511 FORMAT(1X,A52,1PE12.5)
  516 FORMAT(1X,A8,1PE12.5)
! 517 FORMAT(F12.3,2X,F8.3,2X,F8.3,2X,
!     & F8.3,2X,F8.3,2X,F8.3,2X,F8.3)
C
  517 FORMAT(ES9.3, 10(",", ES13.6))
C
  518 FORMAT(ES9.3, 10(",", ES13.6))
C
  519 FORMAT(A, 10(",", A))
  520 FORMAT(A, 3(",", A))
  521 FORMAT(ES9.3, ",", ES13.6)
      END
C
      SUBROUTINE DERIV(RC,FL,E,DJ,H2O,M,O2,YP,Y,DTS,TIME1,RO2,ZP,Z,KZ)
C-----------------------------------------------------------------------
C     PURPOSE:    -  TO EVALUATE CONCENTRATIONS Y FROM RATE COEFFICIENTS
C                    J VALUES AND EMISSION RATES.
C                    DETAILED CHEMISTRY
C
C     INPUTS:     -  SPECIES CONCENTRATIONS (Y), RATE COEFFICIENTS (RC),
C                    PHOTOLYSIS RATES (DJ), AND EMISSIONS (EM).
C
C     OUTPUTS:    -  CONCENTRATIONS FROM CHEMISTRY AND EMISSIONS (DY).
C
C
C     CREATED:    -  1-SEPT-1994   Colin Johnson
C
C     VER 2:      -  20-FEV-1995   Colin Johnson
C                                  New chemical scheme with 50 species
C                                  from chem3.txt mechanism.
C
C     VER NEW:    -  25-NOV-1996  Colin Johnson using Dick's new scheme
C                                  with 70 species.
C                     2-DEC-1996  Dick Derwent edited scheme
C-----------------------------------------------------------------------
      IMPLICIT NONE
C-----------------------------------------------------------------------
      INTEGER NC,NR,NE,NIT,I,J,IZ
      PARAMETER(NC=220,NR=606,NE=14,NIT=6)
      DOUBLE PRECISION EM(NC),RO2,ZP(NC),Z(NC),ZRO2
      DOUBLE PRECISION RC(512),DJ(96),D(NR),E(NE),H2O,M,O2,DTS
      DOUBLE PRECISION Y(NC),YP(NC),FL(NR)
      DOUBLE PRECISION P,L,L1,L2,L3,R1,R2,TIME1,KZ
      CHARACTER*6  CNAMES(NC)
C
C       iteration start
       DO 1000 I=1,NIT
C
C
C
C
C
C      This section written automatically by MECH5GEN
C from the file reducedchem2
C      with 606 equations and219 species.
C
C          O1D              Y(  1)
      P = EM(  1)
     &+(DJ(1)      *Y(6  ))                                             
      L = 0.0
     &+(RC(7)      )       +(RC(8) ) +(RC(16)     *H2O)      
      Y(  1) = P/L
C
C          O                Y(  2)
      P = EM(  2)
     &+(DJ(6)      *Y(5  ))                                             
     &+(DJ(2)      *Y(6  ))       +(DJ(4)      *Y(4  ))                 
     &+(RC(7)      *Y(1  ))       +(RC(8)      *Y(1  ))          
      L = 0.0
     &+(RC(36)     *Y(16 ))                                             
     &+(RC(4)      *Y(8  ))+(RC(5)      *Y(4  ))+(RC(6)      *Y(4  ))   
     &+(RC(1)      )       +(RC(2) )+(RC(3)      *Y(6  ))   
      Y(  2) = P/L
C
C          OH               Y(  3)
      P = EM(  3)
     &+(DJ(95)     *Y(184))       +(DJ(96)     *Y(185))                 
     &+(DJ(93)     *Y(182))       +(DJ(94)     *Y(183))                 
     &+(DJ(91)     *Y(180))       +(DJ(92)     *Y(181))                 
     &+(DJ(89)     *Y(178))       +(DJ(90)     *Y(179))                 
     &+(DJ(87)     *Y(176))       +(DJ(88)     *Y(177))                 
     &+(DJ(85)     *Y(174))       +(DJ(86)     *Y(175))                 
     &+(DJ(83)     *Y(152))       +(DJ(84)     *Y(153))                 
     &+(DJ(81)     *Y(171))       +(DJ(82)     *Y(151))                 
     &+(DJ(79)     *Y(169))       +(DJ(80)     *Y(170))                 
     &+(DJ(77)     *Y(157))       +(DJ(78)     *Y(158))                 
     &+(DJ(75)     *Y(155))       +(DJ(76)     *Y(156))                 
     &+(DJ(73)     *Y(173))       +(DJ(74)     *Y(154))                 
     &+(DJ(71)     *Y(168))       +(DJ(72)     *Y(172))                 
     &+(DJ(69)     *Y(166))       +(DJ(70)     *Y(167))                 
     &+(DJ(67)     *Y(165))       +(DJ(68)     *Y(166))                 
     &+(DJ(65)     *Y(163))       +(DJ(66)     *Y(164))                 
     &+(DJ(63)     *Y(161))       +(DJ(64)     *Y(162))                 
     &+(DJ(61)     *Y(159))       +(DJ(62)     *Y(160))                 
     &+(DJ(59)     *Y(149))       +(DJ(60)     *Y(150))                 
     &+(DJ(57)     *Y(148))       +(DJ(58)     *Y(148))                 
     &+(DJ(55)     *Y(146))       +(DJ(56)     *Y(147))                 
     &+(DJ(53)     *Y(144))       +(DJ(54)     *Y(145))                 
     &+(DJ(7)      *Y(13 ))       +(DJ(8)      *Y(14 ))                 
     &+(RC(464)    *Y(3  )*Y(184))+(DJ(3)      *Y(12 )*2.00)            
     &+(RC(456)    *Y(3  )*Y(175))+(RC(463)    *Y(3  )*Y(183))          
     &+(RC(453)    *Y(3  )*Y(153))+(RC(454)    *Y(3  )*Y(174))          
     &+(RC(451)    *Y(3  )*Y(151))+(RC(452)    *Y(3  )*Y(152))          
     &+(RC(449)    *Y(3  )*Y(170))+(RC(450)    *Y(3  )*Y(171))          
     &+(RC(447)    *Y(3  )*Y(158))+(RC(448)    *Y(3  )*Y(169))          
     &+(RC(445)    *Y(3  )*Y(156))+(RC(446)    *Y(3  )*Y(157))          
     &+(RC(443)    *Y(3  )*Y(154))+(RC(444)    *Y(3  )*Y(155))          
     &+(RC(441)    *Y(3  )*Y(172))+(RC(442)    *Y(3  )*Y(173))          
     &+(RC(437)    *Y(3  )*Y(165))+(RC(438)    *Y(3  )*Y(166))          
     &+(RC(435)    *Y(3  )*Y(163))+(RC(436)    *Y(3  )*Y(164))          
     &+(RC(430)    *Y(3  )*Y(150))+(RC(434)    *Y(3  )*Y(162))          
     &+(RC(428)    *Y(3  )*Y(148))+(RC(429)    *Y(3  )*Y(149))          
     &+(RC(426)    *Y(3  )*Y(146))+(RC(427)    *Y(3  )*Y(147))          
     &+(RC(424)    *Y(3  )*Y(144))+(RC(425)    *Y(3  )*Y(145))          
     &+(RC(362)    *Y(6  )*Y(46 ))+(RC(374)    *Y(6  )*Y(109))          
     &+(RC(70)     *Y(53 )*Y(6  ))+(RC(75)     *Y(59 )*Y(3  ))          
     &+(RC(61)     *Y(6  )*Y(43 ))+(RC(65)     *Y(47 )*Y(6  ))          
     &+(RC(55)     *Y(6  )*Y(32 ))+(RC(57)     *Y(6  )*Y(34 ))          
     &+(RC(33)     *Y(9  )*Y(5  ))+(RC(53)     *Y(6  )*Y(30 ))          
     &+(RC(21)     *Y(9  )*Y(6  ))+(RC(29)     *Y(9  )*Y(8  ))          
     &+(RC(16)     *Y(1  )*H2O*2.00)                                    
      L = 0.0
     &+(RC(481)    *Y(201))+(RC(484)    *Y(203))                        
     &+(RC(474)    *Y(199))+(RC(475)    *Y(200))+(RC(480)    *Y(202))   
     &+(RC(465)    *Y(185))+(RC(466)    *Y(192))+(RC(473)    *Y(198))   
     &+(RC(462)    *Y(182))+(RC(463)    *Y(183))+(RC(464)    *Y(184))   
     &+(RC(459)    *Y(179))+(RC(460)    *Y(180))+(RC(461)    *Y(181))   
     &+(RC(456)    *Y(175))+(RC(457)    *Y(177))+(RC(458)    *Y(178))   
     &+(RC(453)    *Y(153))+(RC(454)    *Y(174))+(RC(455)    *Y(176))   
     &+(RC(450)    *Y(171))+(RC(451)    *Y(151))+(RC(452)    *Y(152))   
     &+(RC(447)    *Y(158))+(RC(448)    *Y(169))+(RC(449)    *Y(170))   
     &+(RC(444)    *Y(155))+(RC(445)    *Y(156))+(RC(446)    *Y(157))   
     &+(RC(441)    *Y(172))+(RC(442)    *Y(173))+(RC(443)    *Y(154))   
     &+(RC(438)    *Y(166))+(RC(439)    *Y(167))+(RC(440)    *Y(168))   
     &+(RC(435)    *Y(163))+(RC(436)    *Y(164))+(RC(437)    *Y(165))   
     &+(RC(432)    *Y(160))+(RC(433)    *Y(161))+(RC(434)    *Y(162))   
     &+(RC(429)    *Y(149))+(RC(430)    *Y(150))+(RC(431)    *Y(159))   
     &+(RC(426)    *Y(146))+(RC(427)    *Y(147))+(RC(428)    *Y(148))   
     &+(RC(423)    *Y(144))+(RC(424)    *Y(144))+(RC(425)    *Y(145))   
     &+(RC(416)    *Y(195))+(RC(418)    *Y(66 ))+(RC(421)    *Y(197))   
     &+(RC(411)    *Y(142))+(RC(412)    *Y(143))+(RC(413)    *Y(63 ))   
     &+(RC(408)    *Y(139))+(RC(409)    *Y(140))+(RC(410)    *Y(141))   
     &+(RC(405)    *Y(136))+(RC(406)    *Y(137))+(RC(407)    *Y(138))   
     &+(RC(402)    *Y(133))+(RC(403)    *Y(134))+(RC(404)    *Y(135))   
     &+(RC(399)    *Y(130))+(RC(400)    *Y(131))+(RC(401)    *Y(132))   
     &+(RC(396)    *Y(127))+(RC(397)    *Y(128))+(RC(398)    *Y(129))   
     &+(RC(393)    *Y(124))+(RC(394)    *Y(125))+(RC(395)    *Y(126))   
     &+(RC(390)    *Y(57 ))+(RC(391)    *Y(58 ))+(RC(392)    *Y(123))   
     &+(RC(385)    *Y(193))+(RC(386)    *Y(120))+(RC(389)    *Y(52 ))   
     &+(RC(382)    *Y(99 ))+(RC(383)    *Y(99 ))+(RC(384)    *Y(51 ))   
     &+(RC(379)    *Y(96 ))+(RC(380)    *Y(97 ))+(RC(381)    *Y(97 ))   
     &+(RC(376)    *Y(113))+(RC(377)    *Y(115))+(RC(378)    *Y(96 ))   
     &+(RC(370)    *Y(190))+(RC(371)    *Y(191))+(RC(372)    *Y(109))   
     &+(RC(367)    *Y(98 ))+(RC(368)    *Y(100))+(RC(369)    *Y(189))   
     &+(RC(360)    *Y(46 ))+(RC(364)    *Y(102))+(RC(366)    *Y(60 ))   
     &+(RC(357)    *Y(188))+(RC(358)    *Y(104))+(RC(359)    *Y(105))   
     &+(RC(354)    *Y(187))+(RC(355)    *Y(88 ))+(RC(356)    *Y(111))   
     &+(RC(105)    *Y(86 ))+(RC(106)    *Y(87 ))+(RC(353)    *Y(186))   
     &+(RC(102)    *Y(83 ))+(RC(103)    *Y(84 ))+(RC(104)    *Y(85 ))   
     &+(RC(99)     *Y(80 ))+(RC(100)    *Y(81 ))+(RC(101)    *Y(82 ))   
     &+(RC(96)     *Y(79 ))+(RC(97)     *Y(40 ))+(RC(98)     *Y(41 ))   
     &+(RC(93)     *Y(78 ))+(RC(94)     *Y(78 ))+(RC(95)     *Y(79 ))   
     &+(RC(90)     *Y(76 ))+(RC(91)     *Y(77 ))+(RC(92)     *Y(77 ))   
     &+(RC(84)     *Y(71 ))+(RC(88)     *Y(73 ))+(RC(89)     *Y(101))   
     &+(RC(81)     *Y(67 ))+(RC(82)     *Y(39 ))+(RC(83)     *Y(42 ))   
     &+(RC(78)     *Y(64 ))+(RC(79)     *Y(64 ))+(RC(80)     *Y(67 ))   
     &+(RC(75)     *Y(59 ))+(RC(76)     *Y(61 ))+(RC(77)     *Y(61 ))   
     &+(RC(63)     *Y(47 ))+(RC(68)     *Y(53 ))+(RC(74)     *Y(59 ))   
     &+(RC(48)     *Y(32 ))+(RC(49)     *Y(34 ))+(RC(59)     *Y(43 ))   
     &+(RC(45)     *Y(25 ))+(RC(46)     *Y(28 ))+(RC(47)     *Y(30 ))   
     &+(RC(42)     *Y(21 ))+(RC(43)     *Y(23 ))+(RC(44)     *Y(25 ))   
     &+(RC(34)     *Y(13 ))+(RC(35)     *Y(14 ))+(RC(37)     *Y(16 ))   
     &+(RC(27)     *Y(4  ))+(RC(28)     *Y(5  ))+(RC(32)     *Y(15 ))   
     &+(RC(20)     *Y(12 ))+(RC(22)     *Y(9  ))+(RC(25)     *Y(8  ))   
     &+(RC(17)     *Y(6  ))+(RC(18)     *Y(10 ))+(RC(19)     *Y(11 ))   
      Y(  3) = P/L
C
C          NO2              Y(  4)
      P = EM(  4)
     &+(DJ(86)     *Y(175))       +(DJ(96)     *Y(185))                 
     &+(DJ(80)     *Y(170))       +(DJ(81)     *Y(171))                 
     &+(DJ(52)     *Y(142))       +(DJ(79)     *Y(169))                 
     &+(DJ(50)     *Y(137))       +(DJ(51)     *Y(138))                 
     &+(DJ(48)     *Y(129))       +(DJ(49)     *Y(136))                 
     &+(DJ(46)     *Y(127))       +(DJ(47)     *Y(128))                 
     &+(DJ(44)     *Y(126))       +(DJ(45)     *Y(127))                 
     &+(DJ(42)     *Y(124))       +(DJ(43)     *Y(125))                 
     &+(DJ(32)     *Y(115))       +(DJ(41)     *Y(123))                 
     &+(DJ(8)      *Y(14 ))       +(DJ(31)     *Y(115))                 
     &+(RC(484)    *Y(3  )*Y(203))+(DJ(6)      *Y(5  ))                 
     &+(RC(481)    *Y(3  )*Y(201))+(RC(483)    *Y(203))                 
     &+(RC(479)    *Y(202))       +(RC(480)    *Y(3  )*Y(202))          
     &+(RC(475)    *Y(3  )*Y(200))+(RC(477)    *Y(201))                 
     &+(RC(473)    *Y(3  )*Y(198))+(RC(474)    *Y(3  )*Y(199))          
     &+(RC(470)    *Y(199))       +(RC(472)    *Y(200))                 
     &+(RC(456)    *Y(3  )*Y(175))+(RC(468)    *Y(198))                 
     &+(RC(449)    *Y(3  )*Y(170))+(RC(450)    *Y(3  )*Y(171))          
     &+(RC(422)    *Y(5  )*Y(197))+(RC(448)    *Y(3  )*Y(169))          
     &+(RC(417)    *Y(5  )*Y(195))+(RC(421)    *Y(3  )*Y(197))          
     &+(RC(412)    *Y(3  )*Y(143))+(RC(416)    *Y(3  )*Y(195))          
     &+(RC(410)    *Y(3  )*Y(141))+(RC(411)    *Y(3  )*Y(142))          
     &+(RC(408)    *Y(3  )*Y(139))+(RC(409)    *Y(3  )*Y(140))          
     &+(RC(406)    *Y(3  )*Y(137))+(RC(407)    *Y(3  )*Y(138))          
     &+(RC(404)    *Y(3  )*Y(135))+(RC(405)    *Y(3  )*Y(136))          
     &+(RC(402)    *Y(3  )*Y(133))+(RC(403)    *Y(3  )*Y(134))          
     &+(RC(400)    *Y(3  )*Y(131))+(RC(401)    *Y(3  )*Y(132))          
     &+(RC(398)    *Y(3  )*Y(129))+(RC(399)    *Y(3  )*Y(130))          
     &+(RC(396)    *Y(3  )*Y(127))+(RC(397)    *Y(3  )*Y(128))          
     &+(RC(394)    *Y(3  )*Y(125))+(RC(395)    *Y(3  )*Y(126))          
     &+(RC(392)    *Y(3  )*Y(123))+(RC(393)    *Y(3  )*Y(124))          
     &+(RC(352)    *Y(55 ))       +(RC(377)    *Y(3  )*Y(115))          
     &+(RC(338)    *Y(38 ))       +(RC(342)    *Y(49 ))                 
     &+(RC(336)    *Y(36 ))       +(RC(337)    *Y(37 ))                 
     &+(RC(242)    *Y(122)*Y(5  ))+(RC(243)    *Y(55 )*Y(5  )*2.00)     
     &+(RC(240)    *Y(54 )*Y(5  ))+(RC(241)    *Y(56 )*Y(5  ))          
     &+(RC(238)    *Y(119)*Y(5  ))+(RC(239)    *Y(121)*Y(5  ))          
     &+(RC(236)    *Y(117)*Y(5  ))+(RC(237)    *Y(118)*Y(5  ))          
     &+(RC(234)    *Y(50 )*Y(5  ))+(RC(235)    *Y(116)*Y(5  ))          
     &+(RC(232)    *Y(48 )*Y(5  ))+(RC(233)    *Y(49 )*Y(5  )*2.00)     
     &+(RC(230)    *Y(45 )*Y(5  ))+(RC(231)    *Y(114)*Y(5  ))          
     &+(RC(229)    *Y(38 )*Y(5  )*2.00)                                 
     &+(RC(228)    *Y(37 )*Y(5  )*2.00)                                 
     &+(RC(226)    *Y(112)*Y(5  ))+(RC(227)    *Y(36 )*Y(5  )*2.00)     
     &+(RC(224)    *Y(112)*Y(5  ))+(RC(225)    *Y(112)*Y(5  ))          
     &+(RC(222)    *Y(110)*Y(5  ))+(RC(223)    *Y(110)*Y(5  ))          
     &+(RC(220)    *Y(44 )*Y(5  ))+(RC(221)    *Y(44 )*Y(5  ))          
     &+(RC(218)    *Y(107)*Y(5  ))+(RC(219)    *Y(108)*Y(5  ))          
     &+(RC(216)    *Y(74 )*Y(5  ))+(RC(217)    *Y(75 )*Y(5  ))          
     &+(RC(214)    *Y(72 )*Y(5  ))+(RC(215)    *Y(106)*Y(5  ))          
     &+(RC(212)    *Y(92 )*Y(5  ))+(RC(213)    *Y(70 )*Y(5  ))          
     &+(RC(210)    *Y(103)*Y(5  ))+(RC(211)    *Y(90 )*Y(5  ))          
     &+(RC(208)    *Y(35 )*Y(5  ))+(RC(209)    *Y(95 )*Y(5  ))          
     &+(RC(206)    *Y(31 )*Y(5  ))+(RC(207)    *Y(33 )*Y(5  ))          
     &+(RC(204)    *Y(69 )*Y(5  ))+(RC(205)    *Y(31 )*Y(5  ))          
     &+(RC(202)    *Y(65 )*Y(5  ))+(RC(203)    *Y(68 )*Y(5  ))          
     &+(RC(200)    *Y(62 )*Y(5  ))+(RC(201)    *Y(65 )*Y(5  ))          
     &+(RC(198)    *Y(93 )*Y(5  ))+(RC(199)    *Y(94 )*Y(5  ))          
     &+(RC(196)    *Y(89 )*Y(5  ))+(RC(197)    *Y(91 )*Y(5  ))          
     &+(RC(194)    *Y(29 )*Y(5  ))+(RC(195)    *Y(29 )*Y(5  ))          
     &+(RC(192)    *Y(27 )*Y(5  ))+(RC(193)    *Y(26 )*Y(5  ))          
     &+(RC(190)    *Y(22 )*Y(5  ))+(RC(191)    *Y(24 )*Y(5  ))          
     &+(RC(163)    *Y(122)*Y(8  ))+(RC(165)    *Y(217))                 
     &+(RC(161)    *Y(56 )*Y(8  ))+(RC(162)    *Y(56 )*Y(8  ))          
     &+(RC(160)    *Y(55 )*Y(8  )*2.00)                                 
     &+(RC(158)    *Y(54 )*Y(8  ))+(RC(159)    *Y(54 )*Y(8  ))          
     &+(RC(156)    *Y(119)*Y(8  ))+(RC(157)    *Y(121)*Y(8  ))          
     &+(RC(154)    *Y(117)*Y(8  ))+(RC(155)    *Y(118)*Y(8  ))          
     &+(RC(152)    *Y(50 )*Y(8  ))+(RC(153)    *Y(116)*Y(8  ))          
     &+(RC(151)    *Y(49 )*Y(8  )*2.00)                                 
     &+(RC(149)    *Y(48 )*Y(8  ))+(RC(150)    *Y(48 )*Y(8  ))          
     &+(RC(147)    *Y(45 )*Y(8  ))+(RC(148)    *Y(114)*Y(8  ))          
     &+(RC(146)    *Y(38 )*Y(8  )*2.00)                                 
     &+(RC(145)    *Y(37 )*Y(8  )*2.00)                                 
     &+(RC(143)    *Y(112)*Y(8  ))+(RC(144)    *Y(36 )*Y(8  )*2.00)     
     &+(RC(141)    *Y(112)*Y(8  ))+(RC(142)    *Y(112)*Y(8  ))          
     &+(RC(139)    *Y(110)*Y(8  ))+(RC(140)    *Y(110)*Y(8  ))          
     &+(RC(137)    *Y(44 )*Y(8  ))+(RC(138)    *Y(44 )*Y(8  ))          
     &+(RC(135)    *Y(107)*Y(8  ))+(RC(136)    *Y(108)*Y(8  ))          
     &+(RC(133)    *Y(74 )*Y(8  ))+(RC(134)    *Y(75 )*Y(8  ))          
     &+(RC(131)    *Y(72 )*Y(8  ))+(RC(132)    *Y(106)*Y(8  ))          
     &+(RC(129)    *Y(92 )*Y(8  ))+(RC(130)    *Y(70 )*Y(8  ))          
     &+(RC(127)    *Y(103)*Y(8  ))+(RC(128)    *Y(90 )*Y(8  ))          
     &+(RC(125)    *Y(35 )*Y(8  ))+(RC(126)    *Y(95 )*Y(8  ))          
     &+(RC(123)    *Y(31 )*Y(8  ))+(RC(124)    *Y(33 )*Y(8  ))          
     &+(RC(121)    *Y(69 )*Y(8  ))+(RC(122)    *Y(31 )*Y(8  ))          
     &+(RC(119)    *Y(65 )*Y(8  ))+(RC(120)    *Y(68 )*Y(8  ))          
     &+(RC(117)    *Y(62 )*Y(8  ))+(RC(118)    *Y(65 )*Y(8  ))          
     &+(RC(115)    *Y(93 )*Y(8  ))+(RC(116)    *Y(94 )*Y(8  ))          
     &+(RC(113)    *Y(89 )*Y(8  ))+(RC(114)    *Y(91 )*Y(8  ))          
     &+(RC(111)    *Y(29 )*Y(8  ))+(RC(112)    *Y(29 )*Y(8  ))          
     &+(RC(109)    *Y(27 )*Y(8  ))+(RC(110)    *Y(26 )*Y(8  ))          
     &+(RC(107)    *Y(22 )*Y(8  ))+(RC(108)    *Y(24 )*Y(8  ))          
     &+(RC(33)     *Y(9  )*Y(5  ))+(RC(34)     *Y(3  )*Y(13 ))          
     &+(RC(31)     *Y(15 ))       +(RC(32)     *Y(3  )*Y(15 ))          
     &+(RC(28)     *Y(3  )*Y(5  ))+(RC(29)     *Y(9  )*Y(8  ))          
     &+(RC(13)     *Y(4  )*Y(5  ))+(RC(15)     *Y(7  ))                 
     &+(RC(12)     *Y(8  )*Y(5  )*2.00)                                 
     &+(RC(11)     *Y(8  )*Y(8  )*2.00)                                 
     &+(RC(4)      *Y(2  )*Y(8  ))+(RC(9)      *Y(8  )*Y(6  ))
C     
C          
      L = 0.0 
     &+(RC(478)    *Y(112))+(RC(482)    *Y(50 ))+(DJ(4)      )          
     &+(RC(469)    *Y(72 ))+(RC(471)    *Y(106))+(RC(476)    *Y(110))   
     &+(RC(415)    *Y(194))+(RC(420)    *Y(196))+(RC(467)    *Y(70 ))   
     &+(RC(27)     *Y(3  ))+(RC(30)     *Y(9  ))+(RC(164)    *Y(22 ))   
     &+(RC(13)     *Y(5  ))+(RC(14)     *Y(5  ))+(RC(26)     )          
     &+(RC(5)      *Y(2  ))+(RC(6)      *Y(2  ))+(RC(10)     *Y(6  ))   
      Y(  4) = (YP(  4)+DTS*P)/(1.0+DTS*L)
C
C          NO3              Y(  5)
      P = EM(  5)
     &+(RC(15)     *Y(7  ))       +(RC(35)     *Y(3  )*Y(14 ))          
     &+(RC(6)      *Y(2  )*Y(4  ))+(RC(10)     *Y(4  )*Y(6  ))          
      L = 0.0
     &+(DJ(6)      )                                                    
     &+(RC(419)    *Y(66 ))+(RC(422)    *Y(197))+(DJ(5)      )          
     &+(RC(388)    *Y(120))+(RC(414)    *Y(63 ))+(RC(417)    *Y(195))   
     &+(RC(365)    *Y(102))+(RC(373)    *Y(109))+(RC(387)    *Y(51 ))   
     &+(RC(242)    *Y(122))+(RC(243)    *Y(55 ))+(RC(361)    *Y(46 ))   
     &+(RC(239)    *Y(121))+(RC(240)    *Y(54 ))+(RC(241)    *Y(56 ))   
     &+(RC(236)    *Y(117))+(RC(237)    *Y(118))+(RC(238)    *Y(119))   
     &+(RC(233)    *Y(49 ))+(RC(234)    *Y(50 ))+(RC(235)    *Y(116))   
     &+(RC(230)    *Y(45 ))+(RC(231)    *Y(114))+(RC(232)    *Y(48 ))   
     &+(RC(227)    *Y(36 ))+(RC(228)    *Y(37 ))+(RC(229)    *Y(38 ))   
     &+(RC(224)    *Y(112))+(RC(225)    *Y(112))+(RC(226)    *Y(112))   
     &+(RC(221)    *Y(44 ))+(RC(222)    *Y(110))+(RC(223)    *Y(110))   
     &+(RC(218)    *Y(107))+(RC(219)    *Y(108))+(RC(220)    *Y(44 ))   
     &+(RC(215)    *Y(106))+(RC(216)    *Y(74 ))+(RC(217)    *Y(75 ))   
     &+(RC(212)    *Y(92 ))+(RC(213)    *Y(70 ))+(RC(214)    *Y(72 ))   
     &+(RC(209)    *Y(95 ))+(RC(210)    *Y(103))+(RC(211)    *Y(90 ))   
     &+(RC(206)    *Y(31 ))+(RC(207)    *Y(33 ))+(RC(208)    *Y(35 ))   
     &+(RC(203)    *Y(68 ))+(RC(204)    *Y(69 ))+(RC(205)    *Y(31 ))   
     &+(RC(200)    *Y(62 ))+(RC(201)    *Y(65 ))+(RC(202)    *Y(65 ))   
     &+(RC(197)    *Y(91 ))+(RC(198)    *Y(93 ))+(RC(199)    *Y(94 ))   
     &+(RC(194)    *Y(29 ))+(RC(195)    *Y(29 ))+(RC(196)    *Y(89 ))   
     &+(RC(191)    *Y(24 ))+(RC(192)    *Y(27 ))+(RC(193)    *Y(26 ))   
     &+(RC(86)     *Y(42 ))+(RC(87)     *Y(71 ))+(RC(190)    *Y(22 ))   
     &+(RC(64)     *Y(47 ))+(RC(69)     *Y(53 ))+(RC(85)     *Y(39 ))   
     &+(RC(51)     *Y(32 ))+(RC(52)     *Y(34 ))+(RC(60)     *Y(43 ))   
     &+(RC(28)     *Y(3  ))+(RC(33)     *Y(9  ))+(RC(50)     *Y(30 ))   
     &+(RC(12)     *Y(8  ))+(RC(13)     *Y(4  ))+(RC(14)     *Y(4  ))   
      Y(  5) = (YP(  5)+DTS*P)/(1.0+DTS*L)
C
C          O3               Y(  6)
      P = EM(  6)
     &+(RC(1)      *Y(2  ))       +(RC(2)      *Y(2  ))          
C     
C
      L = 0.0 
     &+(DJ(1)      )       +(DJ(2)      )                               
     &+(RC(363)    *Y(46 ))+(RC(374)    *Y(109))+(RC(375)    *Y(109))   
     &+(RC(72)     *Y(53 ))+(RC(73)     *Y(53 ))+(RC(362)    *Y(46 ))   
     &+(RC(67)     *Y(47 ))+(RC(70)     *Y(53 ))+(RC(71)     *Y(53 ))   
     &+(RC(62)     *Y(43 ))+(RC(65)     *Y(47 ))+(RC(66)     *Y(47 ))   
     &+(RC(57)     *Y(34 ))+(RC(58)     *Y(34 ))+(RC(61)     *Y(43 ))   
     &+(RC(54)     *Y(30 ))+(RC(55)     *Y(32 ))+(RC(56)     *Y(32 ))   
     &+(RC(17)     *Y(3  ))+(RC(21)     *Y(9  ))+(RC(53)     *Y(30 ))   
     &+(RC(3)      *Y(2  ))+(RC(9)      *Y(8  ))+(RC(10)     *Y(4  ))   
      Y(  6) = (YP(  6)+DTS*P)/(1.0+DTS*L)
C
C          N2O5             Y(  7)
      P = EM(  7)
     &+(RC(14)     *Y(4  )*Y(5  ))                                      
C     
C
      L = 0.0 
     &+(RC(15)     )       +(RC(40)     )                               
      Y(  7) = (YP(  7)+DTS*P)/(1.0+DTS*L)
C
C          NO               Y(  8)
      P = EM(  8)
     &+(DJ(7)      *Y(13 ))                                             
     &+(DJ(4)      *Y(4  ))       +(DJ(5)      *Y(5  ))                 
     &+(RC(5)      *Y(2  )*Y(4  ))+(RC(13)     *Y(4  )*Y(5  ))          
      L = 0.0
     &+(RC(189)    *Y(122))                                             
     &+(RC(186)    *Y(116))+(RC(187)    *Y(54 ))+(RC(188)    *Y(56 ))   
     &+(RC(183)    *Y(68 ))+(RC(184)    *Y(69 ))+(RC(185)    *Y(48 ))   
     &+(RC(180)    *Y(44 ))+(RC(181)    *Y(62 ))+(RC(182)    *Y(65 ))   
     &+(RC(177)    *Y(103))+(RC(178)    *Y(90 ))+(RC(179)    *Y(92 ))   
     &+(RC(174)    *Y(33 ))+(RC(175)    *Y(35 ))+(RC(176)    *Y(95 ))   
     &+(RC(171)    *Y(89 ))+(RC(172)    *Y(91 ))+(RC(173)    *Y(31 ))   
     &+(RC(168)    *Y(27 ))+(RC(169)    *Y(26 ))+(RC(170)    *Y(29 ))   
     &+(RC(163)    *Y(122))+(RC(166)    *Y(22 ))+(RC(167)    *Y(24 ))   
     &+(RC(160)    *Y(55 ))+(RC(161)    *Y(56 ))+(RC(162)    *Y(56 ))   
     &+(RC(157)    *Y(121))+(RC(158)    *Y(54 ))+(RC(159)    *Y(54 ))   
     &+(RC(154)    *Y(117))+(RC(155)    *Y(118))+(RC(156)    *Y(119))   
     &+(RC(151)    *Y(49 ))+(RC(152)    *Y(50 ))+(RC(153)    *Y(116))   
     &+(RC(148)    *Y(114))+(RC(149)    *Y(48 ))+(RC(150)    *Y(48 ))   
     &+(RC(145)    *Y(37 ))+(RC(146)    *Y(38 ))+(RC(147)    *Y(45 ))   
     &+(RC(142)    *Y(112))+(RC(143)    *Y(112))+(RC(144)    *Y(36 ))   
     &+(RC(139)    *Y(110))+(RC(140)    *Y(110))+(RC(141)    *Y(112))   
     &+(RC(136)    *Y(108))+(RC(137)    *Y(44 ))+(RC(138)    *Y(44 ))   
     &+(RC(133)    *Y(74 ))+(RC(134)    *Y(75 ))+(RC(135)    *Y(107))   
     &+(RC(130)    *Y(70 ))+(RC(131)    *Y(72 ))+(RC(132)    *Y(106))   
     &+(RC(127)    *Y(103))+(RC(128)    *Y(90 ))+(RC(129)    *Y(92 ))   
     &+(RC(124)    *Y(33 ))+(RC(125)    *Y(35 ))+(RC(126)    *Y(95 ))   
     &+(RC(121)    *Y(69 ))+(RC(122)    *Y(31 ))+(RC(123)    *Y(31 ))   
     &+(RC(118)    *Y(65 ))+(RC(119)    *Y(65 ))+(RC(120)    *Y(68 ))   
     &+(RC(115)    *Y(93 ))+(RC(116)    *Y(94 ))+(RC(117)    *Y(62 ))   
     &+(RC(112)    *Y(29 ))+(RC(113)    *Y(89 ))+(RC(114)    *Y(91 ))   
     &+(RC(109)    *Y(27 ))+(RC(110)    *Y(26 ))+(RC(111)    *Y(29 ))   
     &+(RC(29)     *Y(9  ))+(RC(107)    *Y(22 ))+(RC(108)    *Y(24 ))   
     &+(RC(11)     *Y(8  ))+(RC(12)     *Y(5  ))+(RC(25)     *Y(3  ))   
     &+(RC(4)      *Y(2  ))+(RC(9)      *Y(6  ))+(RC(11)     *Y(8  ))   
      Y(  8) = (YP(  8)+DTS*P)/(1.0+DTS*L)
C
C          HO2              Y(  9)
      P = EM(  9)
     &+(DJ(94)     *Y(183))                                             
     &+(DJ(91)     *Y(180))       +(DJ(93)     *Y(182))                 
     &+(DJ(84)     *Y(153))       +(DJ(85)     *Y(174))                 
     &+(DJ(82)     *Y(151))       +(DJ(83)     *Y(152))                 
     &+(DJ(77)     *Y(157))       +(DJ(78)     *Y(158))                 
     &+(DJ(75)     *Y(155))       +(DJ(76)     *Y(156))                 
     &+(DJ(73)     *Y(173))       +(DJ(74)     *Y(154))                 
     &+(DJ(70)     *Y(167))       +(DJ(72)     *Y(172))                 
     &+(DJ(68)     *Y(166))       +(DJ(69)     *Y(166))                 
     &+(DJ(58)     *Y(148))       +(DJ(63)     *Y(161))                 
     &+(DJ(55)     *Y(146))       +(DJ(56)     *Y(147))                 
     &+(DJ(53)     *Y(144))       +(DJ(54)     *Y(145))                 
     &+(DJ(51)     *Y(138))       +(DJ(52)     *Y(142))                 
     &+(DJ(49)     *Y(136))       +(DJ(50)     *Y(137))                 
     &+(DJ(44)     *Y(126))       +(DJ(46)     *Y(127))                 
     &+(DJ(42)     *Y(124))       +(DJ(43)     *Y(125))                 
     &+(DJ(39)     *Y(51 ))       +(DJ(41)     *Y(123))                 
     &+(DJ(37)     *Y(99 ))       +(DJ(38)     *Y(99 ))                 
     &+(DJ(35)     *Y(97 ))       +(DJ(36)     *Y(97 ))                 
     &+(DJ(33)     *Y(96 ))       +(DJ(34)     *Y(96 )*2.00)            
     &+(DJ(30)     *Y(113)*2.00)                                        
     &+(DJ(25)     *Y(98 ))       +(DJ(29)     *Y(109))                 
     &+(DJ(23)     *Y(46 ))       +(DJ(24)     *Y(60 )*2.00)            
     &+(DJ(22)     *Y(102)*2.00)                                        
     &+(DJ(20)     *Y(104))       +(DJ(21)     *Y(105))                 
     &+(DJ(18)     *Y(111))       +(DJ(19)     *Y(188))                 
     &+(DJ(11)     *Y(42 ))       +(DJ(12)     *Y(71 ))                 
     &+(RC(379)    *Y(3  )*Y(96 ))+(DJ(9)      *Y(39 )*2.00)            
     &+(RC(357)    *Y(3  )*Y(188))+(RC(366)    *Y(3  )*Y(60 ))          
     &+(RC(350)    *Y(56 ))       +(RC(356)    *Y(3  )*Y(111))          
     &+(RC(347)    *Y(119))       +(RC(349)    *Y(54 ))                 
     &+(RC(340)    *Y(114))       +(RC(341)    *Y(48 ))                 
     &+(RC(335)    *Y(112))       +(RC(339)    *Y(45 ))                 
     &+(RC(332)    *Y(110))       +(RC(334)    *Y(112))                 
     &+(RC(329)    *Y(44 ))       +(RC(330)    *Y(44 ))                 
     &+(RC(321)    *Y(92 ))       +(RC(324)    *Y(106))                 
     &+(RC(319)    *Y(103))       +(RC(320)    *Y(90 ))                 
     &+(RC(317)    *Y(35 ))       +(RC(318)    *Y(95 ))                 
     &+(RC(315)    *Y(31 ))       +(RC(316)    *Y(33 ))                 
     &+(RC(311)    *Y(69 ))       +(RC(314)    *Y(31 ))                 
     &+(RC(309)    *Y(65 ))       +(RC(310)    *Y(68 ))                 
     &+(RC(307)    *Y(62 ))       +(RC(308)    *Y(65 ))                 
     &+(RC(300)    *Y(26 ))       +(RC(304)    *Y(29 ))                 
     &+(RC(294)    *Y(24 ))       +(RC(297)    *Y(27 ))                 
     &+(RC(241)    *Y(56 )*Y(5  ))+(RC(291)    *Y(22 ))                 
     &+(RC(238)    *Y(119)*Y(5  ))+(RC(240)    *Y(54 )*Y(5  ))          
     &+(RC(231)    *Y(114)*Y(5  ))+(RC(232)    *Y(48 )*Y(5  ))          
     &+(RC(226)    *Y(112)*Y(5  ))+(RC(230)    *Y(45 )*Y(5  ))          
     &+(RC(223)    *Y(110)*Y(5  ))+(RC(225)    *Y(112)*Y(5  ))          
     &+(RC(220)    *Y(44 )*Y(5  ))+(RC(221)    *Y(44 )*Y(5  ))          
     &+(RC(212)    *Y(92 )*Y(5  ))+(RC(215)    *Y(106)*Y(5  ))          
     &+(RC(210)    *Y(103)*Y(5  ))+(RC(211)    *Y(90 )*Y(5  ))          
     &+(RC(208)    *Y(35 )*Y(5  ))+(RC(209)    *Y(95 )*Y(5  ))          
     &+(RC(206)    *Y(31 )*Y(5  ))+(RC(207)    *Y(33 )*Y(5  ))          
     &+(RC(204)    *Y(69 )*Y(5  ))+(RC(205)    *Y(31 )*Y(5  ))          
     &+(RC(202)    *Y(65 )*Y(5  ))+(RC(203)    *Y(68 )*Y(5  ))          
     &+(RC(200)    *Y(62 )*Y(5  ))+(RC(201)    *Y(65 )*Y(5  ))          
     &+(RC(193)    *Y(26 )*Y(5  ))+(RC(195)    *Y(29 )*Y(5  ))          
     &+(RC(191)    *Y(24 )*Y(5  ))+(RC(192)    *Y(27 )*Y(5  ))          
     &+(RC(161)    *Y(56 )*Y(8  ))+(RC(190)    *Y(22 )*Y(5  ))          
     &+(RC(156)    *Y(119)*Y(8  ))+(RC(158)    *Y(54 )*Y(8  ))          
     &+(RC(148)    *Y(114)*Y(8  ))+(RC(149)    *Y(48 )*Y(8  ))          
     &+(RC(143)    *Y(112)*Y(8  ))+(RC(147)    *Y(45 )*Y(8  ))          
     &+(RC(140)    *Y(110)*Y(8  ))+(RC(142)    *Y(112)*Y(8  ))          
     &+(RC(137)    *Y(44 )*Y(8  ))+(RC(138)    *Y(44 )*Y(8  ))          
     &+(RC(129)    *Y(92 )*Y(8  ))+(RC(132)    *Y(106)*Y(8  ))          
     &+(RC(127)    *Y(103)*Y(8  ))+(RC(128)    *Y(90 )*Y(8  ))          
     &+(RC(125)    *Y(35 )*Y(8  ))+(RC(126)    *Y(95 )*Y(8  ))          
     &+(RC(123)    *Y(31 )*Y(8  ))+(RC(124)    *Y(33 )*Y(8  ))          
     &+(RC(121)    *Y(69 )*Y(8  ))+(RC(122)    *Y(31 )*Y(8  ))          
     &+(RC(119)    *Y(65 )*Y(8  ))+(RC(120)    *Y(68 )*Y(8  ))          
     &+(RC(117)    *Y(62 )*Y(8  ))+(RC(118)    *Y(65 )*Y(8  ))          
     &+(RC(110)    *Y(26 )*Y(8  ))+(RC(112)    *Y(29 )*Y(8  ))          
     &+(RC(108)    *Y(24 )*Y(8  ))+(RC(109)    *Y(27 )*Y(8  ))          
     &+(RC(97)     *Y(40 )*Y(3  ))+(RC(107)    *Y(22 )*Y(8  ))          
     &+(RC(93)     *Y(78 )*Y(3  ))+(RC(95)     *Y(3  )*Y(79 ))          
     &+(RC(90)     *Y(3  )*Y(76 ))+(RC(91)     *Y(3  )*Y(77 ))          
     &+(RC(82)     *Y(3  )*Y(39 ))+(RC(85)     *Y(5  )*Y(39 ))          
     &+(RC(77)     *Y(61 )*Y(3  ))+(RC(79)     *Y(64 )*Y(3  ))          
     &+(RC(61)     *Y(6  )*Y(43 ))+(RC(74)     *Y(59 )*Y(3  ))          
     &+(RC(38)     *Y(18 ))       +(RC(53)     *Y(6  )*Y(30 ))          
     &+(RC(28)     *Y(3  )*Y(5  ))+(RC(31)     *Y(15 ))                 
     &+(RC(19)     *Y(3  )*Y(11 ))+(RC(20)     *Y(3  )*Y(12 ))          
     &+(RC(17)     *Y(3  )*Y(6  ))+(RC(18)     *Y(3  )*Y(10 ))          
      L = 0.0
     &+(RC(289)    *Y(122))+(RC(290)    *Y(55 ))                        
     &+(RC(286)    *Y(121))+(RC(287)    *Y(54 ))+(RC(288)    *Y(56 ))   
     &+(RC(283)    *Y(117))+(RC(284)    *Y(118))+(RC(285)    *Y(119))   
     &+(RC(280)    *Y(49 ))+(RC(281)    *Y(50 ))+(RC(282)    *Y(116))   
     &+(RC(277)    *Y(45 ))+(RC(278)    *Y(114))+(RC(279)    *Y(48 ))   
     &+(RC(274)    *Y(36 ))+(RC(275)    *Y(37 ))+(RC(276)    *Y(38 ))   
     &+(RC(271)    *Y(44 ))+(RC(272)    *Y(110))+(RC(273)    *Y(112))   
     &+(RC(268)    *Y(75 ))+(RC(269)    *Y(107))+(RC(270)    *Y(108))   
     &+(RC(265)    *Y(72 ))+(RC(266)    *Y(106))+(RC(267)    *Y(74 ))   
     &+(RC(262)    *Y(90 ))+(RC(263)    *Y(92 ))+(RC(264)    *Y(70 ))   
     &+(RC(259)    *Y(35 ))+(RC(260)    *Y(95 ))+(RC(261)    *Y(103))   
     &+(RC(256)    *Y(69 ))+(RC(257)    *Y(31 ))+(RC(258)    *Y(33 ))   
     &+(RC(253)    *Y(62 ))+(RC(254)    *Y(65 ))+(RC(255)    *Y(68 ))   
     &+(RC(250)    *Y(91 ))+(RC(251)    *Y(93 ))+(RC(252)    *Y(94 ))   
     &+(RC(247)    *Y(26 ))+(RC(248)    *Y(29 ))+(RC(249)    *Y(89 ))   
     &+(RC(244)    *Y(22 ))+(RC(245)    *Y(24 ))+(RC(246)    *Y(27 ))   
     &+(RC(29)     *Y(8  ))+(RC(30)     *Y(4  ))+(RC(33)     *Y(5  ))   
     &+(RC(23)     *Y(9  ))+(RC(24)     *Y(9  ))+(RC(24)     *Y(9  ))   
     &+(RC(21)     *Y(6  ))+(RC(22)     *Y(3  ))+(RC(23)     *Y(9  ))   
      Y(  9) = (YP(  9)+DTS*P)/(1.0+DTS*L)
C
C          H2               Y( 10)
      P = EM( 10)
     &+(DJ(10)     *Y(39 ))                                             
      L = 0.0
     &+(RC(18)     *Y(3  ))                                             
      Y( 10) = (YP( 10)+DTS*P)/(1.0+DTS*L)
C
C          CO               Y( 11)
      P = EM( 11)
     &+(DJ(92)     *Y(181))                                             
     &+(DJ(40)     *Y(120))       +(DJ(73)     *Y(173))                 
     &+(DJ(30)     *Y(113)*2.00)                                        
     &+(DJ(25)     *Y(98 ))       +(DJ(29)     *Y(109))                 
     &+(DJ(24)     *Y(60 )*2.00)                                        
     &+(DJ(12)     *Y(71 ))       +(DJ(22)     *Y(102))                 
     &+(DJ(10)     *Y(39 ))       +(DJ(11)     *Y(42 ))                 
     &+(RC(480)    *Y(3  )*Y(202))+(DJ(9)      *Y(39 ))                 
     &+(RC(474)    *Y(3  )*Y(199))+(RC(475)    *Y(3  )*Y(200))          
     &+(RC(442)    *Y(3  )*Y(173))+(RC(473)    *Y(3  )*Y(198))          
     &+(RC(367)    *Y(3  )*Y(98 ))+(RC(374)    *Y(6  )*Y(109))          
     &+(RC(362)    *Y(6  )*Y(46 ))+(RC(366)    *Y(3  )*Y(60 )*2.00)     
     &+(RC(340)    *Y(114))       +(RC(348)    *Y(121))                 
     &+(RC(231)    *Y(114)*Y(5  ))+(RC(239)    *Y(121)*Y(5  ))          
     &+(RC(157)    *Y(121)*Y(8  ))+(RC(223)    *Y(110)*Y(5  ))          
     &+(RC(140)    *Y(110)*Y(8  ))+(RC(148)    *Y(114)*Y(8  ))          
     &+(RC(82)     *Y(3  )*Y(39 ))+(RC(85)     *Y(5  )*Y(39 ))          
     &+(RC(73)     *Y(53 )*Y(6  ))+(RC(74)     *Y(59 )*Y(3  ))          
     &+(RC(57)     *Y(6  )*Y(34 ))+(RC(61)     *Y(6  )*Y(43 ))          
     &+(RC(53)     *Y(6  )*Y(30 ))+(RC(55)     *Y(6  )*Y(32 ))          
      L = 0.0
     &+(RC(19)     *Y(3  ))                                             
      Y( 11) = (YP( 11)+DTS*P)/(1.0+DTS*L)
C
C          H2O2             Y( 12)
      P = EM( 12)
     &+(RC(363)    *Y(6  )*Y(46 ))+(RC(375)    *Y(6  )*Y(109))          
     &+(RC(66)     *Y(47 )*Y(6  ))+(RC(71)     *Y(53 )*Y(6  ))          
     &+(RC(23)     *Y(9  )*Y(9  ))+(RC(24)     *Y(9  )*Y(9  ))          
      L = 0.0 
     &+(RC(20)     *Y(3  ))+(DJ(3)      )                               
      Y( 12) = (YP( 12)+DTS*P)/(1.0+DTS*L)
C
C          HONO             Y( 13)
      P = EM( 13)
     &+(RC(25)     *Y(3  )*Y(8  ))+(RC(26)     *Y(4  ))                 
      L = 0.0
     &+(RC(34)     *Y(3  ))+(DJ(7)      )                               
      Y( 13) = (YP( 13)+DTS*P)/(1.0+DTS*L)
C
C          HNO3             Y( 14)
      P = EM( 14)
     &+(RC(422)    *Y(5  )*Y(197))                                      
     &+(RC(417)    *Y(5  )*Y(195))+(RC(419)    *Y(5  )*Y(66 ))          
     &+(RC(388)    *Y(5  )*Y(120))+(RC(414)    *Y(5  )*Y(63 ))          
     &+(RC(373)    *Y(5  )*Y(109))+(RC(387)    *Y(5  )*Y(51 ))          
     &+(RC(361)    *Y(5  )*Y(46 ))+(RC(365)    *Y(5  )*Y(102))          
     &+(RC(86)     *Y(5  )*Y(42 ))+(RC(87)     *Y(5  )*Y(71 ))          
     &+(RC(27)     *Y(3  )*Y(4  ))+(RC(85)     *Y(5  )*Y(39 ))          
      L = 0.0 
     &+(RC(35)     *Y(3  ))+(RC(39)     )       +(DJ(8)      )          
      Y( 14) = (YP( 14)+DTS*P)/(1.0+DTS*L)
C
C          HO2NO2           Y( 15)
      P = EM( 15)
     &+(RC(30)     *Y(9  )*Y(4  ))                                      
      L = 0.0
     &+(RC(31)     )       +(RC(32)     *Y(3  ))                        
      Y( 15) = (YP( 15)+DTS*P)/(1.0+DTS*L)
C
C          SO2              Y( 16)
      P = EM( 16)
      L = 0.0 
     &+(RC(36)     *Y(2  ))+(RC(37)     *Y(3  ))                        
      Y( 16) = (YP( 16)+DTS*P)/(1.0+DTS*L)
C
C          SO3              Y( 17)
      P = EM( 17)
     &+(RC(36)     *Y(2  )*Y(16 ))+(RC(38)     *Y(18 ))                 
      L = 0.0
     &+(RC(41)     )                                                    
      Y( 17) = (YP( 17)+DTS*P)/(1.0+DTS*L)
C
C          HSO3             Y( 18)
      P = EM( 18)
     &+(RC(37)     *Y(3  )*Y(16 ))                                      
      L = 0.0
     &+(RC(38)     )                                                    
      Y( 18) = (YP( 18)+DTS*P)/(1.0+DTS*L)
C
C          NA               Y( 19)
      P = EM( 19)
     &+(RC(40)     *Y(7  ))                                             
     &+(RC(39)     *Y(14 ))       +(RC(40)     *Y(7  ))                 
      L = 0.0 
      Y( 19) = (YP( 19)+DTS*P)/(1.0+DTS*L)
C
C          SA               Y( 20)
      P = EM( 20)
     &+(RC(41)     *Y(17 ))                                             
      L = 0.0 
      Y( 20) = (YP( 20)+DTS*P)/(1.0+DTS*L)
C
C          CH4              Y( 21)
      P = EM( 21)
      L = 0.0
     &+(RC(42)     *Y(3  ))                                             
      Y( 21) = (YP( 21)+DTS*P)/(1.0+DTS*L)
C
C          CH3O2            Y( 22)
      P = EM( 22)
     &+(DJ(61)     *Y(159))                                             
     &+(DJ(13)     *Y(73 ))       +(DJ(36)     *Y(97 ))                 
     &+(RC(423)    *Y(3  )*Y(144))+(DJ(11)     *Y(42 ))                 
     &+(RC(322)    *Y(70 ))       +(RC(381)    *Y(3  )*Y(97 ))          
     &+(RC(165)    *Y(217))       +(RC(213)    *Y(70 )*Y(5  ))          
     &+(RC(101)    *Y(3  )*Y(82 ))+(RC(130)    *Y(70 )*Y(8  ))          
     &+(RC(99)     *Y(3  )*Y(80 ))+(RC(100)    *Y(3  )*Y(81 ))          
     &+(RC(57)     *Y(6  )*Y(34 ))+(RC(98)     *Y(41 )*Y(3  ))          
     &+(RC(42)     *Y(3  )*Y(21 ))+(RC(55)     *Y(6  )*Y(32 ))          
      L = 0.0
     &+(RC(292)    )       +(RC(293)    )                               
     &+(RC(190)    *Y(5  ))+(RC(244)    *Y(9  ))+(RC(291)    )          
     &+(RC(107)    *Y(8  ))+(RC(164)    *Y(4  ))+(RC(166)    *Y(8  ))   
      Y( 22) = (YP( 22)+DTS*P)/(1.0+DTS*L)
C
C          C2H6             Y( 23)
      P = EM( 23)
      L = 0.0
     &+(RC(43)     *Y(3  ))                                             
      Y( 23) = (YP( 23)+DTS*P)/(1.0+DTS*L)
C
C          C2H5O2           Y( 24)
      P = EM( 24)
     &+(DJ(64)     *Y(162))                                             
     &+(DJ(57)     *Y(148))       +(DJ(62)     *Y(160))                 
     &+(DJ(38)     *Y(99 ))       +(DJ(45)     *Y(127))                 
     &+(DJ(17)     *Y(88 ))       +(DJ(33)     *Y(96 ))                 
     &+(DJ(12)     *Y(71 ))       +(DJ(14)     *Y(101))                 
     &+(RC(378)    *Y(3  )*Y(96 ))+(RC(383)    *Y(3  )*Y(99 ))          
     &+(RC(303)    *Y(29 ))       +(RC(323)    *Y(72 ))                 
     &+(RC(194)    *Y(29 )*Y(5  ))+(RC(214)    *Y(72 )*Y(5  ))          
     &+(RC(111)    *Y(29 )*Y(8  ))+(RC(131)    *Y(72 )*Y(8  ))          
     &+(RC(43)     *Y(3  )*Y(23 ))+(RC(102)    *Y(3  )*Y(83 ))          
      L = 0.0
     &+(RC(296)    )                                                    
     &+(RC(245)    *Y(9  ))+(RC(294)    )       +(RC(295)    )          
     &+(RC(108)    *Y(8  ))+(RC(167)    *Y(8  ))+(RC(191)    *Y(5  ))   
      Y( 24) = (YP( 24)+DTS*P)/(1.0+DTS*L)
C
C          C3H8             Y( 25)
      P = EM( 25)
      L = 0.0
     &+(RC(44)     *Y(3  ))+(RC(45)     *Y(3  ))                        
      Y( 25) = (YP( 25)+DTS*P)/(1.0+DTS*L)
C
C          IC3H7O2          Y( 26)
      P = EM( 26)
     &+(RC(44)     *Y(3  )*Y(25 ))                                      
      L = 0.0
     &+(RC(302)    )                                                    
     &+(RC(247)    *Y(9  ))+(RC(300)    )       +(RC(301)    )          
     &+(RC(110)    *Y(8  ))+(RC(169)    *Y(8  ))+(RC(193)    *Y(5  ))   
      Y( 26) = (YP( 26)+DTS*P)/(1.0+DTS*L)
C
C          RN10O2           Y( 27)
      P = EM( 27)
     &+(DJ(35)     *Y(97 ))       +(DJ(65)     *Y(163))                 
     &+(DJ(15)     *Y(186))       +(DJ(16)     *Y(187))                 
     &+(RC(45)     *Y(3  )*Y(25 ))+(RC(380)    *Y(3  )*Y(97 ))          
      L = 0.0
     &+(RC(299)    )                                                    
     &+(RC(246)    *Y(9  ))+(RC(297)    )       +(RC(298)    )          
     &+(RC(109)    *Y(8  ))+(RC(168)    *Y(8  ))+(RC(192)    *Y(5  ))   
      Y( 27) = (YP( 27)+DTS*P)/(1.0+DTS*L)
C
C          NC4H10           Y( 28)
      P = EM( 28)
      L = 0.0
     &+(RC(46)     *Y(3  ))                                             
      Y( 28) = (YP( 28)+DTS*P)/(1.0+DTS*L)
C
C          RN13O2           Y( 29)
      P = EM( 29)
     &+(DJ(95)     *Y(184))                                             
     &+(DJ(37)     *Y(99 ))       +(DJ(66)     *Y(164))                 
     &+(RC(358)    *Y(3  )*Y(104))+(RC(382)    *Y(3  )*Y(99 ))          
     &+(RC(242)    *Y(122)*Y(5  ))+(RC(351)    *Y(122))                 
     &+(RC(46)     *Y(3  )*Y(28 ))+(RC(163)    *Y(122)*Y(8  ))          
      L = 0.0
     &+(RC(303)    )       +(RC(304)    )                               
     &+(RC(194)    *Y(5  ))+(RC(195)    *Y(5  ))+(RC(248)    *Y(9  ))   
     &+(RC(111)    *Y(8  ))+(RC(112)    *Y(8  ))+(RC(170)    *Y(8  ))   
      Y( 29) = (YP( 29)+DTS*P)/(1.0+DTS*L)
C
C          C2H4             Y( 30)
      P = EM( 30)
      L = 0.0
     &+(RC(54)     *Y(6  ))                                             
     &+(RC(47)     *Y(3  ))+(RC(50)     *Y(5  ))+(RC(53)     *Y(6  ))   
      Y( 30) = (YP( 30)+DTS*P)/(1.0+DTS*L)
C
C          HOCH2CH2O2       Y( 31)
      P = EM( 31)
     &+(RC(466)    *Y(3  )*Y(192))                                      
     &+(RC(105)    *Y(3  )*Y(86 ))+(RC(106)    *Y(3  )*Y(87 ))          
     &+(RC(103)    *Y(3  )*Y(84 ))+(RC(104)    *Y(3  )*Y(85 ))          
     &+(RC(47)     *Y(3  )*Y(30 ))+(RC(92)     *Y(3  )*Y(77 ))          
      L = 0.0
     &+(RC(314)    )       +(RC(315)    )                               
     &+(RC(205)    *Y(5  ))+(RC(206)    *Y(5  ))+(RC(257)    *Y(9  ))   
     &+(RC(122)    *Y(8  ))+(RC(123)    *Y(8  ))+(RC(173)    *Y(8  ))   
      Y( 31) = (YP( 31)+DTS*P)/(1.0+DTS*L)
C
C          C3H6             Y( 32)
      P = EM( 32)
      L = 0.0
     &+(RC(56)     *Y(6  ))                                             
     &+(RC(48)     *Y(3  ))+(RC(51)     *Y(5  ))+(RC(55)     *Y(6  ))   
      Y( 32) = (YP( 32)+DTS*P)/(1.0+DTS*L)
C
C          RN9O2            Y( 33)
      P = EM( 33)
     &+(RC(96)     *Y(3  )*Y(79 ))+(RC(368)    *Y(3  )*Y(100))          
     &+(RC(48)     *Y(3  )*Y(32 ))+(RC(94)     *Y(78 )*Y(3  ))          
      L = 0.0
     &+(RC(258)    *Y(9  ))+(RC(316)    )                               
     &+(RC(124)    *Y(8  ))+(RC(174)    *Y(8  ))+(RC(207)    *Y(5  ))   
      Y( 33) = (YP( 33)+DTS*P)/(1.0+DTS*L)
C
C          TBUT2ENE         Y( 34)
      P = EM( 34)
      L = 0.0
     &+(RC(58)     *Y(6  ))                                             
     &+(RC(49)     *Y(3  ))+(RC(52)     *Y(5  ))+(RC(57)     *Y(6  ))   
      Y( 34) = (YP( 34)+DTS*P)/(1.0+DTS*L)
C
C          RN12O2           Y( 35)
      P = EM( 35)
     &+(RC(369)    *Y(3  )*Y(189))+(RC(371)    *Y(3  )*Y(191))          
     &+(RC(198)    *Y(93 )*Y(5  ))+(RC(305)    *Y(93 ))                 
     &+(RC(49)     *Y(3  )*Y(34 ))+(RC(115)    *Y(93 )*Y(8  ))          
      L = 0.0
     &+(RC(259)    *Y(9  ))+(RC(317)    )                               
     &+(RC(125)    *Y(8  ))+(RC(175)    *Y(8  ))+(RC(208)    *Y(5  ))   
      Y( 35) = (YP( 35)+DTS*P)/(1.0+DTS*L)
C
C          NRN6O2           Y( 36)
      P = EM( 36)
     &+(RC(50)     *Y(5  )*Y(30 ))                                      
      L = 0.0
     &+(RC(336)    )                                                    
     &+(RC(144)    *Y(8  ))+(RC(227)    *Y(5  ))+(RC(274)    *Y(9  ))   
      Y( 36) = (YP( 36)+DTS*P)/(1.0+DTS*L)
C
C          NRN9O2           Y( 37)
      P = EM( 37)
     &+(RC(51)     *Y(5  )*Y(32 ))                                      
      L = 0.0
     &+(RC(337)    )                                                    
     &+(RC(145)    *Y(8  ))+(RC(228)    *Y(5  ))+(RC(275)    *Y(9  ))   
      Y( 37) = (YP( 37)+DTS*P)/(1.0+DTS*L)
C
C          NRN12O2          Y( 38)
      P = EM( 38)
     &+(RC(52)     *Y(5  )*Y(34 ))                                      
      L = 0.0
     &+(RC(338)    )                                                    
     &+(RC(146)    *Y(8  ))+(RC(229)    *Y(5  ))+(RC(276)    *Y(9  ))   
      Y( 38) = (YP( 38)+DTS*P)/(1.0+DTS*L)
C
C          HCHO             Y( 39)
      P = EM( 39)
     &+(DJ(93)     *Y(182))       +(DJ(96)     *Y(185))                 
     &+(DJ(80)     *Y(170))       +(DJ(91)     *Y(180))                 
     &+(DJ(75)     *Y(155))       +(DJ(79)     *Y(169)*2.00)            
     &+(DJ(74)     *Y(154)*2.00)                                        
     &+(DJ(63)     *Y(161))       +(DJ(69)     *Y(166))                 
     &+(DJ(41)     *Y(123))       +(DJ(53)     *Y(144))                 
     &+(DJ(31)     *Y(115))       +(DJ(32)     *Y(115))                 
     &+(DJ(22)     *Y(102))       +(DJ(23)     *Y(46 ))                 
     &+(RC(475)    *Y(3  )*Y(200))+(DJ(18)     *Y(111))                 
     &+(RC(449)    *Y(3  )*Y(170))+(RC(473)    *Y(3  )*Y(198))          
     &+(RC(424)    *Y(3  )*Y(144))+(RC(448)    *Y(3  )*Y(169)*2.00)     
     &+(RC(392)    *Y(3  )*Y(123))+(RC(410)    *Y(3  )*Y(141))          
     &+(RC(362)    *Y(6  )*Y(46 ))+(RC(363)    *Y(6  )*Y(46 ))          
     &+(RC(349)    *Y(54 ))       +(RC(352)    *Y(55 ))                 
     &+(RC(337)    *Y(37 ))       +(RC(347)    *Y(119))                 
     &+(RC(336)    *Y(36 )*2.00)                                        
     &+(RC(334)    *Y(112))       +(RC(335)    *Y(112))                 
     &+(RC(325)    *Y(74 ))       +(RC(330)    *Y(44 ))                 
     &+(RC(316)    *Y(33 ))       +(RC(324)    *Y(106))                 
     &+(RC(314)    *Y(31 )*2.00)                                        
     &+(RC(291)    *Y(22 ))       +(RC(292)    *Y(22 ))                 
     &+(RC(240)    *Y(54 )*Y(5  ))+(RC(243)    *Y(55 )*Y(5  ))          
     &+(RC(228)    *Y(37 )*Y(5  ))+(RC(238)    *Y(119)*Y(5  ))          
     &+(RC(227)    *Y(36 )*Y(5  ))+(RC(227)    *Y(36 )*Y(5  ))          
     &+(RC(225)    *Y(112)*Y(5  ))+(RC(226)    *Y(112)*Y(5  ))          
     &+(RC(216)    *Y(74 )*Y(5  ))+(RC(221)    *Y(44 )*Y(5  ))          
     &+(RC(207)    *Y(33 )*Y(5  ))+(RC(215)    *Y(106)*Y(5  ))          
     &+(RC(205)    *Y(31 )*Y(5  )*2.00)                                 
     &+(RC(162)    *Y(56 )*Y(8  ))+(RC(190)    *Y(22 )*Y(5  ))          
     &+(RC(158)    *Y(54 )*Y(8  ))+(RC(160)    *Y(55 )*Y(8  ))          
     &+(RC(145)    *Y(37 )*Y(8  ))+(RC(156)    *Y(119)*Y(8  ))          
     &+(RC(144)    *Y(36 )*Y(8  ))+(RC(144)    *Y(36 )*Y(8  ))          
     &+(RC(142)    *Y(112)*Y(8  ))+(RC(143)    *Y(112)*Y(8  ))          
     &+(RC(133)    *Y(74 )*Y(8  ))+(RC(138)    *Y(44 )*Y(8  ))          
     &+(RC(124)    *Y(33 )*Y(8  ))+(RC(132)    *Y(106)*Y(8  ))          
     &+(RC(122)    *Y(31 )*Y(8  )*2.00)                                 
     &+(RC(90)     *Y(3  )*Y(76 ))+(RC(107)    *Y(22 )*Y(8  ))          
     &+(RC(71)     *Y(53 )*Y(6  ))+(RC(72)     *Y(53 )*Y(6  ))          
     &+(RC(55)     *Y(6  )*Y(32 ))+(RC(56)     *Y(6  )*Y(32 ))          
     &+(RC(53)     *Y(6  )*Y(30 ))+(RC(54)     *Y(6  )*Y(30 ))          
      L = 0.0
     &+(DJ(10)     )                                                    
     &+(RC(82)     *Y(3  ))+(RC(85)     *Y(5  ))+(DJ(9)      )          
      Y( 39) = (YP( 39)+DTS*P)/(1.0+DTS*L)
C
C          HCOOH            Y( 40)
      P = EM( 40)
     &+(RC(74)     *Y(59 )*Y(3  ))                                      
     &+(RC(54)     *Y(6  )*Y(30 ))+(RC(62)     *Y(6  )*Y(43 ))          
      L = 0.0
     &+(RC(97)     *Y(3  ))                                             
      Y( 40) = (YP( 40)+DTS*P)/(1.0+DTS*L)
C
C          CH3CO2H          Y( 41)
      P = EM( 41)
     &+(RC(56)     *Y(6  )*Y(32 ))+(RC(58)     *Y(6  )*Y(34 ))          
      L = 0.0
     &+(RC(98)     *Y(3  ))                                             
      Y( 41) = (YP( 41)+DTS*P)/(1.0+DTS*L)
C
C          CH3CHO           Y( 42)
      P = EM( 42)
     &+(DJ(81)     *Y(171)*2.00)                                        
     &+(DJ(77)     *Y(157))       +(DJ(80)     *Y(170))                 
     &+(DJ(76)     *Y(156)*2.00)                                        
     &+(DJ(57)     *Y(148))       +(DJ(75)     *Y(155))                 
     &+(DJ(45)     *Y(127))       +(DJ(54)     *Y(145))                 
     &+(DJ(20)     *Y(104))       +(DJ(42)     *Y(124))                 
     &+(RC(474)    *Y(3  )*Y(199))+(DJ(19)     *Y(188))                 
     &+(RC(449)    *Y(3  )*Y(170))+(RC(450)    *Y(3  )*Y(171)*2.00)     
     &+(RC(393)    *Y(3  )*Y(124))+(RC(425)    *Y(3  )*Y(145))          
     &+(RC(338)    *Y(38 )*2.00)                                        
     &+(RC(327)    *Y(107))       +(RC(337)    *Y(37 ))                 
     &+(RC(318)    *Y(95 ))       +(RC(326)    *Y(75 ))                 
     &+(RC(317)    *Y(35 )*2.00)                                        
     &+(RC(303)    *Y(29 ))       +(RC(316)    *Y(33 ))                 
     &+(RC(294)    *Y(24 ))       +(RC(295)    *Y(24 ))                 
     &+(RC(229)    *Y(38 )*Y(5  )*2.00)                                 
     &+(RC(218)    *Y(107)*Y(5  ))+(RC(228)    *Y(37 )*Y(5  ))          
     &+(RC(209)    *Y(95 )*Y(5  ))+(RC(217)    *Y(75 )*Y(5  ))          
     &+(RC(207)    *Y(33 )*Y(5  ))+(RC(208)    *Y(35 )*Y(5  )*2.00)     
     &+(RC(191)    *Y(24 )*Y(5  ))+(RC(194)    *Y(29 )*Y(5  ))          
     &+(RC(146)    *Y(38 )*Y(8  )*2.00)                                 
     &+(RC(135)    *Y(107)*Y(8  ))+(RC(145)    *Y(37 )*Y(8  ))          
     &+(RC(126)    *Y(95 )*Y(8  ))+(RC(134)    *Y(75 )*Y(8  ))          
     &+(RC(125)    *Y(35 )*Y(8  )*2.00)                                 
     &+(RC(111)    *Y(29 )*Y(8  ))+(RC(124)    *Y(33 )*Y(8  ))          
     &+(RC(91)     *Y(3  )*Y(77 ))+(RC(108)    *Y(24 )*Y(8  ))          
     &+(RC(57)     *Y(6  )*Y(34 ))+(RC(58)     *Y(6  )*Y(34 ))          
      L = 0.0
     &+(RC(83)     *Y(3  ))+(RC(86)     *Y(5  ))+(DJ(11)     )          
      Y( 42) = (YP( 42)+DTS*P)/(1.0+DTS*L)
C
C          C5H8             Y( 43)
      P = EM( 43)
      L = 0.0
     &+(RC(62)     *Y(6  ))                                             
     &+(RC(59)     *Y(3  ))+(RC(60)     *Y(5  ))+(RC(61)     *Y(6  ))   
      Y( 43) = (YP( 43)+DTS*P)/(1.0+DTS*L)
C
C          RU14O2           Y( 44)
      P = EM( 44)
     &+(RC(59)     *Y(3  )*Y(43 ))                                      
      L = 0.0
     &+(RC(329)    )       +(RC(330)    )                               
     &+(RC(220)    *Y(5  ))+(RC(221)    *Y(5  ))+(RC(271)    *Y(9  ))   
     &+(RC(137)    *Y(8  ))+(RC(138)    *Y(8  ))+(RC(180)    *Y(8  ))   
      Y( 44) = (YP( 44)+DTS*P)/(1.0+DTS*L)
C
C          NRU14O2          Y( 45)
      P = EM( 45)
     &+(RC(60)     *Y(5  )*Y(43 ))                                      
      L = 0.0
     &+(RC(339)    )                                                    
     &+(RC(147)    *Y(8  ))+(RC(230)    *Y(5  ))+(RC(277)    *Y(9  ))   
      Y( 45) = (YP( 45)+DTS*P)/(1.0+DTS*L)
C
C          UCARB10          Y( 46)
      P = EM( 46)
     &+(DJ(69)     *Y(166))                                             
     &+(RC(330)    *Y(44 ))       +(RC(481)    *Y(3  )*Y(201))          
     &+(RC(138)    *Y(44 )*Y(8  ))+(RC(221)    *Y(44 )*Y(5  ))          
     &+(RC(61)     *Y(6  )*Y(43 ))+(RC(62)     *Y(6  )*Y(43 ))          
      L = 0.0
     &+(RC(363)    *Y(6  ))+(DJ(23)     )                               
     &+(RC(360)    *Y(3  ))+(RC(361)    *Y(5  ))+(RC(362)    *Y(6  ))   
      Y( 46) = (YP( 46)+DTS*P)/(1.0+DTS*L)
C
C          APINENE          Y( 47)
      P = EM( 47)
      L = 0.0
     &+(RC(66)     *Y(6  ))+(RC(67)     *Y(6  ))                        
     &+(RC(63)     *Y(3  ))+(RC(64)     *Y(5  ))+(RC(65)     *Y(6  ))   
      Y( 47) = (YP( 47)+DTS*P)/(1.0+DTS*L)
C
C          RTN28O2          Y( 48)
      P = EM( 48)
     &+(RC(63)     *Y(47 )*Y(3  ))                                      
      L = 0.0
     &+(RC(232)    *Y(5  ))+(RC(279)    *Y(9  ))+(RC(341)    )          
     &+(RC(149)    *Y(8  ))+(RC(150)    *Y(8  ))+(RC(185)    *Y(8  ))   
      Y( 48) = (YP( 48)+DTS*P)/(1.0+DTS*L)
C
C          NRTN28O2         Y( 49)
      P = EM( 49)
     &+(RC(64)     *Y(47 )*Y(5  ))                                      
      L = 0.0
     &+(RC(342)    )                                                    
     &+(RC(151)    *Y(8  ))+(RC(233)    *Y(5  ))+(RC(280)    *Y(9  ))   
      Y( 49) = (YP( 49)+DTS*P)/(1.0+DTS*L)
C
C          RTN26O2          Y( 50)
      P = EM( 50)
     &+(RC(483)    *Y(203))       +(DJ(39)     *Y(51 ))                 
     &+(RC(387)    *Y(5  )*Y(51 ))+(RC(455)    *Y(3  )*Y(176))          
     &+(RC(65)     *Y(47 )*Y(6  ))+(RC(384)    *Y(3  )*Y(51 ))          
      L = 0.0
     &+(RC(343)    )       +(RC(482)    *Y(4  ))                        
     &+(RC(152)    *Y(8  ))+(RC(234)    *Y(5  ))+(RC(281)    *Y(9  ))   
      Y( 50) = (YP( 50)+DTS*P)/(1.0+DTS*L)
C
C          TNCARB26         Y( 51)
      P = EM( 51)
     &+(DJ(85)     *Y(174))       +(DJ(86)     *Y(175))                 
     &+(RC(454)    *Y(3  )*Y(174))+(RC(456)    *Y(3  )*Y(175))          
     &+(RC(342)    *Y(49 ))       +(RC(408)    *Y(3  )*Y(139))          
     &+(RC(233)    *Y(49 )*Y(5  ))+(RC(341)    *Y(48 ))                 
     &+(RC(151)    *Y(49 )*Y(8  ))+(RC(232)    *Y(48 )*Y(5  ))          
     &+(RC(66)     *Y(47 )*Y(6  ))+(RC(149)    *Y(48 )*Y(8  ))          
      L = 0.0
     &+(RC(384)    *Y(3  ))+(RC(387)    *Y(5  ))+(DJ(39)     )          
      Y( 51) = (YP( 51)+DTS*P)/(1.0+DTS*L)
C
C          RCOOH25          Y( 52)
      P = EM( 52)
     &+(RC(67)     *Y(47 )*Y(6  ))+(RC(490)    *Y(206))                 
      L = 0.0
     &+(RC(389)    *Y(3  ))+(RC(489)    )                               
      Y( 52) = (YP( 52)+DTS*P)/(1.0+DTS*L)
C
C          BPINENE          Y( 53)
      P = EM( 53)
      L = 0.0
     &+(RC(71)     *Y(6  ))+(RC(72)     *Y(6  ))+(RC(73)     *Y(6  ))   
     &+(RC(68)     *Y(3  ))+(RC(69)     *Y(5  ))+(RC(70)     *Y(6  ))   
      Y( 53) = (YP( 53)+DTS*P)/(1.0+DTS*L)
C
C          RTX28O2          Y( 54)
      P = EM( 54)
     &+(RC(68)     *Y(53 )*Y(3  ))+(RC(462)    *Y(3  )*Y(182))          
      L = 0.0
     &+(RC(240)    *Y(5  ))+(RC(287)    *Y(9  ))+(RC(349)    )          
     &+(RC(158)    *Y(8  ))+(RC(159)    *Y(8  ))+(RC(187)    *Y(8  ))   
      Y( 54) = (YP( 54)+DTS*P)/(1.0+DTS*L)
C
C          NRTX28O2         Y( 55)
      P = EM( 55)
     &+(RC(69)     *Y(53 )*Y(5  ))+(RC(465)    *Y(3  )*Y(185))          
      L = 0.0
     &+(RC(352)    )                                                    
     &+(RC(160)    *Y(8  ))+(RC(243)    *Y(5  ))+(RC(290)    *Y(9  ))   
      Y( 55) = (YP( 55)+DTS*P)/(1.0+DTS*L)
C
C          RTX24O2          Y( 56)
      P = EM( 56)
     &+(RC(70)     *Y(53 )*Y(6  ))+(RC(390)    *Y(3  )*Y(57 ))          
      L = 0.0
     &+(RC(241)    *Y(5  ))+(RC(288)    *Y(9  ))+(RC(350)    )          
     &+(RC(161)    *Y(8  ))+(RC(162)    *Y(8  ))+(RC(188)    *Y(8  ))   
      Y( 56) = (YP( 56)+DTS*P)/(1.0+DTS*L)
C
C          TXCARB24         Y( 57)
      P = EM( 57)
     &+(DJ(96)     *Y(185))                                             
     &+(RC(410)    *Y(3  )*Y(141))+(DJ(93)     *Y(182))                 
     &+(RC(349)    *Y(54 ))       +(RC(352)    *Y(55 ))                 
     &+(RC(240)    *Y(54 )*Y(5  ))+(RC(243)    *Y(55 )*Y(5  ))          
     &+(RC(158)    *Y(54 )*Y(8  ))+(RC(160)    *Y(55 )*Y(8  ))          
     &+(RC(71)     *Y(53 )*Y(6  ))+(RC(73)     *Y(53 )*Y(6  ))          
      L = 0.0
     &+(RC(390)    *Y(3  ))                                             
      Y( 57) = (YP( 57)+DTS*P)/(1.0+DTS*L)
C
C          TXCARB22         Y( 58)
      P = EM( 58)
     &+(DJ(52)     *Y(142))       +(DJ(94)     *Y(183))                 
     &+(RC(411)    *Y(3  )*Y(142))+(RC(463)    *Y(3  )*Y(183))          
     &+(RC(241)    *Y(56 )*Y(5  ))+(RC(350)    *Y(56 ))                 
     &+(RC(72)     *Y(53 )*Y(6  ))+(RC(161)    *Y(56 )*Y(8  ))          
      L = 0.0
     &+(RC(391)    *Y(3  ))                                             
      Y( 58) = (YP( 58)+DTS*P)/(1.0+DTS*L)
C
C          C2H2             Y( 59)
      P = EM( 59)
      L = 0.0
     &+(RC(74)     *Y(3  ))+(RC(75)     *Y(3  ))                        
      Y( 59) = (YP( 59)+DTS*P)/(1.0+DTS*L)
C
C          CARB3            Y( 60)
      P = EM( 60)
     &+(DJ(83)     *Y(152))                                             
     &+(DJ(50)     *Y(137))       +(DJ(82)     *Y(151))                 
     &+(RC(452)    *Y(3  )*Y(152))+(DJ(49)     *Y(136))                 
     &+(RC(406)    *Y(3  )*Y(137))+(RC(451)    *Y(3  )*Y(151))          
     &+(RC(311)    *Y(69 ))       +(RC(405)    *Y(3  )*Y(136))          
     &+(RC(308)    *Y(65 ))       +(RC(310)    *Y(68 ))                 
     &+(RC(203)    *Y(68 )*Y(5  ))+(RC(307)    *Y(62 ))                 
     &+(RC(200)    *Y(62 )*Y(5  ))+(RC(201)    *Y(65 )*Y(5  ))          
     &+(RC(118)    *Y(65 )*Y(8  ))+(RC(120)    *Y(68 )*Y(8  ))          
     &+(RC(75)     *Y(59 )*Y(3  ))+(RC(117)    *Y(62 )*Y(8  ))          
      L = 0.0
     &+(RC(366)    *Y(3  ))+(DJ(24)     )                               
      Y( 60) = (YP( 60)+DTS*P)/(1.0+DTS*L)
C
C          BENZENE          Y( 61)
      P = EM( 61)
      L = 0.0
     &+(RC(76)     *Y(3  ))+(RC(77)     *Y(3  ))                        
      Y( 61) = (YP( 61)+DTS*P)/(1.0+DTS*L)
C
C          RA13O2           Y( 62)
      P = EM( 62)
     &+(RC(76)     *Y(61 )*Y(3  ))                                      
      L = 0.0
     &+(RC(253)    *Y(9  ))+(RC(307)    )                               
     &+(RC(117)    *Y(8  ))+(RC(181)    *Y(8  ))+(RC(200)    *Y(5  ))   
      Y( 62) = (YP( 62)+DTS*P)/(1.0+DTS*L)
C
C          AROH14           Y( 63)
      P = EM( 63)
     &+(RC(77)     *Y(61 )*Y(3  ))                                      
      L = 0.0
     &+(RC(413)    *Y(3  ))+(RC(414)    *Y(5  ))                        
      Y( 63) = (YP( 63)+DTS*P)/(1.0+DTS*L)
C
C          TOLUENE          Y( 64)
      P = EM( 64)
      L = 0.0
     &+(RC(78)     *Y(3  ))+(RC(79)     *Y(3  ))                        
      Y( 64) = (YP( 64)+DTS*P)/(1.0+DTS*L)
C
C          RA16O2           Y( 65)
      P = EM( 65)
     &+(RC(78)     *Y(64 )*Y(3  ))                                      
      L = 0.0
     &+(RC(308)    )       +(RC(309)    )                               
     &+(RC(201)    *Y(5  ))+(RC(202)    *Y(5  ))+(RC(254)    *Y(9  ))   
     &+(RC(118)    *Y(8  ))+(RC(119)    *Y(8  ))+(RC(182)    *Y(8  ))   
      Y( 65) = (YP( 65)+DTS*P)/(1.0+DTS*L)
C
C          AROH17           Y( 66)
      P = EM( 66)
     &+(RC(79)     *Y(64 )*Y(3  ))                                      
      L = 0.0
     &+(RC(418)    *Y(3  ))+(RC(419)    *Y(5  ))                        
      Y( 66) = (YP( 66)+DTS*P)/(1.0+DTS*L)
C
C          OXYL             Y( 67)
      P = EM( 67)
      L = 0.0
     &+(RC(80)     *Y(3  ))+(RC(81)     *Y(3  ))                        
      Y( 67) = (YP( 67)+DTS*P)/(1.0+DTS*L)
C
C          RA19AO2          Y( 68)
      P = EM( 68)
     &+(RC(80)     *Y(67 )*Y(3  ))                                      
      L = 0.0
     &+(RC(255)    *Y(9  ))+(RC(310)    )                               
     &+(RC(120)    *Y(8  ))+(RC(183)    *Y(8  ))+(RC(203)    *Y(5  ))   
      Y( 68) = (YP( 68)+DTS*P)/(1.0+DTS*L)
C
C          RA19CO2          Y( 69)
      P = EM( 69)
     &+(RC(81)     *Y(67 )*Y(3  ))                                      
      L = 0.0
     &+(RC(256)    *Y(9  ))+(RC(311)    )                               
     &+(RC(121)    *Y(8  ))+(RC(184)    *Y(8  ))+(RC(204)    *Y(5  ))   
      Y( 69) = (YP( 69)+DTS*P)/(1.0+DTS*L)
C
C          CH3CO3           Y( 70)
      P = EM( 70)
     &+(DJ(71)     *Y(168))                                             
     &+(DJ(40)     *Y(120)*2.00)                                        
     &+(DJ(31)     *Y(115))       +(DJ(32)     *Y(115))                 
     &+(DJ(27)     *Y(189))       +(DJ(29)     *Y(109))                 
     &+(DJ(25)     *Y(98 ))       +(DJ(26)     *Y(100)*2.00)            
     &+(DJ(19)     *Y(188))       +(DJ(23)     *Y(46 ))                 
     &+(DJ(17)     *Y(88 ))       +(DJ(18)     *Y(111))                 
     &+(DJ(14)     *Y(101))       +(DJ(15)     *Y(186))                 
     &+(RC(468)    *Y(198))       +(DJ(13)     *Y(73 ))                 
     &+(RC(374)    *Y(6  )*Y(109))+(RC(431)    *Y(3  )*Y(159))          
     &+(RC(362)    *Y(6  )*Y(46 ))+(RC(367)    *Y(3  )*Y(98 ))          
     &+(RC(331)    *Y(110))       +(RC(333)    *Y(112))                 
     &+(RC(325)    *Y(74 ))       +(RC(326)    *Y(75 ))                 
     &+(RC(222)    *Y(110)*Y(5  ))+(RC(224)    *Y(112)*Y(5  ))          
     &+(RC(216)    *Y(74 )*Y(5  ))+(RC(217)    *Y(75 )*Y(5  ))          
     &+(RC(139)    *Y(110)*Y(8  ))+(RC(141)    *Y(112)*Y(8  ))          
     &+(RC(133)    *Y(74 )*Y(8  ))+(RC(134)    *Y(75 )*Y(8  ))          
     &+(RC(83)     *Y(3  )*Y(42 ))+(RC(86)     *Y(5  )*Y(42 ))          
      L = 0.0
     &+(RC(322)    )       +(RC(467)    *Y(4  ))                        
     &+(RC(130)    *Y(8  ))+(RC(213)    *Y(5  ))+(RC(264)    *Y(9  ))   
      Y( 70) = (YP( 70)+DTS*P)/(1.0+DTS*L)
C
C          C2H5CHO          Y( 71)
      P = EM( 71)
     &+(DJ(78)     *Y(158)*2.00)                                        
     &+(DJ(55)     *Y(146))       +(DJ(77)     *Y(157))                 
     &+(DJ(21)     *Y(105))       +(DJ(43)     *Y(125))                 
     &+(RC(394)    *Y(3  )*Y(125))+(RC(426)    *Y(3  )*Y(146))          
     &+(RC(318)    *Y(95 ))       +(RC(319)    *Y(103)*2.00)            
     &+(RC(297)    *Y(27 ))       +(RC(298)    *Y(27 ))                 
     &+(RC(210)    *Y(103)*Y(5  )*2.00)                                 
     &+(RC(192)    *Y(27 )*Y(5  ))+(RC(209)    *Y(95 )*Y(5  ))          
     &+(RC(126)    *Y(95 )*Y(8  ))+(RC(127)    *Y(103)*Y(8  )*2.00)     
     &+(RC(93)     *Y(78 )*Y(3  ))+(RC(109)    *Y(27 )*Y(8  ))          
      L = 0.0
     &+(RC(84)     *Y(3  ))+(RC(87)     *Y(5  ))+(DJ(12)     )          
      Y( 71) = (YP( 71)+DTS*P)/(1.0+DTS*L)
C
C          C2H5CO3          Y( 72)
      P = EM( 72)
     &+(RC(470)    *Y(199))                                             
     &+(RC(327)    *Y(107))       +(RC(432)    *Y(3  )*Y(160))          
     &+(RC(135)    *Y(107)*Y(8  ))+(RC(218)    *Y(107)*Y(5  ))          
     &+(RC(84)     *Y(3  )*Y(71 ))+(RC(87)     *Y(5  )*Y(71 ))          
      L = 0.0
     &+(RC(323)    )       +(RC(469)    *Y(4  ))                        
     &+(RC(131)    *Y(8  ))+(RC(214)    *Y(5  ))+(RC(265)    *Y(9  ))   
      Y( 72) = (YP( 72)+DTS*P)/(1.0+DTS*L)
C
C          CH3COCH3         Y( 73)
      P = EM( 73)
     &+(DJ(90)     *Y(179))       +(DJ(95)     *Y(184))                 
     &+(DJ(44)     *Y(126))       +(DJ(56)     *Y(147))                 
     &+(RC(464)    *Y(3  )*Y(184))+(RC(484)    *Y(3  )*Y(203))          
     &+(RC(412)    *Y(3  )*Y(143))+(RC(427)    *Y(3  )*Y(147))          
     &+(RC(395)    *Y(3  )*Y(126))+(RC(409)    *Y(3  )*Y(140))          
     &+(RC(346)    *Y(118))       +(RC(351)    *Y(122))                 
     &+(RC(300)    *Y(26 ))       +(RC(301)    *Y(26 ))                 
     &+(RC(237)    *Y(118)*Y(5  ))+(RC(242)    *Y(122)*Y(5  ))          
     &+(RC(163)    *Y(122)*Y(8  ))+(RC(193)    *Y(26 )*Y(5  ))          
     &+(RC(159)    *Y(54 )*Y(8  ))+(RC(162)    *Y(56 )*Y(8  ))          
     &+(RC(150)    *Y(48 )*Y(8  ))+(RC(155)    *Y(118)*Y(8  ))          
     &+(RC(95)     *Y(3  )*Y(79 ))+(RC(110)    *Y(26 )*Y(8  ))          
      L = 0.0
     &+(RC(88)     *Y(3  ))+(DJ(13)     )                               
      Y( 73) = (YP( 73)+DTS*P)/(1.0+DTS*L)
C
C          RN8O2            Y( 74)
      P = EM( 74)
     &+(DJ(92)     *Y(181))                                             
     &+(DJ(28)     *Y(190)*2.00)                                        
     &+(DJ(21)     *Y(105))       +(DJ(27)     *Y(189))                 
     &+(DJ(16)     *Y(187))       +(DJ(20)     *Y(104))                 
     &+(RC(239)    *Y(121)*Y(5  ))+(RC(348)    *Y(121))                 
     &+(RC(88)     *Y(3  )*Y(73 ))+(RC(157)    *Y(121)*Y(8  ))          
      L = 0.0
     &+(RC(325)    )                                                    
     &+(RC(133)    *Y(8  ))+(RC(216)    *Y(5  ))+(RC(267)    *Y(9  ))   
      Y( 74) = (YP( 74)+DTS*P)/(1.0+DTS*L)
C
C          RN11O2           Y( 75)
      P = EM( 75)
     &+(RC(89)     *Y(101)*Y(3  ))+(RC(355)    *Y(3  )*Y(88 ))          
      L = 0.0
     &+(RC(326)    )                                                    
     &+(RC(134)    *Y(8  ))+(RC(217)    *Y(5  ))+(RC(268)    *Y(9  ))   
      Y( 75) = (YP( 75)+DTS*P)/(1.0+DTS*L)
C
C          CH3OH            Y( 76)
      P = EM( 76)
     &+(RC(293)    *Y(22 ))                                             
      L = 0.0
     &+(RC(90)     *Y(3  ))                                             
      Y( 76) = (YP( 76)+DTS*P)/(1.0+DTS*L)
C
C          C2H5OH           Y( 77)
      P = EM( 77)
     &+(RC(296)    *Y(24 ))                                             
      L = 0.0
     &+(RC(91)     *Y(3  ))+(RC(92)     *Y(3  ))                        
      Y( 77) = (YP( 77)+DTS*P)/(1.0+DTS*L)
C
C          NPROPOL          Y( 78)
      P = EM( 78)
     &+(RC(299)    *Y(27 ))                                             
      L = 0.0
     &+(RC(93)     *Y(3  ))+(RC(94)     *Y(3  ))                        
      Y( 78) = (YP( 78)+DTS*P)/(1.0+DTS*L)
C
C          IPROPOL          Y( 79)
      P = EM( 79)
     &+(RC(302)    *Y(26 ))                                             
      L = 0.0
     &+(RC(95)     *Y(3  ))+(RC(96)     *Y(3  ))                        
      Y( 79) = (YP( 79)+DTS*P)/(1.0+DTS*L)
C
C          CH3CL            Y( 80)
      P = EM( 80)
      L = 0.0
     &+(RC(99)     *Y(3  ))                                             
      Y( 80) = (YP( 80)+DTS*P)/(1.0+DTS*L)
C
C          CH2CL2           Y( 81)
      P = EM( 81)
      L = 0.0
     &+(RC(100)    *Y(3  ))                                             
      Y( 81) = (YP( 81)+DTS*P)/(1.0+DTS*L)
C
C          CHCL3            Y( 82)
      P = EM( 82)
      L = 0.0
     &+(RC(101)    *Y(3  ))                                             
      Y( 82) = (YP( 82)+DTS*P)/(1.0+DTS*L)
C
C          CH3CCL3          Y( 83)
      P = EM( 83)
      L = 0.0
     &+(RC(102)    *Y(3  ))                                             
      Y( 83) = (YP( 83)+DTS*P)/(1.0+DTS*L)
C
C          TCE              Y( 84)
      P = EM( 84)
      L = 0.0
     &+(RC(103)    *Y(3  ))                                             
      Y( 84) = (YP( 84)+DTS*P)/(1.0+DTS*L)
C
C          TRICLETH         Y( 85)
      P = EM( 85)
      L = 0.0
     &+(RC(104)    *Y(3  ))                                             
      Y( 85) = (YP( 85)+DTS*P)/(1.0+DTS*L)
C
C          CDICLETH         Y( 86)
      P = EM( 86)
      L = 0.0
     &+(RC(105)    *Y(3  ))                                             
      Y( 86) = (YP( 86)+DTS*P)/(1.0+DTS*L)
C
C          TDICLETH         Y( 87)
      P = EM( 87)
      L = 0.0
     &+(RC(106)    *Y(3  ))                                             
      Y( 87) = (YP( 87)+DTS*P)/(1.0+DTS*L)
C
C          CARB11A          Y( 88)
      P = EM( 88)
     &+(DJ(58)     *Y(148))                                             
     &+(RC(428)    *Y(3  )*Y(148))+(DJ(46)     *Y(127))                 
     &+(RC(304)    *Y(29 ))       +(RC(396)    *Y(3  )*Y(127))          
     &+(RC(112)    *Y(29 )*Y(8  ))+(RC(195)    *Y(29 )*Y(5  ))          
      L = 0.0
     &+(RC(355)    *Y(3  ))+(DJ(17)     )                               
      Y( 88) = (YP( 88)+DTS*P)/(1.0+DTS*L)
C
C          RN16O2           Y( 89)
      P = EM( 89)
     &+(RC(359)    *Y(3  )*Y(105))+(DJ(67)     *Y(165))                 
      L = 0.0
     &+(RC(249)    *Y(9  ))+(RC(312)    )                               
     &+(RC(113)    *Y(8  ))+(RC(171)    *Y(8  ))+(RC(196)    *Y(5  ))   
      Y( 89) = (YP( 89)+DTS*P)/(1.0+DTS*L)
C
C          RN15AO2          Y( 90)
      P = EM( 90)
     &+(DJ(59)     *Y(149))                                             
     &+(RC(312)    *Y(89 ))       +(RC(385)    *Y(3  )*Y(193))          
     &+(RC(113)    *Y(89 )*Y(8  ))+(RC(196)    *Y(89 )*Y(5  ))          
      L = 0.0
     &+(RC(262)    *Y(9  ))+(RC(320)    )                               
     &+(RC(128)    *Y(8  ))+(RC(178)    *Y(8  ))+(RC(211)    *Y(5  ))   
      Y( 90) = (YP( 90)+DTS*P)/(1.0+DTS*L)
C
C          RN19O2           Y( 91)
      P = EM( 91)
     &+(RC(150)    *Y(48 )*Y(8  ))+(RC(159)    *Y(54 )*Y(8  ))          
      L = 0.0
     &+(RC(250)    *Y(9  ))+(RC(313)    )                               
     &+(RC(114)    *Y(8  ))+(RC(172)    *Y(8  ))+(RC(197)    *Y(5  ))   
      Y( 91) = (YP( 91)+DTS*P)/(1.0+DTS*L)
C
C          RN18AO2          Y( 92)
      P = EM( 92)
     &+(RC(313)    *Y(91 ))       +(DJ(60)     *Y(150))                 
     &+(RC(114)    *Y(91 )*Y(8  ))+(RC(197)    *Y(91 )*Y(5  ))          
      L = 0.0
     &+(RC(263)    *Y(9  ))+(RC(321)    )                               
     &+(RC(129)    *Y(8  ))+(RC(179)    *Y(8  ))+(RC(212)    *Y(5  ))   
      Y( 92) = (YP( 92)+DTS*P)/(1.0+DTS*L)
C
C          RN13AO2          Y( 93)
      P = EM( 93)
     &+(RC(162)    *Y(56 )*Y(8  ))                                      
      L = 0.0
     &+(RC(305)    )                                                    
     &+(RC(115)    *Y(8  ))+(RC(198)    *Y(5  ))+(RC(251)    *Y(9  ))   
      Y( 93) = (YP( 93)+DTS*P)/(1.0+DTS*L)
C
C          RN16AO2          Y( 94)
      P = EM( 94)
     &+(RC(328)    *Y(108))                                             
     &+(RC(136)    *Y(108)*Y(8  ))+(RC(219)    *Y(108)*Y(5  ))          
      L = 0.0
     &+(RC(306)    )                                                    
     &+(RC(116)    *Y(8  ))+(RC(199)    *Y(5  ))+(RC(252)    *Y(9  ))   
      Y( 94) = (YP( 94)+DTS*P)/(1.0+DTS*L)
C
C          RN15O2           Y( 95)
      P = EM( 95)
     &+(DJ(47)     *Y(128))                                             
     &+(RC(306)    *Y(94 ))       +(RC(370)    *Y(3  )*Y(190))          
     &+(RC(116)    *Y(94 )*Y(8  ))+(RC(199)    *Y(94 )*Y(5  ))          
      L = 0.0
     &+(RC(260)    *Y(9  ))+(RC(318)    )                               
     &+(RC(126)    *Y(8  ))+(RC(176)    *Y(8  ))+(RC(209)    *Y(5  ))   
      Y( 95) = (YP( 95)+DTS*P)/(1.0+DTS*L)
C
C          UDCARB8          Y( 96)
      P = EM( 96)
     &+(DJ(49)     *Y(136))       +(DJ(82)     *Y(151))                 
     &+(RC(405)    *Y(3  )*Y(136))+(RC(451)    *Y(3  )*Y(151))          
     &+(RC(307)    *Y(62 ))       +(RC(309)    *Y(65 ))                 
     &+(RC(202)    *Y(65 )*Y(5  ))+(RC(204)    *Y(69 )*Y(5  ))          
     &+(RC(121)    *Y(69 )*Y(8  ))+(RC(200)    *Y(62 )*Y(5  ))          
     &+(RC(117)    *Y(62 )*Y(8  ))+(RC(119)    *Y(65 )*Y(8  ))          
      L = 0.0
     &+(DJ(34)     )                                                    
     &+(RC(378)    *Y(3  ))+(RC(379)    *Y(3  ))+(DJ(33)     )          
      Y( 96) = (YP( 96)+DTS*P)/(1.0+DTS*L)
C
C          UDCARB11         Y( 97)
      P = EM( 97)
     &+(DJ(84)     *Y(153))                                             
     &+(DJ(51)     *Y(138))       +(DJ(83)     *Y(152))                 
     &+(RC(453)    *Y(3  )*Y(153))+(DJ(50)     *Y(137))                 
     &+(RC(407)    *Y(3  )*Y(138))+(RC(452)    *Y(3  )*Y(152))          
     &+(RC(308)    *Y(65 ))       +(RC(406)    *Y(3  )*Y(137))          
     &+(RC(118)    *Y(65 )*Y(8  ))+(RC(201)    *Y(65 )*Y(5  ))          
      L = 0.0
     &+(DJ(36)     )                                                    
     &+(RC(380)    *Y(3  ))+(RC(381)    *Y(3  ))+(DJ(35)     )          
      Y( 97) = (YP( 97)+DTS*P)/(1.0+DTS*L)
C
C          CARB6            Y( 98)
      P = EM( 98)
     &+(DJ(70)     *Y(167))       +(DJ(84)     *Y(153))                 
     &+(RC(453)    *Y(3  )*Y(153))+(DJ(51)     *Y(138))                 
     &+(RC(407)    *Y(3  )*Y(138))+(RC(434)    *Y(3  )*Y(162))          
     &+(RC(375)    *Y(6  )*Y(109))+(RC(377)    *Y(3  )*Y(115))          
     &+(RC(356)    *Y(3  )*Y(111))+(RC(363)    *Y(6  )*Y(46 ))          
     &+(RC(309)    *Y(65 ))       +(RC(334)    *Y(112))                 
     &+(RC(202)    *Y(65 )*Y(5  ))+(RC(225)    *Y(112)*Y(5  ))          
     &+(RC(119)    *Y(65 )*Y(8  ))+(RC(142)    *Y(112)*Y(8  ))          
      L = 0.0
     &+(RC(367)    *Y(3  ))+(DJ(25)     )                               
      Y( 98) = (YP( 98)+DTS*P)/(1.0+DTS*L)
C
C          UDCARB14         Y( 99)
      P = EM( 99)
     &+(RC(310)    *Y(68 ))       +(RC(311)    *Y(69 ))                 
     &+(RC(120)    *Y(68 )*Y(8  ))+(RC(203)    *Y(68 )*Y(5  ))          
      L = 0.0
     &+(DJ(38)     )                                                    
     &+(RC(382)    *Y(3  ))+(RC(383)    *Y(3  ))+(DJ(37)     )          
      Y( 99) = (YP( 99)+DTS*P)/(1.0+DTS*L)
C
C          CARB9            Y(100)
      P = EM(100)
     &+(RC(357)    *Y(3  )*Y(188))+(RC(435)    *Y(3  )*Y(163))          
     &+(RC(121)    *Y(69 )*Y(8  ))+(RC(204)    *Y(69 )*Y(5  ))          
      L = 0.0
     &+(RC(368)    *Y(3  ))+(DJ(26)     )                               
      Y(100) = (YP(100)+DTS*P)/(1.0+DTS*L)
C
C          MEK              Y(101)
      P = EM(101)
      L = 0.0
     &+(RC(89)     *Y(3  ))+(DJ(14)     )                               
      Y(101) = (YP(101)+DTS*P)/(1.0+DTS*L)
C
C          HOCH2CHO         Y(102)
      P = EM(102)
     &+(DJ(71)     *Y(168))                                             
     &+(DJ(29)     *Y(109))       +(DJ(70)     *Y(167))                 
     &+(RC(399)    *Y(3  )*Y(130))+(RC(443)    *Y(3  )*Y(154))          
     &+(RC(374)    *Y(6  )*Y(109))+(RC(375)    *Y(6  )*Y(109))          
     &+(RC(332)    *Y(110))       +(RC(333)    *Y(112))                 
     &+(RC(315)    *Y(31 ))       +(RC(331)    *Y(110))                 
     &+(RC(222)    *Y(110)*Y(5  ))+(RC(224)    *Y(112)*Y(5  ))          
     &+(RC(141)    *Y(112)*Y(8  ))+(RC(206)    *Y(31 )*Y(5  ))          
     &+(RC(123)    *Y(31 )*Y(8  ))+(RC(139)    *Y(110)*Y(8  ))          
      L = 0.0
     &+(RC(364)    *Y(3  ))+(RC(365)    *Y(5  ))+(DJ(22)     )          
      Y(102) = (YP(102)+DTS*P)/(1.0+DTS*L)
C
C          RN18O2           Y(103)
      P = EM(103)
     &+(DJ(48)     *Y(129))                                             
      L = 0.0
     &+(RC(261)    *Y(9  ))+(RC(319)    )                               
     &+(RC(127)    *Y(8  ))+(RC(177)    *Y(8  ))+(RC(210)    *Y(5  ))   
      Y(103) = (YP(103)+DTS*P)/(1.0+DTS*L)
C
C          CARB13           Y(104)
      P = EM(104)
     &+(RC(446)    *Y(3  )*Y(157))                                      
     &+(RC(416)    *Y(3  )*Y(195))+(RC(417)    *Y(5  )*Y(195))          
     &+(RC(320)    *Y(90 ))       +(RC(402)    *Y(3  )*Y(133))          
     &+(RC(128)    *Y(90 )*Y(8  ))+(RC(211)    *Y(90 )*Y(5  ))          
      L = 0.0
     &+(RC(358)    *Y(3  ))+(DJ(20)     )                               
      Y(104) = (YP(104)+DTS*P)/(1.0+DTS*L)
C
C          CARB16           Y(105)
      P = EM(105)
     &+(RC(447)    *Y(3  )*Y(158))+(RC(484)    *Y(3  )*Y(203))          
     &+(RC(421)    *Y(3  )*Y(197))+(RC(422)    *Y(5  )*Y(197))          
     &+(RC(321)    *Y(92 ))       +(RC(403)    *Y(3  )*Y(134))          
     &+(RC(129)    *Y(92 )*Y(8  ))+(RC(212)    *Y(92 )*Y(5  ))          
      L = 0.0
     &+(RC(359)    *Y(3  ))+(DJ(21)     )                               
      Y(105) = (YP(105)+DTS*P)/(1.0+DTS*L)
C
C          HOCH2CO3         Y(106)
      P = EM(106)
     &+(RC(433)    *Y(3  )*Y(161))+(RC(472)    *Y(200))                 
     &+(RC(364)    *Y(3  )*Y(102))+(RC(365)    *Y(5  )*Y(102))          
      L = 0.0
     &+(RC(324)    )       +(RC(471)    *Y(4  ))                        
     &+(RC(132)    *Y(8  ))+(RC(215)    *Y(5  ))+(RC(266)    *Y(9  ))   
      Y(106) = (YP(106)+DTS*P)/(1.0+DTS*L)
C
C          RN14O2           Y(107)
      P = EM(107)
     &+(RC(353)    *Y(3  )*Y(186))                                      
      L = 0.0
     &+(RC(327)    )                                                    
     &+(RC(135)    *Y(8  ))+(RC(218)    *Y(5  ))+(RC(269)    *Y(9  ))   
      Y(107) = (YP(107)+DTS*P)/(1.0+DTS*L)
C
C          RN17O2           Y(108)
      P = EM(108)
     &+(RC(354)    *Y(3  )*Y(187))                                      
      L = 0.0
     &+(RC(328)    )                                                    
     &+(RC(136)    *Y(8  ))+(RC(219)    *Y(5  ))+(RC(270)    *Y(9  ))   
      Y(108) = (YP(108)+DTS*P)/(1.0+DTS*L)
C
C          UCARB12          Y(109)
      P = EM(109)
     &+(RC(438)    *Y(3  )*Y(166))+(DJ(68)     *Y(166))                 
     &+(RC(329)    *Y(44 ))       +(RC(404)    *Y(3  )*Y(135))          
     &+(RC(137)    *Y(44 )*Y(8  ))+(RC(220)    *Y(44 )*Y(5  ))          
      L = 0.0
     &+(RC(375)    *Y(6  ))+(DJ(29)     )                               
     &+(RC(372)    *Y(3  ))+(RC(373)    *Y(5  ))+(RC(374)    *Y(6  ))   
      Y(109) = (YP(109)+DTS*P)/(1.0+DTS*L)
C
C          RU12O2           Y(110)
      P = EM(110)
     &+(RC(439)    *Y(3  )*Y(167))+(RC(477)    *Y(201))                 
     &+(RC(372)    *Y(3  )*Y(109))+(RC(373)    *Y(5  )*Y(109))          
      L = 0.0
     &+(RC(332)    )       +(RC(476)    *Y(4  ))                        
     &+(RC(223)    *Y(5  ))+(RC(272)    *Y(9  ))+(RC(331)    )          
     &+(RC(139)    *Y(8  ))+(RC(140)    *Y(8  ))+(RC(222)    *Y(5  ))   
      Y(110) = (YP(110)+DTS*P)/(1.0+DTS*L)
C
C          CARB7            Y(111)
      P = EM(111)
     &+(RC(480)    *Y(3  )*Y(202))                                      
     &+(RC(400)    *Y(3  )*Y(131))+(RC(444)    *Y(3  )*Y(155))          
     &+(RC(332)    *Y(110))       +(RC(335)    *Y(112))                 
     &+(RC(223)    *Y(110)*Y(5  ))+(RC(226)    *Y(112)*Y(5  ))          
     &+(RC(140)    *Y(110)*Y(8  ))+(RC(143)    *Y(112)*Y(8  ))          
      L = 0.0
     &+(RC(356)    *Y(3  ))+(DJ(18)     )                               
      Y(111) = (YP(111)+DTS*P)/(1.0+DTS*L)
C
C          RU10O2           Y(112)
      P = EM(112)
     &+(RC(440)    *Y(3  )*Y(168))+(RC(479)    *Y(202))                 
     &+(RC(360)    *Y(3  )*Y(46 ))+(RC(361)    *Y(5  )*Y(46 ))          
      L = 0.0
     &+(RC(335)    )       +(RC(478)    *Y(4  ))                        
     &+(RC(273)    *Y(9  ))+(RC(333)    )       +(RC(334)    )          
     &+(RC(224)    *Y(5  ))+(RC(225)    *Y(5  ))+(RC(226)    *Y(5  ))   
     &+(RC(141)    *Y(8  ))+(RC(142)    *Y(8  ))+(RC(143)    *Y(8  ))   
      Y(112) = (YP(112)+DTS*P)/(1.0+DTS*L)
C
C          NUCARB12         Y(113)
      P = EM(113)
     &+(DJ(72)     *Y(172))                                             
     &+(RC(339)    *Y(45 ))       +(RC(441)    *Y(3  )*Y(172))          
     &+(RC(147)    *Y(45 )*Y(8  ))+(RC(230)    *Y(45 )*Y(5  ))          
      L = 0.0
     &+(RC(376)    *Y(3  ))+(DJ(30)     )                               
      Y(113) = (YP(113)+DTS*P)/(1.0+DTS*L)
C
C          NRU12O2          Y(114)
      P = EM(114)
     &+(RC(376)    *Y(3  )*Y(113))                                      
      L = 0.0
     &+(RC(340)    )                                                    
     &+(RC(148)    *Y(8  ))+(RC(231)    *Y(5  ))+(RC(278)    *Y(9  ))   
      Y(114) = (YP(114)+DTS*P)/(1.0+DTS*L)
C
C          NOA              Y(115)
      P = EM(115)
     &+(DJ(30)     *Y(113))       +(DJ(73)     *Y(173))                 
     &+(RC(340)    *Y(114))       +(RC(442)    *Y(3  )*Y(173))          
     &+(RC(148)    *Y(114)*Y(8  ))+(RC(231)    *Y(114)*Y(5  ))          
      L = 0.0
     &+(RC(377)    *Y(3  ))+(DJ(31)     )       +(DJ(32)     )          
      Y(115) = (YP(115)+DTS*P)/(1.0+DTS*L)
C
C          RTN25O2          Y(116)
      P = EM(116)
     &+(RC(457)    *Y(3  )*Y(177))+(DJ(87)     *Y(176))                 
     &+(RC(343)    *Y(50 ))       +(RC(389)    *Y(3  )*Y(52 ))          
     &+(RC(152)    *Y(50 )*Y(8  ))+(RC(234)    *Y(50 )*Y(5  ))          
      L = 0.0
     &+(RC(282)    *Y(9  ))+(RC(344)    )                               
     &+(RC(153)    *Y(8  ))+(RC(186)    *Y(8  ))+(RC(235)    *Y(5  ))   
      Y(116) = (YP(116)+DTS*P)/(1.0+DTS*L)
C
C          RTN24O2          Y(117)
      P = EM(117)
     &+(DJ(88)     *Y(177))                                             
     &+(RC(344)    *Y(116))       +(RC(458)    *Y(3  )*Y(178))          
     &+(RC(153)    *Y(116)*Y(8  ))+(RC(235)    *Y(116)*Y(5  ))          
      L = 0.0
     &+(RC(345)    )                                                    
     &+(RC(154)    *Y(8  ))+(RC(236)    *Y(5  ))+(RC(283)    *Y(9  ))   
      Y(117) = (YP(117)+DTS*P)/(1.0+DTS*L)
C
C          RTN23O2          Y(118)
      P = EM(118)
     &+(DJ(89)     *Y(178))                                             
     &+(RC(345)    *Y(117))       +(RC(459)    *Y(3  )*Y(179))          
     &+(RC(154)    *Y(117)*Y(8  ))+(RC(236)    *Y(117)*Y(5  ))          
      L = 0.0
     &+(RC(346)    )                                                    
     &+(RC(155)    *Y(8  ))+(RC(237)    *Y(5  ))+(RC(284)    *Y(9  ))   
      Y(118) = (YP(118)+DTS*P)/(1.0+DTS*L)
C
C          RTN14O2          Y(119)
      P = EM(119)
     &+(DJ(90)     *Y(179))                                             
     &+(RC(346)    *Y(118))       +(RC(460)    *Y(3  )*Y(180))          
     &+(RC(155)    *Y(118)*Y(8  ))+(RC(237)    *Y(118)*Y(5  ))          
      L = 0.0
     &+(RC(347)    )                                                    
     &+(RC(156)    *Y(8  ))+(RC(238)    *Y(5  ))+(RC(285)    *Y(9  ))   
      Y(119) = (YP(119)+DTS*P)/(1.0+DTS*L)
C
C          TNCARB10         Y(120)
      P = EM(120)
     &+(RC(347)    *Y(119))       +(DJ(91)     *Y(180))                 
     &+(RC(156)    *Y(119)*Y(8  ))+(RC(238)    *Y(119)*Y(5  ))          
      L = 0.0
     &+(RC(386)    *Y(3  ))+(RC(388)    *Y(5  ))+(DJ(40)     )          
      Y(120) = (YP(120)+DTS*P)/(1.0+DTS*L)
C
C          RTN10O2          Y(121)
      P = EM(121)
     &+(RC(461)    *Y(3  )*Y(181))                                      
     &+(RC(386)    *Y(3  )*Y(120))+(RC(388)    *Y(5  )*Y(120))          
      L = 0.0
     &+(RC(348)    )                                                    
     &+(RC(157)    *Y(8  ))+(RC(239)    *Y(5  ))+(RC(286)    *Y(9  ))   
      Y(121) = (YP(121)+DTS*P)/(1.0+DTS*L)
C
C          RTX22O2          Y(122)
      P = EM(122)
     &+(RC(391)    *Y(3  )*Y(58 ))                                      
      L = 0.0
     &+(RC(289)    *Y(9  ))+(RC(351)    )                               
     &+(RC(163)    *Y(8  ))+(RC(189)    *Y(8  ))+(RC(242)    *Y(5  ))   
      Y(122) = (YP(122)+DTS*P)/(1.0+DTS*L)
C
C          CH3NO3           Y(123)
      P = EM(123)
     &+(RC(166)    *Y(22 )*Y(8  ))                                      
      L = 0.0
     &+(RC(392)    *Y(3  ))+(DJ(41)     )                               
      Y(123) = (YP(123)+DTS*P)/(1.0+DTS*L)
C
C          C2H5NO3          Y(124)
      P = EM(124)
     &+(RC(167)    *Y(24 )*Y(8  ))                                      
      L = 0.0
     &+(RC(393)    *Y(3  ))+(DJ(42)     )                               
      Y(124) = (YP(124)+DTS*P)/(1.0+DTS*L)
C
C          RN10NO3          Y(125)
      P = EM(125)
     &+(RC(168)    *Y(27 )*Y(8  ))                                      
      L = 0.0
     &+(RC(394)    *Y(3  ))+(DJ(43)     )                               
      Y(125) = (YP(125)+DTS*P)/(1.0+DTS*L)
C
C          IC3H7NO3         Y(126)
      P = EM(126)
     &+(RC(169)    *Y(26 )*Y(8  ))                                      
      L = 0.0
     &+(RC(395)    *Y(3  ))+(DJ(44)     )                               
      Y(126) = (YP(126)+DTS*P)/(1.0+DTS*L)
C
C          RN13NO3          Y(127)
      P = EM(127)
     &+(RC(170)    *Y(29 )*Y(8  ))                                      
      L = 0.0
     &+(RC(396)    *Y(3  ))+(DJ(45)     )       +(DJ(46)     )          
      Y(127) = (YP(127)+DTS*P)/(1.0+DTS*L)
C
C          RN16NO3          Y(128)
      P = EM(128)
     &+(RC(171)    *Y(89 )*Y(8  ))                                      
      L = 0.0
     &+(RC(397)    *Y(3  ))+(DJ(47)     )                               
      Y(128) = (YP(128)+DTS*P)/(1.0+DTS*L)
C
C          RN19NO3          Y(129)
      P = EM(129)
     &+(RC(172)    *Y(91 )*Y(8  ))                                      
      L = 0.0
     &+(RC(398)    *Y(3  ))+(DJ(48)     )                               
      Y(129) = (YP(129)+DTS*P)/(1.0+DTS*L)
C
C          HOC2H4NO3        Y(130)
      P = EM(130)
     &+(RC(173)    *Y(31 )*Y(8  ))                                      
      L = 0.0
     &+(RC(399)    *Y(3  ))                                             
      Y(130) = (YP(130)+DTS*P)/(1.0+DTS*L)
C
C          RN9NO3           Y(131)
      P = EM(131)
     &+(RC(174)    *Y(33 )*Y(8  ))                                      
      L = 0.0
     &+(RC(400)    *Y(3  ))                                             
      Y(131) = (YP(131)+DTS*P)/(1.0+DTS*L)
C
C          RN12NO3          Y(132)
      P = EM(132)
     &+(RC(175)    *Y(35 )*Y(8  ))                                      
      L = 0.0
     &+(RC(401)    *Y(3  ))                                             
      Y(132) = (YP(132)+DTS*P)/(1.0+DTS*L)
C
C          RN15NO3          Y(133)
      P = EM(133)
     &+(RC(176)    *Y(95 )*Y(8  ))+(RC(178)    *Y(90 )*Y(8  ))          
      L = 0.0
     &+(RC(402)    *Y(3  ))                                             
      Y(133) = (YP(133)+DTS*P)/(1.0+DTS*L)
C
C          RN18NO3          Y(134)
      P = EM(134)
     &+(RC(177)    *Y(103)*Y(8  ))+(RC(179)    *Y(92 )*Y(8  ))          
      L = 0.0
     &+(RC(403)    *Y(3  ))                                             
      Y(134) = (YP(134)+DTS*P)/(1.0+DTS*L)
C
C          RU14NO3          Y(135)
      P = EM(135)
     &+(RC(180)    *Y(44 )*Y(8  ))                                      
      L = 0.0
     &+(RC(404)    *Y(3  ))                                             
      Y(135) = (YP(135)+DTS*P)/(1.0+DTS*L)
C
C          RA13NO3          Y(136)
      P = EM(136)
     &+(RC(181)    *Y(62 )*Y(8  ))                                      
      L = 0.0
     &+(RC(405)    *Y(3  ))+(DJ(49)     )                               
      Y(136) = (YP(136)+DTS*P)/(1.0+DTS*L)
C
C          RA16NO3          Y(137)
      P = EM(137)
     &+(RC(182)    *Y(65 )*Y(8  ))                                      
      L = 0.0
     &+(RC(406)    *Y(3  ))+(DJ(50)     )                               
      Y(137) = (YP(137)+DTS*P)/(1.0+DTS*L)
C
C          RA19NO3          Y(138)
      P = EM(138)
     &+(RC(183)    *Y(68 )*Y(8  ))+(RC(184)    *Y(69 )*Y(8  ))          
      L = 0.0
     &+(RC(407)    *Y(3  ))+(DJ(51)     )                               
      Y(138) = (YP(138)+DTS*P)/(1.0+DTS*L)
C
C          RTN28NO3         Y(139)
      P = EM(139)
     &+(RC(185)    *Y(48 )*Y(8  ))+(RC(486)    *Y(204))                 
      L = 0.0
     &+(RC(408)    *Y(3  ))+(RC(485)    )                               
      Y(139) = (YP(139)+DTS*P)/(1.0+DTS*L)
C
C          RTN25NO3         Y(140)
      P = EM(140)
     &+(RC(186)    *Y(116)*Y(8  ))                                      
      L = 0.0
     &+(RC(409)    *Y(3  ))                                             
      Y(140) = (YP(140)+DTS*P)/(1.0+DTS*L)
C
C          RTX28NO3         Y(141)
      P = EM(141)
     &+(RC(187)    *Y(54 )*Y(8  ))+(RC(488)    *Y(205))                 
      L = 0.0
     &+(RC(410)    *Y(3  ))+(RC(487)    )                               
      Y(141) = (YP(141)+DTS*P)/(1.0+DTS*L)
C
C          RTX24NO3         Y(142)
      P = EM(142)
     &+(RC(188)    *Y(56 )*Y(8  ))                                      
      L = 0.0
     &+(RC(411)    *Y(3  ))+(DJ(52)     )                               
      Y(142) = (YP(142)+DTS*P)/(1.0+DTS*L)
C
C          RTX22NO3         Y(143)
      P = EM(143)
     &+(RC(189)    *Y(122)*Y(8  ))                                      
      L = 0.0
     &+(RC(412)    *Y(3  ))                                             
      Y(143) = (YP(143)+DTS*P)/(1.0+DTS*L)
C
C          CH3OOH           Y(144)
      P = EM(144)
     &+(RC(244)    *Y(22 )*Y(9  ))                                      
      L = 0.0
     &+(RC(423)    *Y(3  ))+(RC(424)    *Y(3  ))+(DJ(53)     )          
      Y(144) = (YP(144)+DTS*P)/(1.0+DTS*L)
C
C          C2H5OOH          Y(145)
      P = EM(145)
     &+(RC(245)    *Y(24 )*Y(9  ))                                      
      L = 0.0
     &+(RC(425)    *Y(3  ))+(DJ(54)     )                               
      Y(145) = (YP(145)+DTS*P)/(1.0+DTS*L)
C
C          RN10OOH          Y(146)
      P = EM(146)
     &+(RC(246)    *Y(27 )*Y(9  ))                                      
      L = 0.0
     &+(RC(426)    *Y(3  ))+(DJ(55)     )                               
      Y(146) = (YP(146)+DTS*P)/(1.0+DTS*L)
C
C          IC3H7OOH         Y(147)
      P = EM(147)
     &+(RC(247)    *Y(26 )*Y(9  ))                                      
      L = 0.0
     &+(RC(427)    *Y(3  ))+(DJ(56)     )                               
      Y(147) = (YP(147)+DTS*P)/(1.0+DTS*L)
C
C          RN13OOH          Y(148)
      P = EM(148)
     &+(RC(248)    *Y(29 )*Y(9  ))+(RC(251)    *Y(93 )*Y(9  ))          
      L = 0.0
     &+(RC(428)    *Y(3  ))+(DJ(57)     )       +(DJ(58)     )          
      Y(148) = (YP(148)+DTS*P)/(1.0+DTS*L)
C
C          RN16OOH          Y(149)
      P = EM(149)
     &+(RC(249)    *Y(89 )*Y(9  ))+(RC(252)    *Y(94 )*Y(9  ))          
      L = 0.0
     &+(RC(429)    *Y(3  ))+(DJ(59)     )                               
      Y(149) = (YP(149)+DTS*P)/(1.0+DTS*L)
C
C          RN19OOH          Y(150)
      P = EM(150)
     &+(RC(250)    *Y(91 )*Y(9  ))                                      
      L = 0.0
     &+(RC(430)    *Y(3  ))+(DJ(60)     )                               
      Y(150) = (YP(150)+DTS*P)/(1.0+DTS*L)
C
C          RA13OOH          Y(151)
      P = EM(151)
     &+(RC(253)    *Y(62 )*Y(9  ))                                      
      L = 0.0
     &+(RC(451)    *Y(3  ))+(DJ(82)     )                               
      Y(151) = (YP(151)+DTS*P)/(1.0+DTS*L)
C
C          RA16OOH          Y(152)
      P = EM(152)
     &+(RC(254)    *Y(65 )*Y(9  ))                                      
      L = 0.0
     &+(RC(452)    *Y(3  ))+(DJ(83)     )                               
      Y(152) = (YP(152)+DTS*P)/(1.0+DTS*L)
C
C          RA19OOH          Y(153)
      P = EM(153)
     &+(RC(255)    *Y(68 )*Y(9  ))+(RC(256)    *Y(69 )*Y(9  ))          
      L = 0.0
     &+(RC(453)    *Y(3  ))+(DJ(84)     )                               
      Y(153) = (YP(153)+DTS*P)/(1.0+DTS*L)
C
C          HOC2H4OOH        Y(154)
      P = EM(154)
     &+(RC(257)    *Y(31 )*Y(9  ))                                      
      L = 0.0
     &+(RC(443)    *Y(3  ))+(DJ(74)     )                               
      Y(154) = (YP(154)+DTS*P)/(1.0+DTS*L)
C
C          RN9OOH           Y(155)
      P = EM(155)
     &+(RC(258)    *Y(33 )*Y(9  ))                                      
      L = 0.0
     &+(RC(444)    *Y(3  ))+(DJ(75)     )                               
      Y(155) = (YP(155)+DTS*P)/(1.0+DTS*L)
C
C          RN12OOH          Y(156)
      P = EM(156)
     &+(RC(259)    *Y(35 )*Y(9  ))                                      
      L = 0.0
     &+(RC(445)    *Y(3  ))+(DJ(76)     )                               
      Y(156) = (YP(156)+DTS*P)/(1.0+DTS*L)
C
C          RN15OOH          Y(157)
      P = EM(157)
     &+(RC(260)    *Y(95 )*Y(9  ))+(RC(262)    *Y(90 )*Y(9  ))          
      L = 0.0
     &+(RC(446)    *Y(3  ))+(DJ(77)     )                               
      Y(157) = (YP(157)+DTS*P)/(1.0+DTS*L)
C
C          RN18OOH          Y(158)
      P = EM(158)
     &+(RC(261)    *Y(103)*Y(9  ))+(RC(263)    *Y(92 )*Y(9  ))          
      L = 0.0
     &+(RC(447)    *Y(3  ))+(DJ(78)     )                               
      Y(158) = (YP(158)+DTS*P)/(1.0+DTS*L)
C
C          CH3CO3H          Y(159)
      P = EM(159)
     &+(RC(264)    *Y(70 )*Y(9  ))                                      
      L = 0.0
     &+(RC(431)    *Y(3  ))+(DJ(61)     )                               
      Y(159) = (YP(159)+DTS*P)/(1.0+DTS*L)
C
C          C2H5CO3H         Y(160)
      P = EM(160)
     &+(RC(265)    *Y(72 )*Y(9  ))                                      
      L = 0.0
     &+(RC(432)    *Y(3  ))+(DJ(62)     )                               
      Y(160) = (YP(160)+DTS*P)/(1.0+DTS*L)
C
C          HOCH2CO3H        Y(161)
      P = EM(161)
     &+(RC(266)    *Y(106)*Y(9  ))                                      
      L = 0.0
     &+(RC(433)    *Y(3  ))+(DJ(63)     )                               
      Y(161) = (YP(161)+DTS*P)/(1.0+DTS*L)
C
C          RN8OOH           Y(162)
      P = EM(162)
     &+(RC(267)    *Y(74 )*Y(9  ))                                      
      L = 0.0
     &+(RC(434)    *Y(3  ))+(DJ(64)     )                               
      Y(162) = (YP(162)+DTS*P)/(1.0+DTS*L)
C
C          RN11OOH          Y(163)
      P = EM(163)
     &+(RC(268)    *Y(75 )*Y(9  ))                                      
      L = 0.0
     &+(RC(435)    *Y(3  ))+(DJ(65)     )                               
      Y(163) = (YP(163)+DTS*P)/(1.0+DTS*L)
C
C          RN14OOH          Y(164)
      P = EM(164)
     &+(RC(269)    *Y(107)*Y(9  ))                                      
      L = 0.0
     &+(RC(436)    *Y(3  ))+(DJ(66)     )                               
      Y(164) = (YP(164)+DTS*P)/(1.0+DTS*L)
C
C          RN17OOH          Y(165)
      P = EM(165)
     &+(RC(270)    *Y(108)*Y(9  ))                                      
      L = 0.0
     &+(RC(437)    *Y(3  ))+(DJ(67)     )                               
      Y(165) = (YP(165)+DTS*P)/(1.0+DTS*L)
C
C          RU14OOH          Y(166)
      P = EM(166)
     &+(RC(271)    *Y(44 )*Y(9  ))                                      
      L = 0.0
     &+(RC(438)    *Y(3  ))+(DJ(68)     )       +(DJ(69)     )          
      Y(166) = (YP(166)+DTS*P)/(1.0+DTS*L)
C
C          RU12OOH          Y(167)
      P = EM(167)
     &+(RC(272)    *Y(110)*Y(9  ))                                      
      L = 0.0
     &+(RC(439)    *Y(3  ))+(DJ(70)     )                               
      Y(167) = (YP(167)+DTS*P)/(1.0+DTS*L)
C
C          RU10OOH          Y(168)
      P = EM(168)
     &+(RC(273)    *Y(112)*Y(9  ))                                      
      L = 0.0
     &+(RC(440)    *Y(3  ))+(DJ(71)     )                               
      Y(168) = (YP(168)+DTS*P)/(1.0+DTS*L)
C
C          NRN6OOH          Y(169)
      P = EM(169)
     &+(RC(274)    *Y(36 )*Y(9  ))                                      
      L = 0.0
     &+(RC(448)    *Y(3  ))+(DJ(79)     )                               
      Y(169) = (YP(169)+DTS*P)/(1.0+DTS*L)
C
C          NRN9OOH          Y(170)
      P = EM(170)
     &+(RC(275)    *Y(37 )*Y(9  ))                                      
      L = 0.0
     &+(RC(449)    *Y(3  ))+(DJ(80)     )                               
      Y(170) = (YP(170)+DTS*P)/(1.0+DTS*L)
C
C          NRN12OOH         Y(171)
      P = EM(171)
     &+(RC(276)    *Y(38 )*Y(9  ))                                      
      L = 0.0
     &+(RC(450)    *Y(3  ))+(DJ(81)     )                               
      Y(171) = (YP(171)+DTS*P)/(1.0+DTS*L)
C
C          NRU14OOH         Y(172)
      P = EM(172)
     &+(RC(277)    *Y(45 )*Y(9  ))                                      
      L = 0.0
     &+(RC(441)    *Y(3  ))+(DJ(72)     )                               
      Y(172) = (YP(172)+DTS*P)/(1.0+DTS*L)
C
C          NRU12OOH         Y(173)
      P = EM(173)
     &+(RC(278)    *Y(114)*Y(9  ))                                      
      L = 0.0
     &+(RC(442)    *Y(3  ))+(DJ(73)     )                               
      Y(173) = (YP(173)+DTS*P)/(1.0+DTS*L)
C
C          RTN28OOH         Y(174)
      P = EM(174)
     &+(RC(279)    *Y(48 )*Y(9  ))+(RC(496)    *Y(209))                 
      L = 0.0
     &+(RC(454)    *Y(3  ))+(RC(495)    )       +(DJ(85)     )          
      Y(174) = (YP(174)+DTS*P)/(1.0+DTS*L)
C
C          NRTN28OOH        Y(175)
      P = EM(175)
     &+(RC(280)    *Y(49 )*Y(9  ))                                      
      L = 0.0
     &+(RC(456)    *Y(3  ))+(DJ(86)     )                               
      Y(175) = (YP(175)+DTS*P)/(1.0+DTS*L)
C
C          RTN26OOH         Y(176)
      P = EM(176)
     &+(RC(281)    *Y(50 )*Y(9  ))+(RC(498)    *Y(210))                 
      L = 0.0
     &+(RC(455)    *Y(3  ))+(RC(497)    )       +(DJ(87)     )          
      Y(176) = (YP(176)+DTS*P)/(1.0+DTS*L)
C
C          RTN25OOH         Y(177)
      P = EM(177)
     &+(RC(282)    *Y(116)*Y(9  ))+(RC(502)    *Y(212))                 
      L = 0.0
     &+(RC(457)    *Y(3  ))+(RC(501)    )       +(DJ(88)     )          
      Y(177) = (YP(177)+DTS*P)/(1.0+DTS*L)
C
C          RTN24OOH         Y(178)
      P = EM(178)
     &+(RC(283)    *Y(117)*Y(9  ))+(RC(492)    *Y(207))                 
      L = 0.0
     &+(RC(458)    *Y(3  ))+(RC(491)    )       +(DJ(89)     )          
      Y(178) = (YP(178)+DTS*P)/(1.0+DTS*L)
C
C          RTN23OOH         Y(179)
      P = EM(179)
     &+(RC(284)    *Y(118)*Y(9  ))+(RC(504)    *Y(213))                 
      L = 0.0
     &+(RC(459)    *Y(3  ))+(RC(503)    )       +(DJ(90)     )          
      Y(179) = (YP(179)+DTS*P)/(1.0+DTS*L)
C
C          RTN14OOH         Y(180)
      P = EM(180)
     &+(RC(285)    *Y(119)*Y(9  ))                                      
      L = 0.0
     &+(RC(460)    *Y(3  ))+(DJ(91)     )                               
      Y(180) = (YP(180)+DTS*P)/(1.0+DTS*L)
C
C          RTN10OOH         Y(181)
      P = EM(181)
     &+(RC(286)    *Y(121)*Y(9  ))                                      
      L = 0.0
     &+(RC(461)    *Y(3  ))+(DJ(92)     )                               
      Y(181) = (YP(181)+DTS*P)/(1.0+DTS*L)
C
C          RTX28OOH         Y(182)
      P = EM(182)
     &+(RC(287)    *Y(54 )*Y(9  ))+(RC(494)    *Y(208))                 
      L = 0.0
     &+(RC(462)    *Y(3  ))+(RC(493)    )       +(DJ(93)     )          
      Y(182) = (YP(182)+DTS*P)/(1.0+DTS*L)
C
C          RTX24OOH         Y(183)
      P = EM(183)
     &+(RC(288)    *Y(56 )*Y(9  ))                                      
      L = 0.0
     &+(RC(463)    *Y(3  ))+(DJ(94)     )                               
      Y(183) = (YP(183)+DTS*P)/(1.0+DTS*L)
C
C          RTX22OOH         Y(184)
      P = EM(184)
     &+(RC(289)    *Y(122)*Y(9  ))                                      
      L = 0.0
     &+(RC(464)    *Y(3  ))+(DJ(95)     )                               
      Y(184) = (YP(184)+DTS*P)/(1.0+DTS*L)
C
C          NRTX28OOH        Y(185)
      P = EM(185)
     &+(RC(290)    *Y(55 )*Y(9  ))                                      
      L = 0.0
     &+(RC(465)    *Y(3  ))+(DJ(96)     )                               
      Y(185) = (YP(185)+DTS*P)/(1.0+DTS*L)
C
C          CARB14           Y(186)
      P = EM(186)
     &+(RC(397)    *Y(3  )*Y(128))+(RC(429)    *Y(3  )*Y(149))          
      L = 0.0
     &+(RC(353)    *Y(3  ))+(DJ(15)     )                               
      Y(186) = (YP(186)+DTS*P)/(1.0+DTS*L)
C
C          CARB17           Y(187)
      P = EM(187)
     &+(RC(398)    *Y(3  )*Y(129))+(RC(430)    *Y(3  )*Y(150))          
      L = 0.0
     &+(RC(354)    *Y(3  ))+(DJ(16)     )                               
      Y(187) = (YP(187)+DTS*P)/(1.0+DTS*L)
C
C          CARB10           Y(188)
      P = EM(188)
     &+(RC(401)    *Y(3  )*Y(132))+(RC(445)    *Y(3  )*Y(156))          
      L = 0.0
     &+(RC(357)    *Y(3  ))+(DJ(19)     )                               
      Y(188) = (YP(188)+DTS*P)/(1.0+DTS*L)
C
C          CARB12           Y(189)
      P = EM(189)
     &+(RC(436)    *Y(3  )*Y(164))                                      
      L = 0.0
     &+(RC(369)    *Y(3  ))+(DJ(27)     )                               
      Y(189) = (YP(189)+DTS*P)/(1.0+DTS*L)
C
C          CARB15           Y(190)
      P = EM(190)
     &+(RC(437)    *Y(3  )*Y(165))                                      
      L = 0.0
     &+(RC(370)    *Y(3  ))+(DJ(28)     )                               
      Y(190) = (YP(190)+DTS*P)/(1.0+DTS*L)
C
C          CCARB12          Y(191)
      P = EM(191)
     &+(RC(412)    *Y(3  )*Y(143))+(RC(464)    *Y(3  )*Y(184))          
      L = 0.0
     &+(RC(371)    *Y(3  ))                                             
      Y(191) = (YP(191)+DTS*P)/(1.0+DTS*L)
C
C          ANHY             Y(192)
      P = EM(192)
     &+(DJ(38)     *Y(99 ))                                             
     &+(DJ(34)     *Y(96 ))       +(DJ(36)     *Y(97 ))                 
     &+(RC(383)    *Y(3  )*Y(99 ))+(RC(510)    *Y(216))                 
     &+(RC(379)    *Y(3  )*Y(96 ))+(RC(381)    *Y(3  )*Y(97 ))          
      L = 0.0
     &+(RC(466)    *Y(3  ))+(RC(509)    )                               
      Y(192) = (YP(192)+DTS*P)/(1.0+DTS*L)
C
C          TNCARB15         Y(193)
      P = EM(193)
     &+(RC(409)    *Y(3  )*Y(140))                                      
      L = 0.0
     &+(RC(385)    *Y(3  ))                                             
      Y(193) = (YP(193)+DTS*P)/(1.0+DTS*L)
C
C          RAROH14          Y(194)
      P = EM(194)
     &+(RC(413)    *Y(3  )*Y(63 ))+(RC(414)    *Y(5  )*Y(63 ))          
      L = 0.0
     &+(RC(415)    *Y(4  ))                                             
      Y(194) = (YP(194)+DTS*P)/(1.0+DTS*L)
C
C          ARNOH14          Y(195)
      P = EM(195)
     &+(RC(415)    *Y(194)*Y(4  ))+(RC(506)    *Y(214))                 
      L = 0.0
     &+(RC(416)    *Y(3  ))+(RC(417)    *Y(5  ))+(RC(505)    )          
      Y(195) = (YP(195)+DTS*P)/(1.0+DTS*L)
C
C          RAROH17          Y(196)
      P = EM(196)
     &+(RC(418)    *Y(3  )*Y(66 ))+(RC(419)    *Y(5  )*Y(66 ))          
      L = 0.0
     &+(RC(420)    *Y(4  ))                                             
      Y(196) = (YP(196)+DTS*P)/(1.0+DTS*L)
C
C          ARNOH17          Y(197)
      P = EM(197)
     &+(RC(420)    *Y(196)*Y(4  ))+(RC(508)    *Y(215))                 
      L = 0.0
     &+(RC(421)    *Y(3  ))+(RC(422)    *Y(5  ))+(RC(507)    )          
      Y(197) = (YP(197)+DTS*P)/(1.0+DTS*L)
C
C          PAN              Y(198)
      P = EM(198)
     &+(RC(467)    *Y(70 )*Y(4  ))                                      
      L = 0.0
     &+(RC(468)    )       +(RC(473)    *Y(3  ))                        
      Y(198) = (YP(198)+DTS*P)/(1.0+DTS*L)
C
C          PPN              Y(199)
      P = EM(199)
     &+(RC(469)    *Y(72 )*Y(4  ))                                      
      L = 0.0
     &+(RC(470)    )       +(RC(474)    *Y(3  ))                        
      Y(199) = (YP(199)+DTS*P)/(1.0+DTS*L)
C
C          PHAN             Y(200)
      P = EM(200)
     &+(RC(471)    *Y(106)*Y(4  ))                                      
      L = 0.0
     &+(RC(472)    )       +(RC(475)    *Y(3  ))                        
      Y(200) = (YP(200)+DTS*P)/(1.0+DTS*L)
C
C          RU12PAN          Y(201)
      P = EM(201)
     &+(RC(476)    *Y(110)*Y(4  ))                                      
      L = 0.0
     &+(RC(477)    )       +(RC(481)    *Y(3  ))                        
      Y(201) = (YP(201)+DTS*P)/(1.0+DTS*L)
C
C          MPAN             Y(202)
      P = EM(202)
     &+(RC(478)    *Y(112)*Y(4  ))                                      
      L = 0.0
     &+(RC(479)    )       +(RC(480)    *Y(3  ))                        
      Y(202) = (YP(202)+DTS*P)/(1.0+DTS*L)
C
C          RTN26PAN         Y(203)
      P = EM(203)
     &+(RC(482)    *Y(50 )*Y(4  ))+(RC(500)    *Y(211))                 
      L = 0.0
     &+(RC(483)    )       +(RC(484)    *Y(3  ))+(RC(499)    )          
      Y(203) = (YP(203)+DTS*P)/(1.0+DTS*L)
C
C          P2604            Y(204)
      P = EM(204)
     &+(RC(485)    *Y(139))                                             
      L = 0.0
     &+(RC(486)    )                                                    
      Y(204) = (YP(204)+DTS*P)/(1.0+DTS*L)
C
C          P4608            Y(205)
      P = EM(205)
     &+(RC(487)    *Y(141))                                             
      L = 0.0
     &+(RC(488)    )                                                    
      Y(205) = (YP(205)+DTS*P)/(1.0+DTS*L)
C
C          P2631            Y(206)
      P = EM(206)
     &+(RC(489)    *Y(52 ))                                             
      L = 0.0
     &+(RC(490)    )                                                    
      Y(206) = (YP(206)+DTS*P)/(1.0+DTS*L)
C
C          P2635            Y(207)
      P = EM(207)
     &+(RC(491)    *Y(178))                                             
      L = 0.0
     &+(RC(492)    )                                                    
      Y(207) = (YP(207)+DTS*P)/(1.0+DTS*L)
C
C          P4610            Y(208)
      P = EM(208)
     &+(RC(493)    *Y(182))                                             
      L = 0.0
     &+(RC(494)    )                                                    
      Y(208) = (YP(208)+DTS*P)/(1.0+DTS*L)
C
C          P2605            Y(209)
      P = EM(209)
     &+(RC(495)    *Y(174))                                             
      L = 0.0
     &+(RC(496)    )                                                    
      Y(209) = (YP(209)+DTS*P)/(1.0+DTS*L)
C
C          P2630            Y(210)
      P = EM(210)
     &+(RC(497)    *Y(176))                                             
      L = 0.0
     &+(RC(498)    )                                                    
      Y(210) = (YP(210)+DTS*P)/(1.0+DTS*L)
C
C          P2629            Y(211)
      P = EM(211)
     &+(RC(499)    *Y(203))                                             
      L = 0.0
     &+(RC(500)    )                                                    
      Y(211) = (YP(211)+DTS*P)/(1.0+DTS*L)
C
C          P2632            Y(212)
      P = EM(212)
     &+(RC(501)    *Y(177))                                             
      L = 0.0
     &+(RC(502)    )                                                    
      Y(212) = (YP(212)+DTS*P)/(1.0+DTS*L)
C
C          P2637            Y(213)
      P = EM(213)
     &+(RC(503)    *Y(179))                                             
      L = 0.0
     &+(RC(504)    )                                                    
      Y(213) = (YP(213)+DTS*P)/(1.0+DTS*L)
C
C          P3612            Y(214)
      P = EM(214)
     &+(RC(505)    *Y(195))                                             
      L = 0.0
     &+(RC(506)    )                                                    
      Y(214) = (YP(214)+DTS*P)/(1.0+DTS*L)
C
C          P3613            Y(215)
      P = EM(215)
     &+(RC(507)    *Y(197))                                             
      L = 0.0
     &+(RC(508)    )                                                    
      Y(215) = (YP(215)+DTS*P)/(1.0+DTS*L)
C
C          P3442            Y(216)
      P = EM(216)
     &+(RC(509)    *Y(192))                                             
      L = 0.0
     &+(RC(510)    )                                                    
      Y(216) = (YP(216)+DTS*P)/(1.0+DTS*L)
C
C          CH3O2NO2         Y(217)
      P = EM(217)
     &+(RC(164)    *Y(22 )*Y(4  ))                                      
      L = 0.0
     &+(RC(165)    )                                                    
      Y(217) = (YP(217)+DTS*P)/(1.0+DTS*L)
C
C          EMPOA            Y(218)
      P = EM(218)
C     dry deposition of EMPOA
C
	L = 0.0
      Y(218) = (YP(218)+DTS*P)/(1.0+DTS*L)
C
C          P2007            Y(219)
      P = EM(219)
     &+(RC(511)    *Y(167))                                             
      L = 0.0
     &+(RC(512)    )                                                    
      Y(219) = (YP(219)+DTS*P)/(1.0+DTS*L)
C
C

C
C
C  DEFINE TOTAL CONCENTRATION OF PEROXY RADICALS
C
       RO2 =  Y(22) + Y(24)+ Y(27) + Y(26) + Y(29)+
     & Y(93) + Y(94) + Y(89) + Y(91) + Y(31) + Y(33) +
     & Y(35) + Y(95) + Y(103) + Y(90) + Y(92) + Y(70) +
     & Y(72) + Y(75) + Y(107) + Y(108) + Y(106) + Y(44) +
     & Y(110) + Y(112) + Y(36) + Y(37) + Y(38) + Y(48) +
     & Y(45) + Y(114) + Y(62) + Y(65) + Y(68) + Y(69) +
     & Y(74) + Y(50) + Y(49) + Y(116) + Y(117) + Y(118) +
     & Y(119) + Y(121) + Y(54) + Y(56) + Y(122) + Y(55) 

 1000 CONTINUE
C
C      -----------------
C      THERMAL REACTIONS
C      -----------------
C
C      at end of iteration, calculate flux terms.
C
C      O + O2 + M = O3 + M
      FL(1)=FL(1)+RC(1)*Y(2)*DTS
C      O + N2 + M = O3 + M
      FL(2)=FL(2)+RC(2)*Y(2)*DTS
C      O + O3 = 
      FL(3)=FL(3)+RC(3)*Y(2)*Y(6)*DTS
C       O + NO = NO2
      FL(4)=FL(4)+RC(4)*Y(2)*Y(8)*DTS
C      O + NO2 = NO
      FL(5)=FL(5)+RC(5)*Y(2)*Y(4)*DTS
C      O + NO2 = NO3
      FL(6)=FL(6)+RC(6)*Y(2)*Y(4)*DTS
C      O1D + O2 + M = O + M
      FL(7)=FL(7)+RC(7)*Y(1)*DTS
C      O1D + N2 + M = O + M
      FL(8)=FL(8)+RC(8)*Y(1)*DTS
C      NO + O3 = NO2
      FL(9)=FL(9) + RC(9)*Y(8)*Y(6)*DTS
C      NO2 + O3 = NO3
      FL(10)=FL(10)+RC(10)*Y(4)*Y(6)*DTS
C      NO + NO = NO2 + NO2
      FL(11)=FL(11)+RC(11)*Y(4)*Y(4)*DTS
C      NO + NO3 = NO2 + NO2
      FL(12)=FL(12)+RC(12)*Y(4)*Y(5)*DTS
C      NO2 + NO3 = NO + NO2
      FL(13)=FL(13)+RC(13)*Y(4)*Y(5)*DTS
C      NO2 + NO3 = N2O5
      FL(14)=FL(14)+RC(14)*Y(4)*Y(5)*DTS
C      N2O5 = NO2 + NO3
      FL(15)=FL(15)+RC(15)*Y(7)*DTS
C      O1D = OH + OH
      FL(16)=FL(16)+RC(16)*Y(1)*DTS
C      OH + O3 = HO2
      FL(17)=FL(17)+RC(17)*Y(3)*Y(6)*DTS
C      OH + H2 = HO2
      FL(18)=FL(18)+RC(18)*Y(3)*Y(10)*DTS
C       OH + CO = HO2
      FL(19)=FL(19)+RC(19)*Y(3)*Y(11)*DTS
C      OH + H2O2 = HO2
      FL(20)=FL(20)+RC(20)*Y(3)*Y(12)*DTS
C      HO2 + O3 = OH
      FL(21)=FL(21)+RC(21)*Y(9)*Y(6)*DTS
C      OH + HO2 = 
      FL(22)=FL(22)+RC(22)*Y(3)*Y(9)*DTS
C      HO2 + HO2 = H2O2
      FL(23)=FL(23)+RC(23)*Y(9)*Y(9)*DTS
C      HO2 + HO2 = H2O2
      FL(24)=FL(24)+RC(24)*Y(9)*Y(9)*DTS
C      OH + NO = HONO
      FL(25)=FL(25)+RC(25)*Y(3)*Y(8)*DTS
C      NO2 = HONO
      FL(26)=FL(26)+RC(26)*Y(4)*DTS
C      OH + NO2 = HNO3
      FL(27)=FL(27)+RC(27)*Y(3)*Y(4)*DTS
C      OH + NO3 = HO2 + NO2
      FL(28)=FL(28)+RC(28)*Y(3)*Y(5)*DTS
C      HO2 + NO = OH + NO2
      FL(29)=FL(29)+RC(29)*Y(9)*Y(8)*DTS
C      HO2 + NO2 = HO2NO2
      FL(30)=FL(30)+RC(30)*Y(9)*Y(4)*DTS
C      HO2NO2 = HO2 + NO2
      FL(31)=FL(31)+RC(31)*Y(15)*DTS
C      OH + HO2NO2 = NO2 
      FL(32)=FL(32)+RC(32)*Y(3)*Y(15)*DTS
C      HO2 + NO3 = OH + NO2
      FL(33)=FL(33)+RC(33)*Y(9)*Y(5)*DTS
C      OH + HONO = NO2
      FL(34)=FL(34)+RC(34)*Y(3)*Y(13)*DTS
C      OH + HNO3 = NO3
      FL(35)=FL(35)+RC(35)*Y(3)*Y(14)*DTS
C      O + SO2 = SO3
      FL(36)=FL(36)+RC(36)*Y(2)*Y(16)*DTS
C      OH + SO2 = HSO3 
      FL(37)=FL(37)+RC(37)*Y(3)*Y(16)*DTS
C      HSO3 = HO2 + SO3
      FL(38)=FL(38)+RC(38)*Y(18)*DTS
C      HNO3 = NA
      FL(39)=FL(39)+RC(39)*Y(14)*DTS
C      N2O5 = NA + NA
      FL(40)=FL(40)+RC(40)*Y(7)*DTS
C      SO3 = SA
      FL(41)=FL(41)+RC(41)*Y(17)*DTS
C      OH + CH4 = CH3O2
      FL(42)=FL(42)+RC(42)*Y(3)*Y(21)*DTS
C      OH + C2H6 = C2H5O2
      FL(43)=FL(43)+RC(43)*Y(3)*Y(26)*DTS
C      OH + C3H8 = IC3H7O2
      FL(44)=FL(44)+RC(44)*Y(3)*Y(25)*DTS
C      OH + C3H8 = RN10O2 
      FL(45)=FL(45)+RC(45)*Y(3)*Y(25)*DTS
C      OH + NC4H10 = RN13O2
      FL(46)=FL(46)+RC(46)*Y(3)*Y(28)*DTS
C      OH + C2H4 = HOCH2CH2O2
      FL(47)=FL(47)+RC(47)*Y(3)*Y(30)*DTS
C      OH + C3H6 = RN9O2
      FL(48)=FL(48)+RC(48)*Y(3)*Y(32)*DTS
C      OH + TBUT2ENE = RN12O2
      FL(49)=FL(49)+RC(49)*Y(3)*Y(34)*DTS
C      NO3 + C2H4 = NRN6O2
      FL(50)=FL(50)+RC(50)*Y(3)*Y(30)*DTS
C      NO3 + C3H6 = NRN9O2
      FL(51)=FL(51)+RC(51)*Y(5)*Y(32)*DTS
C      NO3 + TBUT2ENE = NRN12O2
      FL(52)=FL(52)+RC(52)*Y(5)*Y(34)*DTS
C      O3 + C2H4 = HCHO + CO + HO2 + OH
      FL(53)=FL(53)+RC(53)*Y(6)*Y(30)*DTS
C      O3 + C2H4 = HCHO + HCOOH
      FL(54)=FL(54)+RC(54)*Y(6)*Y(30)*DTS
C      O3 + C3H6 = HCHO + CO + CH3O2 + OH
      FL(55)=FL(55)+RC(55)*Y(6)*Y(32)*DTS
C      O3 + C3H6 = HCHO + CH3CO2H
      FL(56)=FL(56)+RC(56)*Y(6)*Y(32)*DTS
C      O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH
      FL(57)=FL(57)+RC(57)*Y(6)*Y(34)*DTS
C      O3 + TBUT2ENE = CH3CHO + CH3CO2H
      FL(58)=FL(58)+RC(58)*Y(6)*Y(34)*DTS
C      OH + C5H8 = RU14O2
      FL(59)=FL(59)+RC(59)*Y(3)*Y(43)*DTS
C      NO3 + C5H8 = NRU14O2
      FL(60)=FL(60)+RC(60)*Y(5)*Y(43)*DTS
C      O3 + C5H8 = UCARB10 + CO + HO2 + OH
      FL(61)=FL(61)+RC(61)*Y(6)*Y(43)*DTS
C      O3 + C5H8 = UCARB10 + HCOOH
      FL(62)=FL(62)+RC(62)*Y(6)*Y(43)*DTS
C      APINENE + OH = RTN28O2
      FL(63)=FL(63)+RC(63)*Y(47)*Y(3)*DTS
C      APINENE + NO3 = NRTN28O2
      FL(64)=FL(64)+RC(64)*Y(47)*Y(5)*DTS
C      APINENE + O3 = OH + RTN26O2 
      FL(65)=FL(65)+RC(65)*Y(47)*Y(6)*DTS
C      APINENE + O3 = TNCARB26 + H2O2
      FL(66)=FL(66)+RC(66)*Y(47)*Y(6)*DTS
C      APINENE + O3 = RCOOH25 
      FL(67)=FL(67)+RC(67)*Y(47)*Y(6)*DTS
C      BPINENE + OH = RTX28O2
      FL(68)=FL(68)+RC(68)*Y(53)*Y(3)*DTS
C      BPINENE + NO3 = NRTX28O2
      FL(69)=FL(69)+RC(69)*Y(53)*Y(5)*DTS
C      BPINENE + O3 =  RTX24O2 + OH
      FL(70)=FL(70)+RC(70)*Y(53)*Y(6)*DTS
C      BPINENE + O3 =  HCHO + TXCARB24 + H2O2
      FL(71)=FL(71)+RC(71)*Y(53)*Y(6)*DTS
C      BPINENE + O3 =  HCHO + TXCARB22
      FL(72)=FL(72)+RC(72)*Y(53)*Y(6)*DTS
C      BPINENE + O3 =  TXCARB24 + CO 
      FL(73)=FL(73)+RC(73)*Y(53)*Y(6)*DTS
C      C2H2 + OH = HCOOH + CO + HO2
      FL(74)=FL(74)+RC(74)*Y(59)*Y(3)*DTS
C      C2H2 + OH = CARB3 + OH
      FL(75)=FL(75)+RC(75)*Y(59)*Y(3)*DTS
C      BENZENE + OH = RA13O2
      FL(76)=FL(76)+RC(76)*Y(61)*Y(3)*DTS
C      BENZENE + OH = AROH14 + HO2
      FL(77)=FL(77)+RC(77)*Y(61)*Y(3)*DTS
C      TOLUENE + OH = RA16O2
      FL(78)=FL(78)+RC(78)*Y(64)*Y(3)*DTS
C      TOLUENE + OH = AROH17 + HO2
      FL(79)=FL(79)+RC(79)*Y(64)*Y(3)*DTS
C      OXYL + OH = RA19AO2
      FL(80)=FL(80)+RC(80)*Y(67)*Y(3)*DTS
C      OXYL + OH = RA19CO2
      FL(81)=FL(81)+RC(81)*Y(67)*Y(3)*DTS
C      OH + HCHO = HO2 + CO
      FL(82)=FL(82)+RC(82)*Y(3)*Y(39)*DTS
C      OH + CH3CHO = CH3CO3
      FL(83)=FL(83)+RC(83)*Y(3)*Y(42)*DTS
C      OH + C2H5CHO = C2H5CO3
      FL(84)=FL(84)+RC(84)*Y(3)*Y(71)*DTS
C      NO3 + HCHO = HO2 + CO + HNO3
      FL(85)=FL(85)+RC(85)*Y(5)*Y(39)*DTS
C      NO3 + CH3CHO = CH3CO3 + HNO3
      FL(86)=FL(86)+RC(86)*Y(5)*Y(42)*DTS
C      NO3 + C2H5CHO = C2H5CO3 + HNO3
      FL(87)=FL(87)+RC(87)*Y(5)*Y(71)*DTS
C      OH + CH3COCH3 = RN8O2
      FL(88)=FL(88)+RC(88)*Y(3)*Y(73)*DTS
C      MEK + OH = RN11O2
      FL(89)=FL(89)+RC(89)*Y(101)*Y(3)*DTS
C      OH + CH3OH = HO2 + HCHO
      FL(90)=FL(90)+RC(90)*Y(3)*Y(76)*DTS
C      OH + C2H5OH = CH3CHO + HO2
      FL(91)=FL(91)+RC(91)*Y(3)*Y(76)*DTS
C      OH + C2H5OH = HOCH2CH2O2 
      FL(92)=FL(92)+RC(92)*Y(3)*Y(77)*DTS
C     NPROPOL + OH = C2H5CHO + HO2 
      FL(93)=FL(93)+RC(93)*Y(3)*Y(78)*DTS
C      NPROPOL + OH = RN9O2
      FL(94)=FL(94)+RC(94)*Y(3)*Y(78)*DTS
C      OH + IPROPOL = CH3COCH3 + HO2
      FL(95)=FL(95)+RC(95)*Y(3)*Y(79)*DTS
C      OH + IPROPOL = RN9O2
      FL(96)=FL(96)+RC(96)*Y(3)*Y(79)*DTS
C      HCOOH + OH = HO2
      FL(97)=FL(97)+RC(97)*Y(3)*Y(40)*DTS
C      CH3CO2H + OH = CH3O2
      FL(98)=FL(98)+RC(98)*Y(3)*Y(41)*DTS
C      OH + CH3CL = CH3O2 
      FL(99)=FL(99)+RC(99)*Y(3)*Y(80)*DTS
C      OH + CH2CL2 = CH3O2
      FL(100)=FL(100)+RC(100)*Y(3)*Y(81)*DTS
C      OH + CHCL3 = CH3O2
      FL(101)=FL(101)+RC(101)*Y(3)*Y(80)*DTS
C      OH + CH3CCL3 = C2H5O2
      FL(102)=FL(102)+RC(102)*Y(3)*Y(83)*DTS
C      OH + TCE = HOCH2CH2O2 
      FL(103)=FL(103)+RC(103)*Y(3)*Y(84)*DTS
C      OH + TRICLETH = HOCH2CH2O2
      FL(104)=FL(104)+RC(104)*Y(3)*Y(85)*DTS
C      OH + CDICLETH = HOCH2CH2O2
      FL(105)=FL(105)+RC(105)*Y(3)*Y(86)*DTS
C      OH + TDICLETH = HOCH2CH2O2
      FL(106)=FL(106)+RC(106)*Y(3)*Y(87)*DTS
C      CH3O2 + NO = HCHO + HO2 + NO2
      FL(107)=FL(107)+RC(107)*Y(8)*Y(22)*DTS
C      C2H5O2 + NO = CH3CHO + HO2 + NO2
      FL(108)=FL(108)+RC(108)*Y(8)*Y(24)*DTS
C      RN10O2 + NO = C2H5CHO + HO2 + NO2
      FL(109)=FL(109)+RC(109)*Y(8)*Y(27)*DTS
C      IC3H7O2 + NO = CH3COCH3 + HO2 + NO2
      FL(110)=FL(110)+RC(110)*Y(8)*Y(26)*DTS
C      RN13O2 + NO = CH3CHO + C2H5O2 + NO2 
      FL(111)=FL(111)+RC(111)*Y(8)*Y(29)*DTS
C      RN13O2 + NO = CARB11A + HO2 + NO2
      FL(112)=FL(112)+RC(112)*Y(8)*Y(29)*DTS
C      RN16O2 + NO = RN15AO2 + NO2 
      FL(113)=FL(113)+RC(113)*Y(8)*Y(89)*DTS
C      RN19O2 + NO = RN18AO2 + NO2
      FL(114)=FL(114)+RC(114)*Y(8)*Y(91)*DTS
C      RN13AO2 + NO = RN12O2 + NO2
      FL(115)=FL(115)+RC(115)*Y(8)*Y(93)*DTS
C      RN16AO2 + NO = RN15O2 + NO2 
      FL(116)=FL(116)+RC(116)*Y(8)*Y(94)*DTS
C      RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2
      FL(117)=FL(117)+RC(117)*Y(8)*Y(62)*DTS
C      RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2 
      FL(118)=FL(118)+RC(118)*Y(8)*Y(65)*DTS
C      RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2  
      FL(119)=FL(119)+RC(119)*Y(8)*Y(65)*DTS
C      RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2
      FL(120)=FL(120)+RC(120)*Y(8)*Y(68)*DTS
C      RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2
      FL(121)=FL(121)+RC(121)*Y(8)*Y(69)*DTS
C      HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2
      FL(122)=FL(122)+RC(122)*Y(8)*Y(31)*DTS
C      HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2
      FL(123)=FL(123)+RC(123)*Y(8)*Y(31)*DTS
C       RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2 
      FL(124)=FL(124)+RC(124)*Y(8)*Y(33)*DTS
C      RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2
      FL(125)=FL(125)+RC(125)*Y(8)*Y(35)*DTS
C      RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2
      FL(126)=FL(126)+RC(126)*Y(8)*Y(95)*DTS
C      RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2 
      FL(127)=FL(127)+RC(127)*Y(8)*Y(103)*DTS
C      RN15AO2 + NO = CARB13 + HO2 + NO2 
      FL(128)=FL(128)+RC(128)*Y(8)*Y(90)*DTS
C      RN18AO2 + NO = CARB16 + HO2 + NO2
      FL(129)=FL(129)+RC(129)*Y(8)*Y(92)*DTS
C      CH3CO3 + NO = CH3O2 + NO2 
      FL(130)=FL(130)+RC(130)*Y(8)*Y(70)*DTS
C
      RETURN
      END
C#######################################################################
      SUBROUTINE CHEMCO(RC,TEMP,M,O2,H2O,RO2,MOM,BR01)
C----------------------------------------------------------------------
C-
C-   Purpose and Methods : CALCULATES RATE COEFFICIENTS
C-
C-   Inputs  : TEMP,M,H2O
C-   Outputs : RC,DJ
C-   Controls:

C----------------------------------------------------------------------
      IMPLICIT NONE
C----------------------------------------------------------------------
      DOUBLE PRECISION RC(512),TEMP,M,RKLOW,RKHIGH,FC,BRN,FAC1,FAC2,FAC3
      DOUBLE PRECISION DJ(96),J(70)
      DOUBLE PRECISION RK0,RK2,RK3
      DOUBLE PRECISION H2O,O2,RO2
      DOUBLE PRECISION K0,KI,F
      DOUBLE PRECISION KRO2NO,KRO2HO2,KRO2NO3,KNO3AL,KDEC
      DOUBLE PRECISION KAPHO2,KFPAN,KBPAN,KAPNO
      DOUBLE PRECISION KC0,KCI,KRC,FCC
      DOUBLE PRECISION KD0,KDI,KRD,FCD,FD,K10,K1I,KR1,FC1,F1
      DOUBLE PRECISION K20,K2I,KR2,FC2,Fa2,K30,K3I,KR3,FC3,F3
      DOUBLE PRECISION K40,K4I,KR4,FC4,Fa4,K70,K7I,KR7,FC7,F7
      DOUBLE PRECISION K80,K8I,KR8,FC8,F8,K90,K9I,KR9,FC9,F9
      DOUBLE PRECISION K100,K10I,KR10,FC10,F10,K130,K13I,KR13,FC13,F13
      DOUBLE PRECISION K140,K14I,KR14,FC14,F14,K160,K16I,KR16,FC16,F16
      DOUBLE PRECISION K1,K2,K3,K4,KMT01,KMT02,KMT03,KMT04,KMT05
      DOUBLE PRECISION KMT06,KMT07,KMT08,KMT09,KMT10,KMT11
      DOUBLE PRECISION KMT12,KMT13,KMT14,KMT15,KMT16,KMT17 
      DOUBLE PRECISION KROPRIM,KROSEC,PANTOT,KDEC1,KTOT1,BR01 
      DOUBLE PRECISION FAC4,SOA,SOAM,YY,SC,OM,MOM,KIN,KOUT2604,KOUT4608
	DOUBLE PRECISION KOUT2631,KOUT2635,KOUT2641
      DOUBLE PRECISION KOUT4610,KOUT2605,KOUT4830,KOUT4829,KOUT3442
	DOUBLE PRECISION KOUT2630,KOUT2671,KOUT4834,KOUT5276,KOUT2617 
	DOUBLE PRECISION KOUT5236,KOUT4552,KOUT2703,KOUT2629
      DOUBLE PRECISION KOUT2669,KOUT3613,KOUT3612,KOUT2637,KOUT2632
      DOUBLE PRECISION R, BGOAM,KALKOXY,KALKPXY,K150,K15I,KR15,FC15,F15
	DOUBLE PRECISION K170, K17I, KR17,FC17,F17
	DOUBLE PRECISION N2
	
      INTEGER I
      N2       = 7.809E-01*M 
      O2       = 2.079E-01*M 
      R = 8.314
      DO 317 I=1,512
        RC(I)=0.0
  317 CONTINUE

C    SIMPLE RATE COEFFICIENTS                     
C                                                                     
      KRO2NO  = 2.70D-12*EXP(360/TEMP) 
      KAPNO   = 7.50D-12*EXP(290/TEMP) 
      KRO2NO3 = 2.30D-12 
      KRO2HO2 = 2.91D-13*EXP(1300/TEMP) 
      KAPHO2  = 5.20D-13*EXP(980/TEMP) 
      KNO3AL  = 1.44D-12*EXP(-1862/TEMP) 
      KDEC    = 1.0D+06
      KALKOXY = 3.70D-14*EXP(-460/TEMP)*O2 
      KALKPXY = 1.80D-14*EXP(-260/TEMP)*O2 
      BR01 = (0.156 + 9.77D+08*EXP(-6415/TEMP)) 
C
      KIN = 6.2E-03*MOM
      KOUT2604 = 4.34*EXP(-7776/(R*TEMP))
      KOUT4608 = 4.34*EXP(-9765/(R*TEMP))
      KOUT2631 = 4.34*EXP(-14500/(R*TEMP))
      KOUT2635 = 4.34*EXP(-12541/(R*TEMP))
      KOUT4610 = 4.34*EXP(-10513/(R*TEMP))
      KOUT2605 = 4.34*EXP(-8879/(R*TEMP))
      KOUT2630 = 4.34*EXP(-12639/(R*TEMP))
      KOUT2629 = 4.34*EXP(-4954/(R*TEMP))
      KOUT2632 = 4.34*EXP(-3801/(R*TEMP))
      KOUT2637 = 4.34*EXP(-16752/(R*TEMP))
      KOUT3612 = 4.34*EXP(-8362/(R*TEMP))
      KOUT3613 = 4.34*EXP(-11003/(R*TEMP))
      KOUT3442 = 4.34*EXP(-12763/(R*TEMP))
C
C    COMPLEX RATE COEFFICIENTS                    
C                                                                     
C    KFPAN                                                   
C                                                                     
      KC0     = 3.28D-28*M*(TEMP/300)**(-7.1) 
      KCI     = 1.12D-11*(TEMP/300)**(-1.105)    
      KRC     = KC0/KCI    
      FCC     = 0.30       
      FC      = 10**(LOG10(FCC)/(1+(LOG10(KRC))**2)) 
      KFPAN   = (KC0*KCI)*FC/(KC0+KCI) 
C                                                                   
C    KBPAN                                                   
      KD0     = 1.1D-05*M*EXP(-10100/TEMP) 
      KDI     = 1.90D+17*EXP(-14100/TEMP) 
      KRD     = KD0/KDI    
      FCD     = 0.30       
      FD      = 10**(LOG10(FCD)/(1+(LOG10(KRD))**2)) 
      KBPAN   = (KD0*KDI)*FD/(KD0+KDI) 
C                                                                     
C     KMT01                                                   
      K10     = 9.00D-32*M*(TEMP/300)**(-1.5)
      K1I     = 3.00D-11*(TEMP/300)**0.3    
      KR1     = K10/K1I    
      FC1     = 0.6 
      F1      = 10**(LOG10(FC1)/(1+(LOG10(KR1))**2)) 
      KMT01   = (K10*K1I)*F1/(K10+K1I) 
C                                                                     
C     KMT02                                                   
      K20 = 9.00D-32*((temp/300)**(-2.0))*M 
      K2I = 2.20D-11
      KR2     = K20/K2I    
      FC2 = 0.6 
      Fa2      = 10**(LOG10(FC2)/(1+(LOG10(KR2))**2)) 
      KMT02   = (K20*K2I)*Fa2/(K20+K2I) 
C                                                                     
C      KMT03  : NO2      + NO3     = N2O5                               
C    IUPAC 2001                                                       
      K30     = 2.70D-30*M*(TEMP/300)**(-3.4)
      K3I     = 2.00D-12*(TEMP/300)**0.2    
      KR3     = K30/K3I    
      FC3     = (EXP(-TEMP/250) + EXP(-1050/TEMP)) 
      F3      = 10**(LOG10(FC3)/(1+(LOG10(KR3))**2)) 
      KMT03   = (K30*K3I)*F3/(K30+K3I) 
C                                                                     
C     KMT04  : N2O5               = NO2     + NO3                     
C IUPAC 1997/2001                                                 
      K40     = (2.20D-03*M*(TEMP/300)**(-4.34))*(EXP(-11080/TEMP))
      K4I     = (9.70D+14*(TEMP/300)**0.1)*EXP(-11080/TEMP)    
      KR4     = K40/K4I    
      FC4     = (EXP(-TEMP/250) + EXP(-1050/TEMP))
      Fa4      = 10**(LOG10(FC4)/(1+(LOG10(KR4))**2)) 
      KMT04   = (K40*K4I)*Fa4/(K40+K4I)       
C	WRITE(6,*)'KMTO4=',KMT04                                                               
C    KMT05                                                   
      KMT05  =  1 + ((0.6*M)/(2.687D+19*(273/TEMP))) 
C                                                                     
C    KMT06                                                   
      KMT06  =  1 + (1.40D-21*EXP(2200/TEMP)*H2O) 
C                                                                     
C    KMT07  : OH       + NO      = HONO                              
C    IUPAC 2001                                                      
      K70     = 7.00D-31*M*(TEMP/300)**(-2.6) 
      K7I     = 3.60D-11*(TEMP/300)**0.1    
      KR7     = K70/K7I    
      FC7     = 0.6  
      F7      = 10**(LOG10(FC7)/(1+(LOG10(KR7))**2)) 
      KMT07   = (K70*K7I)*F7/(K70+K7I) 
C                                                                     
C NASA 2000                                                           
  
C    KMT08                                                    
      K80 = 2.50D-30*((temp/300)**(-4.4))*M 
      K8I = 1.60D-11 
      KR8 = K80/K8I 
      FC8 = 0.6 
      F8      = 10**(LOG10(FC8)/(1+(LOG10(KR8))**2)) 
      KMT08   = (K80*K8I)*F8/(K80+K8I) 
C                                                                     
C    KMT09  : HO2      + NO2     = HO2NO2                            
C IUPAC 1997/2001                                                 
      K90     = 1.80D-31*M*(TEMP/300)**(-3.2) 
      K9I     = 4.70D-12    
      KR9     = K90/K9I    
      FC9     = 0.6 
      F9      = 10**(LOG10(FC9)/(1+(LOG10(KR9))**2)) 
      KMT09   = (K90*K9I)*F9/(K90+K9I) 
C                                                                     
C KMT10  : HO2NO2             = HO2     + NO2                     
C IUPAC 2001                                                      
C
      K100     = 4.10D-05*M*EXP(-10650/TEMP) 
      K10I     = 5.70D+15*EXP(-11170/TEMP)   
      KR10     = K100/K10I    
      FC10     = 0.5 
      F10      = 10**(LOG10(FC10)/(1+(LOG10(KR10))**2)) 
      KMT10    = (K100*K10I)*F10/(K100+K10I) 
C                                                                     
C   KMT11  : OH       + HNO3    = H2O     + NO3                     
C   IUPAC 2001                                                      
      K1     = 7.20D-15*EXP(785/TEMP) 
      K3     = 1.90D-33*EXP(725/TEMP) 
      K4     = 4.10D-16*EXP(1440/TEMP) 
      K2     = (K3*M)/(1+(K3*M/K4)) 
      KMT11  = K1 + K2 
C                                                                     
C KMT12 : OH    +   SO2  =  HSO3                                  
C IUPAC 2003                                                      
      K0 = 3.0D-31*((TEMP/300)**(-3.3))*M 
      KI = 1.5D-12 
      KR1 = K0/KI 
      FC = 0.6 
      F=10**(LOG10(FC)/(1+(LOG10(KR1))**2)) 
      KMT12=(K0*KI*F)/(K0+KI) 
C                                                                     
C KMT13  : CH3O2    + NO2     = CH3O2NO2                           
C IUPAC 2003                                                       
      K130     = 1.20D-30*((TEMP/300)**(-6.9))*M 
      K13I     = 1.80D-11 
      KR13     = K130/K13I 
      FC13     = 0.36 
      F13      = 10**(LOG10(FC13)/(1+(LOG10(KR13))**2)) 
      KMT13    = (K130*K13I)*F13/(K130+K13I) 
C                                                                     
C  KMT14  : CH3O2NO2           = CH3O2   + NO2                      
C  IUPAC 2001                                                       
      K140     = 9.00D-05*EXP(-9690/TEMP)*M 
      K14I     = 1.10D+16*EXP(-10560/TEMP) 
      KR14     = K140/K14I 
      FC14     = 0.36 
      F14      = 10**(LOG10(FC14)/(1+(LOG10(KR14))**2)) 
      KMT14    = (K140*K14I)*F14/(K140+K14I) 
C                                                                   
C KMT15  :    OH  +  C2H4  =                                       
C IUPAC 2001                                                      
      K150 = 6.00D-29*((TEMP/298)**(-4.0))*M 
      K15I = 9.00D-12*((TEMP/298)**(-1.1)) 
      KR15 = K150/K15I 
      FC15 = 0.7
      F15      = 10**(LOG10(FC15)/(1+(LOG10(KR15))**2)) 
      KMT15    = (K150*K15I)*F15/(K150+K15I) 
C                                                                    
C KMT16  :  OH  +  C3H6         
C IUPAC 2003                                                     
      K160     = 3.00D-27*((TEMP/298)**(-3.0))*M 
      K16I     = 2.80D-11*((TEMP/298)**(-1.3)) 
      KR16     = K160/K16I 
      FC16     = 0.5 
      F16      = 10**(LOG10(FC16)/(1+(LOG10(KR16))**2)) 
      KMT16    = (K160*K16I)*F16/(K160+K16I) 
C                                                                     
C    KMT17                                                   
      K170 = 5.00D-30*((TEMP/298)**(-1.5))*M 
      K17I = 9.40D-12*EXP(-700/TEMP) 
      KR17     = K170/K17I 
      FC17 = (EXP(-TEMP/580) + EXP(-2320/TEMP)) 
      F17      = 10**(LOG10(FC17)/(1+(LOG10(KR17))**2)) 
      KMT17    = (K170*K17I)*F17/(K170+K17I) 
C
C  LIST OF ALL REACTIONS 
C
C     Reaction (1) O = O3                                                             
         RC(1) = 5.60D-34*O2*N2*((TEMP/300)**(-2.6))
C
C     Reaction (2) O = O3                                                             
         RC(2) = 6.00D-34*O2*O2*((TEMP/300)**(-2.6))
C
C     Reaction (3) O + O3 =                                                           
         RC(3) = 8.00D-12*EXP(-2060/TEMP)         
C
C     Reaction (4) O + NO = NO2                                                       
         RC(4) = KMT01                            
C
C     Reaction (5) O + NO2 = NO                                                       
         RC(5) = 5.50D-12*EXP(188/TEMP)           
C
C     Reaction (6) O + NO2 = NO3                                                      
         RC(6) = KMT02                            
C
C     Reaction (7) O1D = O                                                            
         RC(7) = 3.20D-11*O2*EXP(67/TEMP)         
C
C     Reaction (8) O1D = O                                                            
         RC(8) = 1.80D-11*N2*EXP(107/TEMP)        
C
C     Reaction (9) NO + O3 = NO2                                                      
         RC(9) = 1.40D-12*EXP(-1310/TEMP)         
C
C     Reaction (10) NO2 + O3 = NO3                                                     
         RC(10) = 1.40D-13*EXP(-2470/TEMP)         
C
C     Reaction (11) NO + NO = NO2 + NO2                                                
         RC(11) = 3.30D-39*EXP(530/TEMP)*O2        
C
C     Reaction (12) NO + NO3 = NO2 + NO2                                               
         RC(12) = 1.80D-11*EXP(110/TEMP)           
C
C     Reaction (13) NO2 + NO3 = NO + NO2                                               
         RC(13) = 4.50D-14*EXP(-1260/TEMP)         
C
C     Reaction (14) NO2 + NO3 = N2O5                                                   
         RC(14) = KMT03                            
C
C     Reaction (15) N2O5 = NO2 + NO3                                                   
         RC(15) = KMT04                            
C
C     Reaction (16) O1D = OH + OH                                                      
         RC(16) = 2.20D-10                     
C
C     Reaction (17) OH + O3 = HO2                                                      
         RC(17) = 1.70D-12*EXP(-940/TEMP)          
C
C     Reaction (18) OH + H2 = HO2                                                      
         RC(18) = 7.70D-12*EXP(-2100/TEMP)         
C
C     Reaction (19) OH + CO = HO2                                                      
         RC(19) = 1.30D-13*KMT05                   
C
C     Reaction (20) OH + H2O2 = HO2                                                    
         RC(20) = 2.90D-12*EXP(-160/TEMP)          
C
C     Reaction (21) HO2 + O3 = OH                                                      
         RC(21) = 2.03D-16*((TEMP/300)**4.57)*EXP(693/TEMP)  
C
C     Reaction (22) OH + HO2 =                                                         
         RC(22) = 4.80D-11*EXP(250/TEMP)           
C
C     Reaction (23) HO2 + HO2 = H2O2                                                   
         RC(23) = 2.20D-13*KMT06*EXP(600/TEMP)     
C
C     Reaction (24) HO2 + HO2 = H2O2                                                   
         RC(24) = 1.90D-33*M*KMT06*EXP(980/TEMP)   
C
C     Reaction (25) OH + NO = HONO                                                     
         RC(25) = KMT07                            
C
C     Reaction (26) NO2 = HONO                                                         
         RC(26) = 5.0D-07                          
C
C     Reaction (27) OH + NO2 = HNO3                                                    
         RC(27) = KMT08                            
C
C     Reaction (28) OH + NO3 = HO2 + NO2                                               
         RC(28) = 2.00D-11                         
C
C     Reaction (29) HO2 + NO = OH + NO2                                                
         RC(29) = 3.60D-12*EXP(270/TEMP)           
C
C     Reaction (30) HO2 + NO2 = HO2NO2                                                 
         RC(30) = KMT09                            
C
C     Reaction (31) HO2NO2 = HO2 + NO2                                                 
         RC(31) = KMT10                            
C
C     Reaction (32) OH + HO2NO2 = NO2                                                  
         RC(32) = 1.90D-12*EXP(270/TEMP)           
C
C     Reaction (33) HO2 + NO3 = OH + NO2                                               
         RC(33) = 4.00D-12                         
C
C     Reaction (34) OH + HONO = NO2                                                    
         RC(34) = 2.50D-12*EXP(-260/TEMP)          
C
C     Reaction (35) OH + HNO3 = NO3                                                    
         RC(35) = KMT11                            
C
C     Reaction (36) O + SO2 = SO3                                                      
         RC(36) = 4.00D-32*EXP(-1000/TEMP)*M       
C
C     Reaction (37) OH + SO2 = HSO3                                                    
         RC(37) = KMT12                            
C
C     Reaction (38) HSO3 = HO2 + SO3                                                   
         RC(38) = 1.30D-12*EXP(-330/TEMP)*O2       
C
C     Reaction (39) HNO3 = NA                                                          
         RC(39) = 6.00D-06                         
C
C     Reaction (40) N2O5 = NA + NA                                                     
         RC(40) = 4.00D-05                         
C
C     Reaction (41) SO3 = SA                                                           
         RC(41) = 1.20D-15*H2O                     
C
C     Reaction (42) OH + CH4 = CH3O2                                                   
         RC(42) = 9.65D-20*TEMP**2.58*EXP(-1082/TEMP) 
C
C     Reaction (43) OH + C2H6 = C2H5O2                                                 
         RC(43) = 1.52D-17*TEMP**2*EXP(-498/TEMP) 
C
C     Reaction (44) OH + C3H8 = IC3H7O2                                                
         RC(44) = 1.55D-17*TEMP**2*EXP(-61/TEMP)*0.736  
C
C     Reaction (45) OH + C3H8 = RN10O2                                                 
         RC(45) = 1.55D-17*TEMP**2*EXP(-61/TEMP)*0.264  
C
C     Reaction (46) OH + NC4H10 = RN13O2                                               
         RC(46) = 1.69D-17*TEMP**2*EXP(145/TEMP)  
C
C     Reaction (47) OH + C2H4 = HOCH2CH2O2                                             
         RC(47) = KMT15                        
C
C     Reaction (48) OH + C3H6 = RN9O2                                                  
         RC(48) = KMT16                        
C
C     Reaction (49) OH + TBUT2ENE = RN12O2                                             
         RC(49) = 1.01D-11*EXP(550/TEMP)       
C
C     Reaction (50) NO3 + C2H4 = NRN6O2                                                
         RC(50) = 2.10D-16                     
C
C     Reaction (51) NO3 + C3H6 = NRN9O2                                                
         RC(51) = 9.40D-15                     
C
C     Reaction (52) NO3 + TBUT2ENE = NRN12O2                                           
         RC(52) = 3.90D-13                     
C
C     Reaction (53) O3 + C2H4 = HCHO + CO + HO2 + OH                                   
         RC(53) = 9.14D-15*EXP(-2580/TEMP)*0.13  
C
C     Reaction (54) O3 + C2H4 = HCHO + HCOOH                                           
         RC(54) = 9.14D-15*EXP(-2580/TEMP)*0.87  
C
C     Reaction (55) O3 + C3H6 = HCHO + CO + CH3O2 + OH                                 
         RC(55) = 5.51D-15*EXP(-1878/TEMP)*0.36  
C
C     Reaction (56) O3 + C3H6 = HCHO + CH3CO2H                                         
         RC(56) = 5.51D-15*EXP(-1878/TEMP)*0.64  
C
C     Reaction (57) O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH                           
         RC(57) = 6.64D-15*EXP(-1059/TEMP)*0.69 
C
C     Reaction (58) O3 + TBUT2ENE = CH3CHO + CH3CO2H                                   
         RC(58) = 6.64D-15*EXP(-1059/TEMP)*0.31 
C
C     Reaction (59) OH + C5H8 = RU14O2                                                 
         RC(59) = 2.70D-11*EXP(390/TEMP)       
C
C     Reaction (60) NO3 + C5H8 = NRU14O2                                               
         RC(60) = 3.15D-12*EXP(-450/TEMP)      
C
C     Reaction (61) O3 + C5H8 = UCARB10 + CO + HO2 + OH                                
         RC(61) = 1.03D-14*EXP(-1995/TEMP)*0.27 
C
C     Reaction (62) O3 + C5H8 = UCARB10 + HCOOH                                        
         RC(62) = 1.03D-14*EXP(-1995/TEMP)*0.73 
C
C     Reaction (63) APINENE + OH = RTN28O2                                             
         RC(63) = 1.20D-11*EXP(444/TEMP)           
C
C     Reaction (64) APINENE + NO3 = NRTN28O2                                           
         RC(64) = 1.19D-12*EXP(490/TEMP)           
C
C     Reaction (65) APINENE + O3 = OH + RTN26O2                                        
         RC(65) = 1.01D-15*EXP(-732/TEMP)*0.80  
C
C     Reaction (66) APINENE + O3 = TNCARB26 + H2O2                                     
         RC(66) = 1.01D-15*EXP(-732/TEMP)*0.075  
C
C     Reaction (67) APINENE + O3 = RCOOH25                                             
         RC(67) = 1.01D-15*EXP(-732/TEMP)*0.125  
C
C     Reaction (68) BPINENE + OH = RTX28O2                                             
         RC(68) = 2.38D-11*EXP(357/TEMP) 
C
C     Reaction (69) BPINENE + NO3 = NRTX28O2                                           
         RC(69) = 2.51D-12 
C
C     Reaction (70) BPINENE + O3 =  RTX24O2 + OH                                       
         RC(70) = 1.50D-17*0.35 
C
C     Reaction (71) BPINENE + O3 =  HCHO + TXCARB24 + H2O2                             
         RC(71) = 1.50D-17*0.20 
C
C     Reaction (72) BPINENE + O3 =  HCHO + TXCARB22                                    
         RC(72) = 1.50D-17*0.25 
C
C     Reaction (73) BPINENE + O3 =  TXCARB24 + CO                                      
         RC(73) = 1.50D-17*0.20 
C
C     Reaction (74) C2H2 + OH = HCOOH + CO + HO2                                       
         RC(74) = KMT17*0.364 
C
C     Reaction (75) C2H2 + OH = CARB3 + OH                                             
         RC(75) = KMT17*0.636 
C
C     Reaction (76) BENZENE + OH = RA13O2                                              
         RC(76) = 2.33D-12*EXP(-193/TEMP)*0.47 
C
C     Reaction (77) BENZENE + OH = AROH14 + HO2                                        
         RC(77) = 2.33D-12*EXP(-193/TEMP)*0.53 
C
C     Reaction (78) TOLUENE + OH = RA16O2                                              
         RC(78) = 1.81D-12*EXP(338/TEMP)*0.82 
C
C     Reaction (79) TOLUENE + OH = AROH17 + HO2                                        
         RC(79) = 1.81D-12*EXP(338/TEMP)*0.18 
C
C     Reaction (80) OXYL + OH = RA19AO2                                                
         RC(80) = 1.36D-11*0.70 
C
C     Reaction (81) OXYL + OH = RA19CO2                                                
         RC(81) = 1.36D-11*0.30 
C
C     Reaction (82) OH + HCHO = HO2 + CO                                               
         RC(82) = 1.20D-14*TEMP*EXP(287/TEMP)  
C
C     Reaction (83) OH + CH3CHO = CH3CO3                                               
         RC(83) = 5.55D-12*EXP(311/TEMP)             
C
C     Reaction (84) OH + C2H5CHO = C2H5CO3                                             
         RC(84) = 1.96D-11                                
C
C     Reaction (85) NO3 + HCHO = HO2 + CO + HNO3                                       
         RC(85) = 5.80D-16                  
C
C     Reaction (86) NO3 + CH3CHO = CH3CO3 + HNO3                                       
         RC(86) = KNO3AL                   
C
C     Reaction (87) NO3 + C2H5CHO = C2H5CO3 + HNO3                                     
         RC(87) = KNO3AL*2.4             
C
C     Reaction (88) OH + CH3COCH3 = RN8O2                                              
         RC(88) = 5.34D-18*TEMP**2*EXP(-230/TEMP) 
C
C     Reaction (89) MEK + OH = RN11O2                                                  
         RC(89) = 3.24D-18*TEMP**2*EXP(414/TEMP)
C
C     Reaction (90) OH + CH3OH = HO2 + HCHO                                            
         RC(90) = 6.01D-18*TEMP**2*EXP(170/TEMP)  
C
C     Reaction (91) OH + C2H5OH = CH3CHO + HO2                                         
         RC(91) = 6.18D-18*TEMP**2*EXP(532/TEMP)*0.887 
C
C     Reaction (92) OH + C2H5OH = HOCH2CH2O2                                           
         RC(92) = 6.18D-18*TEMP**2*EXP(532/TEMP)*0.113 
C
C     Reaction (93) NPROPOL + OH = C2H5CHO + HO2                                       
         RC(93) = 5.53D-12*0.49 
C
C     Reaction (94) NPROPOL + OH = RN9O2                                               
         RC(94) = 5.53D-12*0.51 
C
C     Reaction (95) OH + IPROPOL = CH3COCH3 + HO2                                      
         RC(95) = 4.06D-18*TEMP**2*EXP(788/TEMP)*0.86 
C
C     Reaction (96) OH + IPROPOL = RN9O2                                               
         RC(96) = 4.06D-18*TEMP**2*EXP(788/TEMP)*0.14 
C
C     Reaction (97) HCOOH + OH = HO2                                                   
         RC(97) = 4.50D-13 
C
C     Reaction (98) CH3CO2H + OH = CH3O2                                               
         RC(98) = 8.00D-13 
C
C     Reaction (99) OH + CH3CL = CH3O2                                                 
         RC(99) = 7.33D-18*TEMP**2*EXP(-809/TEMP)   
C
C     Reaction (100) OH + CH2CL2 = CH3O2                                                
         RC(100) = 6.14D-18*TEMP**2*EXP(-389/TEMP)   
C
C     Reaction (101) OH + CHCL3 = CH3O2                                                 
         RC(101) = 1.80D-18*TEMP**2*EXP(-129/TEMP)   
C
C     Reaction (102) OH + CH3CCL3 = C2H5O2                                              
         RC(102) = 2.25D-18*TEMP**2*EXP(-910/TEMP)   
C
C     Reaction (103) OH + TCE = HOCH2CH2O2                                              
         RC(103) = 9.64D-12*EXP(-1209/TEMP)         
C
C     Reaction (104) OH + TRICLETH = HOCH2CH2O2                                         
         RC(104) = 5.63D-13*EXP(427/TEMP)            
C
C     Reaction (105) OH + CDICLETH = HOCH2CH2O2                                         
         RC(105) = 1.94D-12*EXP(90/TEMP)            
C
C     Reaction (106) OH + TDICLETH = HOCH2CH2O2                                         
         RC(106) = 1.01D-12*EXP(250/TEMP)           
C
C     Reaction (107) CH3O2 + NO = HCHO + HO2 + NO2                                      
         RC(107) = 3.00D-12*EXP(280/TEMP)*0.999 
C
C     Reaction (108) C2H5O2 + NO = CH3CHO + HO2 + NO2                                   
         RC(108) = 2.60D-12*EXP(365/TEMP)*0.991 
C
C     Reaction (109) RN10O2 + NO = C2H5CHO + HO2 + NO2                                  
         RC(109) = 2.80D-12*EXP(360/TEMP)*0.980 
C
C     Reaction (110) IC3H7O2 + NO = CH3COCH3 + HO2 + NO2                                
         RC(110) = 2.70D-12*EXP(360/TEMP)*0.958 
C
C     Reaction (111) RN13O2 + NO = CH3CHO + C2H5O2 + NO2                                
         RC(111) = KRO2NO*0.917*BR01       
C
C     Reaction (112) RN13O2 + NO = CARB11A + HO2 + NO2                                  
         RC(112) = KRO2NO*0.917*(1-BR01)   
C
C     Reaction (113) RN16O2 + NO = RN15AO2 + NO2                                        
         RC(113) = KRO2NO*0.877                 
C
C     Reaction (114) RN19O2 + NO = RN18AO2 + NO2                                        
         RC(114) = KRO2NO*0.788                 
C
C     Reaction (115) RN13AO2 + NO = RN12O2 + NO2                                        
         RC(115) = KRO2NO                       
C
C     Reaction (116) RN16AO2 + NO = RN15O2 + NO2                                        
         RC(116) = KRO2NO                       
C
C     Reaction (117) RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2                          
         RC(117) = KRO2NO*0.918       
C
C     Reaction (118) RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2                         
         RC(118) = KRO2NO*0.889*0.7 
C
C     Reaction (119) RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2                          
         RC(119) = KRO2NO*0.889*0.3 
C
C     Reaction (120) RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2                        
         RC(120) = KRO2NO*0.862       
C
C     Reaction (121) RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2                         
         RC(121) = KRO2NO*0.862       
C
C     Reaction (122) HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2                          
         RC(122) = KRO2NO*0.995*0.776  
C
C     Reaction (123) HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2                             
         RC(123) = KRO2NO*0.995*0.224  
C
C     Reaction (124) RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2                             
         RC(124) = KRO2NO*0.979     
C
C     Reaction (125) RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2                          
         RC(125) = KRO2NO*0.959     
C
C     Reaction (126) RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2                         
         RC(126) = KRO2NO*0.936     
C
C     Reaction (127) RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2                        
         RC(127) = KRO2NO*0.903     
C
C     Reaction (128) RN15AO2 + NO = CARB13 + HO2 + NO2                                  
         RC(128) = KRO2NO*0.975     
C
C     Reaction (129) RN18AO2 + NO = CARB16 + HO2 + NO2                                  
         RC(129) = KRO2NO*0.946     
C
C     Reaction (130) CH3CO3 + NO = CH3O2 + NO2                                          
         RC(130) = KAPNO                      
C
C     Reaction (131) C2H5CO3 + NO = C2H5O2 + NO2                                        
         RC(131) = KAPNO                      
C
C     Reaction (132) HOCH2CO3 + NO = HO2 + HCHO + NO2                                   
         RC(132) = KAPNO                      
C
C     Reaction (133) RN8O2 + NO = CH3CO3 + HCHO + NO2                                   
         RC(133) = KRO2NO                     
C
C     Reaction (134) RN11O2 + NO = CH3CO3 + CH3CHO + NO2                                
         RC(134) = KRO2NO                     
C
C     Reaction (135) RN14O2 + NO = C2H5CO3 + CH3CHO + NO2                               
         RC(135) = KRO2NO                     
C
C     Reaction (136) RN17O2 + NO = RN16AO2 + NO2                                        
         RC(136) = KRO2NO                     
C
C     Reaction (137) RU14O2 + NO = UCARB12 + HO2 +  NO2                                 
         RC(137) = KRO2NO*0.900*0.32  
C
C     Reaction (138) RU14O2 + NO = UCARB10 + HCHO + HO2 + NO2                           
         RC(138) = KRO2NO*0.900*0.968 
C
C     Reaction (139) RU12O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
         RC(139) = KRO2NO*0.7         
C
C     Reaction (140) RU12O2 + NO = CARB7 + CO + HO2 + NO2                               
         RC(140) = KRO2NO*0.3         
C
C     Reaction (141) RU10O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
         RC(141) = KRO2NO*0.670         
C
C     Reaction (142) RU10O2 + NO = CARB6 + HCHO + HO2 + NO2                             
         RC(142) = KRO2NO*0.295         
C
C     Reaction (143) RU10O2 + NO = CARB7 + HCHO + HO2 + NO2                             
         RC(143) = KRO2NO*0.035          
C
C     Reaction (144) NRN6O2 + NO = HCHO + HCHO + NO2 + NO2                              
         RC(144) = KRO2NO                 
C
C     Reaction (145) NRN9O2 + NO = CH3CHO + HCHO + NO2 + NO2                            
         RC(145) = KRO2NO                 
C
C     Reaction (146) NRN12O2 + NO = CH3CHO + CH3CHO + NO2 + NO2                         
         RC(146) = KRO2NO                 
C
C     Reaction (147) NRU14O2 + NO = NUCARB12 + HO2 + NO2                                
         RC(147) = KRO2NO                 
C
C     Reaction (148) NRU12O2 + NO = NOA + CO + HO2 + NO2                                
         RC(148) = KRO2NO*0.5                 
C
C     Reaction (149) RTN28O2 + NO = TNCARB26 + HO2 + NO2                                
         RC(149) = KRO2NO*0.767*0.915  
C
C     Reaction (150) RTN28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
         RC(150) = KRO2NO*0.767*0.085  
C
C     Reaction (151) NRTN28O2 + NO = TNCARB26 + NO2 + NO2                               
         RC(151) = KRO2NO                  
C
C     Reaction (152) RTN26O2 + NO = RTN25O2 + NO2                                       
         RC(152) = KAPNO                   
C
C     Reaction (153) RTN25O2 + NO = RTN24O2 + NO2                                       
         RC(153) = KRO2NO*0.840        
C
C     Reaction (154) RTN24O2 + NO = RTN23O2 + NO2                                       
         RC(154) = KRO2NO                   
C
C     Reaction (155) RTN23O2 + NO = CH3COCH3 + RTN14O2 + NO2                            
         RC(155) = KRO2NO                  
C
C     Reaction (156) RTN14O2 + NO = HCHO + TNCARB10 + HO2 + NO2                         
         RC(156) = KRO2NO               
C
C     Reaction (157) RTN10O2 + NO = RN8O2 + CO + NO2                                    
         RC(157) = KRO2NO               
C
C     Reaction (158) RTX28O2 + NO = TXCARB24 + HCHO + HO2 + NO2                         
         RC(158) = KRO2NO*0.767*0.915  
C
C     Reaction (159) RTX28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
         RC(159) = KRO2NO*0.767*0.085  
C
C     Reaction (160) NRTX28O2 + NO = TXCARB24 + HCHO + NO2 + NO2                        
         RC(160) = KRO2NO            
C
C     Reaction (161) RTX24O2 + NO = TXCARB22 + HO2 + NO2                                
         RC(161) = KRO2NO*0.843*0.6  
C
C     Reaction (162) RTX24O2 + NO = CH3COCH3 + RN13AO2 + HCHO + NO2                     
         RC(162) = KRO2NO*0.843*0.4  
C
C     Reaction (163) RTX22O2 + NO = CH3COCH3 + RN13O2 + NO2                             
         RC(163) = KRO2NO*0.700         
C
C     Reaction (164) CH3O2    + NO2     = CH3O2NO2                                      
         RC(164) = KMT13         
C
C     Reaction (165) CH3O2NO2           = CH3O2   + NO2                                 
         RC(165) = KMT14         
C
C     Reaction (166) CH3O2 + NO = CH3NO3                                                
         RC(166) = 3.00D-12*EXP(280/TEMP)*0.001 
C
C     Reaction (167) C2H5O2 + NO = C2H5NO3                                              
         RC(167) = 2.60D-12*EXP(365/TEMP)*0.009 
C
C     Reaction (168) RN10O2 + NO = RN10NO3                                              
         RC(168) = 2.80D-12*EXP(360/TEMP)*0.020 
C
C     Reaction (169) IC3H7O2 + NO = IC3H7NO3                                            
         RC(169) = 2.70D-12*EXP(360/TEMP)*0.042 
C
C     Reaction (170) RN13O2 + NO = RN13NO3                                              
         RC(170) = KRO2NO*0.083                 
C
C     Reaction (171) RN16O2 + NO = RN16NO3                                              
         RC(171) = KRO2NO*0.123                 
C
C     Reaction (172) RN19O2 + NO = RN19NO3                                              
         RC(172) = KRO2NO*0.212                 
C
C     Reaction (173) HOCH2CH2O2 + NO = HOC2H4NO3                                        
         RC(173) = KRO2NO*0.005                 
C
C     Reaction (174) RN9O2 + NO = RN9NO3                                                
         RC(174) = KRO2NO*0.021                 
C
C     Reaction (175) RN12O2 + NO = RN12NO3                                              
         RC(175) = KRO2NO*0.041                 
C
C     Reaction (176) RN15O2 + NO = RN15NO3                                              
         RC(176) = KRO2NO*0.064                 
C
C     Reaction (177) RN18O2 + NO = RN18NO3                                              
         RC(177) = KRO2NO*0.097                 
C
C     Reaction (178) RN15AO2 + NO = RN15NO3                                             
         RC(178) = KRO2NO*0.025                 
C
C     Reaction (179) RN18AO2 + NO = RN18NO3                                             
         RC(179) = KRO2NO*0.054                 
C
C     Reaction (180) RU14O2 + NO = RU14NO3                                              
         RC(180) = KRO2NO*0.100                 
C
C     Reaction (181) RA13O2 + NO = RA13NO3                                              
         RC(181) = KRO2NO*0.082                 
C
C     Reaction (182) RA16O2 + NO = RA16NO3                                              
         RC(182) = KRO2NO*0.111                 
C
C     Reaction (183) RA19AO2 + NO = RA19NO3                                             
         RC(183) = KRO2NO*0.138                 
C
C     Reaction (184) RA19CO2 + NO = RA19NO3                                             
         RC(184) = KRO2NO*0.138                 
C
C     Reaction (185) RTN28O2 + NO = RTN28NO3                                            
         RC(185) = KRO2NO*0.233        
C
C     Reaction (186) RTN25O2 + NO = RTN25NO3                                            
         RC(186) = KRO2NO*0.160        
C
C     Reaction (187) RTX28O2 + NO = RTX28NO3                                            
         RC(187) = KRO2NO*0.233        
C
C     Reaction (188) RTX24O2 + NO = RTX24NO3                                            
         RC(188) = KRO2NO*0.157        
C
C     Reaction (189) RTX22O2 + NO = RTX22NO3                                            
         RC(189) = KRO2NO*0.300        
C
C     Reaction (190) CH3O2 + NO3 = HCHO + HO2 + NO2                                     
         RC(190) = KRO2NO3*0.40          
C
C     Reaction (191) C2H5O2 + NO3 = CH3CHO + HO2 + NO2                                  
         RC(191) = KRO2NO3               
C
C     Reaction (192) RN10O2 + NO3 = C2H5CHO + HO2 + NO2                                 
         RC(192) = KRO2NO3               
C
C     Reaction (193) IC3H7O2 + NO3 = CH3COCH3 + HO2 + NO2                               
         RC(193) = KRO2NO3               
C
C     Reaction (194) RN13O2 + NO3 = CH3CHO + C2H5O2 + NO2                               
         RC(194) = KRO2NO3*BR01     
C
C     Reaction (195) RN13O2 + NO3 = CARB11A + HO2 + NO2                                 
         RC(195) = KRO2NO3*(1-BR01) 
C
C     Reaction (196) RN16O2 + NO3 = RN15AO2 + NO2                                       
         RC(196) = KRO2NO3               
C
C     Reaction (197) RN19O2 + NO3 = RN18AO2 + NO2                                       
         RC(197) = KRO2NO3               
C
C     Reaction (198) RN13AO2 + NO3 = RN12O2 + NO2                                       
         RC(198) = KRO2NO3                      
C
C     Reaction (199) RN16AO2 + NO3 = RN15O2 + NO2                                       
         RC(199) = KRO2NO3                      
C
C     Reaction (200) RA13O2 + NO3 = CARB3 + UDCARB8 + HO2 + NO2                         
         RC(200) = KRO2NO3            
C
C     Reaction (201) RA16O2 + NO3 = CARB3 + UDCARB11 + HO2 + NO2                        
         RC(201) = KRO2NO3*0.7     
C
C     Reaction (202) RA16O2 + NO3 = CARB6 + UDCARB8 + HO2 + NO2                         
         RC(202) = KRO2NO3*0.3     
C
C     Reaction (203) RA19AO2 + NO3 = CARB3 + UDCARB14 + HO2 + NO2                       
         RC(203) = KRO2NO3           
C
C     Reaction (204) RA19CO2 + NO3 = CARB9 + UDCARB8 + HO2 + NO2                        
         RC(204) = KRO2NO3           
C
C     Reaction (205) HOCH2CH2O2 + NO3 = HCHO + HCHO + HO2 + NO2                         
         RC(205) = KRO2NO3*0.776  
C
C     Reaction (206) HOCH2CH2O2 + NO3 = HOCH2CHO + HO2 + NO2                            
         RC(206) = KRO2NO3*0.224  
C
C     Reaction (207) RN9O2 + NO3 = CH3CHO + HCHO + HO2 + NO2                            
         RC(207) = KRO2NO3         
C
C     Reaction (208) RN12O2 + NO3 = CH3CHO + CH3CHO + HO2 + NO2                         
         RC(208) = KRO2NO3         
C
C     Reaction (209) RN15O2 + NO3 = C2H5CHO + CH3CHO + HO2 + NO2                        
         RC(209) = KRO2NO3         
C
C     Reaction (210) RN18O2 + NO3 = C2H5CHO + C2H5CHO + HO2 + NO2                       
         RC(210) = KRO2NO3         
C
C     Reaction (211) RN15AO2 + NO3 = CARB13 + HO2 + NO2                                 
         RC(211) = KRO2NO3         
C
C     Reaction (212) RN18AO2 + NO3 = CARB16 + HO2 + NO2                                 
         RC(212) = KRO2NO3         
C
C     Reaction (213) CH3CO3 + NO3 = CH3O2 + NO2                                         
         RC(213) = KRO2NO3*1.60          
C
C     Reaction (214) C2H5CO3 + NO3 = C2H5O2 + NO2                                       
         RC(214) = KRO2NO3*1.60          
C
C     Reaction (215) HOCH2CO3 + NO3 = HO2 + HCHO + NO2                                  
         RC(215) = KRO2NO3*1.60         
C
C     Reaction (216) RN8O2 + NO3 = CH3CO3 + HCHO + NO2                                  
         RC(216) = KRO2NO3               
C
C     Reaction (217) RN11O2 + NO3 = CH3CO3 + CH3CHO + NO2                               
         RC(217) = KRO2NO3               
C
C     Reaction (218) RN14O2 + NO3 = C2H5CO3 + CH3CHO + NO2                              
         RC(218) = KRO2NO3               
C
C     Reaction (219) RN17O2 + NO3 = RN16AO2 + NO2                                       
         RC(219) = KRO2NO3               
C
C     Reaction (220) RU14O2 + NO3 = UCARB12 + HO2 + NO2                                 
         RC(220) = KRO2NO3*0.032     
C
C     Reaction (221) RU14O2 + NO3 = UCARB10 + HCHO + HO2 + NO2                          
         RC(221) = KRO2NO3*0.968     
C
C     Reaction (222) RU12O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
         RC(222) = KRO2NO3*0.7         
C
C     Reaction (223) RU12O2 + NO3 = CARB7 + CO + HO2 + NO2                              
         RC(223) = KRO2NO3*0.3         
C
C     Reaction (224) RU10O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
         RC(224) = KRO2NO3*0.7         
C
C     Reaction (225) RU10O2 + NO3 = CARB6 + HCHO + HO2 + NO2                            
         RC(225) = KRO2NO3*0.3         
C
C     Reaction (226) RU10O2 + NO3 = CARB7 + HCHO + HO2 + NO2                            
         RC(226) = KRO2NO3*0.0         
C
C     Reaction (227) NRN6O2 + NO3 = HCHO + HCHO + NO2 + NO2                             
         RC(227) = KRO2NO3               
C
C     Reaction (228) NRN9O2 + NO3 = CH3CHO + HCHO + NO2 + NO2                           
         RC(228) = KRO2NO3               
C
C     Reaction (229) NRN12O2 + NO3 = CH3CHO + CH3CHO + NO2 + NO2                        
         RC(229) = KRO2NO3               
C
C     Reaction (230) NRU14O2 + NO3 = NUCARB12 + HO2 + NO2                               
         RC(230) = KRO2NO3               
C
C     Reaction (231) NRU12O2 + NO3 = NOA + CO + HO2 + NO2                               
         RC(231) = KRO2NO3               
C
C     Reaction (232) RTN28O2 + NO3 = TNCARB26 + HO2 + NO2                               
         RC(232) = KRO2NO3                
C
C     Reaction (233) NRTN28O2 + NO3 = TNCARB26 + NO2 + NO2                              
         RC(233) = KRO2NO3                
C
C     Reaction (234) RTN26O2 + NO3 = RTN25O2 + NO2                                      
         RC(234) = KRO2NO3*1.60                   
C
C     Reaction (235) RTN25O2 + NO3 = RTN24O2 + NO2                                      
         RC(235) = KRO2NO3                 
C
C     Reaction (236) RTN24O2 + NO3 = RTN23O2 + NO2                                      
         RC(236) = KRO2NO3                   
C
C     Reaction (237) RTN23O2 + NO3 = CH3COCH3 + RTN14O2 + NO2                           
         RC(237) = KRO2NO3                 
C
C     Reaction (238) RTN14O2 + NO3 = HCHO + TNCARB10 + HO2 + NO2                        
         RC(238) = KRO2NO3             
C
C     Reaction (239) RTN10O2 + NO3 = RN8O2 + CO + NO2                                   
         RC(239) = KRO2NO3               
C
C     Reaction (240) RTX28O2 + NO3 = TXCARB24 + HCHO + HO2 + NO2                        
         RC(240) = KRO2NO3             
C
C     Reaction (241) RTX24O2 + NO3 = TXCARB22 + HO2 + NO2                               
         RC(241) = KRO2NO3             
C
C     Reaction (242) RTX22O2 + NO3 = CH3COCH3 + RN13O2 + NO2                            
         RC(242) = KRO2NO3             
C
C     Reaction (243) NRTX28O2 + NO3 = TXCARB24 + HCHO + NO2 + NO2                       
         RC(243) = KRO2NO3            
C
C     Reaction (244) CH3O2 + HO2 = CH3OOH                                               
         RC(244) = 4.10D-13*EXP(790/TEMP)  
C
C     Reaction (245) C2H5O2 + HO2 = C2H5OOH                                             
         RC(245) = 7.50D-13*EXP(700/TEMP)  
C
C     Reaction (246) RN10O2 + HO2 = RN10OOH                                             
         RC(246) = KRO2HO2*0.520           
C
C     Reaction (247) IC3H7O2 + HO2 = IC3H7OOH                                           
         RC(247) = KRO2HO2*0.520           
C
C     Reaction (248) RN13O2 + HO2 = RN13OOH                                             
         RC(248) = KRO2HO2*0.625           
C
C     Reaction (249) RN16O2 + HO2 = RN16OOH                                             
         RC(249) = KRO2HO2*0.706           
C
C     Reaction (250) RN19O2 + HO2 = RN19OOH                                             
         RC(250) = KRO2HO2*0.770           
C
C     Reaction (251) RN13AO2 + HO2 = RN13OOH                                            
         RC(251) = KRO2HO2*0.625           
C
C     Reaction (252) RN16AO2 + HO2 = RN16OOH                                            
         RC(252) = KRO2HO2*0.706           
C
C     Reaction (253) RA13O2 + HO2 = RA13OOH                                             
         RC(253) = KRO2HO2*0.770           
C
C     Reaction (254) RA16O2 + HO2 = RA16OOH                                             
         RC(254) = KRO2HO2*0.820           
C
C     Reaction (255) RA19AO2 + HO2 = RA19OOH                                            
         RC(255) = KRO2HO2*0.859           
C
C     Reaction (256) RA19CO2 + HO2 = RA19OOH                                            
         RC(256) = KRO2HO2*0.859           
C
C     Reaction (257) HOCH2CH2O2 + HO2 = HOC2H4OOH                                       
         RC(257) = 2.03D-13*EXP(1250/TEMP) 
C
C     Reaction (258) RN9O2 + HO2 = RN9OOH                                               
         RC(258) = KRO2HO2*0.520           
C
C     Reaction (259) RN12O2 + HO2 = RN12OOH                                             
         RC(259) = KRO2HO2*0.625           
C
C     Reaction (260) RN15O2 + HO2 = RN15OOH                                             
         RC(260) = KRO2HO2*0.706           
C
C     Reaction (261) RN18O2 + HO2 = RN18OOH                                             
         RC(261) = KRO2HO2*0.770           
C
C     Reaction (262) RN15AO2 + HO2 = RN15OOH                                            
         RC(262) = KRO2HO2*0.706           
C
C     Reaction (263) RN18AO2 + HO2 = RN18OOH                                            
         RC(263) = KRO2HO2*0.770           
C
C     Reaction (264) CH3CO3 + HO2 = CH3CO3H                                             
         RC(264) = KAPHO2*0.560                  
C
C     Reaction (265) C2H5CO3 + HO2 = C2H5CO3H                                           
         RC(265) = KAPHO2*0.560                  
C
C     Reaction (266) HOCH2CO3 + HO2 = HOCH2CO3H                                         
         RC(266) = KAPHO2*0.560                  
C
C     Reaction (267) RN8O2 + HO2 = RN8OOH                                               
         RC(267) = KRO2HO2*0.520           
C
C     Reaction (268) RN11O2 + HO2 = RN11OOH                                             
         RC(268) = KRO2HO2*0.625           
C
C     Reaction (269) RN14O2 + HO2 = RN14OOH                                             
         RC(269) = KRO2HO2*0.706           
C
C     Reaction (270) RN17O2 + HO2 = RN17OOH                                             
         RC(270) = KRO2HO2*0.770           
C
C     Reaction (271) RU14O2 + HO2 = RU14OOH                                             
         RC(271) = KRO2HO2*0.706           
C
C     Reaction (272) RU12O2 + HO2 = RU12OOH                                             
         RC(272) = KRO2HO2*0.706           
C
C     Reaction (273) RU10O2 + HO2 = RU10OOH                                             
         RC(273) = KRO2HO2*0.625           
C
C     Reaction (274) NRN6O2 + HO2 = NRN6OOH                                             
         RC(274) = KRO2HO2*0.387         
C
C     Reaction (275) NRN9O2 + HO2 = NRN9OOH                                             
         RC(275) = KRO2HO2*0.520         
C
C     Reaction (276) NRN12O2 + HO2 = NRN12OOH                                           
         RC(276) = KRO2HO2*0.625         
C
C     Reaction (277) NRU14O2 + HO2 = NRU14OOH                                           
         RC(277) = KRO2HO2*0.706         
C
C     Reaction (278) NRU12O2 + HO2 = NRU12OOH                                           
         RC(278) = KRO2HO2*0.706         
C
C     Reaction (279) RTN28O2 + HO2 = RTN28OOH                                           
         RC(279) = KRO2HO2*0.914         
C
C     Reaction (280) NRTN28O2 + HO2 = NRTN28OOH                                         
         RC(280) = KRO2HO2*0.914         
C
C     Reaction (281) RTN26O2 + HO2 = RTN26OOH                                           
         RC(281) = KAPHO2*0.56                     
C
C     Reaction (282) RTN25O2 + HO2 = RTN25OOH                                           
         RC(282) = KRO2HO2*0.890       
C
C     Reaction (283) RTN24O2 + HO2 = RTN24OOH                                           
         RC(283) = KRO2HO2*0.890       
C
C     Reaction (284) RTN23O2 + HO2 = RTN23OOH                                           
         RC(284) = KRO2HO2*0.890       
C
C     Reaction (285) RTN14O2 + HO2 = RTN14OOH                                           
         RC(285) = KRO2HO2*0.770       
C
C     Reaction (286) RTN10O2 + HO2 = RTN10OOH                                           
         RC(286) = KRO2HO2*0.706       
C
C     Reaction (287) RTX28O2 + HO2 = RTX28OOH                                           
         RC(287) = KRO2HO2*0.914       
C
C     Reaction (288) RTX24O2 + HO2 = RTX24OOH                                           
         RC(288) = KRO2HO2*0.890       
C
C     Reaction (289) RTX22O2 + HO2 = RTX22OOH                                           
         RC(289) = KRO2HO2*0.890       
C
C     Reaction (290) NRTX28O2 + HO2 = NRTX28OOH                                         
         RC(290) = KRO2HO2*0.914       
C
C     Reaction (291) CH3O2 = HCHO + HO2                                                 
         RC(291) = 1.82D-13*EXP(416/TEMP)*0.33*RO2  
C
C     Reaction (292) CH3O2 = HCHO                                                       
         RC(292) = 1.82D-13*EXP(416/TEMP)*0.335*RO2 
C
C     Reaction (293) CH3O2 = CH3OH                                                      
         RC(293) = 1.82D-13*EXP(416/TEMP)*0.335*RO2 
C
C     Reaction (294) C2H5O2 = CH3CHO + HO2                                              
         RC(294) = 3.10D-13*0.6*RO2             
C
C     Reaction (295) C2H5O2 = CH3CHO                                                    
         RC(295) = 3.10D-13*0.2*RO2             
C
C     Reaction (296) C2H5O2 = C2H5OH                                                    
         RC(296) = 3.10D-13*0.2*RO2             
C
C     Reaction (297) RN10O2 = C2H5CHO + HO2                                             
         RC(297) = 6.00D-13*0.6*RO2             
C
C     Reaction (298) RN10O2 = C2H5CHO                                                   
         RC(298) = 6.00D-13*0.2*RO2             
C
C     Reaction (299) RN10O2 = NPROPOL                                                   
         RC(299) = 6.00D-13*0.2*RO2             
C
C     Reaction (300) IC3H7O2 = CH3COCH3 + HO2                                           
         RC(300) = 4.00D-14*0.6*RO2             
C
C     Reaction (301) IC3H7O2 = CH3COCH3                                                 
         RC(301) = 4.00D-14*0.2*RO2             
C
C     Reaction (302) IC3H7O2 = IPROPOL                                                  
         RC(302) = 4.00D-14*0.2*RO2             
C
C     Reaction (303) RN13O2 = CH3CHO + C2H5O2                                           
         RC(303) = 2.50D-13*RO2*BR01       
C
C     Reaction (304) RN13O2 = CARB11A + HO2                                             
         RC(304) = 2.50D-13*RO2*(1-BR01)   
C
C     Reaction (305) RN13AO2 = RN12O2                                                   
         RC(305) = 8.80D-13*RO2                 
C
C     Reaction (306) RN16AO2 = RN15O2                                                   
         RC(306) = 8.80D-13*RO2                 
C
C     Reaction (307) RA13O2 = CARB3 + UDCARB8 + HO2                                     
         RC(307) = 8.80D-13*RO2                 
C
C     Reaction (308) RA16O2 = CARB3 + UDCARB11 + HO2                                    
         RC(308) = 8.80D-13*RO2*0.7          
C
C     Reaction (309) RA16O2 = CARB6 + UDCARB8 + HO2                                     
         RC(309) = 8.80D-13*RO2*0.3          
C
C     Reaction (310) RA19AO2 = CARB3 + UDCARB14 + HO2                                   
         RC(310) = 8.80D-13*RO2                 
C
C     Reaction (311) RA19CO2 = CARB3 + UDCARB14 + HO2                                   
         RC(311) = 8.80D-13*RO2                 
C
C     Reaction (312) RN16O2 = RN15AO2                                                   
         RC(312) = 2.50D-13*RO2                 
C
C     Reaction (313) RN19O2 = RN18AO2                                                   
         RC(313) = 2.50D-13*RO2                 
C
C     Reaction (314) HOCH2CH2O2 = HCHO + HCHO + HO2                                     
         RC(314) = 2.00D-12*RO2*0.776       
C
C     Reaction (315) HOCH2CH2O2 = HOCH2CHO + HO2                                        
         RC(315) = 2.00D-12*RO2*0.224       
C
C     Reaction (316) RN9O2 = CH3CHO + HCHO + HO2                                        
         RC(316) = 8.80D-13*RO2                 
C
C     Reaction (317) RN12O2 = CH3CHO + CH3CHO + HO2                                     
         RC(317) = 8.80D-13*RO2                 
C
C     Reaction (318) RN15O2 = C2H5CHO + CH3CHO + HO2                                    
         RC(318) = 8.80D-13*RO2                 
C
C     Reaction (319) RN18O2 = C2H5CHO + C2H5CHO + HO2                                   
         RC(319) = 8.80D-13*RO2                 
C
C     Reaction (320) RN15AO2 = CARB13 + HO2                                             
         RC(320) = 8.80D-13*RO2                 
C
C     Reaction (321) RN18AO2 = CARB16 + HO2                                             
         RC(321) = 8.80D-13*RO2                 
C
C     Reaction (322) CH3CO3 = CH3O2                                                     
         RC(322) = 1.00D-11*RO2                 
C
C     Reaction (323) C2H5CO3 = C2H5O2                                                   
         RC(323) = 1.00D-11*RO2                 
C
C     Reaction (324) HOCH2CO3 = HCHO + HO2                                              
         RC(324) = 1.00D-11*RO2                 
C
C     Reaction (325) RN8O2 = CH3CO3 + HCHO                                              
         RC(325) = 1.40D-12*RO2                 
C
C     Reaction (326) RN11O2 = CH3CO3 + CH3CHO                                           
         RC(326) = 1.40D-12*RO2                 
C
C     Reaction (327) RN14O2 = C2H5CO3 + CH3CHO                                          
         RC(327) = 1.40D-12*RO2                 
C
C     Reaction (328) RN17O2 = RN16AO2                                                   
         RC(328) = 1.40D-12*RO2                 
C
C     Reaction (329) RU14O2 = UCARB12 + HO2                                             
         RC(329) = 1.26D-12*RO2*0.1        
C
C     Reaction (330) RU14O2 = UCARB10 + HCHO + HO2                                      
         RC(330) = 1.26D-12*RO2*0.9        
C
C     Reaction (331) RU12O2 = CH3CO3 + HOCH2CHO                                         
         RC(331) = 4.20D-13*RO2*0.7            
C
C     Reaction (332) RU12O2 = CARB7 + HOCH2CHO + HO2                                    
         RC(332) = 4.20D-13*RO2*0.3            
C
C     Reaction (333) RU10O2 = CH3CO3 + HOCH2CHO                                         
         RC(333) = 1.83D-12*RO2*0.7            
C
C     Reaction (334) RU10O2 = CARB6 + HCHO + HO2                                        
         RC(334) = 1.83D-12*RO2*0.3            
C
C     Reaction (335) RU10O2 = CARB7 + HCHO + HO2                                        
         RC(335) = 1.83D-12*RO2*0.0            
C
C     Reaction (336) NRN6O2 = HCHO + HCHO + NO2                                         
         RC(336) = 6.00D-13*RO2                 
C
C     Reaction (337) NRN9O2 = CH3CHO + HCHO + NO2                                       
         RC(337) = 2.30D-13*RO2                 
C
C     Reaction (338) NRN12O2 = CH3CHO + CH3CHO + NO2                                    
         RC(338) = 2.50D-13*RO2                 
C
C     Reaction (339) NRU14O2 = NUCARB12 + HO2                                           
         RC(339) = 1.30D-12*RO2                 
C
C     Reaction (340) NRU12O2 = NOA + CO + HO2                                           
         RC(340) = 9.60D-13*RO2*0.5                 
C
C     Reaction (341) RTN28O2 = TNCARB26 + HO2                                           
         RC(341) = 2.85D-13*RO2                 
C
C     Reaction (342) NRTN28O2 = TNCARB26 + NO2                                          
         RC(342) = 1.00D-13*RO2                 
C
C     Reaction (343) RTN26O2 = RTN25O2                                                  
         RC(343) = 1.00D-11*RO2                   
C
C     Reaction (344) RTN25O2 = RTN24O2                                                  
         RC(344) = 1.30D-12*RO2           
C
C     Reaction (345) RTN24O2 = RTN23O2                                                  
         RC(345) = 6.70D-15*RO2             
C
C     Reaction (346) RTN23O2 = CH3COCH3 + RTN14O2                                       
         RC(346) = 6.70D-15*RO2            
C
C     Reaction (347) RTN14O2 = HCHO + TNCARB10 + HO2                                    
         RC(347) = 8.80D-13*RO2        
C
C     Reaction (348) RTN10O2 = RN8O2 + CO                                               
         RC(348) = 2.00D-12*RO2        
C
C     Reaction (349) RTX28O2 = TXCARB24 + HCHO + HO2                                    
         RC(349) = 2.00D-12*RO2       
C
C     Reaction (350) RTX24O2 = TXCARB22 + HO2                                           
         RC(350) = 2.50D-13*RO2       
C
C     Reaction (351) RTX22O2 = CH3COCH3 + RN13O2                                        
         RC(351) = 2.50D-13*RO2       
C
C     Reaction (352) NRTX28O2 = TXCARB24 + HCHO + NO2                                   
         RC(352) = 9.20D-14*RO2       
C
C     Reaction (353) OH + CARB14 = RN14O2                                               
         RC(353) = 1.87D-11       
C
C     Reaction (354) OH + CARB17 = RN17O2                                               
         RC(354) = 4.36D-12       
C
C     Reaction (355) OH + CARB11A = RN11O2                                              
         RC(355) = 3.24D-18*TEMP**2*EXP(414/TEMP)
C
C     Reaction (356) OH + CARB7 = CARB6 + HO2                                           
         RC(356) = 1.60D-12*EXP(305/TEMP)
C
C     Reaction (357) OH + CARB10 = CARB9 + HO2                                          
         RC(357) = 5.86D-12       
C
C     Reaction (358) OH + CARB13 = RN13O2                                               
         RC(358) = 1.65D-11       
C
C     Reaction (359) OH + CARB16 = RN16O2                                               
         RC(359) = 1.25D-11       
C
C     Reaction (360) OH + UCARB10 = RU10O2                                              
         RC(360) = 3.84D-12*EXP(533/TEMP)*0.693
C
C     Reaction (361) NO3 + UCARB10 = RU10O2 + HNO3                                      
         RC(361) = KNO3AL*0.415
C
C     Reaction (362) O3 + UCARB10 = HCHO + CH3CO3 + CO + OH                             
         RC(362) = 1.20D-15*EXP(-1710/TEMP)*0.32       
C
C     Reaction (363) O3 + UCARB10 = HCHO + CARB6 + H2O2                                 
         RC(363) = 1.20D-15*EXP(-1710/TEMP)*0.68
C
C     Reaction (364) OH + HOCH2CHO = HOCH2CO3                                           
         RC(364) = 1.00D-11       
C
C     Reaction (365) NO3 + HOCH2CHO = HOCH2CO3 + HNO3                                   
         RC(365) = KNO3AL        
C
C     Reaction (366) OH + CARB3 = CO + CO + HO2                                         
         RC(366) = 3.10D-12*EXP(340/TEMP)*0.8       
C
C     Reaction (367) OH + CARB6 = CH3CO3 + CO                                           
         RC(367) = 1.90D-12*EXP(575/TEMP)       
C
C     Reaction (368) OH + CARB9 = RN9O2                                                 
         RC(368) = 2.40D-13       
C
C     Reaction (369) OH + CARB12 = RN12O2                                               
         RC(369) = 1.38D-12       
C
C     Reaction (370) OH + CARB15 = RN15O2                                               
         RC(370) = 4.81D-12       
C
C     Reaction (371) OH + CCARB12 = RN12O2                                              
         RC(371) = 4.79D-12       
C
C     Reaction (372) OH + UCARB12 = RU12O2                                              
         RC(372) = 6.42D-11            
C
C     Reaction (373) NO3 + UCARB12 = RU12O2 + HNO3                                      
         RC(373) = KNO3AL*4.25    
C
C     Reaction (374) O3 + UCARB12 = HOCH2CHO + CH3CO3 + CO + OH                         
         RC(374) = 2.40D-17*0.89   
C
C     Reaction (375) O3 + UCARB12 = HOCH2CHO + CARB6 + H2O2                             
         RC(375) = 2.40D-17*0.11   
C
C     Reaction (376) OH + NUCARB12 = NRU12O2                                            
         RC(376) = 4.16D-11            
C
C     Reaction (377) OH + NOA = CARB6 + NO2                                             
         RC(377) = 6.70D-13            
C
C     Reaction (378) OH + UDCARB8 = C2H5O2                                              
         RC(378) = 5.20D-11*0.50        
C
C     Reaction (379) OH + UDCARB8 = ANHY + HO2                                          
         RC(379) = 5.20D-11*0.50        
C
C     Reaction (380) OH + UDCARB11 = RN10O2                                             
         RC(380) = 5.58D-11*0.55     
C
C     Reaction (381) OH + UDCARB11 = ANHY + CH3O2                                       
         RC(381) = 5.58D-11*0.45     
C
C     Reaction (382) OH + UDCARB14 = RN13O2                                             
         RC(382) = 7.00D-11*0.55     
C
C     Reaction (383) OH + UDCARB14 = ANHY + C2H5O2                                      
         RC(383) = 7.00D-11*0.45     
C
C     Reaction (384) OH + TNCARB26 = RTN26O2                                            
         RC(384) = 4.20D-11           
C
C     Reaction (385) OH + TNCARB15 = RN15AO2                                            
         RC(385) = 1.00D-12           
C
C     Reaction (386) OH + TNCARB10 = RTN10O2                                            
         RC(386) = 1.00D-10           
C
C     Reaction (387) NO3 + TNCARB26 = RTN26O2 + HNO3                                    
         RC(387) = 3.80D-14            
C
C     Reaction (388) NO3 + TNCARB10 = RTN10O2 + HNO3                                    
         RC(388) = KNO3AL*5.5      
C
C     Reaction (389) OH + RCOOH25 = RTN25O2                                             
         RC(389) = 6.65D-12            
C
C     Reaction (390) OH + TXCARB24 = RTX24O2                                            
         RC(390) = 1.55D-11           
C
C     Reaction (391) OH + TXCARB22 = RTX22O2                                            
         RC(391) = 4.55D-12           
C
C     Reaction (392) OH + CH3NO3 = HCHO + NO2                                           
         RC(392) = 1.00D-14*EXP(1060/TEMP)      
C
C     Reaction (393) OH + C2H5NO3 = CH3CHO + NO2                                        
         RC(393) = 4.40D-14*EXP(720/TEMP)       
C
C     Reaction (394) OH + RN10NO3 = C2H5CHO + NO2                                       
         RC(394) = 7.30D-13                     
C
C     Reaction (395) OH + IC3H7NO3 = CH3COCH3 + NO2                                     
         RC(395) = 4.90D-13                     
C
C     Reaction (396) OH + RN13NO3 = CARB11A + NO2                                       
         RC(396) = 9.20D-13                     
C
C     Reaction (397) OH + RN16NO3 = CARB14 + NO2                                        
         RC(397) = 1.85D-12                     
C
C     Reaction (398) OH + RN19NO3 = CARB17 + NO2                                        
         RC(398) = 3.02D-12                     
C
C     Reaction (399) OH + HOC2H4NO3 = HOCH2CHO + NO2                                    
         RC(399) = 1.09D-12               
C
C     Reaction (400) OH + RN9NO3 = CARB7 + NO2                                          
         RC(400) = 1.31D-12               
C
C     Reaction (401) OH + RN12NO3 = CARB10 + NO2                                        
         RC(401) = 1.79D-12               
C
C     Reaction (402) OH + RN15NO3 = CARB13 + NO2                                        
         RC(402) = 1.03D-11               
C
C     Reaction (403) OH + RN18NO3 = CARB16 + NO2                                        
         RC(403) = 1.34D-11               
C
C     Reaction (404) OH + RU14NO3 = UCARB12 + NO2                                       
         RC(404) = 3.00D-11*0.34               
C
C     Reaction (405) OH + RA13NO3 = CARB3 + UDCARB8 + NO2                               
         RC(405) = 7.30D-11               
C
C     Reaction (406) OH + RA16NO3 = CARB3 + UDCARB11 + NO2                              
         RC(406) = 7.16D-11               
C
C     Reaction (407) OH + RA19NO3 = CARB6 + UDCARB11 + NO2                              
         RC(407) = 8.31D-11               
C
C     Reaction (408) OH + RTN28NO3 = TNCARB26 + NO2                                     
         RC(408) = 4.35D-12               
C
C     Reaction (409) OH + RTN25NO3 = CH3COCH3 + TNCARB15 + NO2                          
         RC(409) = 2.88D-12               
C
C     Reaction (410) OH + RTX28NO3 = TXCARB24 + HCHO + NO2                              
         RC(410) = 3.53D-12                  
C
C     Reaction (411) OH + RTX24NO3 = TXCARB22 + NO2                                     
         RC(411) = 6.48D-12                  
C
C     Reaction (412) OH + RTX22NO3 = CH3COCH3 + CCARB12 + NO2                           
         RC(412) = 4.74D-12                  
C
C     Reaction (413) OH + AROH14 = RAROH14                                              
         RC(413) = 2.63D-11             
C
C     Reaction (414) NO3 + AROH14 = RAROH14 + HNO3                                      
         RC(414) = 3.78D-12               
C
C     Reaction (415) RAROH14 + NO2 = ARNOH14                                            
         RC(415) = 2.08D-12               
C
C     Reaction (416) OH + ARNOH14 = CARB13 + NO2                                        
         RC(416) = 9.00D-13               
C
C     Reaction (417) NO3 + ARNOH14 = CARB13 + NO2 + HNO3                                
         RC(417) = 9.00D-14               
C
C     Reaction (418) OH + AROH17 = RAROH17                                              
         RC(418) = 4.65D-11               
C
C     Reaction (419) NO3 + AROH17 = RAROH17 + HNO3                                      
         RC(419) = 1.25D-11               
C
C     Reaction (420) RAROH17 + NO2 = ARNOH17                                            
         RC(420) = 2.08D-12               
C
C     Reaction (421) OH + ARNOH17 = CARB16 + NO2                                        
         RC(421) = 1.53D-12               
C
C     Reaction (422) NO3 + ARNOH17 = CARB16 + NO2 + HNO3                                
         RC(422) = 3.13D-13               
C
C     Reaction (423) OH + CH3OOH = CH3O2                                                
         RC(423) = 1.90D-11*EXP(190/TEMP)       
C
C     Reaction (424) OH + CH3OOH = HCHO + OH                                            
         RC(424) = 1.00D-11*EXP(190/TEMP)       
C
C     Reaction (425) OH + C2H5OOH = CH3CHO + OH                                         
         RC(425) = 1.36D-11               
C
C     Reaction (426) OH + RN10OOH = C2H5CHO + OH                                        
         RC(426) = 1.89D-11               
C
C     Reaction (427) OH + IC3H7OOH = CH3COCH3 + OH                                      
         RC(427) = 2.78D-11               
C
C     Reaction (428) OH + RN13OOH = CARB11A + OH                                        
         RC(428) = 3.57D-11               
C
C     Reaction (429) OH + RN16OOH = CARB14 + OH                                         
         RC(429) = 4.21D-11               
C
C     Reaction (430) OH + RN19OOH = CARB17 + OH                                         
         RC(430) = 4.71D-11               
C
C     Reaction (431) OH + CH3CO3H = CH3CO3                                              
         RC(431) = 3.70D-12                     
C
C     Reaction (432) OH + C2H5CO3H = C2H5CO3                                            
         RC(432) = 4.42D-12                     
C
C     Reaction (433) OH + HOCH2CO3H = HOCH2CO3                                          
         RC(433) = 6.19D-12                     
C
C     Reaction (434) OH + RN8OOH = CARB6 + OH                                           
         RC(434) = 1.2D-11                     
C
C     Reaction (435) OH + RN11OOH = CARB9 + OH                                          
         RC(435) = 2.50D-11                     
C
C     Reaction (436) OH + RN14OOH = CARB12 + OH                                         
         RC(436) = 3.20D-11                     
C
C     Reaction (437) OH + RN17OOH = CARB15 + OH                                         
         RC(437) = 3.35D-11                     
C
C     Reaction (438) OH + RU14OOH = UCARB12 + OH                                        
         RC(438) = 7.51D-11                     
C
C     Reaction (439) OH + RU12OOH = RU12O2                                              
         RC(439) = 3.50D-11                     
C
C     Reaction (440) OH + RU10OOH = RU10O2                                              
         RC(440) = 3.50D-11                     
C
C     Reaction (441) OH + NRU14OOH = NUCARB12 + OH                                      
         RC(441) = 1.03D-10                     
C
C     Reaction (442) OH + NRU12OOH = NOA + CO + OH                                      
         RC(442) = 2.65D-11                     
C
C     Reaction (443) OH + HOC2H4OOH = HOCH2CHO + OH                                     
         RC(443) = 2.13D-11               
C
C     Reaction (444) OH + RN9OOH = CARB7 + OH                                           
         RC(444) = 2.50D-11               
C
C     Reaction (445) OH + RN12OOH = CARB10 + OH                                         
         RC(445) = 3.25D-11               
C
C     Reaction (446) OH + RN15OOH = CARB13 + OH                                         
         RC(446) = 3.74D-11               
C
C     Reaction (447) OH + RN18OOH = CARB16 + OH                                         
         RC(447) = 3.83D-11               
C
C     Reaction (448) OH + NRN6OOH = HCHO + HCHO + NO2 + OH                              
         RC(448) = 5.22D-12               
C
C     Reaction (449) OH + NRN9OOH = CH3CHO + HCHO + NO2 + OH                            
         RC(449) = 6.50D-12               
C
C     Reaction (450) OH + NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                         
         RC(450) = 7.15D-12               
C
C     Reaction (451) OH + RA13OOH = CARB3 + UDCARB8 + OH                                
         RC(451) = 9.77D-11               
C
C     Reaction (452) OH + RA16OOH = CARB3 + UDCARB11 + OH                               
         RC(452) = 9.64D-11               
C
C     Reaction (453) OH + RA19OOH = CARB6 + UDCARB11 + OH                               
         RC(453) = 1.12D-10               
C
C     Reaction (454) OH + RTN28OOH = TNCARB26 + OH                                      
         RC(454) = 2.38D-11               
C
C     Reaction (455) OH + RTN26OOH = RTN26O2                                            
         RC(455) = 1.20D-11               
C
C     Reaction (456) OH + NRTN28OOH = TNCARB26 + NO2 + OH                               
         RC(456) = 9.50D-12               
C
C     Reaction (457) OH + RTN25OOH = RTN25O2                                            
         RC(457) = 1.66D-11               
C
C     Reaction (458) OH + RTN24OOH = RTN24O2                                            
         RC(458) = 1.05D-11               
C
C     Reaction (459) OH + RTN23OOH = RTN23O2                                            
         RC(459) = 2.05D-11               
C
C     Reaction (460) OH + RTN14OOH = RTN14O2                                            
         RC(460) = 8.69D-11               
C
C     Reaction (461) OH + RTN10OOH = RTN10O2                                            
         RC(461) = 4.23D-12               
C
C     Reaction (462) OH + RTX28OOH = RTX28O2                                            
         RC(462) = 2.00D-11               
C
C     Reaction (463) OH + RTX24OOH = TXCARB22 + OH                                      
         RC(463) = 8.59D-11               
C
C     Reaction (464) OH + RTX22OOH = CH3COCH3 + CCARB12 + OH                            
         RC(464) = 7.50D-11               
C
C     Reaction (465) OH + NRTX28OOH = NRTX28O2                                          
         RC(465) = 9.58D-12               
C
C     Reaction (466) OH + ANHY = HOCH2CH2O2                                             
         RC(466) = 1.50D-12        
C
C     Reaction (467) CH3CO3 + NO2 = PAN                                                 
         RC(467) = KFPAN                        
C
C     Reaction (468) PAN = CH3CO3 + NO2                                                 
         RC(468) = KBPAN                        
C
C     Reaction (469) C2H5CO3 + NO2 = PPN                                                
         RC(469) = KFPAN                        
C
C     Reaction (470) PPN = C2H5CO3 + NO2                                                
         RC(470) = KBPAN                        
C
C     Reaction (471) HOCH2CO3 + NO2 = PHAN                                              
         RC(471) = 1.125D-11*(TEMP/300)**(-1.105)                       
C
C     Reaction (472) PHAN = HOCH2CO3 + NO2                                              
         RC(472) = 5.2D16*EXP(-13850/TEMP)                        
C
C     Reaction (473) OH + PAN = HCHO + CO + NO2                                         
         RC(473) = 3.00D-14      
C
C     Reaction (474) OH + PPN = CH3CHO + CO + NO2                                       
         RC(474) = 1.27D-12                       
C
C     Reaction (475) OH + PHAN = HCHO + CO + NO2                                        
         RC(475) = 1.12D-12                       
C
C     Reaction (476) RU12O2 + NO2 = RU12PAN                                             
         RC(476) = KFPAN*0.061             
C
C     Reaction (477) RU12PAN = RU12O2 + NO2                                             
         RC(477) = KBPAN                   
C
C     Reaction (478) RU10O2 + NO2 = MPAN                                                
         RC(478) = KFPAN*0.041             
C
C     Reaction (479) MPAN = RU10O2 + NO2                                                
         RC(479) = KBPAN                  
C
C     Reaction (480) OH + MPAN = CARB7 + CO + NO2                                       
         RC(480) = 2.90D-11*0.22 
C
C     Reaction (481) OH + RU12PAN = UCARB10 + NO2                                       
         RC(481) = 2.52D-11 
C
C     Reaction (482) RTN26O2 + NO2 = RTN26PAN                                           
         RC(482) = 1.125D-11*(TEMP/300)**(-1.105)*0.722      
C
C     Reaction (483) RTN26PAN = RTN26O2 + NO2                                           
         RC(483) = 5.2D16*EXP(-13850/TEMP)
C
C     Reaction (484) OH + RTN26PAN = CH3COCH3 + CARB16 + NO2                            
         RC(484) = 3.66D-12  
C
C     Reaction (485) RTN28NO3 = P2604                                                   
         RC(485) = KIN  		
C
C     Reaction (486) P2604 = RTN28NO3                                                   
         RC(486) = KOUT2604 	
C
C     Reaction (487) RTX28NO3 = P4608                                                   
         RC(487) = KIN 		
C
C     Reaction (488) P4608 = RTX28NO3                                                   
         RC(488) = KOUT4608 	
C
C     Reaction (489) RCOOH25 = P2631                                                    
         RC(489) = KIN  		
C
C     Reaction (490) P2631 = RCOOH25                                                    
         RC(490) = KOUT2631 	
C
C     Reaction (491) RTN24OOH = P2635                                                   
         RC(491) = KIN  		
C
C     Reaction (492) P2635 = RTN24OOH                                                   
         RC(492) = KOUT2635 	
C
C     Reaction (493) RTX28OOH = P4610                                                   
         RC(493) = KIN  		
C
C     Reaction (494) P4610 = RTX28OOH                                                   
         RC(494) = KOUT4610 	
C
C     Reaction (495) RTN28OOH = P2605                                                   
         RC(495) = KIN  		
C
C     Reaction (496) P2605 = RTN28OOH                                                   
         RC(496) = KOUT2605 	
C
C     Reaction (497) RTN26OOH = P2630                                                   
         RC(497) = KIN  		
C
C     Reaction (498) P2630 = RTN26OOH                                                   
         RC(498) = KOUT2630 
C
C     Reaction (499) RTN26PAN = P2629                                                   
         RC(499) = KIN  
C
C     Reaction (500) P2629 = RTN26PAN                                                   
         RC(500) = KOUT2629 
C
C     Reaction (501) RTN25OOH = P2632                                                   
         RC(501) = KIN 
C
C     Reaction (502) P2632 = RTN25OOH                                                   
         RC(502) = KOUT2632 
C
C     Reaction (503) RTN23OOH = P2637                                                   
         RC(503) = KIN  
C
C     Reaction (504) P2637 = RTN23OOH                                                   
         RC(504) = KOUT2637 
C
C     Reaction (505) ARNOH14 = P3612                                                    
         RC(505) = KIN  
C
C     Reaction (506) P3612 = ARNOH14                                                    
         RC(506) = KOUT3612 
C
C     Reaction (507) ARNOH17 = P3613                                                    
         RC(507) = KIN 
C
C     Reaction (508) P3613 = ARNOH17                                                    
         RC(508) = KOUT3613 
C
C     Reaction (509) ANHY = P3442                                                       
         RC(509) = KIN  
C
C     Reaction (510) P3442 = ANHY                                                       
         RC(510) = KOUT3442 
C

C
  999 RETURN
      END
C
C
      SUBROUTINE PHOTOL(J,DJ,BR01)
C----------------------------------------------------------------------
C-
C-   Purpose and Methods : CALCULATES RATE COEFFICIENTS
C-
C-   Inputs  : J,BR01
C-   Outputs : DJ
C-   Controls:

C----------------------------------------------------------------------
      IMPLICIT NONE
C----------------------------------------------------------------------
      DOUBLE PRECISION DJ(96),J(70),BR01
      INTEGER I
      DO 317 I=1,96
        DJ(I)=0.0
  317 CONTINUE


C     Photol Reaction (1) O3 = O1D                                                           
         DJ(1) = J(1)                             
C
C     Photol Reaction (2) O3 = O                                                             
         DJ(2) = J(2)                             
C
C     Photol Reaction (3) H2O2 = OH + OH                                                     
         DJ(3) = J(3)                             
C
C     Photol Reaction (4) NO2 = NO + O                                                       
         DJ(4) = J(4)                             
C
C     Photol Reaction (5) NO3 = NO                                                           
         DJ(5) = J(5)                             
C
C     Photol Reaction (6) NO3 = NO2 + O                                                      
         DJ(6) = J(6)                             
C
C     Photol Reaction (7) HONO = OH + NO                                                     
         DJ(7) = J(7)                             
C
C     Photol Reaction (8) HNO3 = OH + NO2                                                    
         DJ(8) = J(8)                             
C
C     Photol Reaction (9) HCHO = CO + HO2 + HO2                                              
         DJ(9) = J(11)                        
C
C     Photol Reaction (10) HCHO = H2 + CO                                                     
         DJ(10) = J(12)                        
C
C     Photol Reaction (11) CH3CHO = CH3O2 + HO2 + CO                                          
         DJ(11) = J(13)                        
C
C     Photol Reaction (12) C2H5CHO = C2H5O2 + CO + HO2                                        
         DJ(12) = J(14)                        
C
C     Photol Reaction (13) CH3COCH3 = CH3CO3 + CH3O2                                          
         DJ(13) = J(21)                        
C
C     Photol Reaction (14) MEK = CH3CO3 + C2H5O2                                              
         DJ(14) = J(22)                        
C
C     Photol Reaction (15) CARB14 = CH3CO3 + RN10O2                                           
         DJ(15) = J(22)*4.74               
C
C     Photol Reaction (16) CARB17 = RN8O2 + RN10O2                                            
         DJ(16) = J(22)*1.33               
C
C     Photol Reaction (17) CARB11A = CH3CO3 + C2H5O2                                          
         DJ(17) = J(22)                        
C
C     Photol Reaction (18) CARB7 = CH3CO3 + HCHO + HO2                                        
         DJ(18) = J(22)                        
C
C     Photol Reaction (19) CARB10 = CH3CO3 + CH3CHO + HO2                                     
         DJ(19) = J(22)                        
C
C     Photol Reaction (20) CARB13 = RN8O2 + CH3CHO + HO2                                      
         DJ(20) = J(22)*3.00               
C
C     Photol Reaction (21) CARB16 = RN8O2 + C2H5CHO + HO2                                     
         DJ(21) = J(22)*3.35               
C
C     Photol Reaction (22) HOCH2CHO = HCHO + CO + HO2 + HO2                                   
         DJ(22) = J(15)                        
C
C     Photol Reaction (23) UCARB10 = CH3CO3 + HCHO + HO2                                      
         DJ(23) = J(18)*2                       
C
C     Photol Reaction (24) CARB3 = CO + CO + HO2 + HO2                                        
         DJ(24) = J(33)                        
C
C     Photol Reaction (25) CARB6 = CH3CO3 + CO + HO2                                          
         DJ(25) = J(34)                        
C
C     Photol Reaction (26) CARB9 = CH3CO3 + CH3CO3                                            
         DJ(26) = J(35)                        
C
C     Photol Reaction (27) CARB12 = CH3CO3 + RN8O2                                            
         DJ(27) = J(35)                        
C
C     Photol Reaction (28) CARB15 = RN8O2 + RN8O2                                             
         DJ(28) = J(35)                        
C
C     Photol Reaction (29) UCARB12 = CH3CO3 + HOCH2CHO + CO + HO2                             
         DJ(29) = J(18)*2           
C
C     Photol Reaction (30) NUCARB12 = NOA + CO + CO + HO2 + HO2                               
         DJ(30) = J(18)             
C
C     Photol Reaction (31) NOA = CH3CO3 + HCHO + NO2                                          
         DJ(31) = J(56)             
C
C     Photol Reaction (32) NOA = CH3CO3 + HCHO + NO2                                          
         DJ(32) = J(57)             
C
C     Photol Reaction (33) UDCARB8 = C2H5O2 + HO2                                             
         DJ(33) = J(4)*0.02*0.64   
C
C     Photol Reaction (34) UDCARB8 = ANHY + HO2 + HO2                                         
         DJ(34) = J(4)*0.02*0.36   
C
C     Photol Reaction (35) UDCARB11 = RN10O2 + HO2                                            
         DJ(35) = J(4)*0.02*0.55   
C
C     Photol Reaction (36) UDCARB11 = ANHY + HO2 + CH3O2                                      
         DJ(36) = J(4)*0.02*0.45   
C
C     Photol Reaction (37) UDCARB14 = RN13O2 + HO2                                            
         DJ(37) = J(4)*0.02*0.55   
C
C     Photol Reaction (38) UDCARB14 = ANHY + HO2 + C2H5O2                                     
         DJ(38) = J(4)*0.02*0.45   
C
C     Photol Reaction (39) TNCARB26 = RTN26O2 + HO2                                           
         DJ(39) = J(15)             
C
C     Photol Reaction (40) TNCARB10 = CH3CO3 + CH3CO3 + CO                                    
         DJ(40) = J(35)*0.5        
C
C     Photol Reaction (41) CH3NO3 = HCHO + HO2 + NO2                                          
         DJ(41) = J(51)                        
C
C     Photol Reaction (42) C2H5NO3 = CH3CHO + HO2 + NO2                                       
         DJ(42) = J(52)                        
C
C     Photol Reaction (43) RN10NO3 = C2H5CHO + HO2 + NO2                                      
         DJ(43) = J(53)                        
C
C     Photol Reaction (44) IC3H7NO3 = CH3COCH3 + HO2 + NO2                                    
         DJ(44) = J(54)                        
C
C     Photol Reaction (45) RN13NO3 =  CH3CHO + C2H5O2 + NO2                                   
         DJ(45) = J(53)*BR01               
C
C     Photol Reaction (46) RN13NO3 =  CARB11A + HO2 + NO2                                     
         DJ(46) = J(53)*(1-BR01)           
C
C     Photol Reaction (47) RN16NO3 = RN15O2 + NO2                                             
         DJ(47) = J(53)                        
C
C     Photol Reaction (48) RN19NO3 = RN18O2 + NO2                                             
         DJ(48) = J(53)                        
C
C     Photol Reaction (49) RA13NO3 = CARB3 + UDCARB8 + HO2 + NO2                              
         DJ(49) = J(54)                    
C
C     Photol Reaction (50) RA16NO3 = CARB3 + UDCARB11 + HO2 + NO2                             
         DJ(50) = J(54)                    
C
C     Photol Reaction (51) RA19NO3 = CARB6 + UDCARB11 + HO2 + NO2                             
         DJ(51) = J(54)                    
C
C     Photol Reaction (52) RTX24NO3 = TXCARB22 + HO2 + NO2                                    
         DJ(52) = J(54)                    
C
C     Photol Reaction (53) CH3OOH = HCHO + HO2 + OH                                           
         DJ(53) = J(41)                        
C
C     Photol Reaction (54) C2H5OOH = CH3CHO + HO2 + OH                                        
         DJ(54) = J(41)                        
C
C     Photol Reaction (55) RN10OOH = C2H5CHO + HO2 + OH                                       
         DJ(55) = J(41)                        
C
C     Photol Reaction (56) IC3H7OOH = CH3COCH3 + HO2 + OH                                     
         DJ(56) = J(41)                        
C
C     Photol Reaction (57) RN13OOH =  CH3CHO + C2H5O2 + OH                                    
         DJ(57) = J(41)*BR01         
C
C     Photol Reaction (58) RN13OOH =  CARB11A + HO2 + OH                                      
         DJ(58) = J(41)*(1-BR01)     
C
C     Photol Reaction (59) RN16OOH = RN15AO2 + OH                                             
         DJ(59) = J(41)                        
C
C     Photol Reaction (60) RN19OOH = RN18AO2 + OH                                             
         DJ(60) = J(41)                        
C
C     Photol Reaction (61) CH3CO3H = CH3O2 + OH                                               
         DJ(61) = J(41)                        
C
C     Photol Reaction (62) C2H5CO3H = C2H5O2 + OH                                             
         DJ(62) = J(41)                        
C
C     Photol Reaction (63) HOCH2CO3H = HCHO + HO2 + OH                                        
         DJ(63) = J(41)                        
C
C     Photol Reaction (64) RN8OOH = C2H5O2 + OH                                               
         DJ(64) = J(41)                        
C
C     Photol Reaction (65) RN11OOH = RN10O2 + OH                                              
         DJ(65) = J(41)                        
C
C     Photol Reaction (66) RN14OOH = RN13O2 + OH                                              
         DJ(66) = J(41)                        
C
C     Photol Reaction (67) RN17OOH = RN16O2 + OH                                              
         DJ(67) = J(41)                        
C
C     Photol Reaction (68) RU14OOH = UCARB12 + HO2 + OH                                       
         DJ(68) = J(41)*0.252              
C
C     Photol Reaction (69) RU14OOH = UCARB10 + HCHO + HO2 + OH                                
         DJ(69) = J(41)*0.748              
C
C     Photol Reaction (70) RU12OOH = CARB6 + HOCH2CHO + HO2 + OH                              
         DJ(70) = J(41)                   
C
C     Photol Reaction (71) RU10OOH = CH3CO3 + HOCH2CHO + OH                                   
         DJ(71) = J(41)                   
C
C     Photol Reaction (72) NRU14OOH = NUCARB12 + HO2 + OH                                     
         DJ(72) = J(41)                   
C
C     Photol Reaction (73) NRU12OOH = NOA + CO + HO2 + OH                                     
         DJ(73) = J(41)                   
C
C     Photol Reaction (74) HOC2H4OOH = HCHO + HCHO + HO2 + OH                                 
         DJ(74) = J(41)                  
C
C     Photol Reaction (75) RN9OOH = CH3CHO + HCHO + HO2 + OH                                  
         DJ(75) = J(41)                  
C
C     Photol Reaction (76) RN12OOH = CH3CHO + CH3CHO + HO2 + OH                               
         DJ(76) = J(41)                  
C
C     Photol Reaction (77) RN15OOH = C2H5CHO + CH3CHO + HO2 + OH                              
         DJ(77) = J(41)                  
C
C     Photol Reaction (78) RN18OOH = C2H5CHO + C2H5CHO + HO2 + OH                             
         DJ(78) = J(41)                 
C
C     Photol Reaction (79) NRN6OOH = HCHO + HCHO + NO2 + OH                                   
         DJ(79) = J(41)                  
C
C     Photol Reaction (80) NRN9OOH = CH3CHO + HCHO + NO2 + OH                                 
         DJ(80) = J(41)                  
C
C     Photol Reaction (81) NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                              
         DJ(81) = J(41)                  
C
C     Photol Reaction (82) RA13OOH = CARB3 + UDCARB8 + HO2 + OH                               
         DJ(82) = J(41)                  
C
C     Photol Reaction (83) RA16OOH = CARB3 + UDCARB11 + HO2 + OH                              
         DJ(83) = J(41)                  
C
C     Photol Reaction (84) RA19OOH = CARB6 + UDCARB11 + HO2 + OH                              
         DJ(84) = J(41)                  
C
C     Photol Reaction (85) RTN28OOH = TNCARB26 + HO2 + OH                                     
         DJ(85) = J(41)                  
C
C     Photol Reaction (86) NRTN28OOH = TNCARB26 + NO2 + OH                                    
         DJ(86) = J(41)                  
C
C     Photol Reaction (87) RTN26OOH = RTN25O2 + OH                                            
         DJ(87) = J(41)             
C
C     Photol Reaction (88) RTN25OOH = RTN24O2 + OH                                            
         DJ(88) = J(41)             
C
C     Photol Reaction (89) RTN24OOH = RTN23O2 + OH                                            
         DJ(89) = J(41)             
C
C     Photol Reaction (90) RTN23OOH = CH3COCH3 + RTN14O2 + OH                                 
         DJ(90) = J(41)             
C
C     Photol Reaction (91) RTN14OOH = TNCARB10 + HCHO + HO2 + OH                              
         DJ(91) = J(41)             
C
C     Photol Reaction (92) RTN10OOH = RN8O2 + CO + OH                                         
         DJ(92) = J(41)             
C
C     Photol Reaction (93) RTX28OOH = TXCARB24 + HCHO + HO2 + OH                              
         DJ(93) = J(41)                  
C
C     Photol Reaction (94) RTX24OOH = TXCARB22 + HO2 + OH                                     
         DJ(94) = J(41)                  
C
C     Photol Reaction (95) RTX22OOH = CH3COCH3 + RN13O2 + OH                                  
         DJ(95) = J(41)                  
C
C     Photol Reaction (96) NRTX28OOH = TXCARB24 + HCHO + NO2 + OH                             
         DJ(96) = J(41)                  
C
         RETURN
         END


C#######################################################################

      SUBROUTINE ZENITH(COSZEN,TTIME,FYEAR,SECYEAR,
     & ZENNOW,LONGRAD,LATRAD,XYEAR)

      DOUBLE PRECISION LAT,PI,RADIAN,DEC,LHA,TTIME,THETA,COSX,SECX
      DOUBLE PRECISION sinld,cosld,FYEAR,XYEAR,SECYEAR,ZENNOW,COSZEN
      DOUBLE PRECISION LONGRAD, FSUN,LATRAD
C SOLAR DECLINATION ANGLE FROM JULY 1ST - HARWELL TRAJ MODEL
 
C LATITUDE
      LAT = (inLAT-1)*5-87.5
      PI = 4.0*ATAN(1.0)
      RADIAN  = 1.80E+02/PI
      XYEAR  = FYEAR+(TTIME/SECYEAR) 
C                                                                     
      FSUN   = 1.00D+00 + (3.40D-02*COS(2.00D+00*PI*FYEAR)) 
C                                                                     
C Calculate declination                                               
C                                                                     
      DEC    = -4.1420D-01*COS(2.00D+00*PI*XYEAR) 
C                                                                     
C Calculate local hour angle from longitude                           
C                                                                     
      LHA    = (1.00D+00+TTIME/4.32D+04)*PI+LONGRAD 
C                                                                     
C Calculate solar zenith angle                                        
C                                                                     
      COSLD   = COS(LATRAD)*COS(DEC) 
      SINLD   = SIN(LATRAD)*SIN(DEC) 
      COSZEN  = (COS(LHA)*COSLD)+SINLD 
C                                                                     
      ZENNOW  = RADIAN*ATAN((SQRT(1.00D+00-COSZEN**2))/COSZEN) 
C                                                                     
C Correct solar zenith angle if zenith angle is negative              
C          
C                                                                     
      IF(ZENNOW.LT.0) THEN                                              
      ZENNOW = 1.80E+02 + ZENNOW                                        
      END IF

      RETURN
      END

C**************** END OF SUBROUTINE ZENITH *****************
