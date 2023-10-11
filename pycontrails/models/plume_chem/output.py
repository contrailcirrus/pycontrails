#          O1D              Y.sel(species="O1D") 
        P = EM.sel(species="O1D") \
        + (DJ.sel(photol_coeffs=1) * Y.sel(species="O3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=7)) + (RC.sel(therm_coeffs=8)) + (RC.sel(therm_coeffs=16) *  H2O) 
        Y.loc[:, :, :, "O1D"] = P/L
        
#          O                Y.sel(species="O")
        P = EM.sel(species="O") \
        + (DJ.sel(photol_coeffs=6) * Y.sel(species="NO3")) \
        + (DJ.sel(photol_coeffs=2) * Y.sel(species="O3")) + (DJ.sel(photol_coeffs=4) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=7) * Y.sel(species="O1D")) + (RC.sel(therm_coeffs=8) * Y.sel(species="O1D")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=36) * Y.sel(species="SO2")) \
        + (RC.sel(therm_coeffs=4) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=5) * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=6) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=1)) + (RC.sel(therm_coeffs=2)) + (RC.sel(therm_coeffs=3) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "O"] = P/L
        
#          OH               Y.sel(species="OH") 
        P = EM.sel(species="OH") \
        + (DJ.sel(photol_coeffs=95) * Y.sel(species="RTX22OOH")) + (DJ.sel(photol_coeffs=96) * Y.sel(species="NRTX28OOH")) \
        + (DJ.sel(photol_coeffs=93) * Y.sel(species="RTX28OOH")) + (DJ.sel(photol_coeffs=94) * Y.sel(species="RTX24OOH")) \
        + (DJ.sel(photol_coeffs=91) * Y.sel(species="RTN14OOH")) + (DJ.sel(photol_coeffs=92) * Y.sel(species="RTN10OOH")) \
        + (DJ.sel(photol_coeffs=89) * Y.sel(species="RTN24OOH")) + (DJ.sel(photol_coeffs=90) * Y.sel(species="RTN23OOH")) \
        + (DJ.sel(photol_coeffs=87) * Y.sel(species="RTN26OOH")) + (DJ.sel(photol_coeffs=88) * Y.sel(species="RTN25OOH")) \
        + (DJ.sel(photol_coeffs=85) * Y.sel(species="RTN28OOH")) + (DJ.sel(photol_coeffs=86) * Y.sel(species="NRTN28OOH")) \
        + (DJ.sel(photol_coeffs=83) * Y.sel(species="RA16OOH")) + (DJ.sel(photol_coeffs=84) * Y.sel(species="RA19OOH")) \
        + (DJ.sel(photol_coeffs=81) * Y.sel(species="NRN12OOH")) + (DJ.sel(photol_coeffs=82) * Y.sel(species="RA13OOH")) \
        + (DJ.sel(photol_coeffs=79) * Y.sel(species="NRN6OOH")) + (DJ.sel(photol_coeffs=80) * Y.sel(species="NRN9OOH")) \
        + (DJ.sel(photol_coeffs=77) * Y.sel(species="RN15OOH")) + (DJ.sel(photol_coeffs=78) * Y.sel(species="RN18OOH")) \
        + (DJ.sel(photol_coeffs=75) * Y.sel(species="RN9OOH")) + (DJ.sel(photol_coeffs=76) * Y.sel(species="RN12OOH")) \
        + (DJ.sel(photol_coeffs=73) * Y.sel(species="NRU12OOH")) + (DJ.sel(photol_coeffs=74) * Y.sel(species="HOC2H4OOH")) \
        + (DJ.sel(photol_coeffs=71) * Y.sel(species="RU10OOH")) + (DJ.sel(photol_coeffs=72) * Y.sel(species="NRU14OOH")) \
        + (DJ.sel(photol_coeffs=69) * Y.sel(species="RU14OOH")) + (DJ.sel(photol_coeffs=70) * Y.sel(species="RU12OOH")) \
        + (DJ.sel(photol_coeffs=67) * Y.sel(species="RN17OOH")) + (DJ.sel(photol_coeffs=68) * Y.sel(species="RU14OOH")) \
        + (DJ.sel(photol_coeffs=65) * Y.sel(species="RN11OOH")) + (DJ.sel(photol_coeffs=66) * Y.sel(species="RN14OOH")) \
        + (DJ.sel(photol_coeffs=63) * Y.sel(species="HOCH2CO3H")) + (DJ.sel(photol_coeffs=64) * Y.sel(species="RN8OOH")) \
        + (DJ.sel(photol_coeffs=61) * Y.sel(species="CH3CO3H")) + (DJ.sel(photol_coeffs=62) * Y.sel(species="C2H5CO3H")) \
        + (DJ.sel(photol_coeffs=59) * Y.sel(species="RN16OOH")) + (DJ.sel(photol_coeffs=60) * Y.sel(species="RN19OOH")) \
        + (DJ.sel(photol_coeffs=57) * Y.sel(species="RN13OOH")) + (DJ.sel(photol_coeffs=58) * Y.sel(species="RN13OOH")) \
        + (DJ.sel(photol_coeffs=55) * Y.sel(species="RN10OOH")) + (DJ.sel(photol_coeffs=56) * Y.sel(species="IC3H7OOH")) \
        + (DJ.sel(photol_coeffs=53) * Y.sel(species="CH3OOH")) + (DJ.sel(photol_coeffs=54) * Y.sel(species="C2H5OOH")) \
        + (DJ.sel(photol_coeffs=7) * Y.sel(species="HONO")) + (DJ.sel(photol_coeffs=8) * Y.sel(species="HNO3")) \
        + (RC.sel(therm_coeffs=464) * Y.sel(species="OH") * Y.sel(species="RTX22OOH")) + (DJ.sel(photol_coeffs=3) * Y.sel(species="H2O2") * 2.00) \
        + (RC.sel(therm_coeffs=456) * Y.sel(species="OH") * Y.sel(species="NRTN28OOH")) + (RC.sel(therm_coeffs=463) * Y.sel(species="OH") * Y.sel(species="RTX24OOH")) \
        + (RC.sel(therm_coeffs=453) * Y.sel(species="OH") * Y.sel(species="RA19OOH")) + (RC.sel(therm_coeffs=454) * Y.sel(species="OH") * Y.sel(species="RTN28OOH")) \
        + (RC.sel(therm_coeffs=451) * Y.sel(species="OH") * Y.sel(species="RA13OOH")) + (RC.sel(therm_coeffs=452) * Y.sel(species="OH") * Y.sel(species="RA16OOH")) \
        + (RC.sel(therm_coeffs=449) * Y.sel(species="OH") * Y.sel(species="NRN9OOH")) + (RC.sel(therm_coeffs=450) * Y.sel(species="OH") * Y.sel(species="NRN12OOH")) \
        + (RC.sel(therm_coeffs=447) * Y.sel(species="OH") * Y.sel(species="RN18OOH")) + (RC.sel(therm_coeffs=448) * Y.sel(species="OH") * Y.sel(species="NRN6OOH")) \
        + (RC.sel(therm_coeffs=445) * Y.sel(species="OH") * Y.sel(species="RN12OOH")) + (RC.sel(therm_coeffs=446) * Y.sel(species="OH") * Y.sel(species="RN15OOH")) \
        + (RC.sel(therm_coeffs=443) * Y.sel(species="OH") * Y.sel(species="HOC2H4OOH")) + (RC.sel(therm_coeffs=444) * Y.sel(species="OH") * Y.sel(species="RN9OOH")) \
        + (RC.sel(therm_coeffs=441) * Y.sel(species="OH") * Y.sel(species="NRU14OOH")) + (RC.sel(therm_coeffs=442) * Y.sel(species="OH") * Y.sel(species="NRU12OOH")) \
        + (RC.sel(therm_coeffs=437) * Y.sel(species="OH") * Y.sel(species="RN17OOH")) + (RC.sel(therm_coeffs=438) * Y.sel(species="OH") * Y.sel(species="RU14OOH")) \
        + (RC.sel(therm_coeffs=435) * Y.sel(species="OH") * Y.sel(species="RN11OOH")) + (RC.sel(therm_coeffs=436) * Y.sel(species="OH") * Y.sel(species="RN14OOH")) \
        + (RC.sel(therm_coeffs=430) * Y.sel(species="OH") * Y.sel(species="RN19OOH")) + (RC.sel(therm_coeffs=434) * Y.sel(species="OH") * Y.sel(species="RN8OOH")) \
        + (RC.sel(therm_coeffs=428) * Y.sel(species="OH") * Y.sel(species="RN13OOH")) + (RC.sel(therm_coeffs=429) * Y.sel(species="OH") * Y.sel(species="RN16OOH")) \
        + (RC.sel(therm_coeffs=426) * Y.sel(species="OH") * Y.sel(species="RN10OOH")) + (RC.sel(therm_coeffs=427) * Y.sel(species="OH") * Y.sel(species="IC3H7OOH")) \
        + (RC.sel(therm_coeffs=424) * Y.sel(species="OH") * Y.sel(species="CH3OOH")) + (RC.sel(therm_coeffs=425) * Y.sel(species="OH") * Y.sel(species="C2H5OOH")) \
        + (RC.sel(therm_coeffs=362) * Y.sel(species="O3") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=374) * Y.sel(species="O3") * Y.sel(species="UCARB12")) \
        + (RC.sel(therm_coeffs=70) * Y.sel(species="BPINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=75) * Y.sel(species="C2H2") * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=61) * Y.sel(species="O3") * Y.sel(species="C5H8")) + (RC.sel(therm_coeffs=65) * Y.sel(species="APINENE") * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=55) * Y.sel(species="O3") * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=57) * Y.sel(species="O3") * Y.sel(species="TBUT2ENE")) \
        + (RC.sel(therm_coeffs=33) * Y.sel(species="HO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=53) * Y.sel(species="O3") * Y.sel(species="C2H4")) \
        + (RC.sel(therm_coeffs=21) * Y.sel(species="HO2") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=29) * Y.sel(species="HO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=16) * Y.sel(species="O1D") *  H2O*2.00) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=481) * Y.sel(species="RU12PAN")) + (RC.sel(therm_coeffs=484) * Y.sel(species="RTN26PAN")) \
        + (RC.sel(therm_coeffs=474) * Y.sel(species="PPN")) + (RC.sel(therm_coeffs=475) * Y.sel(species="PHAN")) + (RC.sel(therm_coeffs=480) * Y.sel(species="MPAN")) \
        + (RC.sel(therm_coeffs=465) * Y.sel(species="NRTX28OOH")) + (RC.sel(therm_coeffs=466) * Y.sel(species="ANHY")) + (RC.sel(therm_coeffs=473) * Y.sel(species="PAN")) \
        + (RC.sel(therm_coeffs=462) * Y.sel(species="RTX28OOH")) + (RC.sel(therm_coeffs=463) * Y.sel(species="RTX24OOH")) + (RC.sel(therm_coeffs=464) * Y.sel(species="RTX22OOH")) \
        + (RC.sel(therm_coeffs=459) * Y.sel(species="RTN23OOH")) + (RC.sel(therm_coeffs=460) * Y.sel(species="RTN14OOH")) + (RC.sel(therm_coeffs=461) * Y.sel(species="RTN10OOH")) \
        + (RC.sel(therm_coeffs=456) * Y.sel(species="NRTN28OOH")) + (RC.sel(therm_coeffs=457) * Y.sel(species="RTN25OOH")) + (RC.sel(therm_coeffs=458) * Y.sel(species="RTN24OOH")) \
        + (RC.sel(therm_coeffs=453) * Y.sel(species="RA19OOH")) + (RC.sel(therm_coeffs=454) * Y.sel(species="RTN28OOH")) + (RC.sel(therm_coeffs=455) * Y.sel(species="RTN26OOH")) \
        + (RC.sel(therm_coeffs=450) * Y.sel(species="NRN12OOH")) + (RC.sel(therm_coeffs=451) * Y.sel(species="RA13OOH")) + (RC.sel(therm_coeffs=452) * Y.sel(species="RA16OOH")) \
        + (RC.sel(therm_coeffs=447) * Y.sel(species="RN18OOH")) + (RC.sel(therm_coeffs=448) * Y.sel(species="NRN6OOH")) + (RC.sel(therm_coeffs=449) * Y.sel(species="NRN9OOH")) \
        + (RC.sel(therm_coeffs=444) * Y.sel(species="RN9OOH")) + (RC.sel(therm_coeffs=445) * Y.sel(species="RN12OOH")) + (RC.sel(therm_coeffs=446) * Y.sel(species="RN15OOH")) \
        + (RC.sel(therm_coeffs=441) * Y.sel(species="NRU14OOH")) + (RC.sel(therm_coeffs=442) * Y.sel(species="NRU12OOH")) + (RC.sel(therm_coeffs=443) * Y.sel(species="HOC2H4OOH")) \
        + (RC.sel(therm_coeffs=438) * Y.sel(species="RU14OOH")) + (RC.sel(therm_coeffs=439) * Y.sel(species="RU12OOH")) + (RC.sel(therm_coeffs=440) * Y.sel(species="RU10OOH")) \
        + (RC.sel(therm_coeffs=435) * Y.sel(species="RN11OOH")) + (RC.sel(therm_coeffs=436) * Y.sel(species="RN14OOH")) + (RC.sel(therm_coeffs=437) * Y.sel(species="RN17OOH")) \
        + (RC.sel(therm_coeffs=432) * Y.sel(species="C2H5CO3H")) + (RC.sel(therm_coeffs=433) * Y.sel(species="HOCH2CO3H")) + (RC.sel(therm_coeffs=434) * Y.sel(species="RN8OOH")) \
        + (RC.sel(therm_coeffs=429) * Y.sel(species="RN16OOH")) + (RC.sel(therm_coeffs=430) * Y.sel(species="RN19OOH")) + (RC.sel(therm_coeffs=431) * Y.sel(species="CH3CO3H")) \
        + (RC.sel(therm_coeffs=426) * Y.sel(species="RN10OOH")) + (RC.sel(therm_coeffs=427) * Y.sel(species="IC3H7OOH")) + (RC.sel(therm_coeffs=428) * Y.sel(species="RN13OOH")) \
        + (RC.sel(therm_coeffs=423) * Y.sel(species="CH3OOH")) + (RC.sel(therm_coeffs=424) * Y.sel(species="CH3OOH")) + (RC.sel(therm_coeffs=425) * Y.sel(species="C2H5OOH")) \
        + (RC.sel(therm_coeffs=416) * Y.sel(species="ARNOH14")) + (RC.sel(therm_coeffs=418) * Y.sel(species="AROH17")) + (RC.sel(therm_coeffs=421) * Y.sel(species="ARNOH17")) \
        + (RC.sel(therm_coeffs=411) * Y.sel(species="RTX24NO3")) + (RC.sel(therm_coeffs=412) * Y.sel(species="RTX22NO3")) + (RC.sel(therm_coeffs=413) * Y.sel(species="AROH14")) \
        + (RC.sel(therm_coeffs=408) * Y.sel(species="RTN28NO3")) + (RC.sel(therm_coeffs=409) * Y.sel(species="RTN25NO3")) + (RC.sel(therm_coeffs=410) * Y.sel(species="RTX28NO3")) \
        + (RC.sel(therm_coeffs=405) * Y.sel(species="RA13NO3")) + (RC.sel(therm_coeffs=406) * Y.sel(species="RA16NO3")) + (RC.sel(therm_coeffs=407) * Y.sel(species="RA19NO3")) \
        + (RC.sel(therm_coeffs=402) * Y.sel(species="RN15NO3")) + (RC.sel(therm_coeffs=403) * Y.sel(species="RN18NO3")) + (RC.sel(therm_coeffs=404) * Y.sel(species="RU14NO3")) \
        + (RC.sel(therm_coeffs=399) * Y.sel(species="HOC2H4NO3")) + (RC.sel(therm_coeffs=400) * Y.sel(species="RN9NO3")) + (RC.sel(therm_coeffs=401) * Y.sel(species="RN12NO3")) \
        + (RC.sel(therm_coeffs=396) * Y.sel(species="RN13NO3")) + (RC.sel(therm_coeffs=397) * Y.sel(species="RN16NO3")) + (RC.sel(therm_coeffs=398) * Y.sel(species="RN19NO3")) \
        + (RC.sel(therm_coeffs=393) * Y.sel(species="C2H5NO3")) + (RC.sel(therm_coeffs=394) * Y.sel(species="RN10NO3")) + (RC.sel(therm_coeffs=395) * Y.sel(species="IC3H7NO3")) \
        + (RC.sel(therm_coeffs=390) * Y.sel(species="TXCARB24")) + (RC.sel(therm_coeffs=391) * Y.sel(species="TXCARB22")) + (RC.sel(therm_coeffs=392) * Y.sel(species="CH3NO3")) \
        + (RC.sel(therm_coeffs=385) * Y.sel(species="TNCARB15")) + (RC.sel(therm_coeffs=386) * Y.sel(species="TNCARB10")) + (RC.sel(therm_coeffs=389) * Y.sel(species="RCOOH25")) \
        + (RC.sel(therm_coeffs=382) * Y.sel(species="UDCARB14")) + (RC.sel(therm_coeffs=383) * Y.sel(species="UDCARB14")) + (RC.sel(therm_coeffs=384) * Y.sel(species="TNCARB26")) \
        + (RC.sel(therm_coeffs=379) * Y.sel(species="UDCARB8")) + (RC.sel(therm_coeffs=380) * Y.sel(species="UDCARB11")) + (RC.sel(therm_coeffs=381) * Y.sel(species="UDCARB11")) \
        + (RC.sel(therm_coeffs=376) * Y.sel(species="NUCARB12")) + (RC.sel(therm_coeffs=377) * Y.sel(species="NOA")) + (RC.sel(therm_coeffs=378) * Y.sel(species="UDCARB8")) \
        + (RC.sel(therm_coeffs=370) * Y.sel(species="CARB15")) + (RC.sel(therm_coeffs=371) * Y.sel(species="CCARB12")) + (RC.sel(therm_coeffs=372) * Y.sel(species="UCARB12")) \
        + (RC.sel(therm_coeffs=367) * Y.sel(species="CARB6")) + (RC.sel(therm_coeffs=368) * Y.sel(species="CARB9")) + (RC.sel(therm_coeffs=369) * Y.sel(species="CARB12")) \
        + (RC.sel(therm_coeffs=360) * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=364) * Y.sel(species="HOCH2CHO")) + (RC.sel(therm_coeffs=366) * Y.sel(species="CARB3")) \
        + (RC.sel(therm_coeffs=357) * Y.sel(species="CARB10")) + (RC.sel(therm_coeffs=358) * Y.sel(species="CARB13")) + (RC.sel(therm_coeffs=359) * Y.sel(species="CARB16")) \
        + (RC.sel(therm_coeffs=354) * Y.sel(species="CARB17")) + (RC.sel(therm_coeffs=355) * Y.sel(species="CARB11A")) + (RC.sel(therm_coeffs=356) * Y.sel(species="CARB7")) \
        + (RC.sel(therm_coeffs=105) * Y.sel(species="CDICLETH")) + (RC.sel(therm_coeffs=106) * Y.sel(species="TDICLETH")) + (RC.sel(therm_coeffs=353) * Y.sel(species="CARB14")) \
        + (RC.sel(therm_coeffs=102) * Y.sel(species="CH3CCL3")) + (RC.sel(therm_coeffs=103) * Y.sel(species="TCE")) + (RC.sel(therm_coeffs=104) * Y.sel(species="TRICLETH")) \
        + (RC.sel(therm_coeffs=99) * Y.sel(species="CH3CL")) + (RC.sel(therm_coeffs=100) * Y.sel(species="CH2CL2")) + (RC.sel(therm_coeffs=101) * Y.sel(species="CHCL3")) \
        + (RC.sel(therm_coeffs=96) * Y.sel(species="IPROPOL")) + (RC.sel(therm_coeffs=97) * Y.sel(species="HCOOH")) + (RC.sel(therm_coeffs=98) * Y.sel(species="CH3CO2H")) \
        + (RC.sel(therm_coeffs=93) * Y.sel(species="NPROPOL")) + (RC.sel(therm_coeffs=94) * Y.sel(species="NPROPOL")) + (RC.sel(therm_coeffs=95) * Y.sel(species="IPROPOL")) \
        + (RC.sel(therm_coeffs=90) * Y.sel(species="CH3OH")) + (RC.sel(therm_coeffs=91) * Y.sel(species="C2H5OH")) + (RC.sel(therm_coeffs=92) * Y.sel(species="C2H5OH")) \
        + (RC.sel(therm_coeffs=84) * Y.sel(species="C2H5CHO")) + (RC.sel(therm_coeffs=88) * Y.sel(species="CH3COCH3")) + (RC.sel(therm_coeffs=89) * Y.sel(species="MEK")) \
        + (RC.sel(therm_coeffs=81) * Y.sel(species="OXYL")) + (RC.sel(therm_coeffs=82) * Y.sel(species="HCHO")) + (RC.sel(therm_coeffs=83) * Y.sel(species="CH3CHO")) \
        + (RC.sel(therm_coeffs=78) * Y.sel(species="TOLUENE")) + (RC.sel(therm_coeffs=79) * Y.sel(species="TOLUENE")) + (RC.sel(therm_coeffs=80) * Y.sel(species="OXYL")) \
        + (RC.sel(therm_coeffs=75) * Y.sel(species="C2H2")) + (RC.sel(therm_coeffs=76) * Y.sel(species="BENZENE")) + (RC.sel(therm_coeffs=77) * Y.sel(species="BENZENE")) \
        + (RC.sel(therm_coeffs=63) * Y.sel(species="APINENE")) + (RC.sel(therm_coeffs=68) * Y.sel(species="BPINENE")) + (RC.sel(therm_coeffs=74) * Y.sel(species="C2H2")) \
        + (RC.sel(therm_coeffs=48) * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=49) * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=59) * Y.sel(species="C5H8")) \
        + (RC.sel(therm_coeffs=45) * Y.sel(species="C3H8")) + (RC.sel(therm_coeffs=46) * Y.sel(species="NC4H10")) + (RC.sel(therm_coeffs=47) * Y.sel(species="C2H4")) \
        + (RC.sel(therm_coeffs=42) * Y.sel(species="CH4")) + (RC.sel(therm_coeffs=43) * Y.sel(species="C2H6")) + (RC.sel(therm_coeffs=44) * Y.sel(species="C3H8")) \
        + (RC.sel(therm_coeffs=34) * Y.sel(species="HONO")) + (RC.sel(therm_coeffs=35) * Y.sel(species="HNO3")) + (RC.sel(therm_coeffs=37) * Y.sel(species="SO2")) \
        + (RC.sel(therm_coeffs=27) * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=28) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=32) * Y.sel(species="HO2NO2")) \
        + (RC.sel(therm_coeffs=20) * Y.sel(species="H2O2")) + (RC.sel(therm_coeffs=22) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=25) * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=17) * Y.sel(species="O3")) + (RC.sel(therm_coeffs=18) * Y.sel(species="H2")) + (RC.sel(therm_coeffs=19) * Y.sel(species="CO")) 
        Y.loc[:, :, :, "OH"] = P/L

        #          NO2              Y.sel(species="NO2") 
        P = EM.sel(species="NO2") \
        + (DJ.sel(photol_coeffs=86) * Y.sel(species="NRTN28OOH")) + (DJ.sel(photol_coeffs=96) * Y.sel(species="NRTX28OOH")) \
        + (DJ.sel(photol_coeffs=80) * Y.sel(species="NRN9OOH")) + (DJ.sel(photol_coeffs=81) * Y.sel(species="NRN12OOH")) \
        + (DJ.sel(photol_coeffs=52) * Y.sel(species="RTX24NO3")) + (DJ.sel(photol_coeffs=79) * Y.sel(species="NRN6OOH")) \
        + (DJ.sel(photol_coeffs=50) * Y.sel(species="RA16NO3")) + (DJ.sel(photol_coeffs=51) * Y.sel(species="RA19NO3")) \
        + (DJ.sel(photol_coeffs=48) * Y.sel(species="RN19NO3")) + (DJ.sel(photol_coeffs=49) * Y.sel(species="RA13NO3")) \
        + (DJ.sel(photol_coeffs=46) * Y.sel(species="RN13NO3")) + (DJ.sel(photol_coeffs=47) * Y.sel(species="RN16NO3")) \
        + (DJ.sel(photol_coeffs=44) * Y.sel(species="IC3H7NO3")) + (DJ.sel(photol_coeffs=45) * Y.sel(species="RN13NO3")) \
        + (DJ.sel(photol_coeffs=42) * Y.sel(species="C2H5NO3")) + (DJ.sel(photol_coeffs=43) * Y.sel(species="RN10NO3")) \
        + (DJ.sel(photol_coeffs=32) * Y.sel(species="NOA")) + (DJ.sel(photol_coeffs=41) * Y.sel(species="CH3NO3")) \
        + (DJ.sel(photol_coeffs=8) * Y.sel(species="HNO3")) + (DJ.sel(photol_coeffs=31) * Y.sel(species="NOA")) \
        + (RC.sel(therm_coeffs=484) * Y.sel(species="OH") * Y.sel(species="RTN26PAN")) + (DJ.sel(photol_coeffs=6) * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=481) * Y.sel(species="OH") * Y.sel(species="RU12PAN")) + (RC.sel(therm_coeffs=483) * Y.sel(species="RTN26PAN")) \
        + (RC.sel(therm_coeffs=479) * Y.sel(species="MPAN")) + (RC.sel(therm_coeffs=480) * Y.sel(species="OH") * Y.sel(species="MPAN")) \
        + (RC.sel(therm_coeffs=475) * Y.sel(species="OH") * Y.sel(species="PHAN")) + (RC.sel(therm_coeffs=477) * Y.sel(species="RU12PAN")) \
        + (RC.sel(therm_coeffs=473) * Y.sel(species="OH") * Y.sel(species="PAN")) + (RC.sel(therm_coeffs=474) * Y.sel(species="OH") * Y.sel(species="PPN")) \
        + (RC.sel(therm_coeffs=470) * Y.sel(species="PPN")) + (RC.sel(therm_coeffs=472) * Y.sel(species="PHAN")) \
        + (RC.sel(therm_coeffs=456) * Y.sel(species="OH") * Y.sel(species="NRTN28OOH")) + (RC.sel(therm_coeffs=468) * Y.sel(species="PAN")) \
        + (RC.sel(therm_coeffs=449) * Y.sel(species="OH") * Y.sel(species="NRN9OOH")) + (RC.sel(therm_coeffs=450) * Y.sel(species="OH") * Y.sel(species="NRN12OOH")) \
        + (RC.sel(therm_coeffs=422) * Y.sel(species="NO3") * Y.sel(species="ARNOH17")) + (RC.sel(therm_coeffs=448) * Y.sel(species="OH") * Y.sel(species="NRN6OOH")) \
        + (RC.sel(therm_coeffs=417) * Y.sel(species="NO3") * Y.sel(species="ARNOH14")) + (RC.sel(therm_coeffs=421) * Y.sel(species="OH") * Y.sel(species="ARNOH17")) \
        + (RC.sel(therm_coeffs=412) * Y.sel(species="OH") * Y.sel(species="RTX22NO3")) + (RC.sel(therm_coeffs=416) * Y.sel(species="OH") * Y.sel(species="ARNOH14")) \
        + (RC.sel(therm_coeffs=410) * Y.sel(species="OH") * Y.sel(species="RTX28NO3")) + (RC.sel(therm_coeffs=411) * Y.sel(species="OH") * Y.sel(species="RTX24NO3")) \
        + (RC.sel(therm_coeffs=408) * Y.sel(species="OH") * Y.sel(species="RTN28NO3")) + (RC.sel(therm_coeffs=409) * Y.sel(species="OH") * Y.sel(species="RTN25NO3")) \
        + (RC.sel(therm_coeffs=406) * Y.sel(species="OH") * Y.sel(species="RA16NO3")) + (RC.sel(therm_coeffs=407) * Y.sel(species="OH") * Y.sel(species="RA19NO3")) \
        + (RC.sel(therm_coeffs=404) * Y.sel(species="OH") * Y.sel(species="RU14NO3")) + (RC.sel(therm_coeffs=405) * Y.sel(species="OH") * Y.sel(species="RA13NO3")) \
        + (RC.sel(therm_coeffs=402) * Y.sel(species="OH") * Y.sel(species="RN15NO3")) + (RC.sel(therm_coeffs=403) * Y.sel(species="OH") * Y.sel(species="RN18NO3")) \
        + (RC.sel(therm_coeffs=400) * Y.sel(species="OH") * Y.sel(species="RN9NO3")) + (RC.sel(therm_coeffs=401) * Y.sel(species="OH") * Y.sel(species="RN12NO3")) \
        + (RC.sel(therm_coeffs=398) * Y.sel(species="OH") * Y.sel(species="RN19NO3")) + (RC.sel(therm_coeffs=399) * Y.sel(species="OH") * Y.sel(species="HOC2H4NO3")) \
        + (RC.sel(therm_coeffs=396) * Y.sel(species="OH") * Y.sel(species="RN13NO3")) + (RC.sel(therm_coeffs=397) * Y.sel(species="OH") * Y.sel(species="RN16NO3")) \
        + (RC.sel(therm_coeffs=394) * Y.sel(species="OH") * Y.sel(species="RN10NO3")) + (RC.sel(therm_coeffs=395) * Y.sel(species="OH") * Y.sel(species="IC3H7NO3")) \
        + (RC.sel(therm_coeffs=392) * Y.sel(species="OH") * Y.sel(species="CH3NO3")) + (RC.sel(therm_coeffs=393) * Y.sel(species="OH") * Y.sel(species="C2H5NO3")) \
        + (RC.sel(therm_coeffs=352) * Y.sel(species="NRTX28O2")) + (RC.sel(therm_coeffs=377) * Y.sel(species="OH") * Y.sel(species="NOA")) \
        + (RC.sel(therm_coeffs=338) * Y.sel(species="NRN12O2")) + (RC.sel(therm_coeffs=342) * Y.sel(species="NRTN28O2")) \
        + (RC.sel(therm_coeffs=336) * Y.sel(species="NRN6O2")) + (RC.sel(therm_coeffs=337) * Y.sel(species="NRN9O2")) \
        + (RC.sel(therm_coeffs=242) * Y.sel(species="RTX22O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=243) * Y.sel(species="NRTX28O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=240) * Y.sel(species="RTX28O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=241) * Y.sel(species="RTX24O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=238) * Y.sel(species="RTN14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=239) * Y.sel(species="RTN10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=236) * Y.sel(species="RTN24O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=237) * Y.sel(species="RTN23O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=234) * Y.sel(species="RTN26O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=235) * Y.sel(species="RTN25O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=232) * Y.sel(species="RTN28O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=233) * Y.sel(species="NRTN28O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=230) * Y.sel(species="NRU14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=231) * Y.sel(species="NRU12O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=229) * Y.sel(species="NRN12O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=228) * Y.sel(species="NRN9O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=226) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=227) * Y.sel(species="NRN6O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=224) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=225) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=222) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=223) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=220) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=221) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=218) * Y.sel(species="RN14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=219) * Y.sel(species="RN17O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=216) * Y.sel(species="RN8O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=217) * Y.sel(species="RN11O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=214) * Y.sel(species="C2H5CO3") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=215) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=212) * Y.sel(species="RN18AO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=213) * Y.sel(species="CH3CO3") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=210) * Y.sel(species="RN18O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=211) * Y.sel(species="RN15AO2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=208) * Y.sel(species="RN12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=209) * Y.sel(species="RN15O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=206) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=207) * Y.sel(species="RN902") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=204) * Y.sel(species="RA19CO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=205) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=202) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=203) * Y.sel(species="RA19AO2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=200) * Y.sel(species="RA13O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=201) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=198) * Y.sel(species="RN13AO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=199) * Y.sel(species="RN16AO2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=196) * Y.sel(species="RN16O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=197) * Y.sel(species="RN19O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=194) * Y.sel(species="RN13O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=195) * Y.sel(species="RN13O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=192) * Y.sel(species="RN10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=193) * Y.sel(species="IC3H7O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=190) * Y.sel(species="CH3O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=191) * Y.sel(species="C2H5O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=163) * Y.sel(species="RTX22O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=165) * Y.sel(species="CH3O2NO2")) \
        + (RC.sel(therm_coeffs=161) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=162) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=160) * Y.sel(species="NRTX28O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=158) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=159) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=156) * Y.sel(species="RTN14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=157) * Y.sel(species="RTN10O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=154) * Y.sel(species="RTN24O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=155) * Y.sel(species="RTN23O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=152) * Y.sel(species="RTN26O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=153) * Y.sel(species="RTN25O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=151) * Y.sel(species="NRTN28O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=149) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=150) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=147) * Y.sel(species="NRU14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=148) * Y.sel(species="NRU12O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=146) * Y.sel(species="NRN12O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=145) * Y.sel(species="NRN9O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=143) * Y.sel(species="RU10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=144) * Y.sel(species="NRN6O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=141) * Y.sel(species="RU10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=142) * Y.sel(species="RU10O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=139) * Y.sel(species="RU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=140) * Y.sel(species="RU12O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=137) * Y.sel(species="RU14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=138) * Y.sel(species="RU14O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=135) * Y.sel(species="RN14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=136) * Y.sel(species="RN17O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=133) * Y.sel(species="RN8O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=134) * Y.sel(species="RN11O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=131) * Y.sel(species="C2H5CO3") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=132) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=129) * Y.sel(species="RN18AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=130) * Y.sel(species="CH3CO3") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=127) * Y.sel(species="RN18O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=128) * Y.sel(species="RN15AO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=125) * Y.sel(species="RN12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=126) * Y.sel(species="RN15O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=123) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=124) * Y.sel(species="RN902") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=121) * Y.sel(species="RA19CO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=122) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=119) * Y.sel(species="RA16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=120) * Y.sel(species="RA19AO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=117) * Y.sel(species="RA13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=118) * Y.sel(species="RA16O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=115) * Y.sel(species="RN13AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=116) * Y.sel(species="RN16AO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=113) * Y.sel(species="RN16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=114) * Y.sel(species="RN19O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=111) * Y.sel(species="RN13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=112) * Y.sel(species="RN13O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=109) * Y.sel(species="RN10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=110) * Y.sel(species="IC3H7O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=107) * Y.sel(species="CH3O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=108) * Y.sel(species="C2H5O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=33) * Y.sel(species="HO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=34) * Y.sel(species="OH") * Y.sel(species="HONO")) \
        + (RC.sel(therm_coeffs=31) * Y.sel(species="HO2NO2")) + (RC.sel(therm_coeffs=32) * Y.sel(species="OH") * Y.sel(species="HO2NO2")) \
        + (RC.sel(therm_coeffs=28) * Y.sel(species="OH") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=29) * Y.sel(species="HO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=13) * Y.sel(species="NO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=15) * Y.sel(species="N2O5")) \
        + (RC.sel(therm_coeffs=12) * Y.sel(species="NO") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=11) * Y.sel(species="NO") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=4) * Y.sel(species="O") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=9) * Y.sel(species="NO") * Y.sel(species="O3")) 
        #     
        #          
        L = 0.0 \
        + (RC.sel(therm_coeffs=478) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=482) * Y.sel(species="RTN26O2")) + (DJ.sel(photol_coeffs=4)) \
        + (RC.sel(therm_coeffs=469) * Y.sel(species="C2H5CO3")) + (RC.sel(therm_coeffs=471) * Y.sel(species="HOCH2CO3")) + (RC.sel(therm_coeffs=476) * Y.sel(species="RU12O2")) \
        + (RC.sel(therm_coeffs=415) * Y.sel(species="RAROH14")) + (RC.sel(therm_coeffs=420) * Y.sel(species="RAROH17")) + (RC.sel(therm_coeffs=467) * Y.sel(species="CH3CO3")) \
        + (RC.sel(therm_coeffs=27) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=30) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=164) * Y.sel(species="CH3O2")) \
        + (RC.sel(therm_coeffs=13) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=14) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=26)) \
        + (RC.sel(therm_coeffs=5) * Y.sel(species="O")) + (RC.sel(therm_coeffs=6) * Y.sel(species="O")) + (RC.sel(therm_coeffs=10) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "NO2"] = (YP.sel(species="NO2") + DTS * P) /(1.0+ DTS * L) 

        #          NO3              Y.sel(species="NO3") 
        P = EM.sel(species="NO3") \
        + (RC.sel(therm_coeffs=15) * Y.sel(species="N2O5")) + (RC.sel(therm_coeffs=35) * Y.sel(species="OH") * Y.sel(species="HNO3")) \
        + (RC.sel(therm_coeffs=6) * Y.sel(species="O") * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=10) * Y.sel(species="NO2") * Y.sel(species="O3")) 
        L = 0.0 \
        + (DJ.sel(photol_coeffs=6)) \
        + (RC.sel(therm_coeffs=419) * Y.sel(species="AROH17")) + (RC.sel(therm_coeffs=422) * Y.sel(species="ARNOH17")) + (DJ.sel(photol_coeffs=5)) \
        + (RC.sel(therm_coeffs=388) * Y.sel(species="TNCARB10")) + (RC.sel(therm_coeffs=414) * Y.sel(species="AROH14")) + (RC.sel(therm_coeffs=417) * Y.sel(species="ARNOH14")) \
        + (RC.sel(therm_coeffs=365) * Y.sel(species="HOCH2CHO")) + (RC.sel(therm_coeffs=373) * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=387) * Y.sel(species="TNCARB26")) \
        + (RC.sel(therm_coeffs=242) * Y.sel(species="RTX22O2")) + (RC.sel(therm_coeffs=243) * Y.sel(species="NRTX28O2")) + (RC.sel(therm_coeffs=361) * Y.sel(species="UCARB10")) \
        + (RC.sel(therm_coeffs=239) * Y.sel(species="RTN10O2")) + (RC.sel(therm_coeffs=240) * Y.sel(species="RTX28O2")) + (RC.sel(therm_coeffs=241) * Y.sel(species="RTX24O2")) \
        + (RC.sel(therm_coeffs=236) * Y.sel(species="RTN24O2")) + (RC.sel(therm_coeffs=237) * Y.sel(species="RTN23O2")) + (RC.sel(therm_coeffs=238) * Y.sel(species="RTN14O2")) \
        + (RC.sel(therm_coeffs=233) * Y.sel(species="NRTN28O2")) + (RC.sel(therm_coeffs=234) * Y.sel(species="RTN26O2")) + (RC.sel(therm_coeffs=235) * Y.sel(species="RTN25O2")) \
        + (RC.sel(therm_coeffs=230) * Y.sel(species="NRU14O2")) + (RC.sel(therm_coeffs=231) * Y.sel(species="NRU12O2")) + (RC.sel(therm_coeffs=232) * Y.sel(species="RTN28O2")) \
        + (RC.sel(therm_coeffs=227) * Y.sel(species="NRN6O2")) + (RC.sel(therm_coeffs=228) * Y.sel(species="NRN9O2")) + (RC.sel(therm_coeffs=229) * Y.sel(species="NRN12O2")) \
        + (RC.sel(therm_coeffs=224) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=225) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=226) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=221) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=222) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=223) * Y.sel(species="RU12O2")) \
        + (RC.sel(therm_coeffs=218) * Y.sel(species="RN14O2")) + (RC.sel(therm_coeffs=219) * Y.sel(species="RN17O2")) + (RC.sel(therm_coeffs=220) * Y.sel(species="RU14O2")) \
        + (RC.sel(therm_coeffs=215) * Y.sel(species="HOCH2CO3")) + (RC.sel(therm_coeffs=216) * Y.sel(species="RN8O2")) + (RC.sel(therm_coeffs=217) * Y.sel(species="RN11O2")) \
        + (RC.sel(therm_coeffs=212) * Y.sel(species="RN18AO2")) + (RC.sel(therm_coeffs=213) * Y.sel(species="CH3CO3")) + (RC.sel(therm_coeffs=214) * Y.sel(species="C2H5CO3")) \
        + (RC.sel(therm_coeffs=209) * Y.sel(species="RN15O2")) + (RC.sel(therm_coeffs=210) * Y.sel(species="RN18O2")) + (RC.sel(therm_coeffs=211) * Y.sel(species="RN15AO2")) \
        + (RC.sel(therm_coeffs=206) * Y.sel(species="HOCH2CH2O2")) + (RC.sel(therm_coeffs=207) * Y.sel(species="RN902")) + (RC.sel(therm_coeffs=208) * Y.sel(species="RN12O2")) \
        + (RC.sel(therm_coeffs=203) * Y.sel(species="RA19AO2")) + (RC.sel(therm_coeffs=204) * Y.sel(species="RA19CO2")) + (RC.sel(therm_coeffs=205) * Y.sel(species="HOCH2CH2O2")) \
        + (RC.sel(therm_coeffs=200) * Y.sel(species="RA13O2")) + (RC.sel(therm_coeffs=201) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=202) * Y.sel(species="RA16O2")) \
        + (RC.sel(therm_coeffs=197) * Y.sel(species="RN19O2")) + (RC.sel(therm_coeffs=198) * Y.sel(species="RN13AO2")) + (RC.sel(therm_coeffs=199) * Y.sel(species="RN16AO2")) \
        + (RC.sel(therm_coeffs=194) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=195) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=196) * Y.sel(species="RN16O2")) \
        + (RC.sel(therm_coeffs=191) * Y.sel(species="C2H5O2")) + (RC.sel(therm_coeffs=192) * Y.sel(species="RN10O2")) + (RC.sel(therm_coeffs=193) * Y.sel(species="IC3H7O2")) \
        + (RC.sel(therm_coeffs=86) * Y.sel(species="CH3CHO")) + (RC.sel(therm_coeffs=87) * Y.sel(species="C2H5CHO")) + (RC.sel(therm_coeffs=190) * Y.sel(species="CH3O2")) \
        + (RC.sel(therm_coeffs=64) * Y.sel(species="APINENE")) + (RC.sel(therm_coeffs=69) * Y.sel(species="BPINENE")) + (RC.sel(therm_coeffs=85) * Y.sel(species="HCHO")) \
        + (RC.sel(therm_coeffs=51) * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=52) * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=60) * Y.sel(species="C5H8")) \
        + (RC.sel(therm_coeffs=28) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=33) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=50) * Y.sel(species="C2H4")) \
        + (RC.sel(therm_coeffs=12) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=13) * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=14) * Y.sel(species="NO2")) 
        Y.loc[:, :, :, "NO3"] = (YP.sel(species="NO3") + DTS * P) /(1.0+ DTS * L) 

        #          O3               Y.sel(species="O3") 
        P = EM.sel(species="O3") \
        + (RC.sel(therm_coeffs=1) * Y.sel(species="O")) + (RC.sel(therm_coeffs=2) * Y.sel(species="O")) 
        #     

        L = 0.0 \
        + (DJ.sel(photol_coeffs=1)) + (DJ.sel(photol_coeffs=2)) \
        + (RC.sel(therm_coeffs=363) * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=374) * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=375) * Y.sel(species="UCARB12")) \
        + (RC.sel(therm_coeffs=72) * Y.sel(species="BPINENE")) + (RC.sel(therm_coeffs=73) * Y.sel(species="BPINENE")) + (RC.sel(therm_coeffs=362) * Y.sel(species="UCARB10")) \
        + (RC.sel(therm_coeffs=67) * Y.sel(species="APINENE")) + (RC.sel(therm_coeffs=70) * Y.sel(species="BPINENE")) + (RC.sel(therm_coeffs=71) * Y.sel(species="BPINENE")) \
        + (RC.sel(therm_coeffs=62) * Y.sel(species="C5H8")) + (RC.sel(therm_coeffs=65) * Y.sel(species="APINENE")) + (RC.sel(therm_coeffs=66) * Y.sel(species="APINENE")) \
        + (RC.sel(therm_coeffs=57) * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=58) * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=61) * Y.sel(species="C5H8")) \
        + (RC.sel(therm_coeffs=54) * Y.sel(species="C2H4")) + (RC.sel(therm_coeffs=55) * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=56) * Y.sel(species="C3H6")) \
        + (RC.sel(therm_coeffs=17) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=21) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=53) * Y.sel(species="C2H4")) \
        + (RC.sel(therm_coeffs=3) * Y.sel(species="O")) + (RC.sel(therm_coeffs=9) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=10) * Y.sel(species="NO2")) 
        Y.loc[:, :, :, "O3"] = (YP.sel(species="O3") + DTS * P) /(1.0+ DTS * L) 

        #          N2O5             Y.sel(species="N2O5") 
        P = EM.sel(species="N2O5") \
        + (RC.sel(therm_coeffs=14) * Y.sel(species="NO2") * Y.sel(species="NO3")) 
        #   

        L = 0.0 \
        + (RC.sel(therm_coeffs=15)) + (RC.sel(therm_coeffs=40)) 
        Y.loc[:, :, :, "N2O5"] = (YP.sel(species="N2O5") + DTS * P) /(1.0+ DTS * L) 

        #          NO               Y.sel(species="NO") 
        P = EM.sel(species="NO") \
        + (DJ.sel(photol_coeffs=7) * Y.sel(species="HONO")) \
        + (DJ.sel(photol_coeffs=4) * Y.sel(species="NO2")) + (DJ.sel(photol_coeffs=5) * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=5) * Y.sel(species="O") * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=13) * Y.sel(species="NO2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=189) * Y.sel(species="RTX22O2")) \
        + (RC.sel(therm_coeffs=186) * Y.sel(species="RTN25O2")) + (RC.sel(therm_coeffs=187) * Y.sel(species="RTX28O2")) + (RC.sel(therm_coeffs=188) * Y.sel(species="RTX24O2")) \
        + (RC.sel(therm_coeffs=183) * Y.sel(species="RA19AO2")) + (RC.sel(therm_coeffs=184) * Y.sel(species="RA19CO2")) + (RC.sel(therm_coeffs=185) * Y.sel(species="RTN28O2")) \
        + (RC.sel(therm_coeffs=180) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=181) * Y.sel(species="RA13O2")) + (RC.sel(therm_coeffs=182) * Y.sel(species="RA16O2")) \
        + (RC.sel(therm_coeffs=177) * Y.sel(species="RN18O2")) + (RC.sel(therm_coeffs=178) * Y.sel(species="RN15AO2")) + (RC.sel(therm_coeffs=179) * Y.sel(species="RN18AO2")) \
        + (RC.sel(therm_coeffs=174) * Y.sel(species="RN902")) + (RC.sel(therm_coeffs=175) * Y.sel(species="RN12O2")) + (RC.sel(therm_coeffs=176) * Y.sel(species="RN15O2")) \
        + (RC.sel(therm_coeffs=171) * Y.sel(species="RN16O2")) + (RC.sel(therm_coeffs=172) * Y.sel(species="RN19O2")) + (RC.sel(therm_coeffs=173) * Y.sel(species="HOCH2CH2O2")) \
        + (RC.sel(therm_coeffs=168) * Y.sel(species="RN10O2")) + (RC.sel(therm_coeffs=169) * Y.sel(species="IC3H7O2")) + (RC.sel(therm_coeffs=170) * Y.sel(species="RN13O2")) \
        + (RC.sel(therm_coeffs=163) * Y.sel(species="RTX22O2")) + (RC.sel(therm_coeffs=166) * Y.sel(species="CH3O2")) + (RC.sel(therm_coeffs=167) * Y.sel(species="C2H5O2")) \
        + (RC.sel(therm_coeffs=160) * Y.sel(species="NRTX28O2")) + (RC.sel(therm_coeffs=161) * Y.sel(species="RTX24O2")) + (RC.sel(therm_coeffs=162) * Y.sel(species="RTX24O2")) \
        + (RC.sel(therm_coeffs=157) * Y.sel(species="RTN10O2")) + (RC.sel(therm_coeffs=158) * Y.sel(species="RTX28O2")) + (RC.sel(therm_coeffs=159) * Y.sel(species="RTX28O2")) \
        + (RC.sel(therm_coeffs=154) * Y.sel(species="RTN24O2")) + (RC.sel(therm_coeffs=155) * Y.sel(species="RTN23O2")) + (RC.sel(therm_coeffs=156) * Y.sel(species="RTN14O2")) \
        + (RC.sel(therm_coeffs=151) * Y.sel(species="NRTN28O2")) + (RC.sel(therm_coeffs=152) * Y.sel(species="RTN26O2")) + (RC.sel(therm_coeffs=153) * Y.sel(species="RTN25O2")) \
        + (RC.sel(therm_coeffs=148) * Y.sel(species="NRU12O2")) + (RC.sel(therm_coeffs=149) * Y.sel(species="RTN28O2")) + (RC.sel(therm_coeffs=150) * Y.sel(species="RTN28O2")) \
        + (RC.sel(therm_coeffs=145) * Y.sel(species="NRN9O2")) + (RC.sel(therm_coeffs=146) * Y.sel(species="NRN12O2")) + (RC.sel(therm_coeffs=147) * Y.sel(species="NRU14O2")) \
        + (RC.sel(therm_coeffs=142) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=143) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=144) * Y.sel(species="NRN6O2")) \
        + (RC.sel(therm_coeffs=139) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=140) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=141) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=136) * Y.sel(species="RN17O2")) + (RC.sel(therm_coeffs=137) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=138) * Y.sel(species="RU14O2")) \
        + (RC.sel(therm_coeffs=133) * Y.sel(species="RN8O2")) + (RC.sel(therm_coeffs=134) * Y.sel(species="RN11O2")) + (RC.sel(therm_coeffs=135) * Y.sel(species="RN14O2")) \
        + (RC.sel(therm_coeffs=130) * Y.sel(species="CH3CO3")) + (RC.sel(therm_coeffs=131) * Y.sel(species="C2H5CO3")) + (RC.sel(therm_coeffs=132) * Y.sel(species="HOCH2CO3")) \
        + (RC.sel(therm_coeffs=127) * Y.sel(species="RN18O2")) + (RC.sel(therm_coeffs=128) * Y.sel(species="RN15AO2")) + (RC.sel(therm_coeffs=129) * Y.sel(species="RN18AO2")) \
        + (RC.sel(therm_coeffs=124) * Y.sel(species="RN902")) + (RC.sel(therm_coeffs=125) * Y.sel(species="RN12O2")) + (RC.sel(therm_coeffs=126) * Y.sel(species="RN15O2")) \
        + (RC.sel(therm_coeffs=121) * Y.sel(species="RA19CO2")) + (RC.sel(therm_coeffs=122) * Y.sel(species="HOCH2CH2O2")) + (RC.sel(therm_coeffs=123) * Y.sel(species="HOCH2CH2O2")) \
        + (RC.sel(therm_coeffs=118) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=119) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=120) * Y.sel(species="RA19AO2")) \
        + (RC.sel(therm_coeffs=115) * Y.sel(species="RN13AO2")) + (RC.sel(therm_coeffs=116) * Y.sel(species="RN16AO2")) + (RC.sel(therm_coeffs=117) * Y.sel(species="RA13O2")) \
        + (RC.sel(therm_coeffs=112) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=113) * Y.sel(species="RN16O2")) + (RC.sel(therm_coeffs=114) * Y.sel(species="RN19O2")) \
        + (RC.sel(therm_coeffs=109) * Y.sel(species="RN10O2")) + (RC.sel(therm_coeffs=110) * Y.sel(species="IC3H7O2")) + (RC.sel(therm_coeffs=111) * Y.sel(species="RN13O2")) \
        + (RC.sel(therm_coeffs=29) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=107) * Y.sel(species="CH3O2")) + (RC.sel(therm_coeffs=108) * Y.sel(species="C2H5O2")) \
        + (RC.sel(therm_coeffs=11) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=12) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=25) * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=4) * Y.sel(species="O")) + (RC.sel(therm_coeffs=9) * Y.sel(species="O3")) + (RC.sel(therm_coeffs=11) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "NO"] = (YP.sel(species="NO") + DTS * P) /(1.0+ DTS * L) 

        #          HO2              Y.sel(species="HO2") 
        P = EM.sel(species="HO2") \
        + (DJ.sel(photol_coeffs=94) * Y.sel(species="RTX24OOH")) \
        + (DJ.sel(photol_coeffs=91) * Y.sel(species="RTN14OOH")) + (DJ.sel(photol_coeffs=93) * Y.sel(species="RTX28OOH")) \
        + (DJ.sel(photol_coeffs=84) * Y.sel(species="RA19OOH")) + (DJ.sel(photol_coeffs=85) * Y.sel(species="RTN28OOH")) \
        + (DJ.sel(photol_coeffs=82) * Y.sel(species="RA13OOH")) + (DJ.sel(photol_coeffs=83) * Y.sel(species="RA16OOH")) \
        + (DJ.sel(photol_coeffs=77) * Y.sel(species="RN15OOH")) + (DJ.sel(photol_coeffs=78) * Y.sel(species="RN18OOH")) \
        + (DJ.sel(photol_coeffs=75) * Y.sel(species="RN9OOH")) + (DJ.sel(photol_coeffs=76) * Y.sel(species="RN12OOH")) \
        + (DJ.sel(photol_coeffs=73) * Y.sel(species="NRU12OOH")) + (DJ.sel(photol_coeffs=74) * Y.sel(species="HOC2H4OOH")) \
        + (DJ.sel(photol_coeffs=70) * Y.sel(species="RU12OOH")) + (DJ.sel(photol_coeffs=72) * Y.sel(species="NRU14OOH")) \
        + (DJ.sel(photol_coeffs=68) * Y.sel(species="RU14OOH")) + (DJ.sel(photol_coeffs=69) * Y.sel(species="RU14OOH")) \
        + (DJ.sel(photol_coeffs=58) * Y.sel(species="RN13OOH")) + (DJ.sel(photol_coeffs=63) * Y.sel(species="HOCH2CO3H")) \
        + (DJ.sel(photol_coeffs=55) * Y.sel(species="RN10OOH")) + (DJ.sel(photol_coeffs=56) * Y.sel(species="IC3H7OOH")) \
        + (DJ.sel(photol_coeffs=53) * Y.sel(species="CH3OOH")) + (DJ.sel(photol_coeffs=54) * Y.sel(species="C2H5OOH")) \
        + (DJ.sel(photol_coeffs=51) * Y.sel(species="RA19NO3")) + (DJ.sel(photol_coeffs=52) * Y.sel(species="RTX24NO3")) \
        + (DJ.sel(photol_coeffs=49) * Y.sel(species="RA13NO3")) + (DJ.sel(photol_coeffs=50) * Y.sel(species="RA16NO3")) \
        + (DJ.sel(photol_coeffs=44) * Y.sel(species="IC3H7NO3")) + (DJ.sel(photol_coeffs=46) * Y.sel(species="RN13NO3")) \
        + (DJ.sel(photol_coeffs=42) * Y.sel(species="C2H5NO3")) + (DJ.sel(photol_coeffs=43) * Y.sel(species="RN10NO3")) \
        + (DJ.sel(photol_coeffs=39) * Y.sel(species="TNCARB26")) + (DJ.sel(photol_coeffs=41) * Y.sel(species="CH3NO3")) \
        + (DJ.sel(photol_coeffs=37) * Y.sel(species="UDCARB14")) + (DJ.sel(photol_coeffs=38) * Y.sel(species="UDCARB14")) \
        + (DJ.sel(photol_coeffs=35) * Y.sel(species="UDCARB11")) + (DJ.sel(photol_coeffs=36) * Y.sel(species="UDCARB11")) \
        + (DJ.sel(photol_coeffs=33) * Y.sel(species="UDCARB8")) + (DJ.sel(photol_coeffs=34) * Y.sel(species="UDCARB8") * 2.00) \
        + (DJ.sel(photol_coeffs=30) * Y.sel(species="NUCARB12") * 2.00) \
        + (DJ.sel(photol_coeffs=25) * Y.sel(species="CARB6")) + (DJ.sel(photol_coeffs=29) * Y.sel(species="UCARB12")) \
        + (DJ.sel(photol_coeffs=23) * Y.sel(species="UCARB10")) + (DJ.sel(photol_coeffs=24) * Y.sel(species="CARB3") * 2.00) \
        + (DJ.sel(photol_coeffs=22) * Y.sel(species="HOCH2CHO") * 2.00) \
        + (DJ.sel(photol_coeffs=20) * Y.sel(species="CARB13")) + (DJ.sel(photol_coeffs=21) * Y.sel(species="CARB16")) \
        + (DJ.sel(photol_coeffs=18) * Y.sel(species="CARB7")) + (DJ.sel(photol_coeffs=19) * Y.sel(species="CARB10")) \
        + (DJ.sel(photol_coeffs=11) * Y.sel(species="CH3CHO")) + (DJ.sel(photol_coeffs=12) * Y.sel(species="C2H5CHO")) \
        + (RC.sel(therm_coeffs=379) * Y.sel(species="OH") * Y.sel(species="UDCARB8")) + (DJ.sel(photol_coeffs=9) * Y.sel(species="HCHO") * 2.00) \
        + (RC.sel(therm_coeffs=357) * Y.sel(species="OH") * Y.sel(species="CARB10")) + (RC.sel(therm_coeffs=366) * Y.sel(species="OH") * Y.sel(species="CARB3")) \
        + (RC.sel(therm_coeffs=350) * Y.sel(species="RTX24O2")) + (RC.sel(therm_coeffs=356) * Y.sel(species="OH") * Y.sel(species="CARB7")) \
        + (RC.sel(therm_coeffs=347) * Y.sel(species="RTN14O2")) + (RC.sel(therm_coeffs=349) * Y.sel(species="RTX28O2")) \
        + (RC.sel(therm_coeffs=340) * Y.sel(species="NRU12O2")) + (RC.sel(therm_coeffs=341) * Y.sel(species="RTN28O2")) \
        + (RC.sel(therm_coeffs=335) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=339) * Y.sel(species="NRU14O2")) \
        + (RC.sel(therm_coeffs=332) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=334) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=329) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=330) * Y.sel(species="RU14O2")) \
        + (RC.sel(therm_coeffs=321) * Y.sel(species="RN18AO2")) + (RC.sel(therm_coeffs=324) * Y.sel(species="HOCH2CO3")) \
        + (RC.sel(therm_coeffs=319) * Y.sel(species="RN18O2")) + (RC.sel(therm_coeffs=320) * Y.sel(species="RN15AO2")) \
        + (RC.sel(therm_coeffs=317) * Y.sel(species="RN12O2")) + (RC.sel(therm_coeffs=318) * Y.sel(species="RN15O2")) \
        + (RC.sel(therm_coeffs=315) * Y.sel(species="HOCH2CH2O2")) + (RC.sel(therm_coeffs=316) * Y.sel(species="RN902")) \
        + (RC.sel(therm_coeffs=311) * Y.sel(species="RA19CO2")) + (RC.sel(therm_coeffs=314) * Y.sel(species="HOCH2CH2O2")) \
        + (RC.sel(therm_coeffs=309) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=310) * Y.sel(species="RA19AO2")) \
        + (RC.sel(therm_coeffs=307) * Y.sel(species="RA13O2")) + (RC.sel(therm_coeffs=308) * Y.sel(species="RA16O2")) \
        + (RC.sel(therm_coeffs=300) * Y.sel(species="IC3H7O2")) + (RC.sel(therm_coeffs=304) * Y.sel(species="RN13O2")) \
        + (RC.sel(therm_coeffs=294) * Y.sel(species="C2H5O2")) + (RC.sel(therm_coeffs=297) * Y.sel(species="RN10O2")) \
        + (RC.sel(therm_coeffs=241) * Y.sel(species="RTX24O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=291) * Y.sel(species="CH3O2")) \
        + (RC.sel(therm_coeffs=238) * Y.sel(species="RTN14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=240) * Y.sel(species="RTX28O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=231) * Y.sel(species="NRU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=232) * Y.sel(species="RTN28O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=226) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=230) * Y.sel(species="NRU14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=223) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=225) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=220) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=221) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=212) * Y.sel(species="RN18AO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=215) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=210) * Y.sel(species="RN18O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=211) * Y.sel(species="RN15AO2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=208) * Y.sel(species="RN12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=209) * Y.sel(species="RN15O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=206) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=207) * Y.sel(species="RN902") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=204) * Y.sel(species="RA19CO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=205) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=202) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=203) * Y.sel(species="RA19AO2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=200) * Y.sel(species="RA13O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=201) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=193) * Y.sel(species="IC3H7O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=195) * Y.sel(species="RN13O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=191) * Y.sel(species="C2H5O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=192) * Y.sel(species="RN10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=161) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=190) * Y.sel(species="CH3O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=156) * Y.sel(species="RTN14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=158) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=148) * Y.sel(species="NRU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=149) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=143) * Y.sel(species="RU10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=147) * Y.sel(species="NRU14O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=140) * Y.sel(species="RU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=142) * Y.sel(species="RU10O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=137) * Y.sel(species="RU14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=138) * Y.sel(species="RU14O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=129) * Y.sel(species="RN18AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=132) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=127) * Y.sel(species="RN18O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=128) * Y.sel(species="RN15AO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=125) * Y.sel(species="RN12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=126) * Y.sel(species="RN15O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=123) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=124) * Y.sel(species="RN902") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=121) * Y.sel(species="RA19CO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=122) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=119) * Y.sel(species="RA16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=120) * Y.sel(species="RA19AO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=117) * Y.sel(species="RA13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=118) * Y.sel(species="RA16O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=110) * Y.sel(species="IC3H7O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=112) * Y.sel(species="RN13O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=108) * Y.sel(species="C2H5O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=109) * Y.sel(species="RN10O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=97) * Y.sel(species="HCOOH") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=107) * Y.sel(species="CH3O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=93) * Y.sel(species="NPROPOL") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=95) * Y.sel(species="OH") * Y.sel(species="IPROPOL")) \
        + (RC.sel(therm_coeffs=90) * Y.sel(species="OH") * Y.sel(species="CH3OH")) + (RC.sel(therm_coeffs=91) * Y.sel(species="OH") * Y.sel(species="C2H5OH")) \
        + (RC.sel(therm_coeffs=82) * Y.sel(species="OH") * Y.sel(species="HCHO")) + (RC.sel(therm_coeffs=85) * Y.sel(species="NO3") * Y.sel(species="HCHO")) \
        + (RC.sel(therm_coeffs=77) * Y.sel(species="BENZENE") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=79) * Y.sel(species="TOLUENE") * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=61) * Y.sel(species="O3") * Y.sel(species="C5H8")) + (RC.sel(therm_coeffs=74) * Y.sel(species="C2H2") * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=38) * Y.sel(species="HSO3")) + (RC.sel(therm_coeffs=53) * Y.sel(species="O3") * Y.sel(species="C2H4")) \
        + (RC.sel(therm_coeffs=28) * Y.sel(species="OH") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=31) * Y.sel(species="HO2NO2")) \
        + (RC.sel(therm_coeffs=19) * Y.sel(species="OH") * Y.sel(species="CO")) + (RC.sel(therm_coeffs=20) * Y.sel(species="OH") * Y.sel(species="H2O2")) \
        + (RC.sel(therm_coeffs=17) * Y.sel(species="OH") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=18) * Y.sel(species="OH") * Y.sel(species="H2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=289) * Y.sel(species="RTX22O2")) + (RC.sel(therm_coeffs=290) * Y.sel(species="NRTX28O2")) \
        + (RC.sel(therm_coeffs=286) * Y.sel(species="RTN10O2")) + (RC.sel(therm_coeffs=287) * Y.sel(species="RTX28O2")) + (RC.sel(therm_coeffs=288) * Y.sel(species="RTX24O2")) \
        + (RC.sel(therm_coeffs=283) * Y.sel(species="RTN24O2")) + (RC.sel(therm_coeffs=284) * Y.sel(species="RTN23O2")) + (RC.sel(therm_coeffs=285) * Y.sel(species="RTN14O2")) \
        + (RC.sel(therm_coeffs=280) * Y.sel(species="NRTN28O2")) + (RC.sel(therm_coeffs=281) * Y.sel(species="RTN26O2")) + (RC.sel(therm_coeffs=282) * Y.sel(species="RTN25O2")) \
        + (RC.sel(therm_coeffs=277) * Y.sel(species="NRU14O2")) + (RC.sel(therm_coeffs=278) * Y.sel(species="NRU12O2")) + (RC.sel(therm_coeffs=279) * Y.sel(species="RTN28O2")) \
        + (RC.sel(therm_coeffs=274) * Y.sel(species="NRN6O2")) + (RC.sel(therm_coeffs=275) * Y.sel(species="NRN9O2")) + (RC.sel(therm_coeffs=276) * Y.sel(species="NRN12O2")) \
        + (RC.sel(therm_coeffs=271) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=272) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=273) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=268) * Y.sel(species="RN11O2")) + (RC.sel(therm_coeffs=269) * Y.sel(species="RN14O2")) + (RC.sel(therm_coeffs=270) * Y.sel(species="RN17O2")) \
        + (RC.sel(therm_coeffs=265) * Y.sel(species="C2H5CO3")) + (RC.sel(therm_coeffs=266) * Y.sel(species="HOCH2CO3")) + (RC.sel(therm_coeffs=267) * Y.sel(species="RN8O2")) \
        + (RC.sel(therm_coeffs=262) * Y.sel(species="RN15AO2")) + (RC.sel(therm_coeffs=263) * Y.sel(species="RN18AO2")) + (RC.sel(therm_coeffs=264) * Y.sel(species="CH3CO3")) \
        + (RC.sel(therm_coeffs=259) * Y.sel(species="RN12O2")) + (RC.sel(therm_coeffs=260) * Y.sel(species="RN15O2")) + (RC.sel(therm_coeffs=261) * Y.sel(species="RN18O2")) \
        + (RC.sel(therm_coeffs=256) * Y.sel(species="RA19CO2")) + (RC.sel(therm_coeffs=257) * Y.sel(species="HOCH2CH2O2")) + (RC.sel(therm_coeffs=258) * Y.sel(species="RN902")) \
        + (RC.sel(therm_coeffs=253) * Y.sel(species="RA13O2")) + (RC.sel(therm_coeffs=254) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=255) * Y.sel(species="RA19AO2")) \
        + (RC.sel(therm_coeffs=250) * Y.sel(species="RN19O2")) + (RC.sel(therm_coeffs=251) * Y.sel(species="RN13AO2")) + (RC.sel(therm_coeffs=252) * Y.sel(species="RN16AO2")) \
        + (RC.sel(therm_coeffs=247) * Y.sel(species="IC3H7O2")) + (RC.sel(therm_coeffs=248) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=249) * Y.sel(species="RN16O2")) \
        + (RC.sel(therm_coeffs=244) * Y.sel(species="CH3O2")) + (RC.sel(therm_coeffs=245) * Y.sel(species="C2H5O2")) + (RC.sel(therm_coeffs=246) * Y.sel(species="RN10O2")) \
        + (RC.sel(therm_coeffs=29) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=30) * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=33) * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=23) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=24) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=24) * Y.sel(species="HO2")) \
        + (RC.sel(therm_coeffs=21) * Y.sel(species="O3")) + (RC.sel(therm_coeffs=22) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=23) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "HO2"] = (YP.sel(species="HO2") + DTS * P) /(1.0+ DTS * L) 

        #          H2               Y.sel(species="H2") 
        P = EM.sel(species="H2") \
        + (DJ.sel(photol_coeffs=10) * Y.sel(species="HCHO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=18) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "H2"] = (YP.sel(species="H2") + DTS * P) /(1.0+ DTS * L) 

        #          CO               Y.sel(species="CO") 
        P = EM.sel(species="CO") \
        + (DJ.sel(photol_coeffs=92) * Y.sel(species="RTN10OOH")) \
        + (DJ.sel(photol_coeffs=40) * Y.sel(species="TNCARB10")) + (DJ.sel(photol_coeffs=73) * Y.sel(species="NRU12OOH")) \
        + (DJ.sel(photol_coeffs=30) * Y.sel(species="NUCARB12") * 2.00) \
        + (DJ.sel(photol_coeffs=25) * Y.sel(species="CARB6")) + (DJ.sel(photol_coeffs=29) * Y.sel(species="UCARB12")) \
        + (DJ.sel(photol_coeffs=24) * Y.sel(species="CARB3") * 2.00) \
        + (DJ.sel(photol_coeffs=12) * Y.sel(species="C2H5CHO")) + (DJ.sel(photol_coeffs=22) * Y.sel(species="HOCH2CHO")) \
        + (DJ.sel(photol_coeffs=10) * Y.sel(species="HCHO")) + (DJ.sel(photol_coeffs=11) * Y.sel(species="CH3CHO")) \
        + (RC.sel(therm_coeffs=480) * Y.sel(species="OH") * Y.sel(species="MPAN")) + (DJ.sel(photol_coeffs=9) * Y.sel(species="HCHO")) \
        + (RC.sel(therm_coeffs=474) * Y.sel(species="OH") * Y.sel(species="PPN")) + (RC.sel(therm_coeffs=475) * Y.sel(species="OH") * Y.sel(species="PHAN")) \
        + (RC.sel(therm_coeffs=442) * Y.sel(species="OH") * Y.sel(species="NRU12OOH")) + (RC.sel(therm_coeffs=473) * Y.sel(species="OH") * Y.sel(species="PAN")) \
        + (RC.sel(therm_coeffs=367) * Y.sel(species="OH") * Y.sel(species="CARB6")) + (RC.sel(therm_coeffs=374) * Y.sel(species="O3") * Y.sel(species="UCARB12")) \
        + (RC.sel(therm_coeffs=362) * Y.sel(species="O3") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=366) * Y.sel(species="OH") * Y.sel(species="CARB3") * 2.00) \
        + (RC.sel(therm_coeffs=340) * Y.sel(species="NRU12O2")) + (RC.sel(therm_coeffs=348) * Y.sel(species="RTN10O2")) \
        + (RC.sel(therm_coeffs=231) * Y.sel(species="NRU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=239) * Y.sel(species="RTN10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=157) * Y.sel(species="RTN10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=223) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=140) * Y.sel(species="RU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=148) * Y.sel(species="NRU12O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=82) * Y.sel(species="OH") * Y.sel(species="HCHO")) + (RC.sel(therm_coeffs=85) * Y.sel(species="NO3") * Y.sel(species="HCHO")) \
        + (RC.sel(therm_coeffs=73) * Y.sel(species="BPINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=74) * Y.sel(species="C2H2") * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=57) * Y.sel(species="O3") * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=61) * Y.sel(species="O3") * Y.sel(species="C5H8")) \
        + (RC.sel(therm_coeffs=53) * Y.sel(species="O3") * Y.sel(species="C2H4")) + (RC.sel(therm_coeffs=55) * Y.sel(species="O3") * Y.sel(species="C3H6")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=19) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CO"] = (YP.sel(species="CO") + DTS * P) /(1.0+ DTS * L) 

        #          H2O2             Y.sel(species="H2O2") 
        P = EM.sel(species="H2O2") \
        + (RC.sel(therm_coeffs=363) * Y.sel(species="O3") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=375) * Y.sel(species="O3") * Y.sel(species="UCARB12")) \
        + (RC.sel(therm_coeffs=66) * Y.sel(species="APINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=71) * Y.sel(species="BPINENE") * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=23) * Y.sel(species="HO2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=24) * Y.sel(species="HO2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=20) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=3)) 
        Y.loc[:, :, :, "H2O2"] = (YP.sel(species="H2O2") + DTS * P) /(1.0+ DTS * L) 

        #          HONO             Y.sel(species="HONO") 
        P = EM.sel(species="HONO") \
        + (RC.sel(therm_coeffs=25) * Y.sel(species="OH") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=26) * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=34) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=7)) 
        Y.loc[:, :, :, "HONO"] = (YP.sel(species="HONO") + DTS * P) /(1.0+ DTS * L) 

        #          HNO3             Y.sel(species="HNO3") 
        P = EM.sel(species="HNO3") \
        + (RC.sel(therm_coeffs=422) * Y.sel(species="NO3") * Y.sel(species="ARNOH17")) \
        + (RC.sel(therm_coeffs=417) * Y.sel(species="NO3") * Y.sel(species="ARNOH14")) + (RC.sel(therm_coeffs=419) * Y.sel(species="NO3") * Y.sel(species="AROH17")) \
        + (RC.sel(therm_coeffs=388) * Y.sel(species="NO3") * Y.sel(species="TNCARB10")) + (RC.sel(therm_coeffs=414) * Y.sel(species="NO3") * Y.sel(species="AROH14")) \
        + (RC.sel(therm_coeffs=373) * Y.sel(species="NO3") * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=387) * Y.sel(species="NO3") * Y.sel(species="TNCARB26")) \
        + (RC.sel(therm_coeffs=361) * Y.sel(species="NO3") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=365) * Y.sel(species="NO3") * Y.sel(species="HOCH2CHO")) \
        + (RC.sel(therm_coeffs=86) * Y.sel(species="NO3") * Y.sel(species="CH3CHO")) + (RC.sel(therm_coeffs=87) * Y.sel(species="NO3") * Y.sel(species="C2H5CHO")) \
        + (RC.sel(therm_coeffs=27) * Y.sel(species="OH") * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=85) * Y.sel(species="NO3") * Y.sel(species="HCHO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=35) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=39)) + (DJ.sel(photol_coeffs=8)) 
        Y.loc[:, :, :, "HNO3"] = (YP.sel(species="HNO3") + DTS * P) /(1.0+ DTS * L) 

        #          HO2NO2           Y.sel(species="HO2NO2") 
        P = EM.sel(species="HO2NO2") \
        + (RC.sel(therm_coeffs=30) * Y.sel(species="HO2") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=31)) + (RC.sel(therm_coeffs=32) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "HO2NO2"] = (YP.sel(species="HO2NO2") + DTS * P) /(1.0+ DTS * L) 

        #          SO2              Y.sel(species="SO2") 
        P = EM.sel(species="SO2") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=36) * Y.sel(species="O")) + (RC.sel(therm_coeffs=37) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "SO2"] = (YP.sel(species="SO2") + DTS * P) /(1.0+ DTS * L) 

        #          SO3              Y.sel(species="SO3") 
        P = EM.sel(species="SO3")  \
        + (RC.sel(therm_coeffs=36) * Y.sel(species="O") * Y.sel(species="SO2")) + (RC.sel(therm_coeffs=38) * Y.sel(species="HSO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=41)) 
        Y.loc[:, :, :, "SO3"] = (YP.sel(species="SO3") + DTS * P) /(1.0+ DTS * L) 

        #          HSO3             Y.sel(species="HSO3") 
        P = EM.sel(species="HSO3")  \
        + (RC.sel(therm_coeffs=37) * Y.sel(species="OH") * Y.sel(species="SO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=38)) 
        Y.loc[:, :, :, "HSO3"] = (YP.sel(species="HSO3") + DTS * P) /(1.0+ DTS * L) 

        #          NA               Y.sel(species="NA") 
        P = EM.sel(species="NA") \
        + (RC.sel(therm_coeffs=40) * Y.sel(species="N2O5")) \
        + (RC.sel(therm_coeffs=39) * Y.sel(species="HNO3")) + (RC.sel(therm_coeffs=40) * Y.sel(species="N2O5")) 
        L = 0.0 
        Y.loc[:, :, :, "NA"] = (YP.sel(species="NA") + DTS * P) /(1.0+ DTS * L) 

        #          SA               Y.sel(species="SA") 
        P = EM.sel(species="SA") \
        + (RC.sel(therm_coeffs=41) * Y.sel(species="SO3")) 
        L = 0.0 
        Y.loc[:, :, :, "SA"] = (YP.sel(species="SA") + DTS * P) /(1.0+ DTS * L) 

        #          CH4              Y.sel(species="CH4") 
        P = EM.sel(species="CH4") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=42) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CH4"] = (YP.sel(species="CH4") + DTS * P) /(1.0+ DTS * L) 

        #          CH3O2            Y.sel(species="CH3O2") 
        P = EM.sel(species="CH3O2") \
        + (DJ.sel(photol_coeffs=61) * Y.sel(species="CH3CO3H")) \
        + (DJ.sel(photol_coeffs=13) * Y.sel(species="CH3COCH3")) + (DJ.sel(photol_coeffs=36) * Y.sel(species="UDCARB11")) \
        + (RC.sel(therm_coeffs=423) * Y.sel(species="OH") * Y.sel(species="CH3OOH")) + (DJ.sel(photol_coeffs=11) * Y.sel(species="CH3CHO")) \
        + (RC.sel(therm_coeffs=322) * Y.sel(species="CH3CO3")) + (RC.sel(therm_coeffs=381) * Y.sel(species="OH") * Y.sel(species="UDCARB11")) \
        + (RC.sel(therm_coeffs=165) * Y.sel(species="CH3O2NO2")) + (RC.sel(therm_coeffs=213) * Y.sel(species="CH3CO3") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=101) * Y.sel(species="OH") * Y.sel(species="CHCL3")) + (RC.sel(therm_coeffs=130) * Y.sel(species="CH3CO3") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=99) * Y.sel(species="OH") * Y.sel(species="CH3CL")) + (RC.sel(therm_coeffs=100) * Y.sel(species="OH") * Y.sel(species="CH2CL2")) \
        + (RC.sel(therm_coeffs=57) * Y.sel(species="O3") * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=98) * Y.sel(species="CH3CO2H") * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=42) * Y.sel(species="OH") * Y.sel(species="CH4")) + (RC.sel(therm_coeffs=55) * Y.sel(species="O3") * Y.sel(species="C3H6")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=292)) + (RC.sel(therm_coeffs=293)) \
        + (RC.sel(therm_coeffs=190) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=244) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=291)) \
        + (RC.sel(therm_coeffs=107) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=164) * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=166) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "CH3O2"] = (YP.sel(species="CH3O2") + DTS * P) /(1.0+ DTS * L) 

        #          C2H6             Y.sel(species="C2H6") 
        P = EM.sel(species="C2H6") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=43) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "C2H6"] = (YP.sel(species="C2H6") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5O2           Y.sel(species="C2H5O2") 
        P = EM.sel(species="C2H5O2") \
        + (DJ.sel(photol_coeffs=64) * Y.sel(species="RN8OOH")) \
        + (DJ.sel(photol_coeffs=57) * Y.sel(species="RN13OOH")) + (DJ.sel(photol_coeffs=62) * Y.sel(species="C2H5CO3H")) \
        + (DJ.sel(photol_coeffs=38) * Y.sel(species="UDCARB14")) + (DJ.sel(photol_coeffs=45) * Y.sel(species="RN13NO3")) \
        + (DJ.sel(photol_coeffs=17) * Y.sel(species="CARB11A")) + (DJ.sel(photol_coeffs=33) * Y.sel(species="UDCARB8")) \
        + (DJ.sel(photol_coeffs=12) * Y.sel(species="C2H5CHO")) + (DJ.sel(photol_coeffs=14) * Y.sel(species="MEK")) \
        + (RC.sel(therm_coeffs=378) * Y.sel(species="OH") * Y.sel(species="UDCARB8")) + (RC.sel(therm_coeffs=383) * Y.sel(species="OH") * Y.sel(species="UDCARB14")) \
        + (RC.sel(therm_coeffs=303) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=323) * Y.sel(species="C2H5CO3")) \
        + (RC.sel(therm_coeffs=194) * Y.sel(species="RN13O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=214) * Y.sel(species="C2H5CO3") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=111) * Y.sel(species="RN13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=131) * Y.sel(species="C2H5CO3") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=43) * Y.sel(species="OH") * Y.sel(species="C2H6")) + (RC.sel(therm_coeffs=102) * Y.sel(species="OH") * Y.sel(species="CH3CCL3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=296)) \
        + (RC.sel(therm_coeffs=245) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=294)) + (RC.sel(therm_coeffs=295)) \
        + (RC.sel(therm_coeffs=108) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=167) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=191) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "C2H5O2"] = (YP.sel(species="C2H5O2") + DTS * P) /(1.0+ DTS * L) 

        #          C3H8             Y.sel(species="C3H8") 
        P = EM.sel(species="C3H8") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=44) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=45) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "C3H8"] = (YP.sel(species="C3H8") + DTS * P) /(1.0+ DTS * L) 

        #          IC3H7O2          Y.sel(species="IC3H7O2") 
        P = EM.sel(species="IC3H7O2") \
        + (RC.sel(therm_coeffs=44) * Y.sel(species="OH") * Y.sel(species="C3H8")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=302)) \
        + (RC.sel(therm_coeffs=247) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=300)) + (RC.sel(therm_coeffs=301)) \
        + (RC.sel(therm_coeffs=110) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=169) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=193) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "IC3H7O2"] = (YP.sel(species="IC3H7O2") + DTS * P) /(1.0+ DTS * L) 

        #          RN10O2           Y.sel(species="RN10O2") 
        P = EM.sel(species="RN10O2") \
        + (DJ.sel(photol_coeffs=35) * Y.sel(species="UDCARB11")) + (DJ.sel(photol_coeffs=65) * Y.sel(species="RN11OOH")) \
        + (DJ.sel(photol_coeffs=15) * Y.sel(species="CARB14")) + (DJ.sel(photol_coeffs=16) * Y.sel(species="CARB17")) \
        + (RC.sel(therm_coeffs=45) * Y.sel(species="OH") * Y.sel(species="C3H8")) + (RC.sel(therm_coeffs=380) * Y.sel(species="OH") * Y.sel(species="UDCARB11")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=299)) \
        + (RC.sel(therm_coeffs=246) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=297)) + (RC.sel(therm_coeffs=298)) \
        + (RC.sel(therm_coeffs=109) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=168) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=192) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN10O2"] = (YP.sel(species="RN10O2") + DTS * P) /(1.0+ DTS * L) 

        #          NC4H10           Y.sel(species="NC4H10") 
        P = EM.sel(species="NC4H10") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=46) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "NC4H10"] = (YP.sel(species="NC4H10") + DTS * P) /(1.0+ DTS * L) 

        #          RN13O2           Y.sel(species="RN13O2") 
        P = EM.sel(species="RN13O2") \
        + (DJ.sel(photol_coeffs=95) * Y.sel(species="RTX22OOH")) \
        + (DJ.sel(photol_coeffs=37) * Y.sel(species="UDCARB14")) + (DJ.sel(photol_coeffs=66) * Y.sel(species="RN14OOH")) \
        + (RC.sel(therm_coeffs=358) * Y.sel(species="OH") * Y.sel(species="CARB13")) + (RC.sel(therm_coeffs=382) * Y.sel(species="OH") * Y.sel(species="UDCARB14")) \
        + (RC.sel(therm_coeffs=242) * Y.sel(species="RTX22O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=351) * Y.sel(species="RTX22O2")) \
        + (RC.sel(therm_coeffs=46) * Y.sel(species="OH") * Y.sel(species="NC4H10")) + (RC.sel(therm_coeffs=163) * Y.sel(species="RTX22O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=303)) + (RC.sel(therm_coeffs=304)) \
        + (RC.sel(therm_coeffs=194) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=195) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=248) * Y.sel(species="HO2")) \
        + (RC.sel(therm_coeffs=111) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=112) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=170) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RN13O2"] = (YP.sel(species="RN13O2") + DTS * P) /(1.0+ DTS * L) 

        #          C2H4             Y.sel(species="C2H4") 
        P = EM.sel(species="C2H4") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=54) * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=47) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=50) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=53) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "C2H4"] = (YP.sel(species="C2H4") + DTS * P) /(1.0+ DTS * L) 

        #          HOCH2CH2O2       Y.sel(species="HOCH2CH2O2") 
        P = EM.sel(species="HOCH2CH2O2") \
        + (RC.sel(therm_coeffs=466) * Y.sel(species="OH") * Y.sel(species="ANHY")) \
        + (RC.sel(therm_coeffs=105) * Y.sel(species="OH") * Y.sel(species="CDICLETH")) + (RC.sel(therm_coeffs=106) * Y.sel(species="OH") * Y.sel(species="TDICLETH")) \
        + (RC.sel(therm_coeffs=103) * Y.sel(species="OH") * Y.sel(species="TCE")) + (RC.sel(therm_coeffs=104) * Y.sel(species="OH") * Y.sel(species="TRICLETH")) \
        + (RC.sel(therm_coeffs=47) * Y.sel(species="OH") * Y.sel(species="C2H4")) + (RC.sel(therm_coeffs=92) * Y.sel(species="OH") * Y.sel(species="C2H5OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=314)) + (RC.sel(therm_coeffs=315)) \
        + (RC.sel(therm_coeffs=205) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=206) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=257) * Y.sel(species="HO2")) \
        + (RC.sel(therm_coeffs=122) *  Y.sel(species="NO")) + (RC.sel(therm_coeffs=123) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=173) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "HOCH2CH2O2"] = (YP.sel(species="HOCH2CH2O2") + DTS * P) /(1.0+ DTS * L) 

        #          C3H6             Y.sel(species="C3H6") 
        P = EM.sel(species="C3H6") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=56) * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=48) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=51) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=55) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "C3H6"] = (YP.sel(species="C3H6") + DTS * P) /(1.0+ DTS * L) 

        #          RN9O2            Y.sel(species="RN902") 
        P = EM.sel(species="RN902") \
        + (RC.sel(therm_coeffs=96) * Y.sel(species="OH") * Y.sel(species="IPROPOL")) + (RC.sel(therm_coeffs=368) * Y.sel(species="OH") * Y.sel(species="CARB9")) \
        + (RC.sel(therm_coeffs=48) * Y.sel(species="OH") * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=94) * Y.sel(species="NPROPOL") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=258) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=316)) \
        + (RC.sel(therm_coeffs=124) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=174) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=207) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN902"] = (YP.sel(species="RN902") + DTS * P) /(1.0+ DTS * L) 

        #          TBUT2ENE         Y.sel(species="TBUT2ENE") 
        P = EM.sel(species="TBUT2ENE") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=58) * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=49) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=52) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=57) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "TBUT2ENE"] = (YP.sel(species="TBUT2ENE") + DTS * P) /(1.0+ DTS * L) 

        #          RN12O2           Y.sel(species="RN12O2") 
        P = EM.sel(species="RN12O2") \
        + (RC.sel(therm_coeffs=369) * Y.sel(species="OH") * Y.sel(species="CARB12")) + (RC.sel(therm_coeffs=371) * Y.sel(species="OH") * Y.sel(species="CCARB12")) \
        + (RC.sel(therm_coeffs=198) * Y.sel(species="RN13AO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=305) * Y.sel(species="RN13AO2")) \
        + (RC.sel(therm_coeffs=49) * Y.sel(species="OH") * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=115) * Y.sel(species="RN13AO2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=259) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=317)) \
        + (RC.sel(therm_coeffs=125) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=175) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=208) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN12O2"] = (YP.sel(species="RN12O2") + DTS * P) /(1.0+ DTS * L) 

        #          NRN6O2           Y.sel(species="NRN6O2") 
        P = EM.sel(species="NRN6O2") \
        + (RC.sel(therm_coeffs=50) * Y.sel(species="NO3") * Y.sel(species="C2H4")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=336)) \
        + (RC.sel(therm_coeffs=144) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=227) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=274) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRN6O2"] = (YP.sel(species="NRN6O2") + DTS * P) /(1.0+ DTS * L) 

        #          NRN9O2           Y.sel(species="NRN9O2") 
        P = EM.sel(species="NRN9O2") \
        + (RC.sel(therm_coeffs=51) * Y.sel(species="NO3") * Y.sel(species="C3H6")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=337)) \
        + (RC.sel(therm_coeffs=145) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=228) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=275) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRN9O2"] = (YP.sel(species="NRN9O2") + DTS * P) /(1.0+ DTS * L) 

        #          NRN12O2          Y.sel(species="NRN12O2") 
        P = EM.sel(species="NRN12O2") \
        + (RC.sel(therm_coeffs=52) * Y.sel(species="NO3") * Y.sel(species="TBUT2ENE")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=338)) \
        + (RC.sel(therm_coeffs=146) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=229) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=276) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRN12O2"] = (YP.sel(species="NRN12O2") + DTS * P) /(1.0+ DTS * L) 

        #          HCHO             Y.sel(species="HCHO") 
        P = EM.sel(species="HCHO") \
        + (DJ.sel(photol_coeffs=93) * Y.sel(species="RTX28OOH")) + (DJ.sel(photol_coeffs=96) * Y.sel(species="NRTX28OOH")) \
        + (DJ.sel(photol_coeffs=80) * Y.sel(species="NRN9OOH")) + (DJ.sel(photol_coeffs=91) * Y.sel(species="RTN14OOH")) \
        + (DJ.sel(photol_coeffs=75) * Y.sel(species="RN9OOH")) + (DJ.sel(photol_coeffs=79) * Y.sel(species="NRN6OOH") * 2.00) \
        + (DJ.sel(photol_coeffs=74) * Y.sel(species="HOC2H4OOH") * 2.00) \
        + (DJ.sel(photol_coeffs=63) * Y.sel(species="HOCH2CO3H")) + (DJ.sel(photol_coeffs=69) * Y.sel(species="RU14OOH")) \
        + (DJ.sel(photol_coeffs=41) * Y.sel(species="CH3NO3")) + (DJ.sel(photol_coeffs=53) * Y.sel(species="CH3OOH")) \
        + (DJ.sel(photol_coeffs=31) * Y.sel(species="NOA")) + (DJ.sel(photol_coeffs=32) * Y.sel(species="NOA")) \
        + (DJ.sel(photol_coeffs=22) * Y.sel(species="HOCH2CHO")) + (DJ.sel(photol_coeffs=23) * Y.sel(species="UCARB10")) \
        + (RC.sel(therm_coeffs=475) * Y.sel(species="OH") * Y.sel(species="PHAN")) + (DJ.sel(photol_coeffs=18) * Y.sel(species="CARB7")) \
        + (RC.sel(therm_coeffs=449) * Y.sel(species="OH") * Y.sel(species="NRN9OOH")) + (RC.sel(therm_coeffs=473) * Y.sel(species="OH") * Y.sel(species="PAN")) \
        + (RC.sel(therm_coeffs=424) * Y.sel(species="OH") * Y.sel(species="CH3OOH")) + (RC.sel(therm_coeffs=448) * Y.sel(species="OH") * Y.sel(species="NRN6OOH") * 2.00) \
        + (RC.sel(therm_coeffs=392) * Y.sel(species="OH") * Y.sel(species="CH3NO3")) + (RC.sel(therm_coeffs=410) * Y.sel(species="OH") * Y.sel(species="RTX28NO3")) \
        + (RC.sel(therm_coeffs=362) * Y.sel(species="O3") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=363) * Y.sel(species="O3") * Y.sel(species="UCARB10")) \
        + (RC.sel(therm_coeffs=349) * Y.sel(species="RTX28O2")) + (RC.sel(therm_coeffs=352) * Y.sel(species="NRTX28O2")) \
        + (RC.sel(therm_coeffs=337) * Y.sel(species="NRN9O2")) + (RC.sel(therm_coeffs=347) * Y.sel(species="RTN14O2")) \
        + (RC.sel(therm_coeffs=336) * Y.sel(species="NRN6O2") * 2.00) \
        + (RC.sel(therm_coeffs=334) * Y.sel(species="RU10O2")) + (RC.sel(therm_coeffs=335) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=325) * Y.sel(species="RN8O2")) + (RC.sel(therm_coeffs=330) * Y.sel(species="RU14O2")) \
        + (RC.sel(therm_coeffs=316) * Y.sel(species="RN902")) + (RC.sel(therm_coeffs=324) * Y.sel(species="HOCH2CO3")) \
        + (RC.sel(therm_coeffs=314) * Y.sel(species="HOCH2CH2O2") * 2.00) \
        + (RC.sel(therm_coeffs=291) * Y.sel(species="CH3O2")) + (RC.sel(therm_coeffs=292) * Y.sel(species="CH3O2")) \
        + (RC.sel(therm_coeffs=240) * Y.sel(species="RTX28O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=243) * Y.sel(species="NRTX28O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=228) * Y.sel(species="NRN9O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=238) * Y.sel(species="RTN14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=227) * Y.sel(species="NRN6O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=227) * Y.sel(species="NRN6O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=225) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=226) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=216) * Y.sel(species="RN8O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=221) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=207) * Y.sel(species="RN902") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=215) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=205) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=162) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=190) * Y.sel(species="CH3O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=158) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=160) * Y.sel(species="NRTX28O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=145) * Y.sel(species="NRN9O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=156) * Y.sel(species="RTN14O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=144) * Y.sel(species="NRN6O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=144) * Y.sel(species="NRN6O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=142) * Y.sel(species="RU10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=143) * Y.sel(species="RU10O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=133) * Y.sel(species="RN8O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=138) * Y.sel(species="RU14O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=124) * Y.sel(species="RN902") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=132) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=122) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=90) * Y.sel(species="OH") * Y.sel(species="CH3OH")) + (RC.sel(therm_coeffs=107) * Y.sel(species="CH3O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=71) * Y.sel(species="BPINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=72) * Y.sel(species="BPINENE") * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=55) * Y.sel(species="O3") * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=56) * Y.sel(species="O3") * Y.sel(species="C3H6")) \
        + (RC.sel(therm_coeffs=53) * Y.sel(species="O3") * Y.sel(species="C2H4")) + (RC.sel(therm_coeffs=54) * Y.sel(species="O3") * Y.sel(species="C2H4")) 
        L = 0.0 \
        + (DJ.sel(photol_coeffs=10)) \
        + (RC.sel(therm_coeffs=82) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=85) * Y.sel(species="NO3")) + (DJ.sel(photol_coeffs=9)) 
        Y.loc[:, :, :, "HCHO"] = (YP.sel(species="HCHO") + DTS * P) /(1.0+ DTS * L) 

        #          HCOOH            Y.sel(species="HCOOH") 
        P = EM.sel(species="HCOOH") \
        + (RC.sel(therm_coeffs=74) * Y.sel(species="C2H2") * Y.sel(species="OH")) \
        + (RC.sel(therm_coeffs=54) * Y.sel(species="O3") * Y.sel(species="C2H4")) + (RC.sel(therm_coeffs=62) * Y.sel(species="O3") * Y.sel(species="C5H8")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=97) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "HCOOH"] = (YP.sel(species="HCOOH") + DTS * P) /(1.0+ DTS * L) 

        #          CH3CO2H          Y.sel(species="CH3CO2H") 
        P = EM.sel(species="CH3CO2H") \
        + (RC.sel(therm_coeffs=56) * Y.sel(species="O3") * Y.sel(species="C3H6")) + (RC.sel(therm_coeffs=58) * Y.sel(species="O3") * Y.sel(species="TBUT2ENE")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=98) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CH3CO2H"] = (YP.sel(species="CH3CO2H") + DTS * P) /(1.0+ DTS * L) 

        #          CH3CHO           Y.sel(species="CH3CHO") 
        P = EM.sel(species="CH3CHO") \
        + (DJ.sel(photol_coeffs=81) * Y.sel(species="NRN12OOH") * 2.00) \
        + (DJ.sel(photol_coeffs=77) * Y.sel(species="RN15OOH")) + (DJ.sel(photol_coeffs=80) * Y.sel(species="NRN9OOH")) \
        + (DJ.sel(photol_coeffs=76) * Y.sel(species="RN12OOH") * 2.00) \
        + (DJ.sel(photol_coeffs=57) * Y.sel(species="RN13OOH")) + (DJ.sel(photol_coeffs=75) * Y.sel(species="RN9OOH")) \
        + (DJ.sel(photol_coeffs=45) * Y.sel(species="RN13NO3")) + (DJ.sel(photol_coeffs=54) * Y.sel(species="C2H5OOH")) \
        + (DJ.sel(photol_coeffs=20) * Y.sel(species="CARB13")) + (DJ.sel(photol_coeffs=42) * Y.sel(species="C2H5NO3")) \
        + (RC.sel(therm_coeffs=474) * Y.sel(species="OH") * Y.sel(species="PPN")) + (DJ.sel(photol_coeffs=19) * Y.sel(species="CARB10")) \
        + (RC.sel(therm_coeffs=449) * Y.sel(species="OH") * Y.sel(species="NRN9OOH")) + (RC.sel(therm_coeffs=450) * Y.sel(species="OH") * Y.sel(species="NRN12OOH") * 2.00) \
        + (RC.sel(therm_coeffs=393) * Y.sel(species="OH") * Y.sel(species="C2H5NO3")) + (RC.sel(therm_coeffs=425) * Y.sel(species="OH") * Y.sel(species="C2H5OOH")) \
        + (RC.sel(therm_coeffs=338) * Y.sel(species="NRN12O2") * 2.00) \
        + (RC.sel(therm_coeffs=327) * Y.sel(species="RN14O2")) + (RC.sel(therm_coeffs=337) * Y.sel(species="NRN9O2")) \
        + (RC.sel(therm_coeffs=318) * Y.sel(species="RN15O2")) + (RC.sel(therm_coeffs=326) * Y.sel(species="RN11O2")) \
        + (RC.sel(therm_coeffs=317) * Y.sel(species="RN12O2") * 2.00) \
        + (RC.sel(therm_coeffs=303) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=316) * Y.sel(species="RN902")) \
        + (RC.sel(therm_coeffs=294) * Y.sel(species="C2H5O2")) + (RC.sel(therm_coeffs=295) * Y.sel(species="C2H5O2")) \
        + (RC.sel(therm_coeffs=229) * Y.sel(species="NRN12O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=218) * Y.sel(species="RN14O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=228) * Y.sel(species="NRN9O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=209) * Y.sel(species="RN15O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=217) * Y.sel(species="RN11O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=207) * Y.sel(species="RN902") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=208) * Y.sel(species="RN12O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=191) * Y.sel(species="C2H5O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=194) * Y.sel(species="RN13O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=146) * Y.sel(species="NRN12O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=135) * Y.sel(species="RN14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=145) * Y.sel(species="NRN9O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=126) * Y.sel(species="RN15O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=134) * Y.sel(species="RN11O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=125) * Y.sel(species="RN12O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=111) * Y.sel(species="RN13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=124) * Y.sel(species="RN902") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=91) * Y.sel(species="OH") * Y.sel(species="C2H5OH")) + (RC.sel(therm_coeffs=108) * Y.sel(species="C2H5O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=57) * Y.sel(species="O3") * Y.sel(species="TBUT2ENE")) + (RC.sel(therm_coeffs=58) * Y.sel(species="O3") * Y.sel(species="TBUT2ENE")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=83) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=86) * Y.sel(species="NO3")) + (DJ.sel(photol_coeffs=11)) 
        Y.loc[:, :, :, "CH3CHO"] = (YP.sel(species="CH3CHO") + DTS * P) /(1.0+ DTS * L) 

        #          C5H8             Y.sel(species="C5H8") 
        P = EM.sel(species="C5H8") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=62) * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=59) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=60) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=61) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "C5H8"] = (YP.sel(species="C5H8") + DTS * P) /(1.0+ DTS * L) 

        #          RU14O2           Y.sel(species="RU14O2") 
        P = EM.sel(species="RU14O2") \
        + (RC.sel(therm_coeffs=59) * Y.sel(species="OH") * Y.sel(species="C5H8")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=329)) + (RC.sel(therm_coeffs=330)) \
        + (RC.sel(therm_coeffs=220) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=221) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=271) * Y.sel(species="HO2")) \
        + (RC.sel(therm_coeffs=137) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=138) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=180) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RU14O2"] = (YP.sel(species="RU14O2") + DTS * P) /(1.0+ DTS * L) 

        #          NRU14O2          Y.sel(species="NRU14O2") 
        P = EM.sel(species="NRU14O2") \
        + (RC.sel(therm_coeffs=60) * Y.sel(species="NO3") * Y.sel(species="C5H8")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=339)) \
        + (RC.sel(therm_coeffs=147) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=230) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=277) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRU14O2"] = (YP.sel(species="NRU14O2") + DTS * P) /(1.0+ DTS * L) 

        #          UCARB10          Y.sel(species="UCARB10") 
        P = EM.sel(species="UCARB10") \
        + (DJ.sel(photol_coeffs=69) * Y.sel(species="RU14OOH")) \
        + (RC.sel(therm_coeffs=330) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=481) * Y.sel(species="OH") * Y.sel(species="RU12PAN")) \
        + (RC.sel(therm_coeffs=138) * Y.sel(species="RU14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=221) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=61) * Y.sel(species="O3") * Y.sel(species="C5H8")) + (RC.sel(therm_coeffs=62) * Y.sel(species="O3") * Y.sel(species="C5H8")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=363) * Y.sel(species="O3")) + (DJ.sel(photol_coeffs=23)) \
        + (RC.sel(therm_coeffs=360) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=361) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=362) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "UCARB10"] = (YP.sel(species="UCARB10") + DTS * P) /(1.0+ DTS * L) 

        #          APINENE          Y.sel(species="APINENE") 
        P = EM.sel(species="APINENE") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=66) * Y.sel(species="O3")) + (RC.sel(therm_coeffs=67) * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=63) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=64) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=65) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "APINENE"] = (YP.sel(species="APINENE") + DTS * P) /(1.0+ DTS * L) 

        #          RTN28O2          Y.sel(species="RTN28O2") 
        P = EM.sel(species="RTN28O2") \
        + (RC.sel(therm_coeffs=63) * Y.sel(species="APINENE") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=232) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=279) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=341)) \
        + (RC.sel(therm_coeffs=149) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=150) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=185) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RTN28O2"] = (YP.sel(species="RTN28O2") + DTS * P) /(1.0+ DTS * L) 

        #          NRTN28O2         Y.sel(species="NRTN28O2") 
        P = EM.sel(species="NRTN28O2") \
        + (RC.sel(therm_coeffs=64) * Y.sel(species="APINENE") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=342)) \
        + (RC.sel(therm_coeffs=151) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=233) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=280) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRTN28O2"] = (YP.sel(species="NRTN28O2") + DTS * P) /(1.0+ DTS * L) 

        #          RTN26O2          Y.sel(species="RTN26O2") 
        P = EM.sel(species="RTN26O2") \
        + (RC.sel(therm_coeffs=483) * Y.sel(species="RTN26PAN")) + (DJ.sel(photol_coeffs=39) * Y.sel(species="TNCARB26")) \
        + (RC.sel(therm_coeffs=387) * Y.sel(species="NO3") * Y.sel(species="TNCARB26")) + (RC.sel(therm_coeffs=455) * Y.sel(species="OH") * Y.sel(species="RTN26OOH")) \
        + (RC.sel(therm_coeffs=65) * Y.sel(species="APINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=384) * Y.sel(species="OH") * Y.sel(species="TNCARB26")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=343)) + (RC.sel(therm_coeffs=482) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=152) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=234) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=281) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RTN26O2"] = (YP.sel(species="RTN26O2") + DTS * P) /(1.0+ DTS * L) 

        #          TNCARB26         Y.sel(species="TNCARB26") 
        P = EM.sel(species="TNCARB26") \
        + (DJ.sel(photol_coeffs=85) * Y.sel(species="RTN28OOH")) + (DJ.sel(photol_coeffs=86) * Y.sel(species="NRTN28OOH")) \
        + (RC.sel(therm_coeffs=454) * Y.sel(species="OH") * Y.sel(species="RTN28OOH")) + (RC.sel(therm_coeffs=456) * Y.sel(species="OH") * Y.sel(species="NRTN28OOH")) \
        + (RC.sel(therm_coeffs=342) * Y.sel(species="NRTN28O2")) + (RC.sel(therm_coeffs=408) * Y.sel(species="OH") * Y.sel(species="RTN28NO3")) \
        + (RC.sel(therm_coeffs=233) * Y.sel(species="NRTN28O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=341) * Y.sel(species="RTN28O2")) \
        + (RC.sel(therm_coeffs=151) * Y.sel(species="NRTN28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=232) * Y.sel(species="RTN28O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=66) * Y.sel(species="APINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=149) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=384) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=387) * Y.sel(species="NO3")) + (DJ.sel(photol_coeffs=39)) 
        Y.loc[:, :, :, "TNCARB26"] = (YP.sel(species="TNCARB26") + DTS * P) /(1.0+ DTS * L) 

        #          RCOOH25          Y.sel(species="RCOOH25") 
        P = EM.sel(species="RCOOH25") \
        + (RC.sel(therm_coeffs=67) * Y.sel(species="APINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=490) * Y.sel(species="P2631")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=389) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=489)) 
        Y.loc[:, :, :, "RCOOH25"] = (YP.sel(species="RCOOH25") + DTS * P) /(1.0+ DTS * L) 

        #          BPINENE          Y.sel(species="BPINENE") 
        P = EM.sel(species="BPINENE") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=71) * Y.sel(species="O3")) + (RC.sel(therm_coeffs=72) * Y.sel(species="O3")) + (RC.sel(therm_coeffs=73) * Y.sel(species="O3")) \
        + (RC.sel(therm_coeffs=68) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=69) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=70) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "BPINENE"] = (YP.sel(species="BPINENE") + DTS * P) /(1.0+ DTS * L) 

        #          RTX28O2          Y.sel(species="RTX28O2") 
        P = EM.sel(species="RTX28O2") \
        + (RC.sel(therm_coeffs=68) * Y.sel(species="BPINENE") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=462) * Y.sel(species="OH") * Y.sel(species="RTX28OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=240) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=287) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=349)) \
        + (RC.sel(therm_coeffs=158) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=159) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=187) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RTX28O2"] = (YP.sel(species="RTX28O2") + DTS * P) /(1.0+ DTS * L) 

        #          NRTX28O2         Y.sel(species="NRTX28O2") 
        P = EM.sel(species="NRTX28O2") \
        + (RC.sel(therm_coeffs=69) * Y.sel(species="BPINENE") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=465) * Y.sel(species="OH") * Y.sel(species="NRTX28OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=352)) \
        + (RC.sel(therm_coeffs=160) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=243) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=290) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRTX28O2"] = (YP.sel(species="NRTX28O2") + DTS * P) /(1.0+ DTS * L) 

        #          RTX24O2          Y.sel(species="RTX24O2") 
        P = EM.sel(species="RTX24O2") \
        + (RC.sel(therm_coeffs=70) * Y.sel(species="BPINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=390) * Y.sel(species="OH") * Y.sel(species="TXCARB24")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=241) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=288) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=350)) \
        + (RC.sel(therm_coeffs=161) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=162) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=188) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RTX24O2"] = (YP.sel(species="RTX24O2") + DTS * P) /(1.0+ DTS * L) 

        #          TXCARB24         Y.sel(species="TXCARB24") 
        P = EM.sel(species="TXCARB24") \
        + (DJ.sel(photol_coeffs=96) * Y.sel(species="NRTX28OOH")) \
        + (RC.sel(therm_coeffs=410) * Y.sel(species="OH") * Y.sel(species="RTX28NO3")) + (DJ.sel(photol_coeffs=93) * Y.sel(species="RTX28OOH")) \
        + (RC.sel(therm_coeffs=349) * Y.sel(species="RTX28O2")) + (RC.sel(therm_coeffs=352) * Y.sel(species="NRTX28O2")) \
        + (RC.sel(therm_coeffs=240) * Y.sel(species="RTX28O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=243) * Y.sel(species="NRTX28O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=158) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=160) * Y.sel(species="NRTX28O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=71) * Y.sel(species="BPINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=73) * Y.sel(species="BPINENE") * Y.sel(species="O3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=390) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TXCARB24"] = (YP.sel(species="TXCARB24") + DTS * P) /(1.0+ DTS * L) 

        #          TXCARB22         Y.sel(species="TXCARB22") 
        P = EM.sel(species="TXCARB22") \
        + (DJ.sel(photol_coeffs=52) * Y.sel(species="RTX24NO3")) + (DJ.sel(photol_coeffs=94) * Y.sel(species="RTX24OOH")) \
        + (RC.sel(therm_coeffs=411) * Y.sel(species="OH") * Y.sel(species="RTX24NO3")) + (RC.sel(therm_coeffs=463) * Y.sel(species="OH") * Y.sel(species="RTX24OOH")) \
        + (RC.sel(therm_coeffs=241) * Y.sel(species="RTX24O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=350) * Y.sel(species="RTX24O2")) \
        + (RC.sel(therm_coeffs=72) * Y.sel(species="BPINENE") * Y.sel(species="O3")) + (RC.sel(therm_coeffs=161) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=391) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TXCARB22"] = (YP.sel(species="TXCARB22") + DTS * P) /(1.0+ DTS * L) 

        #          C2H2             Y.sel(species="C2H2") 
        P = EM.sel(species="C2H2") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=74) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=75) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "C2H2"] = (YP.sel(species="C2H2") + DTS * P) /(1.0+ DTS * L) 

        #          CARB3            Y.sel(species="CARB3") 
        P = EM.sel(species="CARB3") \
        + (DJ.sel(photol_coeffs=83) * Y.sel(species="RA16OOH")) \
        + (DJ.sel(photol_coeffs=50) * Y.sel(species="RA16NO3")) + (DJ.sel(photol_coeffs=82) * Y.sel(species="RA13OOH")) \
        + (RC.sel(therm_coeffs=452) * Y.sel(species="OH") * Y.sel(species="RA16OOH")) + (DJ.sel(photol_coeffs=49) * Y.sel(species="RA13NO3")) \
        + (RC.sel(therm_coeffs=406) * Y.sel(species="OH") * Y.sel(species="RA16NO3")) + (RC.sel(therm_coeffs=451) * Y.sel(species="OH") * Y.sel(species="RA13OOH")) \
        + (RC.sel(therm_coeffs=311) * Y.sel(species="RA19CO2")) + (RC.sel(therm_coeffs=405) * Y.sel(species="OH") * Y.sel(species="RA13NO3")) \
        + (RC.sel(therm_coeffs=308) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=310) * Y.sel(species="RA19AO2")) \
        + (RC.sel(therm_coeffs=203) * Y.sel(species="RA19AO2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=307) * Y.sel(species="RA13O2")) \
        + (RC.sel(therm_coeffs=200) * Y.sel(species="RA13O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=201) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=118) * Y.sel(species="RA16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=120) * Y.sel(species="RA19AO2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=75) * Y.sel(species="C2H2") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=117) * Y.sel(species="RA13O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=366) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=24)) 
        Y.loc[:, :, :, "CARB3"] = (YP.sel(species="CARB3") + DTS * P) /(1.0+ DTS * L) 

        #          BENZENE          Y.sel(species="BENZENE") 
        P = EM.sel(species="BENZENE") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=76) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=77) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "BENZENE"] = (YP.sel(species="BENZENE") + DTS * P) /(1.0+ DTS * L) 

        #          RA13O2           Y.sel(species="RA13O2") 
        P = EM.sel(species="RA13O2") \
        + (RC.sel(therm_coeffs=76) * Y.sel(species="BENZENE") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=253) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=307)) \
        + (RC.sel(therm_coeffs=117) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=181) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=200) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RA13O2"] = (YP.sel(species="RA13O2") + DTS * P) /(1.0+ DTS * L) 

        #          AROH14           Y.sel(species="AROH14") 
        P = EM.sel(species="AROH14") \
        + (RC.sel(therm_coeffs=77) * Y.sel(species="BENZENE") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=413) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=414) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "AROH14"] = (YP.sel(species="AROH14") + DTS * P) /(1.0+ DTS * L) 

        #          TOLUENE          Y.sel(species="TOLUENE") 
        P = EM.sel(species="TOLUENE") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=78) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=79) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TOLUENE"] = (YP.sel(species="TOLUENE") + DTS * P) /(1.0+ DTS * L) 

        #          RA16O2           Y.sel(species="RA16O2") 
        P = EM.sel(species="RA16O2") \
        + (RC.sel(therm_coeffs=78) * Y.sel(species="TOLUENE") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=308)) + (RC.sel(therm_coeffs=309)) \
        + (RC.sel(therm_coeffs=201) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=202) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=254) * Y.sel(species="HO2")) \
        + (RC.sel(therm_coeffs=118) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=119) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=182) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RA16O2"] = (YP.sel(species="RA16O2") + DTS * P) /(1.0+ DTS * L) 

        #          AROH17           Y.sel(species="AROH17") 
        P = EM.sel(species="AROH17") \
        + (RC.sel(therm_coeffs=79) * Y.sel(species="TOLUENE") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=418) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=419) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "AROH17"] = (YP.sel(species="AROH17") + DTS * P) /(1.0+ DTS * L) 

        #          OXYL             Y.sel(species="OXYL") 
        P = EM.sel(species="OXYL") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=80) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=81) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "OXYL"] = (YP.sel(species="OXYL") + DTS * P) /(1.0+ DTS * L) 

        #          RA19AO2          Y.sel(species="RA19AO2") 
        P = EM.sel(species="RA19AO2") \
        + (RC.sel(therm_coeffs=80) * Y.sel(species="OXYL") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=255) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=310)) \
        + (RC.sel(therm_coeffs=120) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=183) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=203) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RA19AO2"] = (YP.sel(species="RA19AO2") + DTS * P) /(1.0+ DTS * L) 

        #          RA19CO2          Y.sel(species="RA19CO2") 
        P = EM.sel(species="RA19CO2") \
        + (RC.sel(therm_coeffs=81) * Y.sel(species="OXYL") * Y.sel(species="OH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=256) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=311)) \
        + (RC.sel(therm_coeffs=121) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=184) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=204) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RA19CO2"] = (YP.sel(species="RA19CO2") + DTS * P) /(1.0+ DTS * L) 

        #          CH3CO3           Y.sel(species="CH3CO3") 
        P = EM.sel(species="CH3CO3") \
        + (DJ.sel(photol_coeffs=71) * Y.sel(species="RU10OOH")) \
        + (DJ.sel(photol_coeffs=40) * Y.sel(species="TNCARB10") * 2.00) \
        + (DJ.sel(photol_coeffs=31) * Y.sel(species="NOA")) + (DJ.sel(photol_coeffs=32) * Y.sel(species="NOA")) \
        + (DJ.sel(photol_coeffs=27) * Y.sel(species="CARB12")) + (DJ.sel(photol_coeffs=29) * Y.sel(species="UCARB12")) \
        + (DJ.sel(photol_coeffs=25) * Y.sel(species="CARB6")) + (DJ.sel(photol_coeffs=26) * Y.sel(species="CARB9") * 2.00) \
        + (DJ.sel(photol_coeffs=19) * Y.sel(species="CARB10")) + (DJ.sel(photol_coeffs=23) * Y.sel(species="UCARB10")) \
        + (DJ.sel(photol_coeffs=17) * Y.sel(species="CARB11A")) + (DJ.sel(photol_coeffs=18) * Y.sel(species="CARB7")) \
        + (DJ.sel(photol_coeffs=14) * Y.sel(species="MEK")) + (DJ.sel(photol_coeffs=15) * Y.sel(species="CARB14")) \
        + (RC.sel(therm_coeffs=468) * Y.sel(species="PAN")) + (DJ.sel(photol_coeffs=13) * Y.sel(species="CH3COCH3")) \
        + (RC.sel(therm_coeffs=374) * Y.sel(species="O3") * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=431) * Y.sel(species="OH") * Y.sel(species="CH3CO3H")) \
        + (RC.sel(therm_coeffs=362) * Y.sel(species="O3") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=367) * Y.sel(species="OH") * Y.sel(species="CARB6")) \
        + (RC.sel(therm_coeffs=331) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=333) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=325) * Y.sel(species="RN8O2")) + (RC.sel(therm_coeffs=326) * Y.sel(species="RN11O2")) \
        + (RC.sel(therm_coeffs=222) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=224) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=216) * Y.sel(species="RN8O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=217) * Y.sel(species="RN11O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=139) * Y.sel(species="RU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=141) * Y.sel(species="RU10O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=133) * Y.sel(species="RN8O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=134) * Y.sel(species="RN11O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=83) * Y.sel(species="OH") * Y.sel(species="CH3CHO")) + (RC.sel(therm_coeffs=86) * Y.sel(species="NO3") * Y.sel(species="CH3CHO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=322)) + (RC.sel(therm_coeffs=467) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=130) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=213) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=264) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "CH3CO3"] = (YP.sel(species="CH3CO3") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5CHO          Y.sel(species="C2H5CHO") 
        P = EM.sel(species="C2H5CHO") \
        + (DJ.sel(photol_coeffs=78) * Y.sel(species="RN18OOH") * 2.00) \
        + (DJ.sel(photol_coeffs=55) * Y.sel(species="RN10OOH")) + (DJ.sel(photol_coeffs=77) * Y.sel(species="RN15OOH")) \
        + (DJ.sel(photol_coeffs=21) * Y.sel(species="CARB16")) + (DJ.sel(photol_coeffs=43) * Y.sel(species="RN10NO3")) \
        + (RC.sel(therm_coeffs=394) * Y.sel(species="OH") * Y.sel(species="RN10NO3")) + (RC.sel(therm_coeffs=426) * Y.sel(species="OH") * Y.sel(species="RN10OOH")) \
        + (RC.sel(therm_coeffs=318) * Y.sel(species="RN15O2")) + (RC.sel(therm_coeffs=319) * Y.sel(species="RN18O2") * 2.00) \
        + (RC.sel(therm_coeffs=297) * Y.sel(species="RN10O2")) + (RC.sel(therm_coeffs=298) * Y.sel(species="RN10O2")) \
        + (RC.sel(therm_coeffs=210) * Y.sel(species="RN18O2") * Y.sel(species="NO3") * 2.00) \
        + (RC.sel(therm_coeffs=192) * Y.sel(species="RN10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=209) * Y.sel(species="RN15O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=126) * Y.sel(species="RN15O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=127) * Y.sel(species="RN18O2") * Y.sel(species="NO") * 2.00) \
        + (RC.sel(therm_coeffs=93) * Y.sel(species="NPROPOL") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=109) * Y.sel(species="RN10O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=84) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=87) * Y.sel(species="NO3")) + (DJ.sel(photol_coeffs=12)) 
        Y.loc[:, :, :, "C2H5CHO"] = (YP.sel(species="C2H5CHO") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5CO3          Y.sel(species="C2H5CO3") 
        P = EM.sel(species="C2H5CO3") \
        + (RC.sel(therm_coeffs=470) * Y.sel(species="PPN")) \
        + (RC.sel(therm_coeffs=327) * Y.sel(species="RN14O2")) + (RC.sel(therm_coeffs=432) * Y.sel(species="OH") * Y.sel(species="C2H5CO3H")) \
        + (RC.sel(therm_coeffs=135) * Y.sel(species="RN14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=218) * Y.sel(species="RN14O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=84) * Y.sel(species="OH") * Y.sel(species="C2H5CHO")) + (RC.sel(therm_coeffs=87) * Y.sel(species="NO3") * Y.sel(species="C2H5CHO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=323)) + (RC.sel(therm_coeffs=469) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=131) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=214) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=265) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "C2H5CO3"] = (YP.sel(species="C2H5CO3") + DTS * P) /(1.0+ DTS * L) 

        #          CH3COCH3         Y.sel(species="CH3COCH3") 
        P = EM.sel(species="CH3COCH3") \
        + (DJ.sel(photol_coeffs=90) * Y.sel(species="RTN23OOH")) + (DJ.sel(photol_coeffs=95) * Y.sel(species="RTX22OOH")) \
        + (DJ.sel(photol_coeffs=44) * Y.sel(species="IC3H7NO3")) + (DJ.sel(photol_coeffs=56) * Y.sel(species="IC3H7OOH")) \
        + (RC.sel(therm_coeffs=464) * Y.sel(species="OH") * Y.sel(species="RTX22OOH")) + (RC.sel(therm_coeffs=484) * Y.sel(species="OH") * Y.sel(species="RTN26PAN")) \
        + (RC.sel(therm_coeffs=412) * Y.sel(species="OH") * Y.sel(species="RTX22NO3")) + (RC.sel(therm_coeffs=427) * Y.sel(species="OH") * Y.sel(species="IC3H7OOH")) \
        + (RC.sel(therm_coeffs=395) * Y.sel(species="OH") * Y.sel(species="IC3H7NO3")) + (RC.sel(therm_coeffs=409) * Y.sel(species="OH") * Y.sel(species="RTN25NO3")) \
        + (RC.sel(therm_coeffs=346) * Y.sel(species="RTN23O2")) + (RC.sel(therm_coeffs=351) * Y.sel(species="RTX22O2")) \
        + (RC.sel(therm_coeffs=300) * Y.sel(species="IC3H7O2")) + (RC.sel(therm_coeffs=301) * Y.sel(species="IC3H7O2")) \
        + (RC.sel(therm_coeffs=237) * Y.sel(species="RTN23O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=242) * Y.sel(species="RTX22O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=163) * Y.sel(species="RTX22O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=193) * Y.sel(species="IC3H7O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=159) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=162) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=150) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=155) * Y.sel(species="RTN23O2") * Y.sel(species="NO")) \
        + (RC.sel(therm_coeffs=95) * Y.sel(species="OH") * Y.sel(species="IPROPOL")) + (RC.sel(therm_coeffs=110) * Y.sel(species="IC3H7O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=88) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=13)) 
        Y.loc[:, :, :, "CH3COCH3"] = (YP.sel(species="CH3COCH3") + DTS * P) /(1.0+ DTS * L) 

        #          RN8O2            Y.sel(species="RN8O2") 
        P = EM.sel(species="RN8O2") \
        + (DJ.sel(photol_coeffs=92) * Y.sel(species="RTN10OOH")) \
        + (DJ.sel(photol_coeffs=28) * Y.sel(species="CARB15") * 2.00) \
        + (DJ.sel(photol_coeffs=21) * Y.sel(species="CARB16")) + (DJ.sel(photol_coeffs=27) * Y.sel(species="CARB12")) \
        + (DJ.sel(photol_coeffs=16) * Y.sel(species="CARB17")) + (DJ.sel(photol_coeffs=20) * Y.sel(species="CARB13")) \
        + (RC.sel(therm_coeffs=239) * Y.sel(species="RTN10O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=348) * Y.sel(species="RTN10O2")) \
        + (RC.sel(therm_coeffs=88) * Y.sel(species="OH") * Y.sel(species="CH3COCH3")) + (RC.sel(therm_coeffs=157) * Y.sel(species="RTN10O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=325)) \
        + (RC.sel(therm_coeffs=133) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=216) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=267) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RN8O2"] = (YP.sel(species="RN8O2") + DTS * P) /(1.0+ DTS * L) 

        #          RN11O2           Y.sel(species="RN11O2") 
        P = EM.sel(species="RN11O2") \
        + (RC.sel(therm_coeffs=89) * Y.sel(species="MEK") * Y.sel(species="OH")) + (RC.sel(therm_coeffs=355) * Y.sel(species="OH") * Y.sel(species="CARB11A")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=326)) \
        + (RC.sel(therm_coeffs=134) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=217) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=268) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RN11O2"] = (YP.sel(species="RN11O2") + DTS * P) /(1.0+ DTS * L) 

        #          CH3OH            Y.sel(species="CH3OH") 
        P = EM.sel(species="CH3OH") \
        + (RC.sel(therm_coeffs=293) * Y.sel(species="CH3O2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=90) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CH3OH"] = (YP.sel(species="CH3OH") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5OH           Y.sel(species="C2H5OH") 
        P = EM.sel(species="C2H5OH") \
        + (RC.sel(therm_coeffs=296) * Y.sel(species="C2H5O2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=91) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=92) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "C2H5OH"] = (YP.sel(species="C2H5OH") + DTS * P) /(1.0+ DTS * L) 

        #          NPROPOL          Y.sel(species="NPROPOL") 
        P = EM.sel(species="NPROPOL") \
        + (RC.sel(therm_coeffs=299) * Y.sel(species="RN10O2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=93) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=94) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "NPROPOL"] = (YP.sel(species="NPROPOL") + DTS * P) /(1.0+ DTS * L) 

        #          IPROPOL          Y.sel(species="IPROPOL") 
        P = EM.sel(species="IPROPOL") \
        + (RC.sel(therm_coeffs=302) * Y.sel(species="IC3H7O2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=95) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=96) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "IPROPOL"] = (YP.sel(species="IPROPOL") + DTS * P) /(1.0+ DTS * L) 

        #          CH3CL            Y.sel(species="CH3CL") 
        P = EM.sel(species="CH3CL") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=99) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CH3CL"] = (YP.sel(species="CH3CL") + DTS * P) /(1.0+ DTS * L) 

        #          CH2CL2           Y.sel(species="CH2CL2") 
        P = EM.sel(species="CH2CL2") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=100) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CH2CL2"] = (YP.sel(species="CH2CL2") + DTS * P) /(1.0+ DTS * L) 

        #          CHCL3            Y.sel(species="CHCL3") 
        P = EM.sel(species="CHCL3") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=101) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CHCL3"] = (YP.sel(species="CHCL3") + DTS * P) /(1.0+ DTS * L) 

        #          CH3CCL3          Y.sel(species="CH3CCL3") 
        P = EM.sel(species="CH3CCL3") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=102) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CH3CCL3"] = (YP.sel(species="CH3CCL3") + DTS * P) /(1.0+ DTS * L) 

        #          TCE              Y.sel(species="TCE") 
        P = EM.sel(species="TCE") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=103) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TCE"] = (YP.sel(species="TCE") + DTS * P) /(1.0+ DTS * L) 

        #          TRICLETH         Y.sel(species="TRICLETH") 
        P = EM.sel(species="TRICLETH") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=104) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TRICLETH"] = (YP.sel(species="TRICLETH") + DTS * P) /(1.0+ DTS * L) 

        #          CDICLETH         Y.sel(species="CDICLETH") 
        P = EM.sel(species="CDICLETH") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=105) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CDICLETH"] = (YP.sel(species="CDICLETH") + DTS * P) /(1.0+ DTS * L) 

        #          TDICLETH         Y.sel(species="TDICLETH") 
        P = EM.sel(species="TDICLETH") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=106) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TDICLETH"] = (YP.sel(species="TDICLETH") + DTS * P) /(1.0+ DTS * L) 

        #          CARB11A          Y.sel(species="CARB11A") 
        P = EM.sel(species="CARB11A") \
        + (DJ.sel(photol_coeffs=58) * Y.sel(species="RN13OOH")) \
        + (RC.sel(therm_coeffs=428) * Y.sel(species="OH") * Y.sel(species="RN13OOH")) + (DJ.sel(photol_coeffs=46) * Y.sel(species="RN13NO3")) \
        + (RC.sel(therm_coeffs=304) * Y.sel(species="RN13O2")) + (RC.sel(therm_coeffs=396) * Y.sel(species="OH") * Y.sel(species="RN13NO3")) \
        + (RC.sel(therm_coeffs=112) * Y.sel(species="RN13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=195) * Y.sel(species="RN13O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=355) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=17)) 
        Y.loc[:, :, :, "CARB11A"] = (YP.sel(species="CARB11A") + DTS * P) /(1.0+ DTS * L) 

        #          RN16O2           Y.sel(species="RN16O2") 
        P = EM.sel(species="RN16O2") \
        + (RC.sel(therm_coeffs=359) * Y.sel(species="OH") * Y.sel(species="CARB16")) + (DJ.sel(photol_coeffs=67) * Y.sel(species="RN17OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=249) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=312)) \
        + (RC.sel(therm_coeffs=113) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=171) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=196) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN16O2"] = (YP.sel(species="RN16O2") + DTS * P) /(1.0+ DTS * L) 

        #          RN15AO2          Y.sel(species="RN15AO2") 
        P = EM.sel(species="RN15AO2") \
        + (DJ.sel(photol_coeffs=59) * Y.sel(species="RN16OOH")) \
        + (RC.sel(therm_coeffs=312) * Y.sel(species="RN16O2")) + (RC.sel(therm_coeffs=385) * Y.sel(species="OH") * Y.sel(species="TNCARB15")) \
        + (RC.sel(therm_coeffs=113) * Y.sel(species="RN16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=196) * Y.sel(species="RN16O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=262) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=320)) \
        + (RC.sel(therm_coeffs=128) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=178) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=211) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN15AO2"] = (YP.sel(species="RN15AO2") + DTS * P) /(1.0+ DTS * L) 

        #          RN19O2           Y.sel(species="RN19O2") 
        P = EM.sel(species="RN19O2") \
        + (RC.sel(therm_coeffs=150) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=159) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=250) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=313)) \
        + (RC.sel(therm_coeffs=114) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=172) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=197) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN19O2"] = (YP.sel(species="RN19O2") + DTS * P) /(1.0+ DTS * L) 

        #          RN18AO2          Y.sel(species="RN18AO2") 
        P = EM.sel(species="RN18AO2") \
        + (RC.sel(therm_coeffs=313) * Y.sel(species="RN19O2")) + (DJ.sel(photol_coeffs=60) * Y.sel(species="RN19OOH")) \
        + (RC.sel(therm_coeffs=114) * Y.sel(species="RN19O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=197) * Y.sel(species="RN19O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=263) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=321)) \
        + (RC.sel(therm_coeffs=129) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=179) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=212) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN18AO2"] = (YP.sel(species="RN18AO2") + DTS * P) /(1.0+ DTS * L) 

        #          RN13AO2          Y.sel(species="RN13AO2") 
        P = EM.sel(species="RN13AO2") \
        + (RC.sel(therm_coeffs=162) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=305)) \
        + (RC.sel(therm_coeffs=115) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=198) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=251) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RN13AO2"] = (YP.sel(species="RN13AO2") + DTS * P) /(1.0+ DTS * L) 

        #          RN16AO2          Y.sel(species="RN16AO2") 
        P = EM.sel(species="RN16AO2") \
        + (RC.sel(therm_coeffs=328) * Y.sel(species="RN17O2")) \
        + (RC.sel(therm_coeffs=136) * Y.sel(species="RN17O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=219) * Y.sel(species="RN17O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=306)) \
        + (RC.sel(therm_coeffs=116) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=199) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=252) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RN16AO2"] = (YP.sel(species="RN16AO2") + DTS * P) /(1.0+ DTS * L) 

        #          RN15O2           Y.sel(species="RN15O2") 
        P = EM.sel(species="RN15O2") \
        + (DJ.sel(photol_coeffs=47) * Y.sel(species="RN16NO3")) \
        + (RC.sel(therm_coeffs=306) * Y.sel(species="RN16AO2")) + (RC.sel(therm_coeffs=370) * Y.sel(species="OH") * Y.sel(species="CARB15")) \
        + (RC.sel(therm_coeffs=116) * Y.sel(species="RN16AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=199) * Y.sel(species="RN16AO2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=260) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=318)) \
        + (RC.sel(therm_coeffs=126) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=176) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=209) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN15O2"] = (YP.sel(species="RN15O2") + DTS * P) /(1.0+ DTS * L) 

        #          UDCARB8          Y.sel(species="UDCARB8") 
        P = EM.sel(species="UDCARB8") \
        + (DJ.sel(photol_coeffs=49) * Y.sel(species="RA13NO3")) + (DJ.sel(photol_coeffs=82) * Y.sel(species="RA13OOH")) \
        + (RC.sel(therm_coeffs=405) * Y.sel(species="OH") * Y.sel(species="RA13NO3")) + (RC.sel(therm_coeffs=451) * Y.sel(species="OH") * Y.sel(species="RA13OOH")) \
        + (RC.sel(therm_coeffs=307) * Y.sel(species="RA13O2")) + (RC.sel(therm_coeffs=309) * Y.sel(species="RA16O2")) \
        + (RC.sel(therm_coeffs=202) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=204) * Y.sel(species="RA19CO2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=121) * Y.sel(species="RA19CO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=200) * Y.sel(species="RA13O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=117) * Y.sel(species="RA13O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=119) * Y.sel(species="RA16O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (DJ.sel(photol_coeffs=34)) \
        + (RC.sel(therm_coeffs=378) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=379) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=33)) 
        Y.loc[:, :, :, "UDCARB8"] = (YP.sel(species="UDCARB8") + DTS * P) /(1.0+ DTS * L) 

        #          UDCARB11         Y.sel(species="UDCARB11") 
        P = EM.sel(species="UDCARB11") \
        + (DJ.sel(photol_coeffs=84) * Y.sel(species="RA19OOH")) \
        + (DJ.sel(photol_coeffs=51) * Y.sel(species="RA19NO3")) + (DJ.sel(photol_coeffs=83) * Y.sel(species="RA16OOH")) \
        + (RC.sel(therm_coeffs=453) * Y.sel(species="OH") * Y.sel(species="RA19OOH")) + (DJ.sel(photol_coeffs=50) * Y.sel(species="RA16NO3")) \
        + (RC.sel(therm_coeffs=407) * Y.sel(species="OH") * Y.sel(species="RA19NO3")) + (RC.sel(therm_coeffs=452) * Y.sel(species="OH") * Y.sel(species="RA16OOH")) \
        + (RC.sel(therm_coeffs=308) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=406) * Y.sel(species="OH") * Y.sel(species="RA16NO3")) \
        + (RC.sel(therm_coeffs=118) * Y.sel(species="RA16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=201) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (DJ.sel(photol_coeffs=36)) \
        + (RC.sel(therm_coeffs=380) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=381) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=35)) 
        Y.loc[:, :, :, "UDCARB11"] = (YP.sel(species="UDCARB11") + DTS * P) /(1.0+ DTS * L) 

        #          CARB6            Y.sel(species="CARB6") 
        P = EM.sel(species="CARB6") \
        + (DJ.sel(photol_coeffs=70) * Y.sel(species="RU12OOH")) + (DJ.sel(photol_coeffs=84) * Y.sel(species="RA19OOH")) \
        + (RC.sel(therm_coeffs=453) * Y.sel(species="OH") * Y.sel(species="RA19OOH")) + (DJ.sel(photol_coeffs=51) * Y.sel(species="RA19NO3")) \
        + (RC.sel(therm_coeffs=407) * Y.sel(species="OH") * Y.sel(species="RA19NO3")) + (RC.sel(therm_coeffs=434) * Y.sel(species="OH") * Y.sel(species="RN8OOH")) \
        + (RC.sel(therm_coeffs=375) * Y.sel(species="O3") * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=377) * Y.sel(species="OH") * Y.sel(species="NOA")) \
        + (RC.sel(therm_coeffs=356) * Y.sel(species="OH") * Y.sel(species="CARB7")) + (RC.sel(therm_coeffs=363) * Y.sel(species="O3") * Y.sel(species="UCARB10")) \
        + (RC.sel(therm_coeffs=309) * Y.sel(species="RA16O2")) + (RC.sel(therm_coeffs=334) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=202) * Y.sel(species="RA16O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=225) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=119) * Y.sel(species="RA16O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=142) * Y.sel(species="RU10O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=367) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=25)) 
        Y.loc[:, :, :, "CARB6"] = (YP.sel(species="CARB6") + DTS * P) /(1.0+ DTS * L) 

        #          UDCARB14         Y.sel(species="UDCARB14") 
        P = EM.sel(species="UDCARB14") \
        + (RC.sel(therm_coeffs=310) * Y.sel(species="RA19AO2")) + (RC.sel(therm_coeffs=311) * Y.sel(species="RA19CO2")) \
        + (RC.sel(therm_coeffs=120) * Y.sel(species="RA19AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=203) * Y.sel(species="RA19AO2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (DJ.sel(photol_coeffs=38)) \
        + (RC.sel(therm_coeffs=382) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=383) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=37)) 
        Y.loc[:, :, :, "UDCARB14"] = (YP.sel(species="UDCARB14") + DTS * P) /(1.0+ DTS * L) 

        #          CARB9            Y.sel(species="CARB9") 
        P = EM.sel(species="CARB9") \
        + (RC.sel(therm_coeffs=357) * Y.sel(species="OH") * Y.sel(species="CARB10")) + (RC.sel(therm_coeffs=435) * Y.sel(species="OH") * Y.sel(species="RN11OOH")) \
        + (RC.sel(therm_coeffs=121) * Y.sel(species="RA19CO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=204) * Y.sel(species="RA19CO2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=368) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=26)) 
        Y.loc[:, :, :, "CARB9"] = (YP.sel(species="CARB9") + DTS * P) /(1.0+ DTS * L) 

        #          MEK              Y.sel(species="MEK") 
        P = EM.sel(species="MEK") 
        L = 0.0 \
        + (RC.sel(therm_coeffs=89) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=14)) 
        Y.loc[:, :, :, "MEK"] = (YP.sel(species="MEK") + DTS * P) /(1.0+ DTS * L) 

        #          HOCH2CHO         Y.sel(species="HOCH2CHO") 
        P = EM.sel(species="HOCH2CHO") \
        + (DJ.sel(photol_coeffs=71) * Y.sel(species="RU10OOH")) \
        + (DJ.sel(photol_coeffs=29) * Y.sel(species="UCARB12")) + (DJ.sel(photol_coeffs=70) * Y.sel(species="RU12OOH")) \
        + (RC.sel(therm_coeffs=399) * Y.sel(species="OH") * Y.sel(species="HOC2H4NO3")) + (RC.sel(therm_coeffs=443) * Y.sel(species="OH") * Y.sel(species="HOC2H4OOH")) \
        + (RC.sel(therm_coeffs=374) * Y.sel(species="O3") * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=375) * Y.sel(species="O3") * Y.sel(species="UCARB12")) \
        + (RC.sel(therm_coeffs=332) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=333) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=315) * Y.sel(species="HOCH2CH2O2")) + (RC.sel(therm_coeffs=331) * Y.sel(species="RU12O2")) \
        + (RC.sel(therm_coeffs=222) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=224) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=141) * Y.sel(species="RU10O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=206) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=123) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=139) * Y.sel(species="RU12O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=364) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=365) * Y.sel(species="NO3")) + (DJ.sel(photol_coeffs=22)) 
        Y.loc[:, :, :, "HOCH2CHO"] = (YP.sel(species="HOCH2CHO") + DTS * P) /(1.0+ DTS * L) 

        #          RN18O2           Y.sel(species="RN18O2") 
        P = EM.sel(species="RN18O2") \
        + (DJ.sel(photol_coeffs=48) * Y.sel(species="RN19NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=261) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=319)) \
        + (RC.sel(therm_coeffs=127) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=177) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=210) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RN18O2"] = (YP.sel(species="RN18O2") + DTS * P) /(1.0+ DTS * L) 

        #          CARB13           Y.sel(species="CARB13") 
        P = EM.sel(species="CARB13") \
        + (RC.sel(therm_coeffs=446) * Y.sel(species="OH") * Y.sel(species="RN15OOH")) \
        + (RC.sel(therm_coeffs=416) * Y.sel(species="OH") * Y.sel(species="ARNOH14")) + (RC.sel(therm_coeffs=417) * Y.sel(species="NO3") * Y.sel(species="ARNOH14")) \
        + (RC.sel(therm_coeffs=320) * Y.sel(species="RN15AO2")) + (RC.sel(therm_coeffs=402) * Y.sel(species="OH") * Y.sel(species="RN15NO3")) \
        + (RC.sel(therm_coeffs=128) * Y.sel(species="RN15AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=211) * Y.sel(species="RN15AO2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=358) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=20)) 
        Y.loc[:, :, :, "CARB13"] = (YP.sel(species="CARB13") + DTS * P) /(1.0+ DTS * L) 

        #          CARB16           Y.sel(species="CARB16") 
        P = EM.sel(species="CARB16") \
        + (RC.sel(therm_coeffs=447) * Y.sel(species="OH") * Y.sel(species="RN18OOH")) + (RC.sel(therm_coeffs=484) * Y.sel(species="OH") * Y.sel(species="RTN26PAN")) \
        + (RC.sel(therm_coeffs=421) * Y.sel(species="OH") * Y.sel(species="ARNOH17")) + (RC.sel(therm_coeffs=422) * Y.sel(species="NO3") * Y.sel(species="ARNOH17")) \
        + (RC.sel(therm_coeffs=321) * Y.sel(species="RN18AO2")) + (RC.sel(therm_coeffs=403) * Y.sel(species="OH") * Y.sel(species="RN18NO3")) \
        + (RC.sel(therm_coeffs=129) * Y.sel(species="RN18AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=212) * Y.sel(species="RN18AO2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=359) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=21)) 
        Y.loc[:, :, :, "CARB16"] = (YP.sel(species="CARB16") + DTS * P) /(1.0+ DTS * L) 

        #          HOCH2CO3         Y.sel(species="HOCH2CO3") 
        P = EM.sel(species="HOCH2CO3") \
        + (RC.sel(therm_coeffs=433) * Y.sel(species="OH") * Y.sel(species="HOCH2CO3H")) + (RC.sel(therm_coeffs=472) * Y.sel(species="PHAN")) \
        + (RC.sel(therm_coeffs=364) * Y.sel(species="OH") * Y.sel(species="HOCH2CHO")) + (RC.sel(therm_coeffs=365) * Y.sel(species="NO3") * Y.sel(species="HOCH2CHO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=324)) + (RC.sel(therm_coeffs=471) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=132) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=215) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=266) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "HOCH2CO3"] = (YP.sel(species="HOCH2CO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN14O2           Y.sel(species="RN14O2") 
        P = EM.sel(species="RN14O2") \
        + (RC.sel(therm_coeffs=353) * Y.sel(species="OH") * Y.sel(species="CARB14")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=327)) \
        + (RC.sel(therm_coeffs=135) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=218) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=269) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RN14O2"] = (YP.sel(species="RN14O2") + DTS * P) /(1.0+ DTS * L) 

        #          RN17O2           Y.sel(species="RN17O2") 
        P = EM.sel(species="RN17O2") \
        + (RC.sel(therm_coeffs=354) * Y.sel(species="OH") * Y.sel(species="CARB17")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=328)) \
        + (RC.sel(therm_coeffs=136) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=219) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=270) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RN17O2"] = (YP.sel(species="RN17O2") + DTS * P) /(1.0+ DTS * L) 

        #          UCARB12          Y.sel(species="UCARB12") 
        P = EM.sel(species="UCARB12") \
        + (RC.sel(therm_coeffs=438) * Y.sel(species="OH") * Y.sel(species="RU14OOH")) + (DJ.sel(photol_coeffs=68) * Y.sel(species="RU14OOH")) \
        + (RC.sel(therm_coeffs=329) * Y.sel(species="RU14O2")) + (RC.sel(therm_coeffs=404) * Y.sel(species="OH") * Y.sel(species="RU14NO3")) \
        + (RC.sel(therm_coeffs=137) * Y.sel(species="RU14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=220) * Y.sel(species="RU14O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=375) * Y.sel(species="O3")) + (DJ.sel(photol_coeffs=29)) \
        + (RC.sel(therm_coeffs=372) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=373) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=374) * Y.sel(species="O3")) 
        Y.loc[:, :, :, "UCARB12"] = (YP.sel(species="UCARB12") + DTS * P) /(1.0+ DTS * L) 

        #          RU12O2           Y.sel(species="RU12O2") 
        P = EM.sel(species="RU12O2") \
        + (RC.sel(therm_coeffs=439) * Y.sel(species="OH") * Y.sel(species="RU12OOH")) + (RC.sel(therm_coeffs=477) * Y.sel(species="RU12PAN")) \
        + (RC.sel(therm_coeffs=372) * Y.sel(species="OH") * Y.sel(species="UCARB12")) + (RC.sel(therm_coeffs=373) * Y.sel(species="NO3") * Y.sel(species="UCARB12")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=332)) + (RC.sel(therm_coeffs=476) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=223) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=272) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=331)) \
        + (RC.sel(therm_coeffs=139) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=140) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=222) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RU12O2"] = (YP.sel(species="RU12O2") + DTS * P) /(1.0+ DTS * L) 

        #          CARB7            Y.sel(species="CARB7") 
        P = EM.sel(species="CARB7") \
        + (RC.sel(therm_coeffs=480) * Y.sel(species="OH") * Y.sel(species="MPAN")) \
        + (RC.sel(therm_coeffs=400) * Y.sel(species="OH") * Y.sel(species="RN9NO3")) + (RC.sel(therm_coeffs=444) * Y.sel(species="OH") * Y.sel(species="RN9OOH")) \
        + (RC.sel(therm_coeffs=332) * Y.sel(species="RU12O2")) + (RC.sel(therm_coeffs=335) * Y.sel(species="RU10O2")) \
        + (RC.sel(therm_coeffs=223) * Y.sel(species="RU12O2") * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=226) * Y.sel(species="RU10O2") * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=140) * Y.sel(species="RU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=143) * Y.sel(species="RU10O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=356) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=18)) 
        Y.loc[:, :, :, "CARB7"] = (YP.sel(species="CARB7") + DTS * P) /(1.0+ DTS * L) 

        #          RU10O2           Y.sel(species="RU10O2") 
        P = EM.sel(species="RU10O2") \
        + (RC.sel(therm_coeffs=440) * Y.sel(species="OH") * Y.sel(species="RU10OOH")) + (RC.sel(therm_coeffs=479) * Y.sel(species="MPAN")) \
        + (RC.sel(therm_coeffs=360) * Y.sel(species="OH") * Y.sel(species="UCARB10")) + (RC.sel(therm_coeffs=361) * Y.sel(species="NO3") * Y.sel(species="UCARB10")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=335)) + (RC.sel(therm_coeffs=478) * Y.sel(species="NO2")) \
        + (RC.sel(therm_coeffs=273) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=333)) + (RC.sel(therm_coeffs=334)) \
        + (RC.sel(therm_coeffs=224) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=225) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=226) * Y.sel(species="NO3")) \
        + (RC.sel(therm_coeffs=141) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=142) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=143) * Y.sel(species="NO")) 
        Y.loc[:, :, :, "RU10O2"] = (YP.sel(species="RU10O2") + DTS * P) /(1.0+ DTS * L) 

        #          NUCARB12         Y.sel(species="NUCARB12") 
        P = EM.sel(species="NUCARB12") \
        + (DJ.sel(photol_coeffs=72) * Y.sel(species="NRU14OOH")) \
        + (RC.sel(therm_coeffs=339) * Y.sel(species="NRU14O2")) + (RC.sel(therm_coeffs=441) * Y.sel(species="OH") * Y.sel(species="NRU14OOH")) \
        + (RC.sel(therm_coeffs=147) * Y.sel(species="NRU14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=230) * Y.sel(species="NRU14O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=376) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=30)) 
        Y.loc[:, :, :, "NUCARB12"] = (YP.sel(species="NUCARB12") + DTS * P) /(1.0+ DTS * L) 

        #          NRU12O2          Y.sel(species="NRU12O2") 
        P = EM.sel(species="NRU12O2") \
        + (RC.sel(therm_coeffs=376) * Y.sel(species="OH") * Y.sel(species="NUCARB12")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=340)) \
        + (RC.sel(therm_coeffs=148) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=231) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=278) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "NRU12O2"] = (YP.sel(species="NRU12O2") + DTS * P) /(1.0+ DTS * L) 

        #          NOA              Y.sel(species="NOA") 
        P = EM.sel(species="NOA") \
        + (DJ.sel(photol_coeffs=30) * Y.sel(species="NUCARB12")) + (DJ.sel(photol_coeffs=73) * Y.sel(species="NRU12OOH")) \
        + (RC.sel(therm_coeffs=340) * Y.sel(species="NRU12O2")) + (RC.sel(therm_coeffs=442) * Y.sel(species="OH") * Y.sel(species="NRU12OOH")) \
        + (RC.sel(therm_coeffs=148) * Y.sel(species="NRU12O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=231) * Y.sel(species="NRU12O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=377) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=31)) + (DJ.sel(photol_coeffs=32)) 
        Y.loc[:, :, :, "NOA"] = (YP.sel(species="NOA") + DTS * P) /(1.0+ DTS * L) 

        #          RTN25O2          Y.sel(species="RTN25O2") 
        P = EM.sel(species="RTN25O2") \
        + (RC.sel(therm_coeffs=457) * Y.sel(species="OH") * Y.sel(species="RTN25OOH")) + (DJ.sel(photol_coeffs=87) * Y.sel(species="RTN26OOH")) \
        + (RC.sel(therm_coeffs=343) * Y.sel(species="RTN26O2")) + (RC.sel(therm_coeffs=389) * Y.sel(species="OH") * Y.sel(species="RCOOH25")) \
        + (RC.sel(therm_coeffs=152) * Y.sel(species="RTN26O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=234) * Y.sel(species="RTN26O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=282) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=344)) \
        + (RC.sel(therm_coeffs=153) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=186) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=235) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RTN25O2"] = (YP.sel(species="RTN25O2") + DTS * P) /(1.0+ DTS * L) 

        #          RTN24O2          Y.sel(species="RTN24O2") 
        P = EM.sel(species="RTN24O2") \
        + (DJ.sel(photol_coeffs=88) * Y.sel(species="RTN25OOH")) \
        + (RC.sel(therm_coeffs=344) * Y.sel(species="RTN25O2")) + (RC.sel(therm_coeffs=458) * Y.sel(species="OH") * Y.sel(species="RTN24OOH")) \
        + (RC.sel(therm_coeffs=153) * Y.sel(species="RTN25O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=235) * Y.sel(species="RTN25O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=345)) \
        + (RC.sel(therm_coeffs=154) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=236) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=283) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RTN24O2"] = (YP.sel(species="RTN24O2") + DTS * P) /(1.0+ DTS * L) 

        #          RTN23O2          Y.sel(species="RTN23O2") 
        P = EM.sel(species="RTN23O2") \
        + (DJ.sel(photol_coeffs=89) * Y.sel(species="RTN24OOH")) \
        + (RC.sel(therm_coeffs=345) * Y.sel(species="RTN24O2")) + (RC.sel(therm_coeffs=459) * Y.sel(species="OH") * Y.sel(species="RTN23OOH")) \
        + (RC.sel(therm_coeffs=154) * Y.sel(species="RTN24O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=236) * Y.sel(species="RTN24O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=346)) \
        + (RC.sel(therm_coeffs=155) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=237) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=284) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RTN23O2"] = (YP.sel(species="RTN23O2") + DTS * P) /(1.0+ DTS * L) 

        #          RTN14O2          Y.sel(species="RTN14O2") 
        P = EM.sel(species="RTN14O2") \
        + (DJ.sel(photol_coeffs=90) * Y.sel(species="RTN23OOH")) \
        + (RC.sel(therm_coeffs=346) * Y.sel(species="RTN23O2")) + (RC.sel(therm_coeffs=460) * Y.sel(species="OH") * Y.sel(species="RTN14OOH")) \
        + (RC.sel(therm_coeffs=155) * Y.sel(species="RTN23O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=237) * Y.sel(species="RTN23O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=347)) \
        + (RC.sel(therm_coeffs=156) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=238) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=285) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RTN14O2"] = (YP.sel(species="RTN14O2") + DTS * P) /(1.0+ DTS * L) 

        #          TNCARB10         Y.sel(species="TNCARB10") 
        P = EM.sel(species="TNCARB10") \
        + (RC.sel(therm_coeffs=347) * Y.sel(species="RTN14O2")) + (DJ.sel(photol_coeffs=91) * Y.sel(species="RTN14OOH")) \
        + (RC.sel(therm_coeffs=156) * Y.sel(species="RTN14O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=238) * Y.sel(species="RTN14O2") * Y.sel(species="NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=386) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=388) * Y.sel(species="NO3")) + (DJ.sel(photol_coeffs=40)) 
        Y.loc[:, :, :, "TNCARB10"] = (YP.sel(species="TNCARB10") + DTS * P) /(1.0+ DTS * L) 

        #          RTN10O2          Y.sel(species="RTN10O2") 
        P = EM.sel(species="RTN10O2") \
        + (RC.sel(therm_coeffs=461) * Y.sel(species="OH") * Y.sel(species="RTN10OOH")) \
        + (RC.sel(therm_coeffs=386) * Y.sel(species="OH") * Y.sel(species="TNCARB10")) + (RC.sel(therm_coeffs=388) * Y.sel(species="NO3") * Y.sel(species="TNCARB10")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=348)) \
        + (RC.sel(therm_coeffs=157) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=239) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=286) * Y.sel(species="HO2")) 
        Y.loc[:, :, :, "RTN10O2"] = (YP.sel(species="RTN10O2") + DTS * P) /(1.0+ DTS * L) 

        #          RTX22O2          Y.sel(species="RTX22O2") 
        P = EM.sel(species="RTX22O2") \
        + (RC.sel(therm_coeffs=391) * Y.sel(species="OH") * Y.sel(species="TXCARB22")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=289) * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=351)) \
        + (RC.sel(therm_coeffs=163) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=189) * Y.sel(species="NO")) + (RC.sel(therm_coeffs=242) * Y.sel(species="NO3")) 
        Y.loc[:, :, :, "RTX22O2"] = (YP.sel(species="RTX22O2") + DTS * P) /(1.0+ DTS * L) 

        #          CH3NO3           Y.sel(species="CH3NO3") 
        P = EM.sel(species="CH3NO3") \
        + (RC.sel(therm_coeffs=166) * Y.sel(species="CH3O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=392) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=41)) 
        Y.loc[:, :, :, "CH3NO3"] = (YP.sel(species="CH3NO3") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5NO3          Y.sel(species="C2H5NO3") 
        P = EM.sel(species="C2H5NO3") \
        + (RC.sel(therm_coeffs=167) * Y.sel(species="C2H5O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=393) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=42)) 
        Y.loc[:, :, :, "C2H5NO3"] = (YP.sel(species="C2H5NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN10NO3          Y.sel(species="RN10NO3") 
        P = EM.sel(species="RN10NO3") \
        + (RC.sel(therm_coeffs=168) * Y.sel(species="RN10O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=394) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=43)) 
        Y.loc[:, :, :, "RN10NO3"] = (YP.sel(species="RN10NO3") + DTS * P) /(1.0+ DTS * L) 

        #          IC3H7NO3         Y.sel(species="IC3H7NO3") 
        P = EM.sel(species="IC3H7NO3") \
        + (RC.sel(therm_coeffs=169) * Y.sel(species="IC3H7O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=395) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=44)) 
        Y.loc[:, :, :, "IC3H7NO3"] = (YP.sel(species="IC3H7NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN13NO3          Y.sel(species="RN13NO3") 
        P = EM.sel(species="RN13NO3") \
        + (RC.sel(therm_coeffs=170) * Y.sel(species="RN13O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=396) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=45)) + (DJ.sel(photol_coeffs=46)) 
        Y.loc[:, :, :, "RN13NO3"] = (YP.sel(species="RN13NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN16NO3          Y.sel(species="RN16NO3") 
        P = EM.sel(species="RN16NO3") \
        + (RC.sel(therm_coeffs=171) * Y.sel(species="RN16O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=397) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=47)) 
        Y.loc[:, :, :, "RN16NO3"] = (YP.sel(species="RN16NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN19NO3          Y.sel(species="RN19NO3") 
        P = EM.sel(species="RN19NO3") \
        + (RC.sel(therm_coeffs=172) * Y.sel(species="RN19O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=398) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=48)) 
        Y.loc[:, :, :, "RN19NO3"] = (YP.sel(species="RN19NO3") + DTS * P) /(1.0+ DTS * L) 

        #          HOC2H4NO3        Y.sel(species="HOC2H4NO3") 
        P = EM.sel(species="HOC2H4NO3") \
        + (RC.sel(therm_coeffs=173) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=399) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "HOC2H4NO3"] = (YP.sel(species="HOC2H4NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN9NO3           Y.sel(species="RN9NO3") 
        P = EM.sel(species="RN9NO3") \
        + (RC.sel(therm_coeffs=174) * Y.sel(species="RN902") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=400) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RN9NO3"] = (YP.sel(species="RN9NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN12NO3          Y.sel(species="RN12NO3") 
        P = EM.sel(species="RN12NO3") \
        + (RC.sel(therm_coeffs=175) * Y.sel(species="RN12O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=401) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RN12NO3"] = (YP.sel(species="RN12NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN15NO3          Y.sel(species="RN15NO3") 
        P = EM.sel(species="RN15NO3") \
        + (RC.sel(therm_coeffs=176) * Y.sel(species="RN15O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=178) * Y.sel(species="RN15AO2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=402) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RN15NO3"] = (YP.sel(species="RN15NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RN18NO3          Y.sel(species="RN18NO3") 
        P = EM.sel(species="RN18NO3") \
        + (RC.sel(therm_coeffs=177) * Y.sel(species="RN18O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=179) * Y.sel(species="RN18AO2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=403) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RN18NO3"] = (YP.sel(species="RN18NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RU14NO3          Y.sel(species="RU14NO3") 
        P = EM.sel(species="RU14NO3") \
        + (RC.sel(therm_coeffs=180) * Y.sel(species="RU14O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=404) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RU14NO3"] = (YP.sel(species="RU14NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RA13NO3          Y.sel(species="RA13NO3") 
        P = EM.sel(species="RA13NO3") \
        + (RC.sel(therm_coeffs=181) * Y.sel(species="RA13O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=405) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=49)) 
        Y.loc[:, :, :, "RA13NO3"] = (YP.sel(species="RA13NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RA16NO3          Y.sel(species="RA16NO3") 
        P = EM.sel(species="RA16NO3") \
        + (RC.sel(therm_coeffs=182) * Y.sel(species="RA16O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=406) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=50)) 
        Y.loc[:, :, :, "RA16NO3"] = (YP.sel(species="RA16NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RA19NO3          Y.sel(species="RA19NO3") 
        P = EM.sel(species="RA19NO3") \
        + (RC.sel(therm_coeffs=183) * Y.sel(species="RA19AO2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=184) * Y.sel(species="RA19CO2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=407) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=51)) 
        Y.loc[:, :, :, "RA19NO3"] = (YP.sel(species="RA19NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RTN28NO3         Y.sel(species="RTN28NO3") 
        P = EM.sel(species="RTN28NO3") \
        + (RC.sel(therm_coeffs=185) * Y.sel(species="RTN28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=486) * Y.sel(species="P2604")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=408) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=485)) 
        Y.loc[:, :, :, "RTN28NO3"] = (YP.sel(species="RTN28NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RTN25NO3         Y.sel(species="RTN25NO3") 
        P = EM.sel(species="RTN25NO3") \
        + (RC.sel(therm_coeffs=186) * Y.sel(species="RTN25O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=409) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RTN25NO3"] = (YP.sel(species="RTN25NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RTX28NO3         Y.sel(species="RTX28NO3") 
        P = EM.sel(species="RTX28NO3") \
        + (RC.sel(therm_coeffs=187) * Y.sel(species="RTX28O2") * Y.sel(species="NO")) + (RC.sel(therm_coeffs=488) * Y.sel(species="P4608")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=410) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=487)) 
        Y.loc[:, :, :, "RTX28NO3"] = (YP.sel(species="RTX28NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RTX24NO3         Y.sel(species="RTX24NO3") 
        P = EM.sel(species="RTX24NO3") \
        + (RC.sel(therm_coeffs=188) * Y.sel(species="RTX24O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=411) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=52)) 
        Y.loc[:, :, :, "RTX24NO3"] = (YP.sel(species="RTX24NO3") + DTS * P) /(1.0+ DTS * L) 

        #          RTX22NO3         Y.sel(species="RTX22NO3") 
        P = EM.sel(species="RTX22NO3") \
        + (RC.sel(therm_coeffs=189) * Y.sel(species="RTX22O2") * Y.sel(species="NO")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=412) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RTX22NO3"] = (YP.sel(species="RTX22NO3") + DTS * P) /(1.0+ DTS * L) 

        #          CH3OOH           Y.sel(species="CH3OOH") 
        P = EM.sel(species="CH3OOH") \
        + (RC.sel(therm_coeffs=244) * Y.sel(species="CH3O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=423) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=424) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=53)) 
        Y.loc[:, :, :, "CH3OOH"] = (YP.sel(species="CH3OOH") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5OOH          Y.sel(species="C2H5OOH") 
        P = EM.sel(species="C2H5OOH") \
        + (RC.sel(therm_coeffs=245) * Y.sel(species="C2H5O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=425) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=54)) 
        Y.loc[:, :, :, "C2H5OOH"] = (YP.sel(species="C2H5OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN10OOH          Y.sel(species="RN10OOH") 
        P = EM.sel(species="RN10OOH") \
        + (RC.sel(therm_coeffs=246) * Y.sel(species="RN10O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=426) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=55)) 
        Y.loc[:, :, :, "RN10OOH"] = (YP.sel(species="RN10OOH") + DTS * P) /(1.0+ DTS * L) 

        #          IC3H7OOH         Y.sel(species="IC3H7OOH") 
        P = EM.sel(species="IC3H7OOH") \
        + (RC.sel(therm_coeffs=247) * Y.sel(species="IC3H7O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=427) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=56)) 
        Y.loc[:, :, :, "IC3H7OOH"] = (YP.sel(species="IC3H7OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN13OOH          Y.sel(species="RN13OOH") 
        P = EM.sel(species="RN13OOH") \
        + (RC.sel(therm_coeffs=248) * Y.sel(species="RN13O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=251) * Y.sel(species="RN13AO2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=428) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=57)) + (DJ.sel(photol_coeffs=58)) 
        Y.loc[:, :, :, "RN13OOH"] = (YP.sel(species="RN13OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN16OOH          Y.sel(species="RN16OOH") 
        P = EM.sel(species="RN16OOH") \
        + (RC.sel(therm_coeffs=249) * Y.sel(species="RN16O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=252) * Y.sel(species="RN16AO2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=429) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=59)) 
        Y.loc[:, :, :, "RN16OOH"] = (YP.sel(species="RN16OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN19OOH          Y.sel(species="RN19OOH") 
        P = EM.sel(species="RN19OOH") \
        + (RC.sel(therm_coeffs=250) * Y.sel(species="RN19O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=430) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=60)) 
        Y.loc[:, :, :, "RN19OOH"] = (YP.sel(species="RN19OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RA13OOH          Y.sel(species="RA13OOH") 
        P = EM.sel(species="RA13OOH") \
        + (RC.sel(therm_coeffs=253) * Y.sel(species="RA13O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=451) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=82)) 
        Y.loc[:, :, :, "RA13OOH"] = (YP.sel(species="RA13OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RA16OOH          Y.sel(species="RA16OOH") 
        P = EM.sel(species="RA16OOH") \
        + (RC.sel(therm_coeffs=254) * Y.sel(species="RA16O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=452) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=83)) 
        Y.loc[:, :, :, "RA16OOH"] = (YP.sel(species="RA16OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RA19OOH          Y.sel(species="RA19OOH") 
        P = EM.sel(species="RA19OOH") \
        + (RC.sel(therm_coeffs=255) * Y.sel(species="RA19AO2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=256) * Y.sel(species="RA19CO2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=453) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=84)) 
        Y.loc[:, :, :, "RA19OOH"] = (YP.sel(species="RA19OOH") + DTS * P) /(1.0+ DTS * L) 

        #          HOC2H4OOH        Y.sel(species="HOC2H4OOH") 
        P = EM.sel(species="HOC2H4OOH") \
        + (RC.sel(therm_coeffs=257) * Y.sel(species="HOCH2CH2O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=443) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=74)) 
        Y.loc[:, :, :, "HOC2H4OOH"] = (YP.sel(species="HOC2H4OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN9OOH           Y.sel(species="RN9OOH") 
        P = EM.sel(species="RN9OOH") \
        + (RC.sel(therm_coeffs=258) * Y.sel(species="RN902") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=444) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=75)) 
        Y.loc[:, :, :, "RN9OOH"] = (YP.sel(species="RN9OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN12OOH          Y.sel(species="RN12OOH") 
        P = EM.sel(species="RN12OOH") \
        + (RC.sel(therm_coeffs=259) * Y.sel(species="RN12O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=445) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=76)) 
        Y.loc[:, :, :, "RN12OOH"] = (YP.sel(species="RN12OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN15OOH          Y.sel(species="RN15OOH") 
        P = EM.sel(species="RN15OOH") \
        + (RC.sel(therm_coeffs=260) * Y.sel(species="RN15O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=262) * Y.sel(species="RN15AO2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=446) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=77)) 
        Y.loc[:, :, :, "RN15OOH"] = (YP.sel(species="RN15OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN18OOH          Y.sel(species="RN18OOH") 
        P = EM.sel(species="RN18OOH") \
        + (RC.sel(therm_coeffs=261) * Y.sel(species="RN18O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=263) * Y.sel(species="RN18AO2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=447) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=78)) 
        Y.loc[:, :, :, "RN18OOH"] = (YP.sel(species="RN18OOH") + DTS * P) /(1.0+ DTS * L) 

        #          CH3CO3H          Y.sel(species="CH3CO3H") 
        P = EM.sel(species="CH3CO3H") \
        + (RC.sel(therm_coeffs=264) * Y.sel(species="CH3CO3") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=431) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=61)) 
        Y.loc[:, :, :, "CH3CO3H"] = (YP.sel(species="CH3CO3H") + DTS * P) /(1.0+ DTS * L) 

        #          C2H5CO3H         Y.sel(species="C2H5CO3H") 
        P = EM.sel(species="C2H5CO3H") \
        + (RC.sel(therm_coeffs=265) * Y.sel(species="C2H5CO3") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=432) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=62)) 
        Y.loc[:, :, :, "C2H5CO3H"] = (YP.sel(species="C2H5CO3H") + DTS * P) /(1.0+ DTS * L) 

        #          HOCH2CO3H        Y.sel(species="HOCH2CO3H") 
        P = EM.sel(species="HOCH2CO3H") \
        + (RC.sel(therm_coeffs=266) * Y.sel(species="HOCH2CO3") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=433) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=63)) 
        Y.loc[:, :, :, "HOCH2CO3H"] = (YP.sel(species="HOCH2CO3H") + DTS * P) /(1.0+ DTS * L) 

        #          RN8OOH           Y.sel(species="RN8OOH") 
        P = EM.sel(species="RN8OOH") \
        + (RC.sel(therm_coeffs=267) * Y.sel(species="RN8O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=434) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=64)) 
        Y.loc[:, :, :, "RN8OOH"] = (YP.sel(species="RN8OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN11OOH          Y.sel(species="RN11OOH") 
        P = EM.sel(species="RN11OOH") \
        + (RC.sel(therm_coeffs=268) * Y.sel(species="RN11O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=435) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=65)) 
        Y.loc[:, :, :, "RN11OOH"] = (YP.sel(species="RN11OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN14OOH          Y.sel(species="RN14OOH") 
        P = EM.sel(species="RN14OOH") \
        + (RC.sel(therm_coeffs=269) * Y.sel(species="RN14O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=436) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=66)) 
        Y.loc[:, :, :, "RN14OOH"] = (YP.sel(species="RN14OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RN17OOH          Y.sel(species="RN17OOH") 
        P = EM.sel(species="RN17OOH") \
        + (RC.sel(therm_coeffs=270) * Y.sel(species="RN17O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=437) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=67)) 
        Y.loc[:, :, :, "RN17OOH"] = (YP.sel(species="RN17OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RU14OOH          Y.sel(species="RU14OOH") 
        P = EM.sel(species="RU14OOH") \
        + (RC.sel(therm_coeffs=271) * Y.sel(species="RU14O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=438) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=68)) + (DJ.sel(photol_coeffs=69)) 
        Y.loc[:, :, :, "RU14OOH"] = (YP.sel(species="RU14OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RU12OOH          Y.sel(species="RU12OOH") 
        P = EM.sel(species="RU12OOH") \
        + (RC.sel(therm_coeffs=272) * Y.sel(species="RU12O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=439) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=70)) 
        Y.loc[:, :, :, "RU12OOH"] = (YP.sel(species="RU12OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RU10OOH          Y.sel(species="RU10OOH") 
        P = EM.sel(species="RU10OOH") \
        + (RC.sel(therm_coeffs=273) * Y.sel(species="RU10O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=440) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=71)) 
        Y.loc[:, :, :, "RU10OOH"] = (YP.sel(species="RU10OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRN6OOH          Y.sel(species="NRN6OOH") 
        P = EM.sel(species="NRN6OOH") \
        + (RC.sel(therm_coeffs=274) * Y.sel(species="NRN6O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=448) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=79)) 
        Y.loc[:, :, :, "NRN6OOH"] = (YP.sel(species="NRN6OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRN9OOH          Y.sel(species="NRN9OOH") 
        P = EM.sel(species="NRN9OOH") \
        + (RC.sel(therm_coeffs=275) * Y.sel(species="NRN9O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=449) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=80)) 
        Y.loc[:, :, :, "NRN9OOH"] = (YP.sel(species="NRN9OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRN12OOH         Y.sel(species="NRN12OOH") 
        P = EM.sel(species="NRN12OOH") \
        + (RC.sel(therm_coeffs=276) * Y.sel(species="NRN12O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=450) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=81)) 
        Y.loc[:, :, :, "NRN12OOH"] = (YP.sel(species="NRN12OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRU14OOH         Y.sel(species="NRU14OOH") 
        P = EM.sel(species="NRU14OOH") \
        + (RC.sel(therm_coeffs=277) * Y.sel(species="NRU14O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=441) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=72)) 
        Y.loc[:, :, :, "NRU14OOH"] = (YP.sel(species="NRU14OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRU12OOH         Y.sel(species="NRU12OOH") 
        P = EM.sel(species="NRU12OOH") \
        + (RC.sel(therm_coeffs=278) * Y.sel(species="NRU12O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=442) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=73)) 
        Y.loc[:, :, :, "NRU12OOH"] = (YP.sel(species="NRU12OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN28OOH         Y.sel(species="RTN28OOH") 
        P = EM.sel(species="RTN28OOH") \
        + (RC.sel(therm_coeffs=279) * Y.sel(species="RTN28O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=496) * Y.sel(species="P2605")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=454) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=495)) + (DJ.sel(photol_coeffs=85)) 
        Y.loc[:, :, :, "RTN28OOH"] = (YP.sel(species="RTN28OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRTN28OOH        Y.sel(species="NRTN28OOH") 
        P = EM.sel(species="NRTN28OOH") \
        + (RC.sel(therm_coeffs=280) * Y.sel(species="NRTN28O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=456) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=86)) 
        Y.loc[:, :, :, "NRTN28OOH"] = (YP.sel(species="NRTN28OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN26OOH         Y.sel(species="RTN26OOH") 
        P = EM.sel(species="RTN26OOH") \
        + (RC.sel(therm_coeffs=281) * Y.sel(species="RTN26O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=498) * Y.sel(species="P2630")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=455) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=497)) + (DJ.sel(photol_coeffs=87)) 
        Y.loc[:, :, :, "RTN26OOH"] = (YP.sel(species="RTN26OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN25OOH         Y.sel(species="RTN25OOH") 
        P = EM.sel(species="RTN25OOH") \
        + (RC.sel(therm_coeffs=282) * Y.sel(species="RTN25O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=502) * Y.sel(species="P2632")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=457) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=501)) + (DJ.sel(photol_coeffs=88)) 
        Y.loc[:, :, :, "RTN25OOH"] = (YP.sel(species="RTN25OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN24OOH         Y.sel(species="RTN24OOH") 
        P = EM.sel(species="RTN24OOH") \
        + (RC.sel(therm_coeffs=283) * Y.sel(species="RTN24O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=492) * Y.sel(species="P2635")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=458) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=491)) + (DJ.sel(photol_coeffs=89)) 
        Y.loc[:, :, :, "RTN24OOH"] = (YP.sel(species="RTN24OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN23OOH         Y.sel(species="RTN23OOH") 
        P = EM.sel(species="RTN23OOH") \
        + (RC.sel(therm_coeffs=284) * Y.sel(species="RTN23O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=504) * Y.sel(species="P2637")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=459) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=503)) + (DJ.sel(photol_coeffs=90)) 
        Y.loc[:, :, :, "RTN23OOH"] = (YP.sel(species="RTN23OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN14OOH         Y.sel(species="RTN14OOH") 
        P = EM.sel(species="RTN14OOH") \
        + (RC.sel(therm_coeffs=285) * Y.sel(species="RTN14O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=460) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=91)) 
        Y.loc[:, :, :, "RTN14OOH"] = (YP.sel(species="RTN14OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTN10OOH         Y.sel(species="RTN10OOH") 
        P = EM.sel(species="RTN10OOH") \
        + (RC.sel(therm_coeffs=286) * Y.sel(species="RTN10O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=461) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=92)) 
        Y.loc[:, :, :, "RTN10OOH"] = (YP.sel(species="RTN10OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTX28OOH         Y.sel(species="RTX28OOH") 
        P = EM.sel(species="RTX28OOH") \
        + (RC.sel(therm_coeffs=287) * Y.sel(species="RTX28O2") * Y.sel(species="HO2")) + (RC.sel(therm_coeffs=494) * Y.sel(species="P4610")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=462) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=493)) + (DJ.sel(photol_coeffs=93)) 
        Y.loc[:, :, :, "RTX28OOH"] = (YP.sel(species="RTX28OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTX24OOH         Y.sel(species="RTX24OOH") 
        P = EM.sel(species="RTX24OOH") \
        + (RC.sel(therm_coeffs=288) * Y.sel(species="RTX24O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=463) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=94)) 
        Y.loc[:, :, :, "RTX24OOH"] = (YP.sel(species="RTX24OOH") + DTS * P) /(1.0+ DTS * L) 

        #          RTX22OOH         Y.sel(species="RTX22OOH") 
        P = EM.sel(species="RTX22OOH") \
        + (RC.sel(therm_coeffs=289) * Y.sel(species="RTX22O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=464) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=95)) 
        Y.loc[:, :, :, "RTX22OOH"] = (YP.sel(species="RTX22OOH") + DTS * P) /(1.0+ DTS * L) 

        #          NRTX28OOH        Y.sel(species="NRTX28OOH") 
        P = EM.sel(species="NRTX28OOH") \
        + (RC.sel(therm_coeffs=290) * Y.sel(species="NRTX28O2") * Y.sel(species="HO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=465) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=96)) 
        Y.loc[:, :, :, "NRTX28OOH"] = (YP.sel(species="NRTX28OOH") + DTS * P) /(1.0+ DTS * L) 

        #          CARB14           Y.sel(species="CARB14") 
        P = EM.sel(species="CARB14") \
        + (RC.sel(therm_coeffs=397) * Y.sel(species="OH") * Y.sel(species="RN16NO3")) + (RC.sel(therm_coeffs=429) * Y.sel(species="OH") * Y.sel(species="RN16OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=353) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=15)) 
        Y.loc[:, :, :, "CARB14"] = (YP.sel(species="CARB14") + DTS * P) /(1.0+ DTS * L) 

        #          CARB17           Y.sel(species="CARB17") 
        P = EM.sel(species="CARB17") \
        + (RC.sel(therm_coeffs=398) * Y.sel(species="OH") * Y.sel(species="RN19NO3")) + (RC.sel(therm_coeffs=430) * Y.sel(species="OH") * Y.sel(species="RN19OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=354) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=16)) 
        Y.loc[:, :, :, "CARB17"] = (YP.sel(species="CARB17") + DTS * P) /(1.0+ DTS * L) 

        #          CARB10           Y.sel(species="CARB10") 
        P = EM.sel(species="CARB10") \
        + (RC.sel(therm_coeffs=401) * Y.sel(species="OH") * Y.sel(species="RN12NO3")) + (RC.sel(therm_coeffs=445) * Y.sel(species="OH") * Y.sel(species="RN12OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=357) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=19)) 
        Y.loc[:, :, :, "CARB10"] = (YP.sel(species="CARB10") + DTS * P) /(1.0+ DTS * L) 

        #          CARB12           Y.sel(species="CARB12") 
        P = EM.sel(species="CARB12") \
        + (RC.sel(therm_coeffs=436) * Y.sel(species="OH") * Y.sel(species="RN14OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=369) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=27)) 
        Y.loc[:, :, :, "CARB12"] = (YP.sel(species="CARB12") + DTS * P) /(1.0+ DTS * L) 

        #          CARB15           Y.sel(species="CARB15") 
        P = EM.sel(species="CARB15") \
        + (RC.sel(therm_coeffs=437) * Y.sel(species="OH") * Y.sel(species="RN17OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=370) * Y.sel(species="OH")) + (DJ.sel(photol_coeffs=28)) 
        Y.loc[:, :, :, "CARB15"] = (YP.sel(species="CARB15") + DTS * P) /(1.0+ DTS * L) 

        #          CCARB12          Y.sel(species="CCARB12") 
        P = EM.sel(species="CCARB12") \
        + (RC.sel(therm_coeffs=412) * Y.sel(species="OH") * Y.sel(species="RTX22NO3")) + (RC.sel(therm_coeffs=464) * Y.sel(species="OH") * Y.sel(species="RTX22OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=371) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "CCARB12"] = (YP.sel(species="CCARB12") + DTS * P) /(1.0+ DTS * L) 

        #          ANHY             Y.sel(species="ANHY") 
        P = EM.sel(species="ANHY") \
        + (DJ.sel(photol_coeffs=38) * Y.sel(species="UDCARB14")) \
        + (DJ.sel(photol_coeffs=34) * Y.sel(species="UDCARB8")) + (DJ.sel(photol_coeffs=36) * Y.sel(species="UDCARB11")) \
        + (RC.sel(therm_coeffs=383) * Y.sel(species="OH") * Y.sel(species="UDCARB14")) + (RC.sel(therm_coeffs=510) * Y.sel(species="P3442")) \
        + (RC.sel(therm_coeffs=379) * Y.sel(species="OH") * Y.sel(species="UDCARB8")) + (RC.sel(therm_coeffs=381) * Y.sel(species="OH") * Y.sel(species="UDCARB11")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=466) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=509)) 
        Y.loc[:, :, :, "ANHY"] = (YP.sel(species="ANHY") + DTS * P) /(1.0+ DTS * L) 

        #          TNCARB15         Y.sel(species="TNCARB15") 
        P = EM.sel(species="TNCARB15") \
        + (RC.sel(therm_coeffs=409) * Y.sel(species="OH") * Y.sel(species="RTN25NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=385) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "TNCARB15"] = (YP.sel(species="TNCARB15") + DTS * P) /(1.0+ DTS * L) 

        #          RAROH14          Y.sel(species="RAROH14") 
        P = EM.sel(species="RAROH14") \
        + (RC.sel(therm_coeffs=413) * Y.sel(species="OH") * Y.sel(species="AROH14")) + (RC.sel(therm_coeffs=414) * Y.sel(species="NO3") * Y.sel(species="AROH14")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=415) * Y.sel(species="NO2")) 
        Y.loc[:, :, :, "RAROH14"] = (YP.sel(species="RAROH14") + DTS * P) /(1.0+ DTS * L) 

        #          ARNOH14          Y.sel(species="ARNOH14") 
        P = EM.sel(species="ARNOH14") \
        + (RC.sel(therm_coeffs=415) * Y.sel(species="RAROH14") * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=506) * Y.sel(species="P3612")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=416) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=417) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=505)) 
        Y.loc[:, :, :, "ARNOH14"] = (YP.sel(species="ARNOH14") + DTS * P) /(1.0+ DTS * L) 

        #          RAROH17          Y.sel(species="RAROH17") 
        P = EM.sel(species="RAROH17") \
        + (RC.sel(therm_coeffs=418) * Y.sel(species="OH") * Y.sel(species="AROH17")) + (RC.sel(therm_coeffs=419) * Y.sel(species="NO3") * Y.sel(species="AROH17")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=420) * Y.sel(species="NO2")) 
        Y.loc[:, :, :, "RAROH17"] = (YP.sel(species="RAROH17") + DTS * P) /(1.0+ DTS * L) 

        #          ARNOH17          Y.sel(species="ARNOH17") 
        P = EM.sel(species="ARNOH17") \
        + (RC.sel(therm_coeffs=420) * Y.sel(species="RAROH17") * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=508) * Y.sel(species="P3613")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=421) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=422) * Y.sel(species="NO3")) + (RC.sel(therm_coeffs=507)) 
        Y.loc[:, :, :, "ARNOH17"] = (YP.sel(species="ARNOH17") + DTS * P) /(1.0+ DTS * L) 

        #          PAN              Y.sel(species="PAN") 
        P = EM.sel(species="PAN") \
        + (RC.sel(therm_coeffs=467) * Y.sel(species="CH3CO3") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=468)) + (RC.sel(therm_coeffs=473) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "PAN"] = (YP.sel(species="PAN") + DTS * P) /(1.0+ DTS * L) 

        #          PPN              Y.sel(species="PPN") 
        P = EM.sel(species="PPN") \
        + (RC.sel(therm_coeffs=469) * Y.sel(species="C2H5CO3") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=470)) + (RC.sel(therm_coeffs=474) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "PPN"] = (YP.sel(species="PPN") + DTS * P) /(1.0+ DTS * L) 

        #          PHAN             Y.sel(species="PHAN") 
        P = EM.sel(species="PHAN") \
        + (RC.sel(therm_coeffs=471) * Y.sel(species="HOCH2CO3") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=472)) + (RC.sel(therm_coeffs=475) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "PHAN"] = (YP.sel(species="PHAN") + DTS * P) /(1.0+ DTS * L) 

        #          RU12PAN          Y.sel(species="RU12PAN") 
        P = EM.sel(species="RU12PAN") \
        + (RC.sel(therm_coeffs=476) * Y.sel(species="RU12O2") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=477)) + (RC.sel(therm_coeffs=481) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "RU12PAN"] = (YP.sel(species="RU12PAN") + DTS * P) /(1.0+ DTS * L) 

        #          MPAN             Y.sel(species="MPAN") 
        P = EM.sel(species="MPAN") \
        + (RC.sel(therm_coeffs=478) * Y.sel(species="RU10O2") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=479)) + (RC.sel(therm_coeffs=480) * Y.sel(species="OH")) 
        Y.loc[:, :, :, "MPAN"] = (YP.sel(species="MPAN") + DTS * P) /(1.0+ DTS * L) 

        #          RTN26PAN         Y.sel(species="RTN26PAN") 
        P = EM.sel(species="RTN26PAN") \
        + (RC.sel(therm_coeffs=482) * Y.sel(species="RTN26O2") * Y.sel(species="NO2")) + (RC.sel(therm_coeffs=500) * Y.sel(species="P2629")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=483)) + (RC.sel(therm_coeffs=484) * Y.sel(species="OH")) + (RC.sel(therm_coeffs=499)) 
        Y.loc[:, :, :, "RTN26PAN"] = (YP.sel(species="RTN26PAN") + DTS * P) /(1.0+ DTS * L) 

        #          P2604            Y.sel(species="P2604") 
        P = EM.sel(species="P2604") \
        + (RC.sel(therm_coeffs=485) * Y.sel(species="RTN28NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=486)) 
        Y.loc[:, :, :, "P2604"] = (YP.sel(species="P2604") + DTS * P) /(1.0+ DTS * L) 

        #          P4608            Y.sel(species="P4608") 
        P = EM.sel(species="P4608") \
        + (RC.sel(therm_coeffs=487) * Y.sel(species="RTX28NO3")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=488)) 
        Y.loc[:, :, :, "P4608"] = (YP.sel(species="P4608") + DTS * P) /(1.0+ DTS * L) 

        #          P2631            Y.sel(species="P2631") 
        P = EM.sel(species="P2631") \
        + (RC.sel(therm_coeffs=489) * Y.sel(species="RCOOH25")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=490)) 
        Y.loc[:, :, :, "P2631"] = (YP.sel(species="P2631") + DTS * P) /(1.0+ DTS * L) 

        #          P2635            Y.sel(species="P2635") 
        P = EM.sel(species="P2635") \
        + (RC.sel(therm_coeffs=491) * Y.sel(species="RTN24OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=492)) 
        Y.loc[:, :, :, "P2635"] = (YP.sel(species="P2635") + DTS * P) /(1.0+ DTS * L) 

        #          P4610            Y.sel(species="P4610") 
        P = EM.sel(species="P4610") \
        + (RC.sel(therm_coeffs=493) * Y.sel(species="RTX28OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=494)) 
        Y.loc[:, :, :, "P4610"] = (YP.sel(species="P4610") + DTS * P) /(1.0+ DTS * L) 

        #          P2605            Y.sel(species="P2605") 
        P = EM.sel(species="P2605") \
        + (RC.sel(therm_coeffs=495) * Y.sel(species="RTN28OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=496)) 
        Y.loc[:, :, :, "P2605"] = (YP.sel(species="P2605") + DTS * P) /(1.0+ DTS * L) 

        #          P2630            Y.sel(species="P2630") 
        P = EM.sel(species="P2630") \
        + (RC.sel(therm_coeffs=497) * Y.sel(species="RTN26OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=498)) 
        Y.loc[:, :, :, "P2630"] = (YP.sel(species="P2630") + DTS * P) /(1.0+ DTS * L) 

        #          P2629            Y.sel(species="P2629") 
        P = EM.sel(species="P2629") \
        + (RC.sel(therm_coeffs=499) * Y.sel(species="RTN26PAN")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=500)) 
        Y.loc[:, :, :, "P2629"] = (YP.sel(species="P2629") + DTS * P) /(1.0+ DTS * L) 

        #          P2632            Y.sel(species="P2632") 
        P = EM.sel(species="P2632") \
        + (RC.sel(therm_coeffs=501) * Y.sel(species="RTN25OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=502)) 
        Y.loc[:, :, :, "P2632"] = (YP.sel(species="P2632") + DTS * P) /(1.0+ DTS * L) 

        #          P2637            Y.sel(species="P2637") 
        P = EM.sel(species="P2637") \
        + (RC.sel(therm_coeffs=503) * Y.sel(species="RTN23OOH")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=504)) 
        Y.loc[:, :, :, "P2637"] = (YP.sel(species="P2637") + DTS * P) /(1.0+ DTS * L) 

        #          P3612            Y.sel(species="P3612") 
        P = EM.sel(species="P3612") \
        + (RC.sel(therm_coeffs=505) * Y.sel(species="ARNOH14")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=506)) 
        Y.loc[:, :, :, "P3612"] = (YP.sel(species="P3612") + DTS * P) /(1.0+ DTS * L) 

        #          P3613            Y.sel(species="P3613") 
        P = EM.sel(species="P3613") \
        + (RC.sel(therm_coeffs=507) * Y.sel(species="ARNOH17")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=508)) 
        Y.loc[:, :, :, "P3613"] = (YP.sel(species="P3613") + DTS * P) /(1.0+ DTS * L) 

        #          P3442            Y.sel(species="P3442") 
        P = EM.sel(species="P3442") \
        + (RC.sel(therm_coeffs=509) * Y.sel(species="ANHY")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=510)) 
        Y.loc[:, :, :, "P3442"] = (YP.sel(species="P3442") + DTS * P) /(1.0+ DTS * L) 

        #          CH3O2NO2         Y.sel(species="CH3O2NO2") 
        P = EM.sel(species="CH3O2NO2") \
        + (RC.sel(therm_coeffs=164) * Y.sel(species="CH3O2") * Y.sel(species="NO2")) 
        L = 0.0 \
        + (RC.sel(therm_coeffs=165)) 
        Y.loc[:, :, :, "CH3O2NO2"] = (YP.sel(species="CH3O2NO2") + DTS * P) /(1.0+ DTS * L) 

        #          EMPOA            Y.sel(species="EMPOA") 
        P = EM.sel(species="EMPOA") 
        #     dry deposition of EMPOA

        L = 0.0
        Y.loc[:, :, :, "EMPOA"] = (YP.sel(species="EMPOA") + DTS * P) /(1.0+ DTS * L) 

        #          P2007            Y.sel(species="P2007") 
        P = EM.sel(species="P2007") \
        + (RC(511) * Y.sel(species="RU12OOH")) 
        L = 0.0 \
        + (RC(512)) 
        Y.loc[:, :, :, "P2007"] = (YP.sel(species="P2007") + DTS * P) /(1.0+ DTS * L) 

              #  DEFINE TOTAL CONCENTRATION OF PEROXY RADICALS
        RO2 =  Y.sel(species="CH3O2") + Y.sel(species="C2H5O2") + Y.sel(species="RN10O2") + Y.sel(species="IC3H7O2") + Y.sel(species="RN13O2") +\
        Y.sel(species="RN13AO2") + Y.sel(species="RN16AO2") + Y.sel(species="RN16O2") + Y.sel(species="RN19O2") + Y.sel(species="HOCH2CH2O2") + Y.sel(species="RN902") +\
        Y.sel(species="RN12O2") + Y.sel(species="RN15O2") + Y.sel(species="RN18O2") + Y.sel(species="RN15AO2") + Y.sel(species="RN18AO2") + Y.sel(species="CH3CO3") +\
        Y.sel(species="C2H5CO3") + Y.sel(species="RN11O2") + Y.sel(species="RN14O2") + Y.sel(species="RN17O2") + Y.sel(species="HOCH2CO3") + Y.sel(species="RU14O2") +\
        Y.sel(species="RU12O2") + Y.sel(species="RU10O2") + Y.sel(species="NRN6O2") + Y.sel(species="NRN9O2") + Y.sel(species="NRN12O2") + Y.sel(species="RTN28O2") +\
        Y.sel(species="NRU14O2") + Y.sel(species="NRU12O2") + Y.sel(species="RA13O2") + Y.sel(species="RA16O2") + Y.sel(species="RA19AO2") + Y.sel(species="RA19CO2") +\
        Y.sel(species="RN8O2") + Y.sel(species="RTN26O2") + Y.sel(species="NRTN28O2") + Y.sel(species="RTN25O2") + Y.sel(species="RTN24O2") + Y.sel(species="RTN23O2") +\
        Y.sel(species="RTN14O2") + Y.sel(species="RTN10O2") + Y.sel(species="RTX28O2") + Y.sel(species="RTX24O2") + Y.sel(species="RTX22O2") + Y.sel(species="NRTX28O2") 