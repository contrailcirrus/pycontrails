#!/bin/bash

replace_variables() {
    input_file="$1"
    output_file="$2"

    # Match simple rate coefficients and convert them to arrays
    sed -Ei.bak '49,80 s/(KRO2NO|KAPNO|KRO2NO3|KRO2HO2|KAPHO2|KNO3AL|KDEC|KALKOXY|KALKPXY|BR01|KIN|KOUT2604|KOUT4608|KOUT2631|KOUT2635|KOUT4610|KOUT2605|KOUT2630|KOUT2629|KOUT2632|KOUT2637|KOUT3612|KOUT3613|KOUT3442)/\1\(:,:,:\)/g' "chemco_out.f90"

    # Match complex rate coefficients and convert them to arrays
    sed -Ei.bak '80,234 s/\b([A-Z]+[0-9]*)\s=/\1\(:,:,:\)\s=/g' "chemco_out.f90"
    sed -Ei.bak '80,234 s/\b([A-Z]+[0-9]*+[A-Z]*)\s=/\1\(:,:,:\)\s=/g' "chemco_out.f90"

    # Replace RC
    sed -Ei.bak '236,1765 s/RC\(/RC\(:,:,:,TS,/g' "chemco_out.f90"
    sed -Ei.bak '236,1765 s/\)\s=\s\b([A-Z]+[0-9]*+[A-Z]*)/\)\s=\s\1\(:,:,:\)/g' "chemco_out.f90"

    # Replace temp, h2o, o2, m
    sed -Ei.bak 's/TEMP/TEMP\(:,:,:,TS\)/g;
                 s/H2O/H2O\(:,:,:,TS\)/g;
                 s/O2/O2\(:,:,:,TS\)/g;
                 s/N2/N2\(:,:,:,TS\)/g;
                 s/M/M\(:,:,:,TS\)/g' "chemco_out.f90"

}

# # Call the function to replace variables in input.f90 and save the output to output.py
replace_variables "chemco_in.f90" "chemco_out.f90"