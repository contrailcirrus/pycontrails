#!/bin/bash

replace_variables() {
    input_file="$1"
    output_file="$2"

    # Match simple rate coefficients and convert them to arrays
    sed -Ei.bak 's/(KRO2NO|KAPNO|KRO2NO3|KRO2HO2|KAPHO2|KNO3AL|KDEC|KALKOXY|KALKPXY|BR01|KIN|KOUT2604|KOUT4608|KOUT2631|KOUT2635|KOUT4610|KOUT2605|KOUT2630|KOUT2629|KOUT2632|KOUT2637|KOUT3612|KOUT3613|KOUT3442)/\1\(:,:,:\)/g' "chemco_out.f90"

    # Match complex rate coefficients and convert them to arrays
    
    ||||||||||)
}