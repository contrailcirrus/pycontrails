#!/bin/bash

# Read species names from the txt file into an array
mapfile -t species_names < species_num.txt

replace_variables() {
    input_file="$1"
    output_file="$2"

    for ((i = 0 ; i < ${#species_names[@]} ; i++)); do
        n=$((i+1))
        # Regex to match fortran vars like Y(i), YP(i) and EM(i)
        yeq_pattern='Y\('$n'\) ='
        y_pattern='Y\('$n'\)'
        yp_pattern='YP\('$n'\)'
        em_pattern='EM\('$n'\)'
        echo "Element [$n]: ${species_names[$i]} y_pattern: $y_pattern yp_pattern: $yp_pattern em_pattern: $em_pattern"
        
        # Replace fortran vars with python syntax using sed
        sed -Ei.bak "s/$yeq_pattern/Y.loc[:, :, :, \"${species_names[$i]}\"] =/g;
                     s/$y_pattern/Y.sel(species=\"${species_names[$i]}\")/g;
                     s/$yp_pattern/YP.sel(species=\"${species_names[$i]}\")/g;
                     s/$em_pattern/EM.sel(species=\"${species_names[$i]}\")/g;" "output.py"
                
    done

    for ((i = 1 ; i < 511 ; i++)); do
        # Regex to match RC(i)
        rc_pattern='RC\('$i'\)'
    
        # Replace fortran vars with python syntax using sed
        sed -Ei.bak "s/$rc_pattern/RC.sel(therm_coeffs=$i)/g;" "output.py"

    done

    for ((i = 1 ; i < 97 ; i++)); do
        # Regex to match DJ(i)
        dj_pattern='DJ\('$i'\)'
        
        # Replace fortran vars with python syntax using sed
        sed -Ei.bak "s/$dj_pattern/DJ.sel(photol_coeffs=$i)/g;" "output.py"

    done

}


# # Call the function to replace variables in input.f90 and save the output to output.py
replace_variables "input.f90" "output.py"

