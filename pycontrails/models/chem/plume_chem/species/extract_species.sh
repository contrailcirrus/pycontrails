#!/bin/bash

# Define the dir where files are located
dir="~/pycontrails_kt/pycontrails/models/chem/plume_chem/species"

# Use a for loop to iterate over the files in dir
for file in "$dir"/*; do
	# Extract chem species name from file name
	species=$(basename "$file" | awk -F'_' '{print $1}')

    	# Append the species name to an array
    	if [ -n "$species" ]; then
        	species_list+=("$species")
    	fi
done
	
# Remove duplicates and sort the list
unique_species=($(echo "${species_list[@]}" | tr ' ' '\n' | sort -u))

# Print the list of unique chemical species
for species in "${unique_species[@]}"; do
    echo "$species"
done
