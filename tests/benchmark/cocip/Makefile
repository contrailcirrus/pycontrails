
GCP_BUCKET = gs://contrails-301217-benchmark-data

# Put first so that "make" without argument is like "make help".
help:
	echo "See Makefile for recipe list"

.PHONY: help

# download inputs
download-inputs:
	mkdir -p inputs
	gsutil -m rsync -d -r $(GCP_BUCKET)/cocip/inputs inputs/

# download outputs
download-outputs:
	mkdir -p outputs
	gsutil -m rsync -d -r $(GCP_BUCKET)/cocip/outputs outputs/

# run model and generate output data
generate-outputs:
	ipython output.py

# run model and compare with existing output data
compare:
	ipython compare.py

# sync data after regenerating
upload-inputs:
	gsutil -m rsync -n -x ".*.DS_Store" -d -r inputs/ gs://contrails-301217-benchmark-data/cocip/inputs

upload-outputs:
	gsutil -m rsync -x ".*.DS_Store" -d -r outputs/ gs://contrails-301217-benchmark-data/cocip/outputs

upload: upload-inputs upload-outputs
