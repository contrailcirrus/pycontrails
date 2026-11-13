#!/bin/bash
# Run the CoCiP benchmark on a VM created by launch_vm.sh or the benchmark workflow.
#
# Metadata values:
#   pycontrails-ref                 tag, branch or commit to benchmark
#   cutoff                          its commit time
#   run-benchmark                   contents of run_benchmark.py
#   pycontrails-source              optional, URL of the staged pycontrails source
#   pycontrails-bada-version        optional, version of the staged pycontrails-bada source
#   pycontrails-bada-source         optional, URL of the staged pycontrails-bada source
#   pycontrails-bada-from-registry  optional, "true" to install it from Artifact Registry
#
# On exit, print BENCHMARK_EXIT_CODE=<code> and delete the VM.

set -eu

# Print the value at path $1 of the metadata server, relative to the VM instance
metadata() {
    curl --silent --fail --header "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/$1"
}

finish() {
    echo "BENCHMARK_EXIT_CODE=$?"
    # Give the guest agent time to send the last output lines to Cloud Logging
    sleep 10
    zone=$(metadata zone)
    gcloud compute instances delete "$(hostname)" --zone "${zone##*/}" --quiet
}
trap finish EXIT

mkdir /benchmark
cd /benchmark
metadata attributes/run-benchmark > run_benchmark.py

ref=$(metadata attributes/pycontrails-ref)
cutoff=$(metadata attributes/cutoff)
pycontrails_source=$(metadata attributes/pycontrails-source || true)
pycontrails_bada_version=$(metadata attributes/pycontrails-bada-version || true)
pycontrails_bada_source=$(metadata attributes/pycontrails-bada-source || true)
pycontrails_bada_from_registry=$(metadata attributes/pycontrails-bada-from-registry || echo false)
machine_type=$(metadata machine-type)

# Index URL of the project's Artifact Registry, with the VM's access token, to install
# pycontrails-bada from
pycontrails_bada_index_url=""
if [ "$pycontrails_bada_from_registry" = true ]; then
    token=$(metadata service-accounts/default/token \
        | grep -o '"access_token": *"[^"]*"' | cut -d '"' -f 4)
    registry=us-central1-python.pkg.dev/contrails-301217/pycontrails/simple
    pycontrails_bada_index_url="https://oauth2accesstoken:$token@$registry"
fi

apt-get update
apt-get install --yes docker.io git

if [ -n "$pycontrails_source" ]; then
    mkdir -p /src/pycontrails
    gcloud storage cat "$pycontrails_source" | tar xz -C /src/pycontrails
else
    # Full clone, since setuptools-scm reads tags to set pycontrails.__version__
    git clone --quiet https://github.com/contrailcirrus/pycontrails.git /src/pycontrails
    git -C /src/pycontrails checkout --quiet "$ref"
fi

if [ -n "$pycontrails_bada_source" ]; then
    mkdir -p /src/pycontrails-bada
    gcloud storage cat "$pycontrails_bada_source" | tar xz -C /src/pycontrails-bada
fi

# Date by which all of pyarrow, multidict, yarl, frozenlist, aiohttp, netCDF4, cftime, numpy,
# pandas and scipy had a manylinux x86_64 wheel for each Python version
# 3.8 is left out because pip 26.0, needed for PIP_UPLOADED_PRIOR_TO, requires Python 3.9.
declare -A WHEEL_DATE=(
    [3.9]=2021-01-26
    [3.10]=2021-11-06
    [3.11]=2022-12-03
    [3.12]=2024-02-01
    [3.13]=2024-10-28
    [3.14]=2025-10-24
    [3.15]=2026-10-09
)

# Newest Python version in the classifiers of the benchmarked pycontrails with wheels before the
# commit date
python_version=$(
    cat \
        /src/pycontrails/pyproject.toml \
        /src/pycontrails/setup.cfg \
        /src/pycontrails/setup.py \
        2>/dev/null \
        | grep -o "Programming Language :: Python :: 3\.[0-9]*" \
        | grep -o "3\.[0-9]*" \
        | sort -t. -k2 -nr \
        | while read -r v; do
            if [[ "${WHEEL_DATE[$v]:-9999}" < "${cutoff:0:10}" ]]; then
                echo "$v"
                break
            fi
        done
)
if [ -z "$python_version" ]; then
    echo "No Python version in the classifiers has wheels before $cutoff" >&2
    exit 1
fi

# Download BADA3
gcloud storage cp gs://contrails-301217-bada/bada-tar/bada3.tar.gz .
mkdir bada
tar xzf bada3.tar.gz -C bada

# Airport database (contrailcirrus/ourairports-data), read by pycontrails from its cache
mkdir cache
gcloud storage cp \
    gs://contrails-301217-benchmark-data/inputs/airports.csv \
    cache/ourairports-data_airports.csv

# Write the script to run in the container
cat > container.sh <<'EOF'
# pip 26.0 or later for PIP_UPLOADED_PRIOR_TO
pip install --quiet --upgrade pip

# Install the packages uploaded before the commit date, including build dependencies.
# Older pycontrails-bada import overrides, older pycontrails import tqdm, cdsapi, ecmwfapi,
# and the xarray installed for pycontrails 0.11.0 imports packaging, without declaring
# them as dependencies. pyarrow 19.0.0 cannot read the staged Parquet files.
PIP_UPLOADED_PRIOR_TO=$CUTOFF pip install --quiet \
    /src/pycontrails gcsfs netcdf4 'pyarrow!=19.0.0' overrides tqdm packaging cdsapi \
    ecmwf-api-client

# The pycontrails-bada source has no git metadata, so give setuptools-scm its version
if [ -n "$PYCONTRAILS_BADA_VERSION" ]; then
    SETUPTOOLS_SCM_PRETEND_VERSION=$PYCONTRAILS_BADA_VERSION PIP_UPLOADED_PRIOR_TO=$CUTOFF \
        pip install --quiet --no-deps /src/pycontrails-bada
elif [ -n "$PYCONTRAILS_BADA_INDEX_URL" ]; then
    # Install the newest pycontrails-bada that works with the installed pycontrails. The
    # constraint keeps pip from replacing the installed pycontrails.
    python -c 'import importlib.metadata as m; print("pycontrails==" + m.version("pycontrails"))' \
        > /tmp/constraints.txt
    pip install --quiet --constraint /tmp/constraints.txt \
        --index-url "$PYCONTRAILS_BADA_INDEX_URL" pycontrails-bada
fi

# Copy the files in the pycontrails source that the install leaves out: the static data files
# before pycontrails 0.25.2, and packages without __init__.py
site_packages=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(cd /src/pycontrails && find pycontrails -type f | while read -r f; do
    [ -e "$site_packages/$f" ] || cp --parents "$f" "$site_packages"
done)

# Put BADA3 at the default path, next to the installed pycontrails package
ln -s /benchmark/bada "$(dirname "$site_packages")/bada"

python run_benchmark.py
EOF

# Host networking lets gcsfs reach the metadata server
docker run \
    --network host \
    --volume /src:/src \
    --volume /benchmark:/benchmark \
    --workdir /benchmark \
    --env PYCONTRAILS_CACHE_DIR=/benchmark/cache \
    --env CUTOFF="$cutoff" \
    --env PYCONTRAILS_BADA_VERSION="$pycontrails_bada_version" \
    --env PYCONTRAILS_BADA_INDEX_URL="$pycontrails_bada_index_url" \
    --env MACHINE_TYPE="${machine_type##*/}" \
    --env PIP_ROOT_USER_ACTION=ignore \
    "python:$python_version" \
    bash -eu container.sh
