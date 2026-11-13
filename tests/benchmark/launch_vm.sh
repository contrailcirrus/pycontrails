#!/bin/bash
# Launch one benchmark VM for each pycontrails tag, branch or commit given as an argument.
#
#     tests/benchmark/launch_vm.sh v0.65.0 0.22.4
#
# This runs the same benchmark as the benchmark workflow, and can also run releases that are only
# in pycontrails-archive. The workflow installs the newest compatible pycontrails-bada from Artifact
# Registry; this script uses the newest pycontrails-bada tag from before the pycontrails release.
#
# Each VM runs vm_startup.sh and deletes itself when done. The VM cannot clone private repositories,
# so the pycontrails-bada source, and the pycontrails source of refs only in pycontrails-archive,
# are staged under SRC. Requires clones of both repositories next to this one.


set -eu
cd "$(dirname "$0")/../.."

PYCONTRAILS_ARCHIVE=../pycontrails-archive
PYCONTRAILS_BADA=../pycontrails-bada
SRC=gs://contrails-301217-benchmark-data/src
archive_file=$(mktemp)
trap 'rm -f "$archive_file"' EXIT

git fetch --quiet --tags origin
git -C "$PYCONTRAILS_BADA" fetch --quiet --tags origin

# Upload the source of repository $1 at tag $3 to $SRC/$2-$3.tar.gz unless it is already there,
# and print its URL
stage() {
    local url="$SRC/$2-$3.tar.gz"
    if ! gcloud storage ls "$url" > /dev/null 2>&1; then
        git -C "$1" archive --format=tar.gz --output="$archive_file" "$3"
        gcloud storage cp --quiet "$archive_file" "$url" > /dev/null 2>&1
    fi
    echo "$url"
}

# Succeed if version $1 is not newer than version $2
version_le() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]
}

for ref in "$@"; do
    metadata="pycontrails-ref=$ref"
    if commit=$(git rev-parse --verify --quiet "origin/$ref^{commit}" \
        || git rev-parse --verify --quiet "$ref^{commit}"); then
        repo=.
    else
        repo=$PYCONTRAILS_ARCHIVE
        commit=$(git -C "$repo" rev-parse --verify "$ref^{commit}")
        metadata+=",pycontrails-source=$(stage "$repo" pycontrails-archive "$ref")"
    fi
    cutoff=$(TZ=UTC git -C "$repo" log -1 --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd "$commit")
    metadata+=",cutoff=$cutoff"

    version=$(git -C "$repo" describe --tags --abbrev=0 "$commit")
    bada_cutoff=$(($(git -C "$repo" log -1 --format=%ct "$commit") + 86400))
    bada_tag=$(
        git -C "$PYCONTRAILS_BADA" tag -l "v*" --sort=-v:refname \
            | while read -r tag; do
                required=$(git -C "$PYCONTRAILS_BADA" show "$tag:pyproject.toml" 2>/dev/null \
                    | grep -o "pycontrails>=[0-9.]*" | head -1 | cut -d= -f2)
                if [ "$(git -C "$PYCONTRAILS_BADA" log -1 --format=%ct "$tag")" -lt "$bada_cutoff" ] \
                    && version_le "${required:-0}" "${version#v}"; then
                    echo "$tag"
                    break
                fi
            done
    )
    if [ -n "$bada_tag" ]; then
        metadata+=",pycontrails-bada-version=${bada_tag#v}"
        metadata+=",pycontrails-bada-source=$(stage "$PYCONTRAILS_BADA" pycontrails-bada "$bada_tag")"
    fi

    gcloud compute instances create "pycontrails-benchmark-${ref//[._\/]/-}" \
        --project contrails-301217 \
        --zone us-east1-b \
        --machine-type c3-highmem-4 \
        --image-family debian-12 \
        --image-project debian-cloud \
        --boot-disk-size 50GB \
        --boot-disk-type pd-balanced \
        --service-account ci-service-account@contrails-301217.iam.gserviceaccount.com \
        --scopes cloud-platform \
        --max-run-duration 6h \
        --instance-termination-action DELETE \
        --metadata "$metadata" \
        --metadata-from-file startup-script=tests/benchmark/vm_startup.sh,run-benchmark=tests/benchmark/run_benchmark.py \
        --quiet > /dev/null
    echo "$ref: ${repo##*/}, pycontrails-bada ${bada_tag:-none}"
done
