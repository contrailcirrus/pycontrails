# CoCiP Benchmark Testing

Comparison data output from [`Cocip` model](https://py.contrails.earth/api/pycontrails.models.cocip.Cocip.html#pycontrails.models.cocip.Cocip).

Input/output data stored in [gs://contrails-301217-benchmark-data/cocip](https://console.cloud.google.com/storage/browser/contrails-301217-benchmark-data/cocip?) (public).

Requires [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed to download data.


## Data

See [data.md](data.md) for a dictionary of input and output values.

### Inputs

Download input meteorology data (15 GB) for model from GCP bucket to local `inputs/` directory. From the current directory:

```bash
$ make download-inputs
```

Flight input data is provided in [Apache Parquet (.pq)](https://parquet.apache.org/) format and csv. The data in each file is equivalent.


### Ouputs

Download output data for model from gcp bucket to local `outputs/` directory. From the current directory:

```bash
$ make download-outputs
```

Flight and contrail output data is stored in the [Apache Parquet (.pq)](https://parquet.apache.org/) format.


## Run comparison against previous data

```bash
$ make compare
```

## Generate new output data

```bash
$ make generate-outputs
```

To update the reference data in the bucket

```bash
$ make upload-outputs
```

To update the input data in the bucket

```bash
$ make upload-inputs
```
