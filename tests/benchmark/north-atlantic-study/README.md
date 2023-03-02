# Cocip Output Validation

> This comparison is superceded by `benchmark/cocip` outputs

Single day in the North Atlantic 2019-01-01.

Compared to original python cocip implementation (*pycocip*) output on `feature/nats` @ 7c85975340ec97f8d90d0a1557a47e36ae9aeb74.

Input/output data stored in `gs://contrails-301217-benchmark-data-archive/north-atlantic-study`.


## Requires

Download input data for model from gcp bucket to local `inputs/` directory

```bash
$ gsutil -m cp -r gs://contrails-301217-benchmark-data-archive/north-atlantic-study/inputs .
```

Download output data for model from gcp bucket to local `outputs/` directory

```bash
$ gsutil -m cp -r gs://contrails-301217-benchmark-data-archive/north-atlantic-study/outputs .
```

## Sync validation data

To update the validation data in the bucket

```bash
$ gsutil rsync -d -r inputs/ gs://contrails-301217-benchmark-data-archive/north-atlantic-study/inputs
$ gsutil rsync -d -r outputs/ gs://contrails-301217-benchmark-data-archive/north-atlantic-study/outputs
```
