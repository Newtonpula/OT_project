# OT_project
OT project for Romain's class

## Data

The `.h5` datasets (black-body test series, CMFGEN model spectra and SN spectral time series) are not
stored in git.
Build them locally as described in [data/data_description.md](data/data_description.md#building-the-h5-files-locally).
 
Please note that in the SNDataset, `m['mass']` still not sum to one due to some convention in the original paper. Please divide it by the sum manually. For example,
```
m = SNDataset().ot_measures('SN2011fe')
rho = m['mass'] / m['total'][:, None]   # so that sum of rho is 1
F = m['total']                           # the 'total brightness'
```