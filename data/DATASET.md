# Committed dataset

**Source dataset:** ULB / Worldline credit card fraud, OpenML 1597 v1

**Selection:** Complete dataset: all 284,807 transactions, in source order, split into five CSV files with identical headers.

**Data license:** [ODbL 1.0 database / Database Contents License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). This applies to the data, independently of the project code.

## Sources

- https://www.openml.org/api/v1/json/data/1597
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## CSV files

| File | Rows | Bytes |
|---|---:|---:|
| [creditcard_001.csv](raw/creditcard_001.csv) | 60,000 | 31,596,364 |
| [creditcard_002.csv](raw/creditcard_002.csv) | 60,000 | 31,649,854 |
| [creditcard_003.csv](raw/creditcard_003.csv) | 60,000 | 31,628,575 |
| [creditcard_004.csv](raw/creditcard_004.csv) | 60,000 | 31,696,975 |
| [creditcard_005.csv](raw/creditcard_005.csv) | 44,807 | 23,687,792 |

Attribution: Andrea Dal Pozzolo, Olivier Caelen, Gianluca Bontempi; ULB Machine Learning Group / Worldline. Source labels and numeric strings are preserved; ARFF quoting is removed during CSV conversion. There are 492 fraud labels. The original Kaggle release declares ODbL/DbCL; OpenML labels its mirror Public. The original ARFF remains local and its publisher checksum is recorded separately.

The CSV files contain published observations or their documented format conversion; no training records were invented. Source formats and raw CSV exports are distinguished by the manifest. File checksums and schemas are recorded in [csv-manifest.json](csv-manifest.json).
