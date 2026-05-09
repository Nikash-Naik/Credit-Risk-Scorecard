# Data

This project uses the Lending Club accepted loans dataset (2007–2018 Q4).

## Download instructions

1. Get the dataset from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Download `accepted_2007_to_2018Q4.csv.gz` (~600 MB)
3. Place the file in this folder

The data file is gitignored due to its size and is not included in the repository.

## Schema

See [the Lending Club data dictionary](https://resources.lendingclub.com/LCDataDictionary.xlsx) for full field descriptions. The notebook works with 151 columns and downselects to ~110 features after leakage control.
