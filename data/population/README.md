# Population data

`World_Bank_population_2024.csv` is a frozen World Bank World Development Indicators download for `SP.POP.TOTL` (Population, total). The analysis uses the `2024` column.

Source: World Bank, Population, total:

- https://data.worldbank.org/indicator/SP.POP.TOTL
- CSV download endpoint: https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv

The raw file contains country rows and World Bank aggregate entities. The pipeline matches the `Country Name` column against the source-country names in the case data and reads positive values from `2024`.

The World Bank page identifies the indicator license as CC BY-4.0. This repository keeps the 2024 column as a reproducibility snapshot; it is not a live mirror of the World Bank series.
