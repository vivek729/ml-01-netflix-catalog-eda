# Netflix Catalog - Cleaning and Exploratory Analysis

An exploratory analysis of the [Netflix Movies and TV Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows). The focus is on getting reliable title counts from catalog metadata: resolving likely duplicate listings, correcting malformed values, and accounting for columns that contain multiple values.

[Read the notebook](nb_netflix_da.ipynb)

## Approach

The source has 8,807 rows covering movies and TV shows, with catalog addition dates through September 2021. The notebook audits the raw data, prepares it for analysis, then looks at ratings, genre tags, production countries, credited people, addition dates, and the calendar-year difference between release and addition.

- **Resolve likely duplicate listings.** Repeated descriptions surfaced alternate titles and language variants that were not duplicate rows. The notebook retains the first listing for each description and removes 32 subsequent records. An exact description is a useful matching heuristic here, but it is not a guaranteed work identifier; this choice can discard listing-specific metadata or merge distinct works.
- **Repair and normalize fields.** Three movie durations were stored in `rating`; these were moved to `duration`. The analysis also splits `country`, `director`, `cast`, and `listed_in` into individual values, trims whitespace, and removes empty tokens caused by stray commas. Movie minutes and TV seasons are parsed separately.
- **Count at the right grain.** Exploding the four multi-value columns expands the cleaned data to 201,228 rows. Attribute counts therefore use distinct `(show_id, attribute)` pairs; counting rows directly would overstate titles with larger casts or more tags.
- **Keep uncertainty visible.** The notebook tests director imputation on a copy, then rejects it because it creates an implausible concentration of credits. Missing directors remain missing. Duration values flagged by an IQR rule are inspected and retained when they represent valid shorts, long films, or long-running series.

## What the data shows

TV shows are more often added in their release year than movies (about 52% versus 30%). Friday is the most common recorded addition day for both formats. These describe the entries in this dataset; the year comparison is based on `release_year`, so it is not a precise release-to-addition interval.

The notebook also compares the catalog by maturity rating, genre tag, production country, cast, and director. Each title can contribute to multiple genre and country counts. These are counts of catalog entries, not measures of popularity.

## Limits

This dataset contains no viewing, satisfaction, revenue, or regional availability data. `country` identifies production country, and `rating` is a maturity classification, not a viewer score. It is a snapshot rather than a history of additions and removals. The analysis can describe catalog composition, but recommendations about what to produce or which market to enter would require demand and availability data.

## Run locally

Download the CSV from the [dataset page](https://www.kaggle.com/datasets/shivamb/netflix-shows), save it as `nflx.csv` beside the notebook, and run the cells in order with Jupyter, pandas, NumPy, Matplotlib, and Seaborn. The CSV is not included in this repository.

One inspection cell uses `groupby.apply(include_groups=False)`, which requires pandas 2.2 or later.
