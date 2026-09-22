# Netflix Catalog - Cleaning and Exploratory Analysis

An EDA of the public Netflix titles dataset: 8,807 titles, scraped as a single snapshot dated 25 Sep 2021. Most of the work here is in data preparation rather than in the charts. The dataset looks tidy at first glance and isn't, and the failure modes it hides (multi-valued fields, a shifted column, duplicate titles that aren't row-duplicates) are the kind that quietly corrupt every downstream number.

**Notebook:** [`nb_netflix_da.ipynb`](nb_netflix_da.ipynb)

## Scope, and what this data cannot tell you

The original problem statement for this exercise reads: *"generate insights that could help Netflix decide which type of shows/movies to produce."* That framing overreaches, and it's worth being direct about why before showing any results.

The dataset is catalog metadata only: title, type, cast, director, production country, date added, release year, maturity rating, duration, genre tags. There is no viewership, no watch time, no user rating, no completion rate, no revenue, and no per-region availability.

| The data supports | The data does not support |
|---|---|
| What Netflix added to its catalog, and when | What anyone watched, or liked |
| The composition of the catalog by genre, rating, country, duration | Whether that composition performed well |
| Release-to-catalog lag as an operational pattern | Commissioning or licensing recommendations |

Every number below is a count of **titles**, not of viewers. "International Movies is the largest genre tag" describes supply. A supply-side count cannot be read as demand, and treating it that way reverses the causality: a genre may be large because it's cheap to license, not because it wins audiences.

Two further limits that aren't obvious from the file itself:

- **`country` is the production country, not the availability country.** It says where a title was made, not where it can be streamed. Any conclusion about "growing the business in country X" would need availability data this file doesn't have.
- **It's a snapshot with no removals.** Titles dropped from the catalog before Sep 2021 simply aren't in it, so the apparent growth in additions over time is partly survivorship. Older cohorts have had more time to be pruned.

What follows, then, is an honest description of a catalog and the pipeline needed to describe it correctly.

## Data

Source: [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Kaggle).

- 8,807 rows, 12 columns; 6,131 movies and 2,676 TV shows
- `date_added` spans 2008-01-01 to 2021-09-25; `release_year` spans 1925 to 2021
- No fully duplicated rows; `show_id` is unique and non-null

`description` is dropped early. Using it properly means NLP, which is out of scope here, and it's the widest column in memory.

## Cleaning

The substance of the project. Each step below came out of an observation in the data, not from a checklist.

### 1. Duplicate titles that aren't duplicate rows

`df.duplicated()` returns zero. But `describe(include='all')` showed one `description` occurring four times. Those four rows turned out to be the same film listed under four language variants of its title, identical in every other field. Checking `description` as a duplicate key across the whole frame surfaced **32 such rows**, kept the first of each group, and dropped the rest, leaving 8,775 titles.

Worth noting the assumption: identical free-text descriptions are treated as identifying the same work. That held on inspection for these groups, but it is an assumption, not a guarantee.

### 2. A shifted column

Three titles (`s5542`, `s5795`, `s5814`) carried `74 min`, `84 min`, `66 min` in the **`rating`** column, a parsing error upstream that left `duration` empty. Values were moved to `duration` and `rating` set to null, to be handled with the other missing ratings.

This is the kind of thing a null-count audit misses entirely: the field wasn't missing, it was wrong.

### 3. Multi-valued columns, and the comma trap

`country`, `director`, `cast`, and `listed_in` each pack multiple values into one comma-separated string. Exploding them is straightforward; what isn't is that some values carry a **leading or trailing comma**. `"United States,"` splits into `["United States", ""]` and the explode produces a phantom row with a blank country. That accounted for 7 junk rows, and would have produced more in the other three columns had they not been stripped of stray commas before splitting.

### 4. Fan-out, and counting discipline

Exploding four columns multiplies rows hard:

| Stage | Rows |
|---|---|
| After dedup | 8,775 |
| explode `country` | 10,807 |
| explode `director` | 11,874 |
| explode `cast` | 89,102 |
| explode `listed_in` | **201,228** |

8,775 titles become 201,228 rows. A title with 3 countries, 2 directors, 12 cast members and 3 genres appears 216 times. **Any `value_counts()` on this frame is wrong**: it counts cast-list length and genre breadth, not titles.

The pattern used throughout the analysis is to collapse to unique `(show_id, attribute)` pairs before counting:

```python
df_tv.loc[~df_tv.duplicated(subset=['show_id', 'rating'], keep='first'), 'rating'].value_counts()
```

The missing-value audit has the same trap. Post-explode, `director` reads 25.09% null; on the original frame it's **29.91%**. The exploded figure is diluted by rows that fanned out from well-populated records, and is the misleading one.

### 5. Missing values, including one imputation that was rejected

Null rates on the original frame:

| Column | Null % |
|---|---|
| `director` | 29.91 |
| `country` | 9.44 |
| `cast` | 9.37 |
| `date_added` | 0.11 |
| `rating` | 0.05 |
| `duration` | 0.03 |

`director` was the only one worth attempting. I tried group-mode imputation on `(country, listed_in, type)` in a **separate copy of the frame**, which cut nulls from 2,626 to 991, and then inspected the result rather than accepting the improvement. It had concentrated a handful of prolific directors across large blocks of titles, manufacturing a skew that would have distorted exactly the "top directors" analysis the column feeds.

**The imputation was discarded** and the analysis continued on the un-imputed frame, treating `director` nulls as missing. A 30% gap is honest; a fabricated mode is not. The remaining columns, all under 10%, were left as-is.

### 6. Types and per-type duration

`duration` is two different units in one column, minutes for movies and seasons for TV, so it can't be parsed until the frame is split. After splitting into `df_tv` and `df_mov`: `duration_in_seasons` and `duration_in_min` as `int`, `type` / `rating` / `listed_in` as `category`, `date_added` to datetime with `year`, `month`, `day`, `week`, `dayofweek` derived.

### 7. Outliers that turned out to be correct data

IQR flagged 259 TV shows (9.7%) and 445 movies (7.3%). Inspecting them instead of dropping them:

- TV: bounds were [-0.5, 3.5] seasons, so anything past 3 seasons is "an outlier." The maximum is 17 seasons, which is long-running, not wrong.
- Movies: bounds [46.5, 154.5] min. Below-range entries are shorts and stand-up specials, down to 3 min; above-range are long-form films, up to 312 min. Both are real.

Nothing was treated. The IQR rule was measuring a skewed-but-valid distribution, which is a limitation of the rule, not a property of the data.

## What the catalog looks like

All figures are title counts on the cleaned, dedup-before-count frames.

**Release-to-catalog lag.** 52% of TV shows (1,375 of 2,660) were added in their release year, against 30% of movies (1,848 of 6,105), with another 19% of movies added one year out. TV skews toward near-simultaneous availability; the movie catalog leans more on back-catalog licensing.

A small number of titles have a **negative** lag, added before their stated release year (12 TV shows, 2 movies). Either a release-year error or a pre-release listing; flagged and excluded from the lag figures rather than clipped to zero.

**When titles get added.** Friday dominates both: 35% of TV additions (930) and 25% of movie additions (1,556) land on a Friday, more than double any other day. Monthly variation is mild. TV peaks in December (266) and July (261); movies in July (564), April, December and January. February is lowest for movies (381), though a 28-day month makes that partly an artifact of month length rather than a scheduling decision.

Week 1 is a sharp spike for movies, 315 additions against 243 for the next-highest week, consistent with a New Year batch drop.

**Composition.** TV-MA is the largest maturity rating for both (43% of TV, 34% of movies); the catalog skews adult. Largest genre tags are International TV Shows (1,349), TV Dramas (761) and TV Comedies (581) for TV; International Movies (2,732), Dramas (2,416) and Comedies (1,665) for film.

**Production geography.** The US leads both. The second and third slots differ by format: UK (272) and Japan (199) for TV, India (954) and UK (534) for film. Read as (title, country) pairs, so a co-production counts once per country.

**People.** The most-credited TV performers are almost entirely Japanese voice actors (Takahiro Sakurai, 25 titles). That reflects how anime series are catalogued, a large number of separately-listed titles with recurring voice casts, more than it says anything about prominence. Film credits are led by Indian actors (Anupam Kher, 41; Shah Rukh Khan, 35), tracking India's position as the second-largest production country. Cast lists are unordered and of uneven length, so "credits" here is not "leading roles."

## Notes worth keeping

A few things this dataset taught me that generalise:

- **Pandas has two day-of-week conventions.** `dt.dayofweek` and `dt.weekday` are 0-6 with Monday = 0; `dt.isocalendar().day` is 1-7 with Monday = 1. `isocalendar().week` is 1-53, never 0. Mixing them silently shifts every weekday result by one. The notebook uses `dayofweek` and annotates the convention at each plot.
- **ISO weeks straddle year boundaries.** Early-January dates can return week 52 or 53 of the *previous* ISO year, so `isocalendar().year`, not `dt.year`, is the correct partner for week-level grouping.
- **`groupby(...).apply()` needs `include_groups=False`** from pandas 2.2 onward to keep the grouping column out of the operated-on frame.
- **Improvement is not validation.** The director imputation reduced nulls by 62% and was still the wrong call. The check that mattered was looking at the resulting distribution, not the null count.

## Where this would go next

Roughly in order of value:

1. **Per-country analysis.** The pipeline supports it, and the top-3 countries per format are already identified, but only the global cut is written up. Genre and rating mix by country is the most direct extension.
2. **Additions over time, with the survivorship caveat stated.** Monthly additions by format and genre, framed as catalog-building activity and not as growth.
3. **`description` via NLP.** The column was dropped for scope. Embedding it would allow genre tags to be checked against content, and near-duplicate detection more principled than exact description matching.
4. **Join an external signal.** The honest fix for the demand-side gap: IMDb or TMDB ratings, keyed on title and release year. That turns "what's in the catalog" into something that can actually speak to reception, with the usual caveats about matching on title.
