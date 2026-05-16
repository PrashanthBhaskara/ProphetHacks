---
license: mit
language:
- en
---
# Prophet-Arena-Subset-100

This directory contains both the `Prophet-Arena-Subset-100` dataset itself, and some handy tools for analyzing and running predictions on `Prophet Arena` event data.

## Dataset: An Overview

This dataset contains 100 sample events from the `Prophet Arena` platform with complete source data, market information, and submission details used for `Prophet Arena` benchmarking.

> **Note that:** many event outcomes are predicted more than once, and in the following dataset, we only take the first time of each events' prediction (referred to as the first submission).

### Event Category Distribution

| Category | Count |
|----------|-------|
| `Sports` | 75 |
| `Politics` | 5 |
| `Economics` | 5 |
| `Entertainment` | 5 |
| `Other` | 10 |

> **Note that:** the category distribution of this subset **approximates**, but does **NOT match exactly** the full distribution of events on the `Prophet Arena` platform. The abundance of `Sports` events is due to their high representation on the `Kalshi` platform -- from which our current events are sourced from.

### CSV Schema

The raw data is stored in the CSV format (`subset_data_100.csv`), with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `event_ticker` | String | Unique identifier for the event |
| `title` | String | Human-readable title of the event |
| `category` | String | Category classification of the event |
| `markets` | JSON Array | List of prediction markets for this event |
| `close_time` | DateTime | When the event closes for predictions |
| `market_outcome` | JSON Object | Whether each market was resolved as true (1) or false (0) |
| `sources` | JSON Array | List of sources used in the first submission (see Sources Schema) |
| `market_info` | JSON Object | Market trading data at snapshot time (see Market Info Schema) |
| `snapshot_time` | DateTime | When the market data was captured |
| `submission_id` | String | ID of the first submission for this event (can be ignored) |
| `submission_created_at` | DateTime | When the first submission was created |

### Sources Schema

Each event contains a `sources` field with an array of source objects from the first submission. Each source has:

| Field | Type | Description |
|-------|------|-------------|
| `summary` | String | Summary of the source content |
| `source_id` | String | Unique database ID for the source (can be ignored) |
| `ranking` | Integer | Human-based ranking (1 = most popular among raters) |
| `title` | String | Title of the source article/content |
| `url` | String | URL to the original source |

### Market Info Schema

Each event contains a `market_info` field with trading data for each market **at the time of the first submission**. 

| Field | Type | Description |
|-------|------|-------------|
| `last_price` | Float | Most recent trading price |
| `yes_ask` | Float | Current asking price for "Yes" outcome |
| `no_ask` | Float | Current asking price for "No" outcome |
| Plus additional market metadata (ticker, rules, etc.) |

## Tools

### standalone_predictor.py

Self-contained prediction script that runs LLM predictions on event datasets. 

**Usage:**
```bash
# Run predictions on all events
python3 standalone_predictor.py \
  --input_csv test_dataset_100.csv \
  --output_csv predictions.csv \
  --base_url https://api.openrouter.ai/api/v1 \
  --api_key YOUR_API_KEY \
  --model gpt-4 \
  --run_all

# Run predictions on specific events
python3 standalone_predictor.py \
  --input_csv test_dataset_100.csv \
  --output_csv predictions.csv \
  --base_url https://api.example.com/v1 \
  --api_key YOUR_API_KEY \
  --model custom-model \
  --run_specific EVENT1,EVENT2,EVENT3
```

### standalone_evaluator.py

Once you have obtained the `output_csv` from running the first script (`standalone_predictor.py`), you can perform evaluations on the predictions you've obtained.

Specifically, the current `standalone_evaluator.py` supports two import metrics (both averaged over 100 events): (1) the Brier score, and (2) the average return (using a risk-neutral strategy with $1 per event budget).
Please refer to the [blogpost section](https://www.prophetarena.co/blog/welcome#evaluation-metrics-for-forecasts) if you want to understand these metrics better.

In order to use this evaluator script, make sure that you first install the `pm-rank` package (e.g. via pip):
```bash
# latest version (requires python version >= 3.8)
pip install pm-rank>=0.2.25  
```

**Usage:**
_Assuming that you have run the `standalone_predictor.py` to obtain the outputs in `predictions.csv`_.

Note that the `input_csv` and `output_csv` arguments should point to the same file paths as those used in `standalone_predictor.py`.
```bash
# Get Brier score from an input csv results file
python standalone_evaluator.py \
  --input_csv test_dataset_100.csv \
  --output_csv predictions.csv \
  --metric brier

# Get average return from an input csv results file, with (1) verbose output turned on, and (2) save results to `log.csv`
python standalone_evaluator.py \
  --input_csv test_dataset_100.csv \
  --output_csv predictions.csv \
  --metric average_return \
  --verbose \
  --log_csv log.csv
```

**Features:**
- **Self-contained**: No dependencies on the main app module
- **Flexible API support**: Works with OpenRouter, custom endpoints, etc.
- **Robust parsing**: Handles UUID objects and Python dict representations in CSV data
- **Market data integration**: Extracts `last_price`, `yes_ask`, `no_ask` for LLM context
- **Async processing**: Parallel processing for multiple events with `--run_all`
- **Error handling**: Continues processing other events if one fails
- **Complete prediction storage**: Stores full prediction JSON (probabilities + rationale)

**Output Schema:**
The prediction CSV contains:
- `event_ticker`, `title`, `category`, `markets`: Original event data
- `prediction`: Complete JSON with probabilities array and rationale
- `model`: Model used for prediction
- `status`: `success` or `error`
- `error_message`: Error details if prediction failed


## Notes

- All prompts, sources, and market data are used exactly by the benchmarked LLMs at their time of prediction.
- The dataset captures the **first submission** for each event to provide a consistent baseline
- Market info provides real market consensus data at the time of submission
- Sources are filtered to only those actually used in the specific submission
- The standalone predictor replicates the production prediction pipeline for research use

## Useful Links

- [Prophet Arena Platform](https://prophetarena.co)
- [Blogpost on the scoring/ranking module](https://ai-prophet.github.io/pm_ranking/blogpost/ranking_llm_250727.html#)