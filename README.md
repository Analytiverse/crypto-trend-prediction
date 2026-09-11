# crypto-trend-prediction
## 1. Project Overview

AlphaPulse is an end-to-end machine learning project for predicting short-term cryptocurrency price direction.

The system collects hourly cryptocurrency market data from CoinGecko, stores and maintains the historical dataset in PostgreSQL, validates and repairs data quality issues, performs exploratory data analysis (EDA), constructs future trend targets, and engineers features for machine learning.

The prediction problem is formulated as a 3-class classification task:

- `UP`
- `DOWN`
- `STABLE`

The project currently covers:

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- XRP
- Cardano (ADA)

Prediction horizons:

- 6 hours
- 12 hours
- 24 hours

A key requirement throughout the project is **temporal correctness**: features at time `t` must only use information available at or before `t`.

---

## 2. Current Project Flow

```text
CoinGecko API
      ↓
API Client
      ↓
PostgreSQL / Neon
      ↓
Historical Backfill
      ↓
Automated Daily Ingestion
      ↓
Cleaning & Validation
      ↓
Gap Detection & Repair
      ↓
Exploratory Data Analysis
      ↓
Target Definition
      ↓
Feature Engineering
      ↓
Temporal Modeling Dataset
      ↓
Baseline + ML Models
      ↓
Evaluation
```

The project is currently complete through **Feature Engineering**.

---

## 3. Data Collection

CoinGecko is used as the market data source.

The main historical endpoint provides hourly information including:

- Price
- Market capitalization
- Trading volume
- Timestamp

The database is the primary source for analysis and modeling rather than local CSV files.

Main PostgreSQL tables:

- `coins`
- `market_history_raw`
- `market_hourly`
- `market_snapshots`

`market_history_raw` preserves the raw historical observations, while `market_hourly` contains the cleaned hourly data used for analysis.

---

## 4. Historical Backfill & Automated Ingestion

An initial historical backfill was performed to populate the PostgreSQL database.

After the historical dataset was created, a daily ingestion pipeline was implemented.

The pipeline:

1. Fetches recent CoinGecko observations.
2. Cleans and normalizes the data.
3. Upserts observations into PostgreSQL.
4. Avoids duplicate `(coin_id, timestamp)` records.
5. Validates the updated dataset.

The ingestion process uses overlapping recent history so that recent missing observations can automatically be recovered.

---

## 5. Gap Detection and Repair

During EDA, historical hourly gaps were identified across multiple cryptocurrencies.

This exposed an important limitation in the rolling ingestion approach: if an outage became older than the normal recent-history fetch window, daily ingestion could no longer recover it.

A historical gap-repair process was therefore added.

The repair process:

1. Detects intervals where timestamps are more than one hour apart.
2. Determines the exact missing time range.
3. Requests that historical interval from CoinGecko.
4. Cleans the returned observations.
5. Filters them to the missing interval.
6. Upserts the recovered rows.
7. Re-runs validation.

After repair, the current dataset contains no detected gaps greater than one hour.

---

## 6. Current Dataset

At the completion of the EDA / feature-engineering stage:

- Total observations: **11,700**
- Observations per coin: **2,340**
- Number of cryptocurrencies: **5**
- Frequency: **Hourly**
- Date range: **2026-06-03 to 2026-09-09**

Current data-quality checks show:

- No duplicate coin/timestamp records
- No invalid prices
- No invalid market-cap values
- No invalid negative volume values
- No remaining hourly gaps

---

## 7. Exploratory Data Analysis

EDA focused on understanding whether the data is suitable for short-term directional prediction and which information could potentially be useful for modeling.

The main analyses included:

- 1h, 6h, 12h and 24h historical returns
- Return distributions
- Rolling volatility
- Autocorrelation
- Cross-coin correlation
- Future-return distributions
- Target class distributions
- Volume behavior

### Volatility

The standard deviation of 24-hour returns was approximately:

| Coin | 24h Return Std |
|---|---:|
| BTC | 2.04% |
| ETH | 3.00% |
| SOL | 3.08% |
| XRP | 3.59% |
| ADA | 4.31% |

This shows that the cryptocurrencies have materially different volatility profiles.

### Autocorrelation

Short-term return autocorrelations were generally close to zero.

This suggests that simple rules based only on previous price direction are unlikely to be sufficient, although lagged returns may still contribute when combined with other features.

### Cross-Coin Relationships

Hourly returns showed relatively strong relationships across cryptocurrencies.

Examples:

- BTC / ETH: ~0.87
- BTC / SOL: ~0.81
- ETH / SOL: ~0.82

This suggests that individual cryptocurrencies contain a substantial common market component.

Based on this observation, a market-context feature was created using the average 1-hour return of the **other cryptocurrencies**, excluding the coin currently being predicted.

The correlation analysis itself is not treated as evidence that one coin causes or necessarily predicts another. Whether market context improves future prediction will be evaluated during modeling.

---

## 8. Target Definition

For each timestamp, future returns are calculated as:

```text
future_return_h = price(t+h) / price(t) - 1
```

Targets are then assigned as:

```text
future_return > +threshold  → UP
future_return < -threshold  → DOWN
otherwise                   → STABLE
```

Future returns are used only for target construction and are never included as model features.

### Threshold Selection

Multiple thresholds were evaluated against the empirical future-return distributions.

Threshold selection considered:

1. Whether the movement is large enough to represent meaningful price movement rather than small market noise.
2. The observed distribution of returns at each prediction horizon.
3. Whether the resulting UP/DOWN/STABLE classes remain usable for classification.

The same threshold is used across all cryptocurrencies for a given horizon, while thresholds increase for longer prediction horizons because return distributions become wider over time.

Selected thresholds:

| Horizon | Threshold |
|---|---:|
| 6h | ±0.5% |
| 12h | ±1.0% |
| 24h | ±1.5% |

Resulting pooled class distributions:

| Horizon | DOWN | STABLE | UP |
|---|---:|---:|---:|
| 6h | 28.59% | 41.76% | 29.66% |
| 12h | 24.13% | 51.78% | 24.09% |
| 24h | 24.37% | 51.11% | 24.53% |

Class distributions were also checked separately for each cryptocurrency because different volatility levels cause the same threshold to behave differently across assets.

Coin-specific thresholds may be investigated later as a sensitivity experiment, but the initial models will use consistent thresholds across coins.

---

## 9. Feature Engineering

Feature engineering was performed using only information available at or before prediction time.

The analysis DataFrame contains additional raw, intermediate, diagnostic, target, and experimental columns, but these are not all model inputs.

After feature evaluation, **11 numerical features** were selected for the initial modeling experiments.

### Momentum

```text
return_1h
return_6h
return_12h
return_24h
```

These represent price movement over different historical windows.

### Volatility

```text
volatility_24h
volatility_72h
```

These capture shorter and longer market volatility regimes.

### Trading Activity

```text
log_volume_change_1h
log_volume_change_6h
log_volume_change_12h
log_volume_change_24h
```

Raw volume percentage changes were initially investigated but produced extreme ratios because trading volume can change dramatically between observations.

Log-volume changes were therefore selected as a more stable representation.

### Market Context

```text
other_coins_return_1h
```

This represents the average 1-hour return of the other cryptocurrencies at the same timestamp.

It was introduced after EDA showed strong cross-coin relationships.

---

## 10. Features Evaluated but Rejected

### Raw Volume Changes

Raw volume ratios contained severe outliers and highly skewed values.

They were replaced with log-volume changes.

### Market-Cap Changes

Market-cap change features were also investigated.

Their correlations with the corresponding price-return features were:

| Horizon | Correlation |
|---|---:|
| 1h | 0.999695 |
| 6h | 0.999474 |
| 12h | 0.999263 |
| 24h | 0.998861 |

Because they contain almost the same information as price returns, they were excluded from the initial feature set to avoid unnecessary redundancy.

---

## 11. Final Feature Validation

The final numerical feature set contains **11 features**.

After accounting for historical warm-up periods required by features such as 72-hour rolling volatility:

- Original rows: **11,700**
- Rows with complete feature history: **11,340**
- Rows removed because of feature warm-up: **360**
- Approximately **96.9%** of observations remain usable before horizon-specific target filtering.

No infinite values were detected in the final feature set.

An explicit leakage check was also implemented to ensure that the following cannot accidentally enter the model feature matrix:

```text
future_return_6h
future_return_12h
future_return_24h
target_6h
target_12h
target_24h
```

Current leakage validation passes successfully.

---

## 12. Current Limitations

The main limitations identified so far are:

- The dataset currently contains roughly three months of hourly history, so the initial models will not represent every possible crypto market regime.
- Target class distributions differ between cryptocurrencies because their volatility levels differ.
- Consecutive hourly targets overlap, especially for longer horizons such as 24h.
- Some engineered features are correlated and their actual incremental predictive value still needs to be validated.

These limitations will be considered during model construction and evaluation.

---

## 13. Next Steps

The next phase is **modeling dataset construction and temporal evaluation**.

Planned steps:

1. Construct separate modeling targets for 6h, 12h and 24h.
2. Remove rows without sufficient feature history or known future labels.
3. Preserve coin identity as categorical information.
4. Split data chronologically into train, validation and test periods.
5. Prevent target windows from crossing dataset split boundaries.
6. Establish simple majority-class / naïve baselines.
7. Train initial classification models.
8. Compare performance across horizons and cryptocurrencies.

Initial models planned for comparison:

- Logistic Regression
- Random Forest
- XGBoost

The final model will **not** be selected in advance.

Evaluation will consider:

- Accuracy
- Macro F1
- Precision
- Recall
- Per-class performance
- Confusion matrix
- Performance by cryptocurrency
- Performance by prediction horizon
- Train vs validation/test performance

The main objective of the modeling phase is to determine whether the engineered signals provide meaningful out-of-sample predictive performance beyond simple baseline strategies.

---

## 14. Current Status

```text
[✓] CoinGecko API integration
[✓] PostgreSQL database
[✓] Historical backfill
[✓] Automated ingestion pipeline
[✓] Data cleaning and validation
[✓] Historical gap detection and repair
[✓] Exploratory Data Analysis
[✓] Target definition
[✓] Threshold analysis
[✓] Feature engineering
[✓] Feature validation
[✓] Leakage validation

[✓] Chronological splitting 
[✓] Horizon-specific temporal purge 
[✓] Majority baseline 
[✓] Logistic Regression
 [✓] Random Forest 
 [✓] XGBoost 
 [✓] Model comparison 
 [✓] Final model selection 
 [✓] Final Test evaluation


```
