crypto-trend-prediction

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

The project now includes the complete modeling, production prediction, API/dashboard, LLM explanation, automated ingestion, and cross-platform bootstrap workflow.

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

future_return > +threshold  → UP

future_return < -threshold  → DOWN

otherwise                   → STABLE

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

13. Modeling and Final Evaluation

The modeling workflow is complete for all three prediction horizons. The project uses chronological splitting and a horizon-specific temporal purge so future target windows do not leak across dataset boundaries.

Models evaluated:

Majority-class baseline

Logistic Regression

Random Forest

XGBoost

Balanced Logistic Regression was selected as the production model for the 6h, 12h, and 24h horizons.

Final held-out test Macro-F1:

Horizon

Final Model Macro-F1

Majority Baseline Macro-F1

6h

0.363911

0.200832

12h

0.344609

0.226631

24h

0.325128

0.212721

The final test set is reserved for final evaluation and is not used for model tuning.

14. Production Prediction Pipeline

Production models are stored under artifacts/models/ for the 6h, 12h, and 24h horizons.

The prediction service:

Loads recent hourly PostgreSQL market history.

Rebuilds the same 11 features used during training.

Loads the appropriate horizon-specific production model.

Predicts UP, DOWN, or STABLE.

Returns class probabilities and model confidence.

Can persist predictions to PostgreSQL.

The daily market pipeline performs ingestion, cleaning, hourly upserts, gap repair, and then generates 15 predictions: 5 assets × 3 horizons. Production models are not retrained during the daily pipeline.

## 15. API, Dashboard, and LLM Layer

The FastAPI application exposes endpoints for market data, predictions, daily ingestion, on-demand prediction, chatbot interaction, explanation, and health checks.

The application now has two main user-facing pages:

- `/` - AlphaPulse AI Chatbot
- `/dashboard` - Original analytics and prediction dashboard

The production ML models remain responsible for generating the actual cryptocurrency trend predictions. The LLM layer does not replace the prediction models and does not generate market prices or prediction probabilities.

Important endpoints include:

```text
/api/market-data
/api/predictions
/api/daily-ingestion
/api/predict
/api/chat
/api/explain
/api/health
```

---

## 16. AlphaPulse AI Chatbot

AlphaPulse now includes an AI-powered conversational interface that acts as the primary interaction layer between the user, the production prediction models, and verified cryptocurrency market data.

The chatbot is available on the main application page:

```text
/
```

The original analytics dashboard is preserved as a second page:

```text
/dashboard
```

### Chatbot Architecture

The chatbot uses Groq to understand natural-language requests and extract structured intent and parameters.

Prediction execution, market-data retrieval, calculations, validation, and final grounding remain controlled by the AlphaPulse backend.

```text
User Prompt
    |
    v
AlphaPulse Chat UI
    |
    v
Groq Intent + Parameter Extraction
    |
    v
Deterministic Execution Planner
    |
    +---------------------------+
    |                           |
    v                           v
Prediction Pipeline       PostgreSQL Market Data
    |                           |
    +-------------+-------------+
                  |
                  v
        Grounded Response Builder
                  |
                  v
                User
```

The LLM is therefore used primarily as a natural-language interpretation layer.

It does not generate the underlying ML prediction.

### Supported Chatbot Capabilities

The chatbot currently supports:

- Cryptocurrency trend prediction
- Current market-price queries
- Market analysis
- Multi-coin prediction comparisons
- Investment scenarios
- Model-based investment-signal comparisons
- AlphaPulse capability questions
- Detection and rejection of irrelevant requests

Supported cryptocurrencies:

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- XRP
- Cardano (ADA)

Supported prediction horizons:

- 6 hours
- 12 hours
- 24 hours

### Intent and Parameter Extraction

The chatbot converts natural-language requests into structured parameters that can be executed by the backend.

Example:

```text
User:
"What is the BTC prediction for the next 24 hours?"

Extracted parameters:
intent = prediction
coin = BTC
horizon = 24h
```

Investment amounts can also be extracted.

Example:

```text
User:
"If I invest $5000 in BTC, what could happen in the next 24 hours?"

Extracted parameters:
intent = investment_projection
coin = BTC
horizon = 24h
investment_amount = 5000
currency = USD
```

The extracted parameters are validated before any model or database operation is executed.

### Complex Queries and Multiple Prediction Calls

The chatbot can decompose complex requests into multiple production-model calls.

For example:

```text
"Compare BTC, ETH and SOL for the next 24 hours."
```

is converted into separate prediction operations:

```text
predict(BTC, 24h)
predict(ETH, 24h)
predict(SOL, 24h)
```

The resulting model outputs are then compared.

For example, AlphaPulse can identify which of the checked assets has the highest `UP` class probability.

This is presented as a comparison of model signals and not as a guaranteed investment outcome.

### Market-Data Grounding

When a request requires market information, the chatbot retrieves the latest stored AlphaPulse market data from PostgreSQL.

Grounded market information can include:

- Latest stored price
- Observation timestamp
- 24-hour price change
- 24-hour high
- 24-hour low
- Market capitalization
- Trading volume

These values are retrieved by the backend rather than generated by the LLM.

Prediction-pipeline prices and stored market-snapshot prices may have different timestamps. AlphaPulse preserves the corresponding timestamps so observations from different times are not incorrectly represented as the same value.

### Investment Scenarios

The chatbot can interpret investment-related questions such as:

```text
"If I invest $5000 in BTC, what could happen in the next 24 hours?"
```

For supported requests, AlphaPulse combines:

1. The investment amount supplied by the user.
2. Verified market information.
3. The production model prediction.
4. DOWN, STABLE, and UP class probabilities.

The current production models are directional classifiers.

They predict:

```text
UP
DOWN
STABLE
```

They do not directly predict an exact future percentage return or future cryptocurrency price.

Therefore:

```text
model confidence != expected return

class probability != percentage profit
```

AlphaPulse does not convert model confidence or class probabilities into fabricated financial returns.

### Unsupported Prediction Horizons

The current trained production models support only:

```text
6h
12h
24h
```

If a user requests an unsupported horizon, the chatbot detects and rejects the request.

Example:

```text
"If I invest $5000 in BTC today, what will my gain be in the next 10 days?"
```

AlphaPulse extracts:

```text
coin = BTC
investment_amount = 5000
requested_horizon = 10 days
```

but does not execute a prediction because a 10-day production model does not exist.

The system returns an unsupported-horizon response instead.

AlphaPulse does not chain multiple 24-hour predictions together to simulate unsupported longer horizons.

### Irrelevant Query Filtering

The chatbot is intentionally restricted to the AlphaPulse cryptocurrency prediction and market-analysis domain.

For example:

```text
"How do I write a for loop in Python?"
```

is classified as an irrelevant request.

The chatbot responds that it can assist with AlphaPulse cryptocurrency predictions, market analysis, comparisons, market information, and supported investment scenarios rather than answering the unrelated question.

### Hallucination Controls

A major design requirement of the chatbot is preventing unsupported financial or market claims.

The following grounding rules are applied:

- Market prices must come from AlphaPulse market data.
- Prediction classes must come from the production ML pipeline.
- Prediction probabilities must come from the production ML pipeline.
- Groq cannot generate prediction percentages.
- Model confidence is not treated as an expected financial return.
- Class probabilities are not treated as percentage profits.
- Unsupported prediction horizons are rejected.
- 24-hour forecasts are not chained to simulate longer unsupported horizons.
- Exact future prices are not claimed when the production model does not generate them.
- Exact future investment profits are not claimed when the model does not predict returns.
- Investment comparisons are presented as model signals rather than guaranteed investment recommendations.

### Chat API

The chatbot is exposed through:

```text
POST /api/chat
```

Example request:

```json
{
  "message": "Compare BTC, ETH and SOL for the next 24 hours."
}
```

The request is processed through:

```text
User Message
    |
    v
Intent Parser
    |
    v
Execution Planner
    |
    v
Prediction / Market Data Executor
    |
    v
Grounded Response Builder
    |
    v
API Response
```

The API response contains the user-facing message together with structured intent information and, where applicable, prediction and market-data results.

### Production Verification

The chatbot workflow has been tested with:

- Single-asset predictions
- Current price queries
- Current market data + prediction queries
- Multi-asset prediction comparisons
- Investment scenarios
- Data-backed signal comparisons
- Unsupported prediction horizons
- Irrelevant questions
- Capability questions

Example production comparison:

```text
Compare BTC, ETH and SOL for the next 24 hours.
```

The system successfully performs three separate prediction calls and compares the resulting class probabilities.

Example combined request:

```text
What is the current BTC price and prediction for the next 24 hours?
```

The system retrieves the latest stored BTC market information and combines it with the production 24-hour model output.

The chatbot is deployed as the primary AlphaPulse interface, while the original analytics dashboard remains available at `/dashboard`.

---

## 17. Fresh Database Bootstrap

A clean installation does not require a pre-populated PostgreSQL database.

The repository includes a fixed seven-day hourly seed dataset at:

```text
data/seed/market_history_7d.csv
```

The seed contains 840 observations:

```text
5 supported cryptocurrencies x 168 hourly observations
```

`src/database/bootstrap_database.py` is idempotent:

- If the required PostgreSQL tables are missing, it creates the schema.
- If hourly market data already exists, it leaves the existing database unchanged.
- If the database is empty, it loads the bundled seven-day seed.
- It verifies that all five supported coins exist and each has enough history for prediction features.

The empty-database bootstrap has been tested successfully against an isolated PostgreSQL database:

- 840 raw rows inserted
- 840 clean hourly rows inserted
- 168 hourly observations for each supported coin

The seed is only the starting dataset for a fresh database. Normal CoinGecko ingestion continues to fetch and store newer market data afterward.

---

## 18. Environment Configuration

Each developer or deployment must configure its own database and API credentials.

A developer should not copy another developer's `DATABASE_URL`.

Create a `.env` file in the repository root and configure at least:

```env
DATABASE_URL=postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_HOST/YOUR_DATABASE?sslmode=require
COINGECKO_API_KEY=YOUR_COINGECKO_API_KEY
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-120b
```

`DATABASE_URL` must point to a PostgreSQL database that the current developer or environment is authorized to use.

The database may be empty. The bootstrap process creates the required schema and loads the bundled seven-day seed automatically when necessary.

If the database already contains AlphaPulse market history, the seed step is skipped and existing data is preserved.

Never commit `.env`, database credentials, or API keys to Git.

---

## 19. Quick Start

The operating-system launchers perform local environment setup and then call the shared Python bootstrap in `src/bootstrap.py`.

The shared bootstrap logic:

1. Validates environment variables.
2. Tests PostgreSQL connectivity.
3. Performs the database bootstrap when necessary.
4. Verifies production model artifacts.

### Windows

From PowerShell in the project root:

```powershell
.\quick_start.ps1
```

The launcher creates `.venv` when needed, installs `requirements.txt`, runs the shared bootstrap, and starts the configured application workflow.

For direct local API development, activate the environment and run the FastAPI application according to the project configuration.

### macOS

After cloning the repository, first create and configure `.env` with your own PostgreSQL `DATABASE_URL` and required API keys.

Then from Terminal in the project root:

```bash
chmod +x quick_start.sh
./quick_start.sh
```

The macOS launcher creates `.venv` when needed, installs dependencies, and runs the same shared bootstrap as Windows.

Shell files are stored with LF line endings through `.gitattributes`; PowerShell files use CRLF.

The Windows launcher and shared bootstrap have been executed successfully.

The macOS launcher is prepared for cross-platform use and should receive a final execution test on an actual macOS machine.

---

## 20. Database URL and New-Machine Onboarding

The bootstrap does not provide or share a PostgreSQL connection string.

On a new machine, the developer must first configure a valid `DATABASE_URL` in `.env`.

The intended onboarding sequence is:

```text
Clone repository
      |
      v
Create .env
      |
      v
Set developer's own DATABASE_URL + API keys
      |
      v
Run quick_start.ps1 or quick_start.sh
      |
      v
Create/install virtual environment dependencies
      |
      v
Shared bootstrap tests PostgreSQL connection
      |
      v
Existing DB?
   |       |
  Yes      No
   |       |
   |       v
   |    Create schema
   |       |
   |       v
   |    Load fixed 7-day seed
   |    (840 rows)
   |       |
   +-------+
      |
      v
Normal CoinGecko ingestion continues
with fresh market data
```

This separation is intentional: credentials remain private to each environment, while the seed dataset and bootstrap behavior are reproducible from the repository.

---

## 21. Project Status

AlphaPulse has completed the full development workflow from data ingestion and validation through machine-learning modeling, production prediction, API integration, dashboard development, and the AI chatbot layer.

### Stage 1 - Data Collection & Storage

- [x] CoinGecko API integration
- [x] PostgreSQL / Neon database integration
- [x] Historical market-data backfill
- [x] Automated daily ingestion
- [x] Database upsert and duplicate protection

### Stage 2 - Data Quality & EDA

- [x] Data cleaning and normalization
- [x] Data-quality validation
- [x] Historical gap detection
- [x] Historical gap repair
- [x] Exploratory Data Analysis
- [x] Return and volatility analysis
- [x] Cross-coin correlation analysis

### Stage 3 - Target & Feature Engineering

- [x] 6h, 12h and 24h target definition
- [x] Threshold analysis and selection
- [x] Momentum features
- [x] Volatility features
- [x] Trading-volume features
- [x] Cross-coin market-context features
- [x] Feature validation
- [x] Data-leakage protection

### Stage 4 - Model Development

- [x] Chronological train/validation/test splitting
- [x] Horizon-specific temporal purge
- [x] Majority-class baseline
- [x] Logistic Regression
- [x] Random Forest
- [x] XGBoost
- [x] Model comparison
- [x] Final production-model selection
- [x] Final held-out test evaluation

### Stage 5 - Production Prediction Pipeline

- [x] Production models for 6h, 12h and 24h
- [x] Production feature reconstruction
- [x] UP / DOWN / STABLE predictions
- [x] Class probabilities and confidence
- [x] Prediction persistence
- [x] Automated predictions for all five supported assets

### Stage 6 - API & Dashboard

- [x] FastAPI backend
- [x] Market-data API
- [x] Prediction API
- [x] On-demand prediction endpoint
- [x] Daily-ingestion endpoint
- [x] Health endpoint
- [x] Analytics dashboard
- [x] Production deployment

### Stage 7 - AlphaPulse AI Chatbot

- [x] Groq LLM integration
- [x] Natural-language intent detection
- [x] Coin, horizon and investment-amount extraction
- [x] Deterministic execution planner
- [x] Single-coin prediction queries
- [x] Multi-coin prediction comparisons
- [x] Current market-data queries
- [x] Investment-scenario handling
- [x] Data-backed model-signal comparisons
- [x] Unsupported-horizon protection
- [x] Irrelevant-query filtering
- [x] Hallucination safeguards
- [x] `/api/chat` endpoint
- [x] Chatbot as the primary application page
- [x] Existing dashboard preserved at `/dashboard`
- [x] Local chatbot verification
- [x] Production chatbot verification

### Stage 8 - Reproducibility & Onboarding

- [x] Fixed 7-day bootstrap seed
- [x] Fresh-database schema creation
- [x] Empty-database bootstrap
- [x] Existing-database safe-skip
- [x] Shared Python bootstrap
- [x] Windows quick-start launcher
- [x] Windows end-to-end quick-start test
- [x] macOS quick-start launcher
- [x] Git cross-platform line-ending configuration
- [ ] Final `quick_start.sh` execution test on a real macOS machine

### Remaining Work

The complete AlphaPulse application is implemented and deployed.

The only remaining environment-validation task is to execute the macOS quick-start workflow on a real Mac and confirm the complete setup process end-to-end.