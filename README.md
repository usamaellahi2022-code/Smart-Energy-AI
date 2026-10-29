# SmartEnergy AI

## Overview

SmartEnergy AI is a machine-learning research and portfolio project for hourly household electricity consumption forecasting and potential anomaly detection. It uses a Random Forest regression model and an interactive Streamlit dashboard.

**Important:** Potential anomalies are unusual prediction errors, not confirmed electrical faults.

## Objectives

- Analyze household electricity consumption.
- Forecast hourly active power consumption.
- Compare model predictions with a persistence baseline.
- Identify potentially unusual observations.
- Present results through an interactive dashboard.

## Dataset

This project uses the UCI Individual Household Electric Power Consumption dataset.

Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The prediction target is `mean_power_kw`, measured in kilowatts (kW). The dataset represents one household, so results may not generalize to other households.

## Methodology

### Feature Engineering

The model uses 11 time-based and historical consumption features:

- `hour`, `day_of_week`, `day`, `month`, `year`
- `is_weekend`
- `lag_1h`, `lag_24h`, `lag_168h`
- `rolling_mean_24h`, `rolling_mean_168h`

### Forecasting Model

A Random Forest Regressor with 200 trees was selected based on validation performance. Its results were compared with a persistence baseline.

### Anomaly Detection

The anomaly threshold was calibrated using the 95th percentile of absolute prediction errors on the validation data. Observations exceeding the threshold are flagged for further investigation; these flags do not establish equipment failure.

## Evaluation Results

Final evaluation on the held-out test split:

| Metric | Random Forest | Persistence baseline |
|---|---:|---:|
| MAE | 0.3283 kW | 0.3914 kW |
| RMSE | 0.4740 kW | 0.5913 kW |
| R² | 0.5696 | 0.3301 |

Compared with the persistence baseline, the Random Forest achieved approximately 16.13% lower MAE and 19.84% lower RMSE on this test split.

The held-out test set was used for final evaluation, not model selection.

## Anomaly Results

On the test split:

- Test observations: 3,472
- Potential anomalies: 100 (approximately 2.88%)
- Higher-than-predicted observations: 72
- Lower-than-predicted observations: 28
- Consecutive anomaly groups: 89
- Groups with at least two observations: 10
- Longest consecutive group: 3 hours

These are statistical signals based on prediction errors, not confirmed electrical faults.

## Dashboard Features

The Streamlit dashboard includes:

1. Energy overview
2. Historical electricity consumption
3. Saved model estimates
4. Potential anomaly explorer
5. Consecutive anomaly groups

## Project Structure

```text
rebuild/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── recovery/
│   ├── models/
│   └── results/
└── logs/
```

Large datasets, the trained model binary, and generated results are excluded from GitHub to keep the repository lightweight.

## Installation

Use Python 3.10 or a compatible version, then install the dependencies from the project directory:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

From the directory containing `app.py`, run:

```bash
streamlit run app.py
```

A fresh clone does not include the required data and model artifacts. The dashboard requires the processed feature dataset, trained Random Forest model, test anomaly results, and consecutive anomaly-group results. Obtain these from an authorized backup or regenerate them using the project workflow before running the dashboard.

## Limitations and Responsible Use

- The dataset represents a particular household.
- Results may not generalize to other households or environments.
- Forecast errors may increase during unusual periods.
- Anomaly flags do not prove equipment failure.
- The threshold requires further validation.
- This is a research demonstration, not a certified electrical safety system.
- Protect household consumption data when sharing or deploying the project.

Investigate anomaly flags alongside other evidence before making operational decisions.

## Future Improvements

- Compare additional forecasting models.
- Evaluate performance across seasons.
- Add automated tests and strengthen reproducibility.
- Validate anomalies using expert-reviewed events.
- Deploy the dashboard securely.

## Author

Developed as an AI and Data Science portfolio research project.
