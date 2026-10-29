# SmartEnergy AI

## Overview

SmartEnergy AI is a machine-learning research project for hourly household electricity consumption forecasting and potential anomaly detection. It uses a Random Forest regression model and an interactive Streamlit dashboard.

**Important:** Potential anomalies are unusual prediction errors, not confirmed electrical faults.

## Objectives

- Analyze household electricity consumption.
- Forecast hourly active power consumption.
- Compare predictions with a persistence baseline.
- Identify potentially unusual observations.

## Dataset

The project uses the UCI Individual Household Electric Power Consumption dataset.

Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The target variable is `mean_power_kw`, measured in kW.



## Methodology

### Feature Engineering
The model uses 11 time-based and historical consumption features, including lags of 1, 24, and 168 hours and rolling means over 24 and 168 hours.

### Machine-Learning Model
A Random Forest regressor estimates hourly active power consumption and is compared with a persistence baseline.

### Anomaly Detection
The threshold is calibrated using the 95th percentile of validation absolute prediction errors. Flagged observations require further investigation.

## Evaluation Results

| Metric | Random Forest | Persistence Baseline |
|---|---:|---:|
| MAE | 0.3283 kW | 0.3914 kW |
| RMSE | 0.4740 kW | 0.5913 kW |
| R² | 0.5696 | 0.3301 |

On the evaluated test split, the Random Forest achieved approximately 16.13% lower MAE and 19.84% lower RMSE than the baseline.

## Anomaly Results

- Test observations: 3,472
- Potential anomalies: 100 (approximately 2.88%)
- Higher-than-predicted observations: 72
- Lower-than-predicted observations: 28
- Consecutive anomaly groups: 89
- Groups with at least two observations: 10
- Longest consecutive group: 3 hours

Anomaly flags are statistical signals, not confirmed electrical faults.

## Project Structure

```text
rebuild/
├── app.py
├── README.md
├── data/
├── recovery/
│   ├── models/
│   └── results/
└── logs/
```

## Requirements

Main dependencies: Python, pandas, joblib, matplotlib, scikit-learn, and Streamlit.

Install them using:
```bash
pip install pandas joblib matplotlib scikit-learn streamlit
```

## Run the Dashboard

From the directory containing `app.py`, run:
```bash
streamlit run app.py
```
Ensure the model, dataset, and anomaly result files are available at the paths configured in the application.

## Limitations

- The dataset represents a particular household.
- Results may not generalize to other households.
- Prediction errors may increase during unusual periods.
- Anomaly flags do not prove equipment failure.
- The threshold requires further validation.
- This is a research demonstration, not a certified safety system.

## Future Improvements

- Compare additional forecasting models.
- Evaluate performance across seasons.
- Add automated tests and reproducible dependencies.
- Validate anomalies with expert-reviewed events.
- Deploy the dashboard securely.

## Responsible Use

Protect household consumption data when sharing this project. Investigate anomaly flags alongside other evidence before making operational decisions.

## Author

Developed as an AI and Data Science portfolio research project.

## Reproducing the Dashboard

Install the dependencies listed in `requirements.txt`:

    pip install -r requirements.txt

Start the dashboard from the project directory:

    streamlit run app.py

## Required Project Artifacts

Large datasets, the trained model, and generated results are
excluded from GitHub by the `.gitignore` rules.
Cloning the repository will not download these files.

The dashboard requires the feature dataset, trained Random Forest
model, test anomaly results, and consecutive anomaly-group results.
Obtain them from an authorized backup or regenerate them before
running the dashboard.

## Reproducibility Note

Results may differ if the data, preprocessing, package versions,
or model configuration changes.
