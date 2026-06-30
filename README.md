# 🏡 California Housing Price Predictor

## 📖 Overview
Predicts median house prices in California using the 1990 Census dataset. Built from scratch using **NumPy**.

## 🎯 What This Project Shows
- NumPy mastery (arrays, broadcasting, linear algebra)
- Linear regression from scratch (Normal Equation)
- Data cleaning and feature engineering
- Professional GitHub structure

## 📊 Dataset
- Source: California Housing (1990 Census)
- 20,640 samples, 8 features
- Target: Median house value (in $100,000s)

## 🛠️ Tech Stack
- Python 3.8+
- NumPy (ALL computations)
- Pandas (data loading only)
- Matplotlib (visualizations)

## 📈 Results
| Metric | Value |
|:---|:---|
| **RMSE** | **$75,558** |
| MSE | 0.571 |
| Best Feature | MedInc (Median Income) |

### 🔍 Key Insights
- **Median Income** is the strongest predictor of house prices
- **Location matters** - Latitude and Longitude have significant negative weights
- **Rooms per household** positively correlates with higher prices
- **Average rooms** negatively correlates - more rooms = denser/cheaper areas

## 🚀 How to Run
```bash
git clone https://github.com/SharmanKushwaha/California_House_Prediction/blob/main/main.py
cd california-housing-prediction
pip install -r requirements.txt
python main.py
