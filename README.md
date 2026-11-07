# EV_Charging_Scheduling

## 🔬 Research & Publications

### An Efficient Electric Vehicle Charging Scheduling Based on Deep Reinforcement Learning

**Research Area:** AI | Deep Reinforcement Learning | Smart Energy Management | Electric Vehicles

**Framework:**

This research proposes a day-ahead electric vehicle (EV) charging scheduling framework based on Long Short-Term Memory (LSTM) networks and Deep Q-Networks (DQN). The framework forecasts future electricity prices and energy demand over a 24-hour horizon and optimizes EV charging and discharging decisions under fluctuating electricity prices and energy requirements.

**Proposed Methodology:**

1. **LSTM-Based Forecasting:** Predicting electricity prices and energy consumption.
2. **Deep Q-Network (DQN):** Making optimal charging and discharging decisions based on forecasts, user requirements, and system constraints.

**Key Contributions:**

- Integration of LSTM forecasting and DQN-based scheduling.
- Development of a multi-component reward function.
- Consideration of battery degradation costs.
- Modeling battery swapping through a zero-power action.
- Application of training stabilization strategies.
- Use of real-world electricity price and energy-related data from CAISO.

**Experimental Results:**

The proposed method achieves a **34%–42% reduction in EV charging costs** compared with the evaluated state-of-the-art methods.

**Research Technologies:**

Python, Deep Learning, LSTM, DQN, Deep Reinforcement Learning, Markov Decision Process (MDP), Time-Series Forecasting, EV Charging Optimization, Battery Degradation, and Smart Energy Management.

**Research Scope:**

The study focuses on day-ahead EV charging and discharging scheduling from the user's perspective, considering electricity costs and battery energy requirements. Vehicle-to-Grid (V2G) implementation, aggregator participation, and detailed distribution-grid analysis are outside the scope of this research.

---

## 🛠️ Implementation & Python Libraries

The project is implemented in Python using the following libraries:

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Gym
- Matplotlib
- Seaborn
- Requests
- TensorBoard

**Python Standard Libraries:**

- time
- math
- random
- argparse
- datetime
- os

---

## 📂 Project Structure & Code Description

The project consists of the following main Python modules:

### LSTM Forecasting Models

- `LSTM_Price_Case1.py`
- `LSTM_Enery_case1.py`
- `LSTM_Price_Case2.py`
- `LSTM_Enery_case2.py`

These modules implement LSTM-based forecasting models for electricity prices and energy-related data under different case studies.

### LSTM Model Files

- `LSTM/`

This folder contains the pre-trained PyTorch model files (`.pt`) used by the LSTM forecasting modules for electricity price and energy prediction.

### Data Preparation and Environment

- `Demand_Forecasting.py`
- `Prediction_Price.py`
- `Prediction_Energy.py`
- `DQN_Parse_Parameter.py`

These modules handle demand forecasting, prediction processing, and parameter configuration.

### DQN Scheduling

- `DQN_Main.py`

This module implements the Deep Q-Network algorithm for EV charging and discharging scheduling.

### Testing

- `Prediction_Price_Test.py`

This module evaluates electricity price prediction performance.

### Evaluation Metrics

The project uses the following metrics and indicators:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)
- Safe MAPE
- DQN cumulative reward
- Cumulative EV charging costs
- Training losses

---

## ⚙️ Installation

**Requirements:**

- Python 3.x
- pip

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn torch gym scikit-learn requests tensorboard
```

**Note:** Depending on the Python version, some Gym versions may require compatibility adjustments.

---

## 📚 Data Sources

The three datasets used in this research were obtained from the **California Independent System Operator (CAISO)** through the **Open Access Same-time Information System (OASIS)**.

- **Electricity Price Data:** Day-Ahead Market (DAM) Locational Marginal Prices (LMP).
- **Energy Data:** Historical energy-related data.
- **Electricity Demand Data:** Historical demand data used by the `Demand_Forecasting.py` module.

**Data Source:** [https://oasis.caiso.com/mrioasis/](https://oasis.caiso.com/mrioasis/)

Users should ensure that the required datasets are available and properly prepared before running the forecasting and scheduling modules.

---

## 📄 Citation

If you use this project, its implementation, or any part of its research methodology in your work, please acknowledge the original author:

**Sam Ohammadi**

GitHub Repository: [https://github.com/samohammadi/EV_Charging_Scheduling](https://github.com/samohammadi/EV_Charging_Scheduling)

Please provide appropriate credit to **Sam Ohammadi** when using, adapting, or referencing this work in academic publications, research projects, or other applications.

---

## 👤 Author

**Sam Ohammadi**

GitHub: [https://github.com/samohammadi](https://github.com/samohammadi)

---

## 📜 License

Please refer to the repository's LICENSE file for licensing information.
