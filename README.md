# EV_Charging_Scheduling

## 🔬 Research & Publications

### An Efficient Electric Vehicle Charging Scheduling Based on Deep Reinforcement Learning

**Research Area:** AI | Deep Reinforcement Learning | Smart Energy Management | Electric Vehicles

**Framework:**

This research proposes a day-ahead electric vehicle (EV) charging scheduling framework based on Long Short-Term Memory (LSTM) networks and Deep Q-Networks (DQN). The framework forecasts future electricity prices and energy demand over a 24-hour horizon and optimizes EV charging and discharging decisions based on predicted operating conditions, user energy requirements, and system constraints.

**Proposed Methodology:**

1. **LSTM-Based Forecasting:** LSTM networks are used to forecast electricity prices and energy-related variables over a 24-hour horizon.
2. **Deep Q-Network (DQN):** The DQN agent determines optimal EV charging and discharging decisions based on predicted electricity prices, energy demand, battery conditions, and user requirements.

**Key Contributions:**

- Integration of LSTM forecasting and DQN-based EV charging scheduling.
- Development of a multi-component reward function to improve scheduling decisions.
- Consideration of battery degradation costs in the scheduling framework.
- Modeling battery swapping through a zero-power action.
- Application of training stabilization strategies to improve DQN learning performance.
- Utilization of real-world electricity price and energy-related data obtained from CAISO.

**Experimental Results:**

The proposed method achieves a **34%–42% reduction in EV charging costs** compared with the evaluated state-of-the-art methods.

**Research Technologies:**

Python, PyTorch, Deep Learning, LSTM, DQN, Deep Reinforcement Learning, Markov Decision Process (MDP), Time-Series Forecasting, EV Charging Optimization, Battery Degradation, and Smart Energy Management.

**Research Scope:**

The study focuses on day-ahead EV charging and discharging scheduling from the user's perspective, considering electricity costs and battery energy requirements. Although Vehicle-to-Grid (V2G) is discussed, its actual implementation, aggregator participation, and detailed distribution-grid analysis are outside the scope of this research.

---

## 🛠️ Implementation & Python Libraries

The project is implemented in Python using the following libraries:

| Library | Purpose |
|---|---|
| PyTorch | Implementation and training of LSTM and DQN neural networks |
| NumPy | Numerical computations and array operations |
| Pandas | Data loading, processing, and manipulation |
| Scikit-learn | Data preprocessing and evaluation metrics |
| Gym | Reinforcement learning environment development |
| Matplotlib | Data visualization and plotting |
| Seaborn | Statistical data visualization |
| Requests | HTTP requests and data access |
| TensorBoard | Monitoring and visualization of training performance |

**Python Standard Libraries:**

| Library | Purpose |
|---|---|
| time | Time-related operations |
| math | Mathematical operations |
| random | Random number generation |
| argparse | Command-line argument parsing |
| datetime | Date and time operations |
| os | Operating system and file path operations |

---

## 📂 Project Structure & Code Description

The project consists of several Python modules organized according to their main functionalities.

### 1. LSTM Forecasting Models

The following files implement LSTM-based forecasting models for electricity prices and energy-related variables under different case studies:

- `LSTM_Price_Case1.py`
- `LSTM_Enery_case1.py`
- `LSTM_Price_Case2.py`
- `LSTM_Enery_case2.py`

These modules are responsible for training and evaluating LSTM forecasting models and generating predictions used by the EV charging scheduling framework.

### 2. LSTM Model Files

- `LSTM/`

This folder contains the pre-trained PyTorch model files (`.pt`) used by the LSTM forecasting modules. These files store trained model parameters and are required for loading the corresponding models during prediction and scheduling operations.

### 3. Data Preparation and Environment

- `Demand_Forecasting.py`
- `Prediction_Price.py`
- `Prediction_Energy.py`
- `DQN_Parse_Parameter.py`

These modules are responsible for data preparation, demand forecasting, processing predicted electricity prices and energy values, and configuring the parameters required by the DQN scheduling framework.

### 4. DQN Scheduling

- `DQN_Main.py`

This is the main implementation of the Deep Q-Network algorithm. It manages the EV charging and discharging scheduling process using the forecasting results, reward function, and defined system constraints.

### 5. Testing and Prediction Evaluation

- `Prediction_Price_Test.py`

This module is used to test and evaluate electricity price prediction performance.

### 6. Evaluation Metrics

The project uses several evaluation metrics to assess forecasting accuracy and reinforcement learning performance.

**Forecasting Metrics:**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)
- Safe MAPE

**DQN Performance Indicators:**

- Cumulative reward
- Cumulative EV charging costs
- Training losses

---

## ⚙️ Installation

**Requirements:**

- Python 3.x
- pip

Install the required libraries using the following command:

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
