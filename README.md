# EV_Charging_Scheduling

## 🔬 Research & Publications

### ⚡ An Efficient Electric Vehicle Charging Scheduling Based on Deep Reinforcement Learning

**Research Area:** Artificial Intelligence | Deep Reinforcement Learning | Smart Energy Management | Electric Vehicles

This research focuses on developing an intelligent, data-driven framework for day-ahead Electric Vehicle (EV) charging scheduling using **Long Short-Term Memory (LSTM)** and **Deep Q-Network (DQN)**.

The proposed framework integrates time-series forecasting with deep reinforcement learning to optimize EV charging and discharging decisions under fluctuating electricity prices and varying energy demand.

### 🧠 Proposed Methodology

The proposed framework consists of two main neural network components:

**1. LSTM-Based Forecasting Network**

LSTM networks are used to extract temporal patterns from historical electricity price and energy consumption data. These models forecast future operating conditions over a 24-hour horizon, providing predictive information for the EV charging scheduling framework.

**2. Deep Q-Network (DQN)**

A DQN agent learns an optimal action-value function to determine EV charging and discharging decisions based on predicted operating conditions, user energy requirements, and system constraints.

By combining forecasting and reinforcement learning, the framework aims to make intelligent charging decisions while reducing electricity costs and considering battery-related constraints.

### ⚙️ Key Research Contributions

- Integration of LSTM-based forecasting with DQN for intelligent EV charging scheduling.
- Development of a multi-component reward function for multi-objective optimization.
- Consideration of battery degradation costs in the scheduling framework.
- Modeling battery swapping through a zero-power action.
- Application of training stabilization strategies to improve learning performance and reward convergence.
- Evaluation using real-world electricity price and energy data from the California Independent System Operator (CAISO).

### 📊 Experimental Results

The proposed DRL-based framework achieved a **34%–42% reduction in EV charging costs** compared with the state-of-the-art methods evaluated in the study.

The results demonstrate the potential of combining predictive deep learning and reinforcement learning for cost-efficient EV charging management.

### 🛠️ Research Technologies & Concepts

- Python
- Deep Learning
- Long Short-Term Memory (LSTM)
- Deep Q-Network (DQN)
- Deep Reinforcement Learning (DRL)
- Markov Decision Process (MDP)
- Time-Series Forecasting
- Electric Vehicle Charging Optimization
- Battery Degradation Modeling
- Smart Energy Management

### 📌 Research Scope

This work focuses on day-ahead EV charging and discharging scheduling from the user's perspective, with emphasis on electricity cost and battery energy requirements.

Although Vehicle-to-Grid (V2G) technology is discussed as an important related concept, its actual implementation, aggregator participation, and detailed distribution-grid analysis are outside the scope of this research.

---

## 🛠️ Implementation & Python Libraries

The project is implemented in Python and uses the following libraries for deep learning, data processing, reinforcement learning, and visualization.

| Library | Purpose |
|---|---|
| **PyTorch** | Neural network implementation and training, including LSTM and DQN models |
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data processing, manipulation, and analysis |
| **Scikit-learn** | Data preprocessing, normalization, and model evaluation |
| **Gym** | Reinforcement learning environment and action-space definition |
| **Matplotlib** | Data visualization and plotting |
| **Seaborn** | Statistical data visualization |
| **Requests** | HTTP requests and data retrieval |
| **TensorBoard** | Logging and monitoring training information |

### 🐍 Python Standard Library

The project also uses built-in Python modules, including:

- `time`
- `math`
- `random`
- `argparse`
- `datetime`
- `os`

---

## 📂 Project Structure & Code Description

The project is organized into several Python modules covering LSTM-based forecasting, data preparation, charging environment modeling, DQN implementation, testing, and performance evaluation.

### 1. LSTM-Based Forecasting

This section contains four LSTM-based models developed to forecast electricity prices and energy consumption under two experimental cases.

| File | Description |
|---|---|
| `LSTM_Price_Case1.py` | Implements an LSTM model for electricity price forecasting using a 24-hour sliding window. The model is trained on historical data and evaluated using a separate test dataset. |
| `LSTM_Enery_case1.py` | Implements an LSTM model for energy consumption forecasting using historical energy data. Prediction performance is evaluated using regression metrics. |
| `LSTM_Price_Case2.py` | Implements the electricity price forecasting model with a separate validation dataset. Training and validation losses are monitored, and Early Stopping is applied based on validation performance. |
| `LSTM_Enery_case2.py` | Implements the energy consumption forecasting model using training, validation, and test datasets. Early Stopping is employed, and the best model parameters are restored for final evaluation. |

### LSTM Model Files

| Folder | Description |
|---|---|
| `LSTM/` | Contains the pre-trained PyTorch model files (`.pt`) used by the LSTM forecasting modules. These files store trained model parameters and are used when loading the corresponding models. |

### 2. Data Preparation and Environment Configuration

This section includes the modules responsible for preparing input data and configuring the simulation environments used by the DQN agent.

| File | Description |
|---|---|
| `Demand_Forecasting.py` | Handles the loading and preparation of electricity demand data used in the EV charging scheduling environment. |
| `Prediction_Price.py` | Implements the electricity price environment by loading LSTM-based price predictions. It updates electricity prices at each time step and calculates the corresponding reward for charging decisions. |
| `Prediction_Energy.py` | Implements the energy consumption environment using LSTM-based energy predictions. It updates energy consumption values over time and incorporates energy-related information into the reward function. |
| `DQN_Parse_Parameter.py` | Defines the configuration parameters and command-line arguments for the DQN algorithm, including learning rate, discount factor, exploration settings, replay memory capacity, battery constraints, and reward coefficients. |

### 3. Deep Q-Network (DQN) Implementation

| File | Description |
|---|---|
| `DQN_Main.py` | Implements the main DQN-based EV charging scheduling framework. It includes the Q-network and target network architectures, experience replay memory, action selection, reward processing, and the training loop. The agent interacts with the electricity price, energy consumption, and demand environments to learn charging decisions. Training results, rewards, selected actions, losses, Q-table values, and network parameters are recorded for further analysis. |

### 4. Testing and Performance Evaluation

| File | Description |
|---|---|
| `Prediction_Price_Test.py` | Implements the electricity price testing environment using the predicted price test dataset. It evaluates charging decisions under test conditions and calculates cumulative charging costs based on electricity prices and selected charging actions. |

### 5. Evaluation Metrics

The LSTM forecasting models are evaluated using the following regression metrics:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination
- **MAPE:** Mean Absolute Percentage Error
- **Safe MAPE:** A modified percentage error metric designed to reduce instability caused by very small denominator values.

The DQN-based scheduling framework is evaluated using reward values, cumulative charging costs, and training losses.

---

## 📦 Installation

### Requirements

- Python 3.x
- pip

### Install Dependencies

The required third-party libraries can be installed using:

```bash
pip install numpy pandas matplotlib seaborn torch gym scikit-learn requests tensorboard
```

**Note:** The project uses Gym for reinforcement learning environment implementation. Depending on the installed Gym version, some compatibility adjustments may be required.

---

## 📚 Data Sources

The three datasets used in this research were obtained from the **California Independent System Operator (CAISO)** through the **Open Access Same-time Information System (OASIS)**.

- **Electricity Price Data:** Day-Ahead Market (DAM) Locational Marginal Prices (LMP).
- **Energy Data:** Historical energy-related data.
- **Electricity Demand Data:** Historical demand data used in the demand forecasting process.

**Data Source:** [https://oasis.caiso.com/mrioasis/](https://oasis.caiso.com/mrioasis/)

Users should obtain the required datasets from CAISO OASIS and ensure that they are properly prepared before running the forecasting and scheduling modules.

---

## 📄 Citation

If you use this project, its implementation, or any part of its research methodology in your work, please acknowledge the original author:

**Sam Ohammadi**

GitHub Repository: [https://github.com/samohammadi/EV_Charging_Scheduling](https://github.com/samohammadi/EV_Charging_Scheduling)

Please provide appropriate credit to **Sam Ohammadi** when using, adapting, or referencing this work in academic publications, research projects, or other applications.

---

## 👨‍💻 Author

**Sam Ohammadi**

GitHub: [@samohammadi](https://github.com/samohammadi)

---

## 📜 License

Please refer to the repository's license file for terms of use and distribution.
