# ATS Security Framework (ATS-SF)


This repository contains the implementation of the paper "The Ephemeral Threat: Assessing the Security of Algorithmic Trading Systems powered by Deep Learning," which has been submitted for review. Our research investigates the vulnerability of algorithmic trading systems (ATS) to adversarial attacks, a critical assessment as financial markets increasingly integrate machine learning models.


We use publicly available stock data from Yahoo Finance to construct and evaluate the ATS. Our system is built upon the moving average crossover technique, which compares the predicted value (of the following day) with the previous predictions to calculate the short-term (5 days) and long-term (20 days) moving averages, generating buy/sell signals based on their crossover to identify changes in stock price trends. The portfolio comprises 38 stocks, representing a diverse cross-section of the market.

Below is the table listing the stocks included in our portfolio:
| Industry            | Company                              |
|---------------------|--------------------------------------|
| Technology          | GOOGL, AMZN, AAPL                    |
| Healthcare          | JNJ, PFE, MRK, ABBV                  |
| Consumer Goods      | PEP, PG, KO, HSY                     |
| Retail              | WMT                                  |
| Financial Services  | JPM, BAC, GS, V                      |
| Energy              | XOM, CVX, COP, BP                    |
| Aerospace           | BA                                   |
| Industrial          | MMM, HON, GE                         |
| Telecommunications  | T, VZ, TMUS                          |
| Utilities           | DUK, SO, EXC, AEP                    |
| Real Estate         | AMT, PLD, SPG                        |
| Mining              | BHP, RIO, VALE, FCX                  |


## **Organization**

This repository contains resources for carrying out the technical experiments discussed in the associated paper. The main components are:

### Directories

- `notebooks/`: Contains Jupyter notebooks for demonstrations and analysis.
  - `ats.ipynb`: Main notebook for running the ATS.
  - `indiscriminate_attacks.ipynb`: Notebook for running indiscriminate attacks.
  - `model_analysis.ipynb`: Notebook for analyzing the trained models.
  - `model_training.ipynb`: Notebook for training the models.
  - `targeted_attacks.ipynb`: Notebook for running targeted attacks.
  - `utils.py`: Utility functions used across notebooks.

  
- `src/`: Source code for the project.
  - `data/`: Code related to data handling and processing.
    - `load_data.py`: Functions for loading and preprocessing data.
  - `models/`: Code related to model training and evaluation.
    - `model_training.py`: Model architecture and training functions.
    - `model_evaluation.py`: Functions for making predictions and evaluating models.
  - `trade/`: Trading strategy and backtesting code.
    - `trading_strategy.py`: Trading strategy implementation.
    - `backtest.py`: Backtesting functions.
  - `utils.py`: Utility functions used across the project.
  
- `data/`: Contains sample datasets.
  - `stock_data/`: Directory for storing stock data CSV files.

### Contents

The contents are structured to follow a logical flow from data preparation to model training, prediction, trade simulation and finally, attack simulations.

#### Data Preprocessing
- `src/data/load_data.py`: Script to load the stock data, and data pre-processing.

#### Model Definition
- `src/models/model_training.py`: Defines the LSTM model architecture used for trading predictions.

#### Model Training
- `notebooks/model_training.ipynb`: Contains the training process for the LSTM model.

#### Trading Signals Generation
- `src/trade/trading_strategy.py`: Generates trading signals based on the model's predictions.

#### Trading Simulation
- `src/trade/backtest.py`: Simulates trading sessions using the generated signals.

.
#### Ephemeral Adversarial Perturbations
- `src/attack/ephemeral_attack.py`: Generates ephemeral adversarial perturbations to attack the trading models.
- `src/attack/perform_ephemeral_attack.py`: Defines the adversarial attack model for simulations.

### Results
The results directory stores the output from the experiments:
- `notebooks/results/`: Contains plots and supplementary materials generated from the experiments.
  - `baseline/`: Holds plots related to the system's baseline performance.
  - `attack_plots/`: Contains plots as outcomes of the Google ML model attack with Ephemeral Perturbations (EP).
=======
- `notebooks/`: Contains plots and supplementary materials generated from the experiments.
  - `results/`: Conrains csv and files from analysing attack impacts on all test data
  - `plots/`: Contains pots generated from the experiments
     - `baseline/`: Holds plots related to the system's baseline performance.
     - `attack_plots/`: Contains plots as outcomes of the Google ML model attack with Ephemeral Perturbations (EP).

### Instructions

To replicate our experiments follow these steps:

1. Clone the repository.
2. Navigate to the `notebooks/` directory.
3. Install dependencies: `pip install -r requirements.txt`.
4. Execute the Jupyter notebooks as needed for various stages of the workflow.
=======

The notebooks are designed to be self-contained, handling all aspects of the workflow including preprocessing data, training the LSTM model, generating and executing trading and attack strategies. Please refer to the inline comments within the notebooks for more detailed information on the execution flow and parameter configurations.

This project is designed to be modular and extensible, allowing researchers and developers to easily add new models, parameters, or trading strategies.



The notebooks are designed to be self-contained, handling all aspects of the workflow including preprocessing data, training the LSTM model, generating and executing trading and attack strategies. Please refer to the inline comments within the notebooks for more detailed information on the execution flow and parameter configurations.
