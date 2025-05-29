# TORCS AI Controller Project

**AI2002: Artificial Intelligence Final Project – Spring 2025**

## 🏎 Overview

This repository contains the implementation of an AI controller for the **Simulated Car Racing Championship (SCRC)** using the **TORCS** (The Open Racing Car Simulator) framework. Developed as part of the **AI 2002** course, this project features a **supervised neural network** to predict control actions (acceleration, braking, steering, gear) from real-time telemetry data, along with a **rule-based fallback** for robustness.

---

## 👥 Authors

* **Muhammad Zohaib Raza**
  *Led model training, implemented the neural network, and wrote the report.*
* **Muhammad Awais**
  *Collected telemetry data from high-speed laps and tested controller performance.*
* **Muhammad Haziq Naeem**
  *Recorded data from diverse tracks and assisted in data preprocessing.*

📅 **Date**: May 29, 2025

---

## 📦 Project Deliverables

* ✅ **Deliverable 1**: Telemetry processing implementation (logged in `telemetry_log.csv`)
* ✅ **Final Deliverable**: Python client + trained neural network + report (`TORCS_AI_Controller_Report.docx`)

---

## 📁 Repository Contents

| File / Folder                     | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| `pyclient.py`                     | Main client script to connect to TORCS server        |
| `carControl.py`                   | Class defining car control parameters (actuators)    |
| `carState.py`                     | Class managing car state variables (sensors)         |
| `msgParser.py`                    | UDP message parser utility                           |
| `train_model.py`                  | Neural network training script                       |
| `driver.py`                       | Main driver class (integrates NN + rule-based logic) |
| `race_controller.pth`             | Trained neural network weights                       |
| `scaler.pkl`                      | Feature scaler for input preprocessing               |
| `telemetry_log.csv`               | Collected telemetry and action data                  |
| `TORCS_AI_Controller_Report.docx` | Final report                                         |

---

## ⚙️ Prerequisites

### TORCS

* Version: **1.3.4 with SCRC patch**
* Download: [SourceForge](https://sourceforge.net/projects/torcs/) + [CIG SCRC Patch](http://scr.sandbox.googlecode.com/files/)

### Python 3.x

Install the following libraries:

```bash
pip install numpy pandas torch scikit-learn matplotlib joblib
```

### Operating System

* ✅ **Linux** (Preferred)
* ⚠️ Windows: Requires additional patching (`scr-win-patch.zip`)

### Optional

* CUDA-compatible GPU for faster training

---

## 🛠 Installation

### TORCS with SCRC Patch (Linux)

```bash
sudo apt-get install libgl1-mesa-dev freeglut3-dev libplib-dev libpng-dev zlib1g-dev libopenal-dev libalut-dev
# Unpack, patch, compile:
./configure
make
make install
make datainstall
```

Launch TORCS and verify bots like `scr_server`.

### Python Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt  # or install manually as listed above
```

---

## 🚦 Usage

### 1. Train the Model

Make sure `telemetry_log.csv` exists with enough data:

```bash
python train_model.py
```

Output:

* Trained model: `race_controller.pth`
* Loss plot: `training_loss.png`

### 2. Run the Client

Start TORCS in **Quick Race** with at least one `scr_server` bot.

```bash
python pyclient.py --host localhost --port 3001 --id CAR1
```

**Optional Flags:**

* `--maxEpisodes`: Number of races (default: 1)
* `--maxSteps`: Max steps per episode (default: 0 = unlimited)
* `--track`: Track name (default: None)
* `--stage`: 0 = Warm-Up, 1 = Qualifying, 2 = Race (default: 3)

🕹 Press **`i`** to toggle AI/manual control (requires `keyboard` module)

---

## 📈 Evaluation

* Use TORCS UI to monitor **lap time**, **position**, and **AI control**
* Check:

  * `telemetry_log.csv` – Collected race data
  * `training_loss.png` – Model loss curve

---

## 📄 Report Summary

**TORCS\_AI\_Controller\_Report.docx** includes:

* **Methodology**:

  * 43 input features
  * 4 output actions (steer, accel, brake, gear)

* **Results**:

  * Lap time: *78.5*
  * Race position: *4th*

* **Discussion**:

  * Challenges: Real-time execution, generalization
  * Future Work: Reinforcement learning, opponent prediction

---

## 🤝 Contributing

Contributions are welcome!
Fork the repo and create a pull request with your improvements:

* Add RL-based controller
* Improve model generalization
* Enhance data collection tools

---

## ⚖️ License

**MIT**

---

## 🙏 Acknowledgments

* Thanks to **TORCS** and **SCRC** communities
* Inspired by AI 2002 course material and sample AI bots

---
