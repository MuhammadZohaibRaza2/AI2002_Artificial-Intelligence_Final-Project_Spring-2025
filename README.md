# AI2002_Artificial-Intelligence_Final-Project_Spring-2025

TORCS AI Controller Project
Overview
This repository contains the implementation of an AI controller for the Simulated Car Racing Championship (SCRC) using the TORCS (The Open Racing Car Simulator) framework. The project, developed as part of the Artificial Intelligence AI 2002 course, involves designing a machine learning-based controller to race competitively on various tracks. The controller uses a supervised neural network to predict control actions (acceleration, braking, steering, and gear) based on real-time telemetry data, with a rule-based fallback for robustness.

Authors
Muhammad Zohaib Raza: Led model training, implemented the neural network, and wrote the report.
Muhammad Awais: Collected telemetry data from high-speed laps and tested controller performance.
Muhammad Haziq Naeem: Recorded data from diverse tracks and assisted in data preprocessing.
Date
May 29, 2025

Project Deliverables
Deliverable 1: Implementation of telemetry processing (completed and logged in telemetry_log.csv).
Final Deliverable: A Python client with a trained neural network controller, submitted as a self-contained archive with a two-page report.
Repository Contents
pyclient.py: Main script to connect to the TORCS server and drive the car.
carControl.py: Class defining car control parameters (actuators).
carState.py: Class managing car state variables (sensors).
msgParser.py: Utility for parsing and building UDP messages.
train_model.py: Script to train the neural network model using telemetry data.
driver.py: Core driver class integrating the neural network and rule-based control.
race_controller.pth: Trained neural network model weights.
scaler.pkl: Preprocessing scaler for input features.
telemetry_log.csv: Dataset of telemetry and control actions for training.
TORCS_AI_Controller_Report.docx (or .tex if using LaTeX): Final project report.
Prerequisites
TORCS: Version 1.3.4 with the SCRC patch installed (see Installation for details).
Python 3.x: With the following libraries:
numpy
pandas
torch
sklearn
matplotlib
joblib
socket (built-in)
Operating System: Tested on Linux (preferred); Windows support with adjustments.
GPU (optional): For faster model training (requires CUDA-compatible PyTorch).
Installation
Install TORCS with SCRC Patch:
Download TORCS 1.3.4 source from SourceForge and the SCRC patch from the CIG project page.
On Linux:
Install dependencies: sudo apt-get install libgl1-mesa-dev freeglut3-dev libplib-dev libpng-dev zlib1g-dev libopenal-dev libalut-dev.
Unpack TORCS, apply the patch (do-patch.sh), and compile: ./configure, make, make install, make datainstall.
Verify installation by launching TORCS and checking for scr_server bots.
On Windows:
Install TORCS 1.3.4 using the Windows installer, then apply the scr-win-patch.zip.
Launch wtorcs.exe and verify scr_server bots.
Configure TORCS with -nofuel, -nodamage, and -nolaptime flags for project simplifications.
Set Up Python Environment:
Create a virtual environment: python -m venv venv.
Activate it: source venv/bin/activate (Linux) or venv\Scripts\activate (Windows).
Install dependencies: pip install numpy pandas torch scikit-learn matplotlib joblib.
Clone Repository:
git clone <your-repo-url>.
Navigate to the repository: cd <repo-name>.
Usage
Train the Model:
Ensure telemetry_log.csv contains sufficient data (collect more if needed using the client).
Run: python train_model.py.
This will train the model, save weights to race_controller.pth, and generate a loss plot (training_loss.png).
Run the Client:
Start TORCS in Quick Race mode with at least one scr_server bot configured.
Launch the client: python pyclient.py --host localhost --port 3001 --id CAR1.
Optional arguments:
--maxEpisodes: Number of race episodes (default: 1).
--maxSteps: Maximum steps per episode (default: 0, unlimited).
--track: Track name (default: None).
--stage: Stage (0 = Warm-Up, 1 = Qualifying, 2 = Race, 3 = Unknown, default: 3).
The client will connect, drive the car, and log telemetry. Press 'i' to toggle between AI and manual control (if the keyboard module is available).
Evaluate Performance:
Monitor lap times and race position in TORCS Quick Race mode.
Check telemetry_log.csv for logged data and training_loss.png for training progress.
Report
The final report, TORCS_AI_Controller_Report.docx, details the methodology, results, and discussion. Key points:

Methodology: Supervised neural network with 43 input features and 4 output actions.
Results: Achieved a lap time of [insert measured lap time] and [insert rank] in Quick Race mode.
Discussion: Highlights challenges (real-time constraints, generalization) and future improvements (RL, opponent prediction).
Contributing
Contributions are welcome! Please fork the repository and submit pull requests with improvements (e.g., adding reinforcement learning, optimizing model performance).

License
[Specify license, e.g., MIT License. If none, state "No license specified; all rights reserved."]

Acknowledgments
Thanks to the TORCS and SCRC communities for resources and support.
Inspired by the AI 2002 course guidelines and sample implementations.
