<div align="center">

# 🔋 Battery-Health-Intelligence
### Data-and-Physics Fusion for EV Battery Health Monitoring using Physics-Informed Neural Networks (PINNs)

<p align="center">
An end-to-end Artificial Intelligence framework for Electric Vehicle (EV) Battery Health Monitoring that combines <b>Deep Learning</b>, <b>Physics-Informed Neural Networks (PINNs)</b>, and <b>Machine Learning</b> to accurately estimate battery health and remaining useful life.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-WebApp-black?style=for-the-badge&logo=flask)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![PINN](https://img.shields.io/badge/PINN-Physics_Informed_Neural_Network-success?style=for-the-badge)
![IEEE](https://img.shields.io/badge/IEEE-Published-blue?style=for-the-badge&logo=ieee)

</p>

---

## 📑 Table of Contents

- Project Overview
- Research Publication
- Key Features
- Technology Stack
- Repository Structure
- Physics-Informed Neural Network
- Dataset Processing
- Training Pipeline
- Prediction APIs
- Flask Dashboard
- Outputs
- Installation
- Usage
- API Documentation
- Future Scope
- Citation
- About the Author
- License

---

# 🚀 Project Overview

Battery-Health-Intelligence is an AI-powered Battery Health Monitoring System designed for Lithium-Ion Electric Vehicle batteries.

Unlike conventional machine learning approaches, this project combines:

- 🧠 Deep Learning
- ⚡ Physics-Informed Neural Networks (PINNs)
- 📈 Machine Learning
- 🔬 Battery degradation physics
- 🌐 Flask Web Dashboard

to estimate battery health with improved reliability.

The framework predicts:

- 🔋 State of Health (SOH)
- ⏳ Remaining Useful Life (RUL)
- 🔄 Remaining Useful Life (Cycles)

using both battery operational data and physics-inspired constraints.

---

# 🏆 IEEE Research Publication

This project is based on our IEEE published research paper.

## 📄 Paper

**Data-and-Physics Fusion using PINNs for Lithium-Ion Battery Health Monitoring**

📚 Published in:

**IEEE Xplore Digital Library**

🔗 https://ieeexplore.ieee.org/document/11470440

Conference:

**International Conference on Visual Analytics and Data Visualization (ICVADV-2025)**

---

## 🎤 Research Presentation

This research work was officially presented at

**International Conference on Visual Analytics and Data Visualization (ICVADV-2025)**

held during

**04–06 March 2025**

organized by

Department of Electronics and Communication Engineering

Francis Xavier Engineering College

Tirunelveli, Tamil Nadu, India

in association with IEEE.

---

# 👨‍💻 Author

## Mahanandhi Abhiram

Software Developer | AI & Machine Learning Engineer

Research Interests

- Artificial Intelligence
- Physics-Informed Neural Networks
- Electric Vehicle Technologies
- Deep Learning
- Full Stack Development

---

# ⭐ Key Features

## 🔋 Battery Health Estimation

Predicts

- State of Health (SOH)
- Remaining Useful Life (RUL)
- Remaining Useful Life in Cycles

---

## 🧠 Physics-Informed Neural Network

Instead of learning only from data,

the model also learns battery degradation physics by enforcing

- SOH Monotonicity
- RUL Monotonicity

during training.

---

## 📊 Multi-Output Deep Learning

Single neural network predicts

- SOH
- RUL
- RUL Cycles

simultaneously.

---

## 🌐 Flask Dashboard

Interactive web application with

- Training Dashboard
- Prediction Dashboard
- Graph Visualization
- Metrics Visualization

---

## 🤖 Random Forest Baseline

Includes a complete baseline implementation for comparison.

Comparison between

- PINN
- Random Forest

helps evaluate model improvements.

---

## 📈 Automatic Evaluation

Automatically generates

- Training Curves
- Loss Graphs
- Prediction Graphs
- Evaluation Metrics
- Comparison Reports

---

# 🛠 Technology Stack

## Backend

- Python
- Flask
- Flask-CORS

## Artificial Intelligence

- TensorFlow
- Keras
- Physics-Informed Neural Networks (PINNs)

## Machine Learning

- Scikit-Learn
- Random Forest Regressor

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib

## Utilities

- Joblib
- MinMaxScaler

## Frontend

- HTML
- CSS
- JavaScript
- Flask Templates

---

# 🧠 Physics-Informed Neural Network (PINN)

Traditional neural networks learn purely from data.

Physics-Informed Neural Networks additionally incorporate scientific knowledge into the learning process.

This project enforces two important physical constraints.

## SOH Monotonicity

Battery State of Health should never increase as charging cycles increase.

The model penalizes any violation.

---

## RUL Monotonicity

Remaining Useful Life should always decrease over battery usage.

The model introduces additional loss functions that discourage unrealistic predictions.

---

The overall optimization objective becomes

Total Loss

= Data Loss

+ Physics Loss (SOH)

+ Physics Loss (RUL)

which results in significantly more stable predictions.

---

# 🎯 Prediction Targets

The model predicts

| Output | Description |
|---------|-------------|
| SOH | State of Health |
| RUL | Remaining Useful Life |
| RUL Cycles | Remaining Remaining Life in Charging Cycles |

---

# 📂 Repository Structure

```
Battery-Health-Intelligence
│
├── app.py                              # Flask application
├── train_model.py                      # PINN training pipeline
├── battery_pinn_model.py               # PINN architecture
├── predict.py                          # Prediction script
├── data_processor.py                   # Dataset preprocessing
├── generate_datasets.py                # Synthetic dataset generator
├── model.py                            # Helper model utilities
│
├── frontend/                           # Frontend source
├── templates/                          # Flask HTML Templates
├── scripts/                            # Utility scripts
│
├── data/                               # Battery datasets
│
├── models/                             # Saved AI Models
│     ├── battery_model.keras
│     ├── battery_health_model.h5
│     ├── rf_model.pkl
│     ├── scalers.joblib
│     └── ...
│
├── results/
│     ├── evaluation_metrics.json
│     ├── plots/
│     └── training_history.json
│
├── results_rf/
│
├── evaluation_plots/
│
├── requirements.txt
│
└── README.md
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/mahanandhi-abhiram-22/Battery-Health-Intelligence.git

cd Battery-Health-Intelligence
```

---

## Create Virtual Environment

Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

Start Flask Server

```bash
python app.py
```

Application runs on

```
http://127.0.0.1:8000
```

---

# 🏋 Training Pipeline

The project provides an end-to-end training workflow.

Training performs

- Dataset Loading
- Feature Engineering
- Data Scaling
- PINN Training
- Physics Constraint Optimization
- Model Saving
- Evaluation
- Visualization

Generated artifacts include

```
models/
results/
evaluation_metrics.json
training_loss.png
parity plots
```

---

# 📊 Dataset Processing

Implemented in

```
data_processor.py
```

Responsibilities

✔ Load Battery CSV Files

✔ Feature Engineering

✔ Target Engineering

✔ Missing Value Handling

✔ Battery Type Encoding

✔ Feature Scaling

✔ Dataset Splitting

---

## Expected Input Features

| Feature | Description |
|----------|-------------|
| Voltage | Battery Voltage |
| Current | Charging / Discharging Current |
| Temperature | Battery Temperature |
| Cycle | Charge Cycle |
| Capacity | Battery Capacity |

---

## Output Targets

| Target | Description |
|----------|------------|
| SOH | State of Health |
| RUL | Remaining Useful Life |
| RUL Cycles | Remaining Useful Cycles |

---

# 🔬 Feature Engineering

The preprocessing module automatically

- Normalizes data
- Encodes battery type
- Creates RUL Cycles
- Creates SOH
- Standardizes feature names

making datasets from different battery sources compatible.

---

# ⚡ Model Training Pipeline

```
CSV Dataset
      │
      ▼
Preprocessing
      │
      ▼
Feature Scaling
      │
      ▼
PINN Model
      │
      ▼
Physics Constraints
      │
      ▼
Training
      │
      ▼
Evaluation
      │
      ▼
Model Saving
```

---

# 🌐 Flask Web Dashboard

The project includes a Flask-based dashboard for interacting with the trained models.

Available pages

| Route | Description |
|--------|-------------|
| / | Home Dashboard |
| /training | Training Page |
| /dashboard | Prediction Dashboard |
| /graphs | Visualization Dashboard |

---

# 🔌 REST API Documentation

## Train Model

### Endpoint

```
POST /api/train
```

### Request

```json
{
  "epochs":30,
  "batch_size":32,
  "patience":5
}
```

### Response

```json
{
   "status":"Training Started"
}
```

---

## Predict Battery Health

### Endpoint

```
POST /api/predict
```

### Example Request

```json
{
  "voltage":3.7,
  "current":0.5,
  "temperature":25,
  "cycle":80,
  "capacity":1.0
}
```

---

### Example Response

```json
{
   "SOH":0.94,
   "RUL":0.78,
   "RUL_Cycles":520
}
```

---

# 📈 Compare Models

Endpoint

```
POST /api/compare
```

Returns predictions from

- Physics-Informed Neural Network
- Random Forest Baseline

allowing direct comparison.

---

# 📊 Metrics Endpoint

```
GET /api/compare/metrics
```

Combines

- PINN Metrics
- Random Forest Metrics

for side-by-side evaluation.

---

# 📉 Plot Endpoint

```
GET /api/results/plot/<plot_name>
```

Available plots include

- Training Loss
- SOH Prediction
- RUL Prediction
- Parity Plot
- Evaluation Graphs

---

# 🧪 Generate Synthetic Dataset

Run

```bash
python generate_datasets.py
```

Generated datasets

```
calce_battery_synth.csv

nasa_battery_synth.csv

oxford_battery_synth.csv
```

These datasets are stored under

```
data/
```

---

# 📤 Project Outputs

## Saved Models

```
battery_model.keras

battery_health_model.h5

rf_model.pkl

scalers.joblib
```

---

## Evaluation Results

```
evaluation_metrics.json

training_history.json

training_loss.png

parity_grid.png

training_curves.png
```

---

## Baseline Results

Stored in

```
results_rf/
```

including

```
rf_metrics.json
```

used for comparison with the proposed PINN model.

---


# 📊 Results

The proposed **Physics-Informed Neural Network (PINN)** combines data-driven learning with battery degradation physics, resulting in more stable and physically consistent predictions than conventional machine learning models.

### Key Prediction Outputs

- 🔋 State of Health (SOH)
- ⏳ Remaining Useful Life (RUL)
- 🔄 Remaining Useful Life (RUL Cycles)

---

# 🔬 Physics Constraints

Unlike conventional neural networks, the proposed PINN introduces **physics-inspired monotonicity constraints** during optimization.

## 1️⃣ SOH Monotonicity

Battery State of Health should never increase as the number of charge cycles increases.

The model penalizes predictions violating this physical behavior.

---

## 2️⃣ RUL Monotonicity

Remaining Useful Life should continuously decrease with battery aging.

The PINN introduces an additional loss function that discourages unrealistic increases in RUL during training.

---

## Overall Training Objective

The final optimization objective combines data loss with physics losses.

```
Total Loss

= Data Loss

+ λ₁ × SOH Physics Loss

+ λ₂ × RUL Physics Loss
```

Implemented as

```python
total_loss = data_loss + 0.1 * loss_soh_phy + 0.1 * loss_rul_phy
```

This enables the model to produce predictions that are not only accurate but also physically meaningful.

---

# 📈 Model Comparison

| Feature | PINN (Proposed) | Random Forest |
|----------|-----------------|---------------|
| Multi-output Prediction | ✅ | Partial |
| Physics Constraints | ✅ | ❌ |
| Deep Learning | ✅ | ❌ |
| Battery Aging Knowledge | ✅ | ❌ |
| SOH Prediction | ✅ | ✅ |
| RUL Prediction | ✅ | ✅ |
| RUL Cycles | ✅ | ❌ |

---

# 🏆 Research Publication

This project is based on our IEEE-published research.

## **Data-and-Physics Fusion using PINNs for Lithium-Ion Battery Health Monitoring**

📖 **Published in IEEE Xplore Digital Library**

🔗 https://ieeexplore.ieee.org/document/11470440

Presented at

**International Conference on Visual Analytics and Data Visualization (ICVADV-2025)**

---

# 🎤 Conference Presentation

This research was officially presented at

**International Conference on Visual Analytics and Data Visualization (ICVADV-2025)**

📅 **04–06 March 2025**

Organized by

**Department of Electronics and Communication Engineering**

**Francis Xavier Engineering College**

Tirunelveli, Tamil Nadu, India

in association with **IEEE**.

---

# 📜 Citation

If you use this work in your research, please cite the paper.

```bibtex
@inproceedings{abhiram2025batteryhealth,
  title={Data-and-Physics Fusion using PINNs for Lithium-Ion Battery Health Monitoring},
  author={Mahanandhi Abhiram},
  booktitle={International Conference on Visual Analytics and Data Visualization (ICVADV)},
  year={2025},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/11470440}
}
```

---

# 🚀 Future Enhancements

- Battery Remaining Charging Time Prediction
- Transformer-based Battery Health Estimation
- Explainable AI (XAI)
- Attention-based PINN Architecture
- Cloud Deployment
- Docker Support
- Mobile Application
- Real-time IoT Battery Monitoring
- Battery Digital Twin
- Edge AI Deployment

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 🛣️ Project Roadmap

- [x] Data Processing Pipeline
- [x] PINN Model Development
- [x] Random Forest Baseline
- [x] Flask Dashboard
- [x] REST APIs
- [x] IEEE Research Publication
- [x] Evaluation Framework
- [ ] Docker Support
- [ ] CI/CD Pipeline
- [ ] Cloud Deployment
- [ ] Mobile App Integration

---

# 📷 Screenshots

You can include screenshots of:

- 🏠 Dashboard
- 📈 Training Progress
- 📊 Evaluation Graphs
- 🔋 Battery Prediction Interface
- 📉 Loss Curves
- 📑 API Responses

Example folder

```
assets/

dashboard.png

prediction.png

training.png

graphs.png
```

---

# 📄 License

This project is released for **research and educational purposes**.

Feel free to use, modify, and extend the work with proper attribution.

---

# 🙏 Acknowledgements

Special thanks to

- IEEE
- ICVADV 2025
- Francis Xavier Engineering College
- TensorFlow
- Flask
- Scikit-Learn
- Python Community
- Open Source Community

for enabling this research.

---

# 👨‍💻 About the Author

## **Mahanandhi Abhiram**

**Software Developer | AI Engineer | Machine Learning Researcher**

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Physics-Informed Neural Networks
- Electric Vehicle Technologies
- Full Stack Development
- Computer Vision
- Data Science

---

### 🌐 Connect with Me

**GitHub**

https://github.com/mahanandhi-abhiram-22

**LinkedIn**

https://www.linkedin.com/in/m-abhiram/

**Portfolio**

https://abhiram-dev-portfolio.vercel.app/

---

# ⭐ Support the Project

If you found this project useful,

⭐ **Please consider giving it a Star on GitHub!**

It helps others discover the project and motivates further research and development.

---

<div align="center">

## ⭐ If you like this project, don't forget to Star the Repository ⭐

**Made with ❤️ by Mahanandhi Abhiram**

*Advancing EV Battery Intelligence through Artificial Intelligence and Physics-Informed Learning.*

</div>
