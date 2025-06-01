# 🚀 Quick Start Guide

Get up and running with GTFS Transit Analysis in 5 minutes!

## ⚡ Installation

### 1. Install Dependencies

```bash
pip install pandas numpy networkx geopy scikit-learn matplotlib seaborn plotly folium jupyter
```

### 2. Verify Installation

```bash
python3 -c "import pandas, numpy, networkx, sklearn; print('✅ All dependencies installed!')"
```

## 🏃‍♂️ Running Analysis

### Option 1: Quick Command Line Analysis

```bash
# Full analysis pipeline
python3 main.py --full-analysis

# Network analysis only
python3 main.py --network-only

# Delay prediction only
python3 main.py --predict-delays

# Create dashboard
python3 main.py --create-dashboard
```

### Option 2: Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Open the demo notebook
# Navigate to: notebooks/04_final_demo.ipynb
```

### Option 3: Python Script

```python
import sys
sys.path.append('src')

from data_processing import GTFSProcessor
from routing import TransitRouter
from prediction import DelayPredictor
from visualization import TransitVisualizer

# Load data
gtfs = GTFSProcessor('data/')
data = gtfs.load_data()

# Basic analysis
features = gtfs.create_features()
travel_times = gtfs.calculate_travel_times()

# Network analysis
router = TransitRouter(gtfs)
network = router.build_network()

# Delay prediction
predictor = DelayPredictor(gtfs)
training_data = predictor.prepare_training_data()
results = predictor.train_model(training_data)

# Visualization
viz = TransitVisualizer(gtfs, router)
dashboard = viz.create_interactive_dashboard(predictor)
```

## 📊 Quick Results

After running the analysis, you'll get:

- **Network map**: `outputs/network_map.html`
- **Interactive dashboard**: `outputs/transit_dashboard.html`
- **Console output**: Statistics and performance metrics

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Empty data directory**

```bash
# Ensure GTFS files are in data/ directory:
# agency.txt, routes.txt, stops.txt, trips.txt, stop_times.txt
```

**Memory errors**

```bash
# Reduce dataset size or increase system memory
# Use data sampling in scripts
```

## 📱 View Results

Open generated HTML files in your browser:

```bash
# On macOS
open outputs/transit_dashboard.html

# On Linux
xdg-open outputs/transit_dashboard.html

# On Windows
start outputs/transit_dashboard.html
```

## 🎯 Next Steps

1. **Explore the notebook**: `notebooks/04_final_demo.ipynb`
2. **Customize parameters**: Edit models in `src/` directory
3. **Add your data**: Replace files in `data/` directory
4. **Extend analysis**: Build on existing modules

## ⏱️ Expected Runtime

- **Small dataset** (< 1000 stops): 2-5 minutes
- **Medium dataset** (1000-10000 stops): 5-15 minutes
- **Large dataset** (> 10000 stops): 15+ minutes

Ready to analyze your transit data! 🚊
