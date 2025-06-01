# 🚊 GTFS Transit Analysis System

A comprehensive machine learning and network analysis platform for public transit data using the General Transit Feed Specification (GTFS) format, specifically designed for analyzing the Integrated Transport System of South Moravian Region (IDS JMK) in Brno, Czech Republic.

## 🌟 Overview

This project provides a complete suite of tools for analyzing public transit systems, including route optimization, delay prediction, demand forecasting, and interactive visualizations. Built with modern data science libraries and machine learning techniques, it transforms raw GTFS data into actionable insights for transit agencies and researchers.

### 📍 Data Source

This analysis uses GTFS timetable data from the **Integrated Transport System of South Moravian Region (IDS JMK)**, provided by:

- **Source**: [Statutární město Brno - Open Data Portal](https://data.brno.cz/datasets/379d2e9a7907460c8ca7fda1f3e84328/about)
- **Provider**: KORDIS (transport coordination company)
- **Update Frequency**: Weekly updates every Sunday at 12:00 AM
- **Coverage**: Public transport in South Moravian Region, Czech Republic
- **Format**: Static GTFS format with additional GTFS Realtime data available

The dataset includes comprehensive timetable information for buses, trams, trolleybuses, and regional transport connections in and around Brno.

## ✨ Key Features

### 🔧 Data Processing & Engineering

- **GTFS Data Loading**: Automated parsing of all GTFS files (routes, stops, trips, stop_times, etc.)
- **Feature Engineering**: Advanced time-based, route-based, and spatial features
- **Travel Time Calculation**: Distance and speed analysis between consecutive stops
- **Data Validation**: Comprehensive data quality checks and statistics

### 🗺️ Network Analysis & Routing

- **Graph Construction**: Build transit networks using NetworkX
- **Shortest Path Algorithms**: Dijkstra, A\*, and BFS implementations
- **Centrality Analysis**: Identify key stops using multiple centrality measures
- **Alternative Route Finding**: Multi-path routing capabilities
- **Network Statistics**: Connectivity, clustering, and performance metrics

### 🤖 Machine Learning Models

- **Delay Prediction**: Random Forest and Gradient Boosting models for real-time delay forecasting
- **Demand Forecasting**: Time series analysis for ridership prediction
- **Feature Importance**: Understand key factors affecting transit performance
- **Cross-validation**: Robust model evaluation and selection

### 📊 Interactive Visualizations

- **Network Maps**: Interactive Folium maps with route overlays
- **Real-time Dashboards**: Plotly-based dashboards for system monitoring
- **Performance Analytics**: Route analysis and delay pattern visualization
- **Demand Patterns**: Ridership trend analysis and forecasting charts

## 📁 Project Structure

```
GTFS/
├── data/                          # GTFS data files
│   ├── agency.txt
│   ├── routes.txt
│   ├── stops.txt
│   ├── trips.txt
│   ├── stop_times.txt
│   ├── calendar.txt
│   ├── calendar_dates.txt
│   └── transfers.txt
├── src/                           # Source code modules
│   ├── data_processing.py         # GTFSProcessor class
│   ├── routing.py                 # TransitRouter class
│   ├── prediction.py              # ML models (DelayPredictor, DemandForecaster)
│   └── visualization.py           # TransitVisualizer class
├── notebooks/                     # Jupyter notebooks
│   └── 04_final_demo.ipynb       # Main demonstration notebook
├── outputs/                       # Generated files
│   ├── transit_dashboard.html
│   └── transit_network_map.html
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/tsnyrch/pef-su-gtfs-transit-analysis.git
cd pef-su-gtfs-transit-analysis
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare GTFS data**

   Download the latest IDS JMK GTFS data from [Brno Open Data Portal](https://data.brno.cz/datasets/379d2e9a7907460c8ca7fda1f3e84328/about) and extract the files to the `data/` directory. The required files include:

   - agency.txt, routes.txt, stops.txt, trips.txt, stop_times.txt, calendar.txt, calendar_dates.txt, transfers.txt

4. **Run the analysis**

```bash
# Command line analysis
python3 main.py --full-analysis

# Or use Jupyter notebook
jupyter notebook notebooks/04_final_demo.ipynb
```

## 🎓 Academic Context

This project was developed as part of coursework at **Mendel University in Brno (MENDELU)**, Faculty of Business and Economics (PEF), focusing on advanced data analysis and machine learning applications in transportation systems.

**Authors:**

- Tomáš Snyrch (xsnyrch@mendelu.cz)
- Martin Fiša (xfisa@mendelu.cz)

**Data Source Details:**

- **Dataset**: GTFS Timetable Data for IDS JMK (Integrated Transport System of South Moravian Region)
- **Provider**: Statutární město Brno (Statutory City of Brno)
- **Coordinator**: KORDIS JMK
- **URL**: https://data.brno.cz/datasets/379d2e9a7907460c8ca7fda1f3e84328/about
- **License**: Open Data
- **Coverage**: Public transport network covering Brno and South Moravian Region
- **Transport Modes**: Buses, trams, trolleybuses, regional connections
- **Update Schedule**: Weekly updates every Sunday at 12:00 AM
- **Additional Data**: GTFS Realtime data available for dynamic analysis

## 💻 Usage Examples

### Basic Data Processing

```python
from src.data_processing import GTFSProcessor

# Initialize processor
gtfs = GTFSProcessor('data/')

# Load data
data = gtfs.load_data()

# Create features
features = gtfs.create_features()

# Calculate travel times
travel_times = gtfs.calculate_travel_times()
```

### Network Analysis

```python
from src.routing import TransitRouter

# Initialize router
router = TransitRouter(gtfs)

# Build network
network = router.build_network()

# Find shortest path
path, cost = router.find_shortest_path('stop_1', 'stop_2')

# Analyze centrality
centrality = router.analyze_centrality()
```

### Delay Prediction

```python
from src.prediction import DelayPredictor

# Initialize predictor
predictor = DelayPredictor(gtfs)

# Prepare training data
training_data = predictor.prepare_training_data()

# Train model
results = predictor.train_model(training_data)

# Make predictions
delay = predictor.predict_delay({
    'hour': 8,
    'is_rush_hour': 1,
    'route_type': 3,
    'weather_condition': 0
})
```

### Demand Forecasting

```python
from src.prediction import DemandForecaster

# Initialize forecaster
forecaster = DemandForecaster(gtfs)

# Simulate ridership data
ridership = forecaster.simulate_ridership_data(days=30)

# Generate forecast
forecast = forecaster.forecast_ridership(forecast_days=7)

# Analyze patterns
patterns = forecaster.get_demand_patterns()
```

### Visualization

```python
from src.visualization import TransitVisualizer

# Initialize visualizer
viz = TransitVisualizer(gtfs, router)

# Create network map
network_map = viz.plot_network_map(interactive=True)

# Generate dashboard
dashboard = viz.create_interactive_dashboard(predictor, forecaster)
```

## 🎯 Use Cases

### For Transit Agencies

- **Route Optimization**: Identify underperforming routes and optimization opportunities
- **Delay Management**: Predict and mitigate service delays
- **Capacity Planning**: Forecast ridership demand for service planning
- **Real-time Information**: Provide accurate passenger information systems

### For Researchers

- **Network Analysis**: Study transit network topology and efficiency
- **Machine Learning**: Develop and test predictive models for transit systems
- **Urban Planning**: Analyze accessibility and connectivity patterns
- **Performance Metrics**: Evaluate transit system effectiveness

### For Data Scientists

- **Feature Engineering**: Advanced spatial and temporal feature creation
- **Model Development**: End-to-end ML pipeline for transit prediction
- **Visualization**: Interactive dashboards and mapping capabilities
- **Data Pipeline**: Automated GTFS data processing workflows

## 🛠️ Technical Architecture

### Core Components

1. **GTFSProcessor**: Handles data loading, validation, and feature engineering
2. **TransitRouter**: Implements graph algorithms and network analysis
3. **DelayPredictor**: Machine learning models for delay prediction
4. **DemandForecaster**: Time series forecasting for ridership
5. **TransitVisualizer**: Interactive plotting and dashboard creation

### Machine Learning Pipeline

1. **Data Ingestion**: Automated GTFS file parsing
2. **Feature Engineering**: Time, spatial, and route-based features
3. **Model Training**: Cross-validated model selection
4. **Prediction**: Real-time delay and demand forecasting
5. **Evaluation**: Performance metrics and validation

### Visualization Stack

- **Folium**: Interactive web maps
- **Plotly**: Interactive dashboards and charts
- **Matplotlib/Seaborn**: Statistical plots
- **NetworkX**: Network graph visualization

## 📊 Performance Metrics

The system tracks various performance indicators:

- **Model Accuracy**: R², MAE, RMSE for predictions
- **Network Metrics**: Connectivity, centrality, clustering
- **System Statistics**: Routes, stops, trips, coverage
- **Real-time Performance**: Delay patterns, service reliability

## 🔧 Configuration

### Model Parameters

- **Random Forest**: 100 estimators, max_depth=10
- **Gradient Boosting**: 100 estimators, learning_rate=0.1
- **Network Analysis**: Multiple centrality measures
- **Forecasting**: 7-day prediction horizon

### Visualization Settings

- **Map Tiles**: OpenStreetMap
- **Color Palette**: Plotly Set3
- **Interactive Features**: Zoom, pan, hover tooltips
- **Export Formats**: HTML, PNG, SVG

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows PEP 8 standards
5. Submit a pull request with clear description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest flake8 black

# Run tests
pytest tests/

# Format code
black src/

# Check style
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [GTFS Specification](https://developers.google.com/transit/gtfs)
- [NetworkX Documentation](https://networkx.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python Documentation](https://plotly.com/python/)

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the development team.

## 🙏 Acknowledgments

- Transit agencies providing open GTFS data
- Open source community for excellent libraries
- Contributors and beta testers

---

**Built with ❤️ for better public transportation**
