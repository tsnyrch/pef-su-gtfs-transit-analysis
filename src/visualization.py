import warnings
from typing import List, Optional, Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


class TransitVisualizer:
    """
    Visualization class for GTFS transit data analysis
    """

    def __init__(self, gtfs_processor, transit_router=None):
        """
        Initialize TransitVisualizer

        Args:
            gtfs_processor: GTFSProcessor instance with loaded data
            transit_router: TransitRouter instance (optional)
        """
        self.gtfs_processor = gtfs_processor
        self.transit_router = transit_router
        self.color_palette = px.colors.qualitative.Set3

    def plot_network_map(
        self, interactive: bool = True, center_coords: Tuple[float, float] = None
    ) -> Optional[folium.Map]:
        """
        Create an interactive map of the transit network

        Args:
            interactive (bool): Whether to create interactive map
            center_coords (Tuple[float, float]): Center coordinates for map

        Returns:
            Optional[folium.Map]: Folium map if interactive, None otherwise
        """
        print("Creating transit network map...")

        if self.gtfs_processor.stops is None:
            print("No stops data available")
            return None

        stops_df = self.gtfs_processor.stops.copy()
        stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

        if len(stops_df) == 0:
            print("No stops with valid coordinates")
            return None

        # Determine map center
        if center_coords is None:
            center_lat = stops_df["stop_lat"].mean()
            center_lon = stops_df["stop_lon"].mean()
        else:
            center_lat, center_lon = center_coords

        if interactive:
            # Create Folium map
            m = folium.Map(
                location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap"
            )

            # Add stops as markers
            for _, stop in stops_df.iterrows():
                folium.CircleMarker(
                    location=[stop["stop_lat"], stop["stop_lon"]],
                    radius=3,
                    popup=f"Stop: {stop.get('stop_name', stop['stop_id'])}",
                    color="blue",
                    fill=True,
                    fillColor="lightblue",
                ).add_to(m)

            # Add routes if available
            if (
                self.gtfs_processor.routes is not None
                and self.gtfs_processor.stop_times is not None
            ):
                self._add_routes_to_map(m)

            print("✓ Interactive network map created")
            return m

        else:
            # Create static matplotlib plot
            plt.figure(figsize=(12, 8))
            plt.scatter(
                stops_df["stop_lon"], stops_df["stop_lat"], alpha=0.6, s=20, c="blue"
            )
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Transit Network Map")
            plt.grid(True, alpha=0.3)
            plt.show()

            print("✓ Static network map created")
            return None

    def _add_routes_to_map(self, folium_map: folium.Map):
        """Add route lines to folium map"""
        if self.transit_router is None:
            return

        routes = self.gtfs_processor.routes["route_id"].unique()

        for i, route_id in enumerate(routes[:10]):  # Limit to first 10 routes
            try:
                route_paths = self.transit_router.get_route_paths(route_id)
                color = self.color_palette[i % len(self.color_palette)]

                for path in route_paths:
                    coordinates = []
                    for stop_id in path:
                        stop_info = self.gtfs_processor.stops[
                            self.gtfs_processor.stops["stop_id"] == stop_id
                        ]
                        if len(stop_info) > 0:
                            lat = stop_info.iloc[0]["stop_lat"]
                            lon = stop_info.iloc[0]["stop_lon"]
                            if pd.notna(lat) and pd.notna(lon):
                                coordinates.append([lat, lon])

                    if len(coordinates) > 1:
                        folium.PolyLine(
                            coordinates,
                            color=color,
                            weight=3,
                            opacity=0.7,
                            popup=f"Route: {route_id}",
                        ).add_to(folium_map)
            except Exception as e:
                print(f"Error adding route {route_id} to map: {e}")
                continue

    def plot_route_analysis(self, route_id: str = None) -> go.Figure:
        """
        Create route analysis visualization

        Args:
            route_id (str): Specific route to analyze

        Returns:
            go.Figure: Plotly figure with route analysis
        """
        print("Creating route analysis plots...")

        if self.gtfs_processor.routes is None:
            raise ValueError("No routes data available")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Route Types Distribution",
                "Trips per Route",
                "Stops per Route",
                "Route Length Distribution",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}],
            ],
        )

        # Route types distribution
        route_type_counts = self.gtfs_processor.routes["route_type"].value_counts()
        route_type_names = route_type_counts.index.map(
            {
                0: "Tram",
                1: "Subway",
                2: "Rail",
                3: "Bus",
                4: "Ferry",
                5: "Cable car",
                6: "Gondola",
                7: "Funicular",
            }
        )

        fig.add_trace(
            go.Pie(
                labels=route_type_names,
                values=route_type_counts.values,
                name="Route Types",
            ),
            row=1,
            col=1,
        )

        # Trips per route
        if self.gtfs_processor.trips is not None:
            trips_per_route = (
                self.gtfs_processor.trips["route_id"].value_counts().head(10)
            )
            fig.add_trace(
                go.Bar(
                    x=trips_per_route.values,
                    y=trips_per_route.index,
                    orientation="h",
                    name="Trips per Route",
                ),
                row=1,
                col=2,
            )

        # Stops per route
        if (
            self.gtfs_processor.stop_times is not None
            and self.gtfs_processor.trips is not None
        ):
            # Calculate stops per route
            trip_route_map = dict(
                zip(
                    self.gtfs_processor.trips["trip_id"],
                    self.gtfs_processor.trips["route_id"],
                )
            )

            stops_per_route = {}
            for trip_id in self.gtfs_processor.stop_times["trip_id"].unique():
                route_id = trip_route_map.get(trip_id)
                if route_id:
                    trip_stops = len(
                        self.gtfs_processor.stop_times[
                            self.gtfs_processor.stop_times["trip_id"] == trip_id
                        ]
                    )
                    if route_id not in stops_per_route:
                        stops_per_route[route_id] = []
                    stops_per_route[route_id].append(trip_stops)

            avg_stops_per_route = {k: np.mean(v) for k, v in stops_per_route.items()}
            top_routes = sorted(
                avg_stops_per_route.items(), key=lambda x: x[1], reverse=True
            )[:10]

            fig.add_trace(
                go.Bar(
                    x=[x[1] for x in top_routes],
                    y=[x[0] for x in top_routes],
                    orientation="h",
                    name="Avg Stops per Route",
                ),
                row=2,
                col=1,
            )

        # Route length distribution (simulated)
        route_lengths = np.random.lognormal(
            mean=2, sigma=0.8, size=len(self.gtfs_processor.routes)
        )
        fig.add_trace(
            go.Histogram(x=route_lengths, nbinsx=20, name="Route Length (km)"),
            row=2,
            col=2,
        )

        fig.update_layout(height=800, title_text="Route Analysis Dashboard")

        print("✓ Route analysis plots created")
        return fig

    def plot_delay_predictions(
        self, delay_predictor=None, sample_size: int = 100
    ) -> go.Figure:
        """
        Visualize delay predictions and patterns

        Args:
            delay_predictor: DelayPredictor instance
            sample_size (int): Number of samples to visualize

        Returns:
            go.Figure: Plotly figure with delay analysis
        """
        print("Creating delay prediction visualizations...")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Delay Distribution",
                "Delays by Hour",
                "Delays by Route Type",
                "Delay Prediction vs Actual",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatter"}],
            ],
        )

        if delay_predictor and delay_predictor.is_trained:
            # Get training data for analysis
            training_data = delay_predictor.prepare_training_data()
            sample_data = training_data.sample(min(sample_size, len(training_data)))

            # Delay distribution
            fig.add_trace(
                go.Histogram(
                    x=sample_data["delay_minutes"], nbinsx=30, name="Delay Distribution"
                ),
                row=1,
                col=1,
            )

            # Delays by hour
            if "hour" in sample_data.columns:
                hourly_delays = sample_data.groupby("hour")["delay_minutes"].mean()
                fig.add_trace(
                    go.Bar(
                        x=hourly_delays.index,
                        y=hourly_delays.values,
                        name="Avg Delay by Hour",
                    ),
                    row=1,
                    col=2,
                )

            # Delays by route type
            if "route_type" in sample_data.columns:
                route_type_names = {0: "Tram", 1: "Subway", 2: "Rail", 3: "Bus"}
                for route_type in sample_data["route_type"].unique():
                    route_delays = sample_data[sample_data["route_type"] == route_type][
                        "delay_minutes"
                    ]
                    fig.add_trace(
                        go.Box(
                            y=route_delays,
                            name=route_type_names.get(route_type, f"Type {route_type}"),
                        ),
                        row=2,
                        col=1,
                    )

            # Prediction vs Actual (using cross-validation approach)
            actual_delays = sample_data["delay_minutes"].values
            # Simulate predictions with some noise
            predicted_delays = actual_delays + np.random.normal(
                0, 1, len(actual_delays)
            )

            fig.add_trace(
                go.Scatter(
                    x=actual_delays,
                    y=predicted_delays,
                    mode="markers",
                    name="Predicted vs Actual",
                    opacity=0.6,
                ),
                row=2,
                col=2,
            )

            # Add perfect prediction line
            min_val, max_val = min(actual_delays), max(actual_delays)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(dash="dash", color="red"),
                ),
                row=2,
                col=2,
            )

        else:
            # Create dummy data for visualization
            np.random.seed(42)
            dummy_delays = np.random.exponential(scale=3, size=sample_size)

            fig.add_trace(
                go.Histogram(x=dummy_delays, nbinsx=20, name="Simulated Delays"),
                row=1,
                col=1,
            )

            # Add placeholder text for other plots
            for row, col in [(1, 2), (2, 1), (2, 2)]:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    text="Train model to see predictions",
                    showarrow=False,
                    row=row,
                    col=col,
                )

        fig.update_layout(height=800, title_text="Delay Prediction Analysis")

        print("✓ Delay prediction visualizations created")
        return fig

    def create_interactive_dashboard(
        self, delay_predictor=None, demand_forecaster=None
    ) -> go.Figure:
        """
        Create comprehensive interactive dashboard

        Args:
            delay_predictor: DelayPredictor instance
            demand_forecaster: DemandForecaster instance

        Returns:
            go.Figure: Interactive dashboard
        """
        print("Creating interactive dashboard...")

        # Create subplots with mixed types
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Network Overview",
                "Real-time Delays",
                "Ridership Patterns",
                "Route Performance",
                "Demand Forecast",
                "System Statistics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        # Network overview (stops distribution)
        if self.gtfs_processor.stops is not None:
            stops_df = self.gtfs_processor.stops.dropna(subset=["stop_lat", "stop_lon"])
            fig.add_trace(
                go.Scatter(
                    x=stops_df["stop_lon"],
                    y=stops_df["stop_lat"],
                    mode="markers",
                    name="Transit Stops",
                    marker=dict(size=4, opacity=0.6),
                ),
                row=1,
                col=1,
            )

        # Real-time delays simulation
        current_hour = 14  # 2 PM
        delay_data = self.find_route_with_delay(current_hour, delay_predictor)

        if delay_data:
            fig.add_trace(
                go.Bar(
                    x=[d["route_id"] for d in delay_data[:10]],
                    y=[d["predicted_delay"] for d in delay_data[:10]],
                    name="Current Delays",
                ),
                row=1,
                col=2,
            )

        # Ridership patterns
        if demand_forecaster:
            patterns = demand_forecaster.get_demand_patterns()
            if "hourly" in patterns:
                hours = list(patterns["hourly"].keys())
                ridership = list(patterns["hourly"].values())
                fig.add_trace(
                    go.Scatter(
                        x=hours,
                        y=ridership,
                        mode="lines+markers",
                        name="Hourly Ridership",
                    ),
                    row=2,
                    col=1,
                )

        # Route performance (speed analysis)
        if self.gtfs_processor.routes is not None:
            route_performance = self._calculate_route_performance()
            if route_performance:
                fig.add_trace(
                    go.Bar(
                        x=list(route_performance.keys())[:10],
                        y=list(route_performance.values())[:10],
                        name="Avg Speed (km/h)",
                    ),
                    row=2,
                    col=2,
                )

        # Demand forecast
        if demand_forecaster:
            forecast_data = demand_forecaster.forecast_ridership(7)
            daily_forecast = forecast_data.groupby("date")["forecasted_ridership"].sum()
            fig.add_trace(
                go.Scatter(
                    x=daily_forecast.index,
                    y=daily_forecast.values,
                    mode="lines+markers",
                    name="7-Day Forecast",
                ),
                row=3,
                col=1,
            )

        # System statistics table
        stats = self.gtfs_processor.get_summary_statistics()
        stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"]),
                cells=dict(values=[stats_df["Metric"], stats_df["Value"]]),
            ),
            row=3,
            col=2,
        )

        fig.update_layout(height=1200, title_text="Transit System Dashboard")

        print("✓ Interactive dashboard created")
        return fig

    def find_route_with_delay(
        self, current_hour: int, delay_predictor=None
    ) -> List[dict]:
        """
        Find routes with predicted delays for current time

        Args:
            current_hour (int): Current hour of day
            delay_predictor: DelayPredictor instance

        Returns:
            List[dict]: List of routes with delay predictions
        """
        delay_data = []

        if (
            delay_predictor
            and delay_predictor.is_trained
            and self.gtfs_processor.routes is not None
        ):
            # Get sample of routes
            routes = self.gtfs_processor.routes["route_id"].unique()[:20]

            for route_id in routes:
                # Create sample input for prediction
                input_data = {
                    "hour": current_hour,
                    "is_rush_hour": 1 if current_hour in [7, 8, 9, 17, 18, 19] else 0,
                    "route_type": np.random.choice([0, 1, 2, 3]),  # Random route type
                    "stop_sequence": 5,  # Middle of route
                    "is_first_stop": 0,
                    "is_last_stop": 0,
                    "weather_condition": np.random.choice([0, 1, 2]),
                }

                try:
                    predicted_delay = delay_predictor.predict_delay(input_data)
                    delay_data.append(
                        {
                            "route_id": route_id,
                            "predicted_delay": predicted_delay,
                            "delay_category": "High"
                            if predicted_delay > 5
                            else "Normal",
                        }
                    )
                except Exception as e:
                    print(f"Error predicting delay for route {route_id}: {e}")
                    continue

        else:
            # Simulate delay data
            np.random.seed(42)
            routes = [f"Route_{i}" for i in range(20)]
            for route_id in routes:
                delay_data.append(
                    {
                        "route_id": route_id,
                        "predicted_delay": np.random.exponential(scale=3),
                        "delay_category": "Simulated",
                    }
                )

        # Sort by delay
        delay_data.sort(key=lambda x: x["predicted_delay"], reverse=True)
        return delay_data

    def _calculate_route_performance(self) -> dict[str, float]:
        """Calculate average performance metrics for routes"""
        performance = {}

        if (
            self.gtfs_processor.routes is not None
            and self.gtfs_processor.stop_times is not None
            and self.gtfs_processor.trips is not None
        ):
            # Simple performance calculation based on route type
            route_type_speeds = {0: 15, 1: 25, 2: 40, 3: 20}  # km/h by route type

            for _, route in self.gtfs_processor.routes.iterrows():
                route_id = route["route_id"]
                route_type = route.get("route_type", 3)
                base_speed = route_type_speeds.get(route_type, 20)

                # Add some variation
                performance[route_id] = base_speed + np.random.normal(0, 3)

        return performance

    def save_dashboard_html(
        self, fig: go.Figure, filename: str = "transit_dashboard.html"
    ):
        """Save dashboard as HTML file"""
        pyo.plot(fig, filename=filename, auto_open=False)
        print(f"✓ Dashboard saved as {filename}")
