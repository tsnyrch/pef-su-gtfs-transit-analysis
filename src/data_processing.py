import os
import warnings
from datetime import datetime, timedelta
from typing import Optional

import networkx as nx
import pandas as pd
from geopy.distance import geodesic

warnings.filterwarnings("ignore")


class GTFSProcessor:
    """
    GTFS data processor for loading and preprocessing transit data
    """

    def __init__(self, data_path: str):
        """
        Initialize GTFS processor with data path

        Args:
            data_path (str): Path to directory containing GTFS files
        """
        self.data_path = data_path
        self.agencies = None
        self.routes = None
        self.trips = None
        self.stops = None
        self.stop_times = None
        self.calendar = None
        self.calendar_dates = None
        self.transfers = None

    def load_data(self) -> dict[str, pd.DataFrame]:
        """
        Load all GTFS data files into pandas DataFrames

        Returns:
            dict[str, pd.DataFrame]: Dictionary of loaded GTFS data
        """
        print("Loading GTFS data files...")

        # Define required files and their corresponding attributes
        file_mapping = {
            "agency.txt": "agencies",
            "routes.txt": "routes",
            "trips.txt": "trips",
            "stops.txt": "stops",
            "stop_times.txt": "stop_times",
            "calendar.txt": "calendar",
            "calendar_dates.txt": "calendar_dates",
            "transfers.txt": "transfers",
        }

        loaded_data = {}

        for filename, attr_name in file_mapping.items():
            file_path = os.path.join(self.data_path, filename)

            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    setattr(self, attr_name, df)
                    loaded_data[attr_name] = df
                    print(f"✓ Loaded {filename}: {len(df)} records")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
            else:
                print(f"⚠ File not found: {filename}")

        return loaded_data

    def create_features(self) -> pd.DataFrame:
        """
        Create engineered features from GTFS data

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        print("Creating engineered features...")

        if self.stop_times is None or self.trips is None or self.routes is None:
            raise ValueError("Required data not loaded. Call load_data() first.")

        # Merge stop_times with trips and routes for comprehensive data
        features_df = self.stop_times.merge(self.trips, on="trip_id", how="left")
        features_df = features_df.merge(self.routes, on="route_id", how="left")
        features_df = features_df.merge(self.stops, on="stop_id", how="left")

        # Convert time strings to datetime objects
        def parse_time(time_str):
            if pd.isna(time_str):
                return None
            try:
                time_parts = time_str.split(":")
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2])

                # Handle times > 24:00 (next day)
                if hour >= 24:
                    hour = hour - 24

                return datetime.strptime(
                    f"{hour:02d}:{minute:02d}:{second:02d}", "%H:%M:%S"
                ).time()
            except Exception as e:
                print(f"Error parsing time '{time_str}': {e}")
                return None

        features_df["arrival_time_parsed"] = features_df["arrival_time"].apply(
            parse_time
        )
        features_df["departure_time_parsed"] = features_df["departure_time"].apply(
            parse_time
        )

        # Extract time-based features
        features_df["hour"] = features_df["arrival_time_parsed"].apply(
            lambda x: x.hour if x else None
        )
        features_df["is_rush_hour"] = features_df["hour"].apply(
            lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0 if x is not None else None
        )
        features_df["time_period"] = features_df["hour"].apply(
            self._categorize_time_period
        )

        # Route-based features
        features_df["route_type_name"] = features_df["route_type"].map(
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

        # Stop sequence features
        features_df = features_df.sort_values(["trip_id", "stop_sequence"])
        features_df["is_first_stop"] = (features_df["stop_sequence"] == 1).astype(int)
        features_df["is_last_stop"] = (
            features_df.groupby("trip_id")["stop_sequence"].transform("max")
            == features_df["stop_sequence"]
        )
        features_df["is_last_stop"] = features_df["is_last_stop"].astype(int)

        print(f"✓ Created features for {len(features_df)} records")
        return features_df

    def _categorize_time_period(self, hour):
        """Helper function to categorize time periods"""
        if hour is None:
            return None
        elif 6 <= hour < 10:
            return "Morning Rush"
        elif 10 <= hour < 16:
            return "Midday"
        elif 16 <= hour < 20:
            return "Evening Rush"
        elif 20 <= hour < 24:
            return "Evening"
        else:
            return "Night"

    def calculate_travel_times(self) -> pd.DataFrame:
        """
        Calculate travel times between consecutive stops

        Returns:
            pd.DataFrame: DataFrame with travel time calculations
        """
        print("Calculating travel times...")

        if self.stop_times is None or self.stops is None:
            raise ValueError("Required data not loaded. Call load_data() first.")

        # Merge with stops to get coordinates
        travel_data = self.stop_times.merge(self.stops, on="stop_id", how="left")
        travel_data = travel_data.sort_values(["trip_id", "stop_sequence"])

        travel_times = []

        for trip_id in travel_data["trip_id"].unique():
            trip_data = travel_data[travel_data["trip_id"] == trip_id].copy()

            for i in range(len(trip_data) - 1):
                current_stop = trip_data.iloc[i]
                next_stop = trip_data.iloc[i + 1]

                # Calculate travel time
                try:
                    current_time = self._parse_gtfs_time(current_stop["departure_time"])
                    next_time = self._parse_gtfs_time(next_stop["arrival_time"])

                    if current_time and next_time:
                        travel_time = (
                            next_time - current_time
                        ).total_seconds() / 60  # minutes

                        # Calculate distance if coordinates are available
                        distance = None
                        if all(
                            pd.notna(
                                [
                                    current_stop["stop_lat"],
                                    current_stop["stop_lon"],
                                    next_stop["stop_lat"],
                                    next_stop["stop_lon"],
                                ]
                            )
                        ):
                            distance = geodesic(
                                (current_stop["stop_lat"], current_stop["stop_lon"]),
                                (next_stop["stop_lat"], next_stop["stop_lon"]),
                            ).kilometers

                        travel_times.append(
                            {
                                "trip_id": trip_id,
                                "from_stop_id": current_stop["stop_id"],
                                "to_stop_id": next_stop["stop_id"],
                                "from_stop_sequence": current_stop["stop_sequence"],
                                "to_stop_sequence": next_stop["stop_sequence"],
                                "departure_time": current_stop["departure_time"],
                                "arrival_time": next_stop["arrival_time"],
                                "travel_time_minutes": travel_time,
                                "distance_km": distance,
                                "speed_kmh": distance / (travel_time / 60)
                                if distance and travel_time > 0
                                else None,
                            }
                        )
                except Exception as e:
                    print(f"✗ Error calculating travel time for trip {trip_id}: {e}")
                    continue

        travel_times_df = pd.DataFrame(travel_times)
        print(f"✓ Calculated travel times for {len(travel_times_df)} segments")

        return travel_times_df

    def _parse_gtfs_time(self, time_str: str) -> Optional[datetime]:
        """
        Parse GTFS time string to datetime object

        Args:
            time_str (str): Time string in HH:MM:SS format

        Returns:
            Optional[datetime]: Parsed datetime object
        """
        if pd.isna(time_str):
            return None

        try:
            time_parts = time_str.split(":")
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(time_parts[2])

            # Create base date (arbitrary)
            base_date = datetime(2024, 1, 1)

            # Handle times >= 24:00 (next day service)
            if hour >= 24:
                base_date += timedelta(days=1)
                hour = hour - 24

            return base_date.replace(hour=hour, minute=minute, second=second)
        except Exception as e:
            print(f"✗ Error parsing time '{time_str}': {e}")
            return None

    def get_network_graph(self) -> nx.Graph:
        """
        Create a network graph from GTFS data

        Returns:
            nx.Graph: NetworkX graph representing the transit network
        """
        print("Creating network graph...")

        G = nx.Graph()

        # Add stops as nodes
        if self.stops is not None:
            for _, stop in self.stops.iterrows():
                G.add_node(
                    stop["stop_id"],
                    name=stop.get("stop_name", ""),
                    lat=stop.get("stop_lat", 0),
                    lon=stop.get("stop_lon", 0),
                )

        # Add connections from stop_times
        if self.stop_times is not None:
            travel_times = self.calculate_travel_times()

            for _, segment in travel_times.iterrows():
                if not G.has_edge(segment["from_stop_id"], segment["to_stop_id"]):
                    G.add_edge(
                        segment["from_stop_id"],
                        segment["to_stop_id"],
                        travel_time=segment["travel_time_minutes"],
                        distance=segment.get("distance_km", 0),
                    )

        print(
            f"✓ Created network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def get_summary_statistics(self) -> dict:
        """
        Get summary statistics of the GTFS dataset

        Returns:
            dict: Summary statistics
        """
        stats = {}

        if self.agencies is not None:
            stats["num_agencies"] = len(self.agencies)

        if self.routes is not None:
            stats["num_routes"] = len(self.routes)
            stats["route_types"] = self.routes["route_type"].value_counts().to_dict()

        if self.stops is not None:
            stats["num_stops"] = len(self.stops)

        if self.trips is not None:
            stats["num_trips"] = len(self.trips)

        if self.stop_times is not None:
            stats["num_stop_times"] = len(self.stop_times)

        return stats
