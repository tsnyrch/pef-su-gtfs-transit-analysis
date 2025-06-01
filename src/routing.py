import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import networkx as nx
from geopy.distance import geodesic

warnings.filterwarnings("ignore")


class TransitRouter:
    """
    Transit routing and network analysis class for GTFS data
    """

    def __init__(self, gtfs_processor):
        """
        Initialize TransitRouter with GTFS processor

        Args:
            gtfs_processor: GTFSProcessor instance with loaded data
        """
        self.gtfs_processor = gtfs_processor
        self.network_graph = None
        self.stop_coordinates = {}
        self.route_stop_mapping = {}
        self.centrality_measures = {}

    def build_network(self, weight_type: str = "travel_time") -> nx.Graph:
        """
        Build transit network graph from GTFS data

        Args:
            weight_type (str): Type of edge weight ('travel_time', 'distance', 'uniform')

        Returns:
            nx.Graph: Built network graph
        """
        print(f"Building transit network with {weight_type} weights...")

        G = nx.Graph()

        # Add stops as nodes with attributes
        if self.gtfs_processor.stops is not None:
            for _, stop in self.gtfs_processor.stops.iterrows():
                stop_id = stop["stop_id"]
                G.add_node(
                    stop_id,
                    name=stop.get("stop_name", ""),
                    lat=stop.get("stop_lat", 0),
                    lon=stop.get("stop_lon", 0),
                    zone_id=stop.get("zone_id", ""),
                )

                # Store coordinates for distance calculations
                self.stop_coordinates[stop_id] = (
                    stop.get("stop_lat", 0),
                    stop.get("stop_lon", 0),
                )

        # Build route-stop mapping
        if (
            self.gtfs_processor.stop_times is not None
            and self.gtfs_processor.trips is not None
        ):
            trip_route_map = dict(
                zip(
                    self.gtfs_processor.trips["trip_id"],
                    self.gtfs_processor.trips["route_id"],
                )
            )

            for _, stop_time in self.gtfs_processor.stop_times.iterrows():
                route_id = trip_route_map.get(stop_time["trip_id"])
                if route_id:
                    if route_id not in self.route_stop_mapping:
                        self.route_stop_mapping[route_id] = set()
                    self.route_stop_mapping[route_id].add(stop_time["stop_id"])

        # Add edges based on consecutive stops in trips
        if self.gtfs_processor.stop_times is not None:
            # Group by trip and sort by stop sequence
            stop_times_sorted = self.gtfs_processor.stop_times.sort_values(
                ["trip_id", "stop_sequence"]
            )

            for trip_id in stop_times_sorted["trip_id"].unique():
                trip_stops = stop_times_sorted[stop_times_sorted["trip_id"] == trip_id]

                for i in range(len(trip_stops) - 1):
                    current_stop = trip_stops.iloc[i]
                    next_stop = trip_stops.iloc[i + 1]

                    stop1 = current_stop["stop_id"]
                    stop2 = next_stop["stop_id"]

                    # Calculate edge weight based on weight_type
                    weight = self._calculate_edge_weight(
                        current_stop, next_stop, weight_type
                    )

                    # Add edge or update existing edge with minimum weight
                    if G.has_edge(stop1, stop2):
                        current_weight = G[stop1][stop2].get("weight", float("inf"))
                        if weight < current_weight:
                            G[stop1][stop2]["weight"] = weight
                    else:
                        G.add_edge(stop1, stop2, weight=weight)

        # Add transfer connections if available
        if self.gtfs_processor.transfers is not None:
            for _, transfer in self.gtfs_processor.transfers.iterrows():
                from_stop = transfer["from_stop_id"]
                to_stop = transfer["to_stop_id"]

                if from_stop in G.nodes and to_stop in G.nodes:
                    transfer_time = (
                        transfer.get("min_transfer_time", 120) / 60
                    )  # Convert to minutes

                    if weight_type == "travel_time":
                        weight = transfer_time
                    elif weight_type == "distance":
                        weight = self._calculate_distance(from_stop, to_stop)
                    else:
                        weight = 1

                    G.add_edge(from_stop, to_stop, weight=weight, edge_type="transfer")

        self.network_graph = G
        print(
            f"✓ Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def _calculate_edge_weight(self, stop1_data, stop2_data, weight_type: str) -> float:
        """Calculate edge weight between two stops"""
        if weight_type == "travel_time":
            try:
                time1 = self._parse_time(stop1_data["departure_time"])
                time2 = self._parse_time(stop2_data["arrival_time"])
                if time1 and time2:
                    return (time2 - time1).total_seconds() / 60  # minutes
            except Exception as e:
                print(f"Error calculating travel time: {e}")
                pass
            return 5.0  # Default 5 minutes

        elif weight_type == "distance":
            return self._calculate_distance(
                stop1_data["stop_id"], stop2_data["stop_id"]
            )

        else:  # uniform
            return 1.0

    def _calculate_distance(self, stop1_id: str, stop2_id: str) -> float:
        """Calculate distance between two stops in kilometers"""
        if stop1_id in self.stop_coordinates and stop2_id in self.stop_coordinates:
            coord1 = self.stop_coordinates[stop1_id]
            coord2 = self.stop_coordinates[stop2_id]
            if all(coord1) and all(coord2):
                return geodesic(coord1, coord2).kilometers
        return 1.0  # Default 1 km

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse GTFS time string"""
        try:
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2])

            base = datetime(2024, 1, 1)
            if hour >= 24:
                base += timedelta(days=1)
                hour -= 24

            return base.replace(hour=hour, minute=minute, second=second)
        except Exception as e:
            print(f"Error parsing time '{time_str}': {e}")
            return None

    def find_shortest_path(
        self, start_stop: str, end_stop: str, algorithm: str = "dijkstra"
    ) -> Tuple[List[str], float]:
        """
        Find shortest path between two stops

        Args:
            start_stop (str): Starting stop ID
            end_stop (str): Destination stop ID
            algorithm (str): Algorithm to use ('dijkstra', 'astar', 'bfs')

        Returns:
            Tuple[List[str], float]: Path as list of stop IDs and total cost
        """
        if self.network_graph is None:
            self.build_network()

        if start_stop not in self.network_graph.nodes:
            raise ValueError(f"Start stop {start_stop} not found in network")
        if end_stop not in self.network_graph.nodes:
            raise ValueError(f"End stop {end_stop} not found in network")

        try:
            if algorithm == "dijkstra":
                path = nx.shortest_path(
                    self.network_graph, start_stop, end_stop, weight="weight"
                )
                cost = nx.shortest_path_length(
                    self.network_graph, start_stop, end_stop, weight="weight"
                )

            elif algorithm == "astar":

                def heuristic(node1, node2):
                    return self._calculate_distance(node1, node2)

                path = nx.astar_path(
                    self.network_graph,
                    start_stop,
                    end_stop,
                    heuristic=heuristic,
                    weight="weight",
                )
                cost = nx.astar_path_length(
                    self.network_graph,
                    start_stop,
                    end_stop,
                    heuristic=heuristic,
                    weight="weight",
                )

            elif algorithm == "bfs":
                path = nx.shortest_path(self.network_graph, start_stop, end_stop)
                cost = len(path) - 1

            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            return path, cost

        except nx.NetworkXNoPath:
            return [], float("inf")

    def analyze_centrality(self, measures: List[str] = None) -> dict[str, dict]:
        """
        Analyze network centrality measures

        Args:
            measures (List[str]): List of centrality measures to calculate

        Returns:
            dict[str, dict]: Centrality measures for each node
        """
        if self.network_graph is None:
            self.build_network()

        if measures is None:
            measures = ["degree", "betweenness", "closeness", "eigenvector"]

        print(f"Calculating centrality measures: {measures}")

        centrality_results = {}

        for measure in measures:
            print(f"  Computing {measure} centrality...")

            if measure == "degree":
                centrality_results[measure] = dict(self.network_graph.degree())

            elif measure == "betweenness":
                centrality_results[measure] = nx.betweenness_centrality(
                    self.network_graph, weight="weight"
                )

            elif measure == "closeness":
                centrality_results[measure] = nx.closeness_centrality(
                    self.network_graph, distance="weight"
                )

            elif measure == "eigenvector":
                try:
                    centrality_results[measure] = nx.eigenvector_centrality(
                        self.network_graph, weight="weight", max_iter=1000
                    )
                except Exception as e:
                    print(f"Error computing eigenvector centrality: {e}")
                    print(f"Warning: Could not compute {measure} centrality")
                    centrality_results[measure] = {}

            elif measure == "pagerank":
                centrality_results[measure] = nx.pagerank(
                    self.network_graph, weight="weight"
                )

        self.centrality_measures = centrality_results
        print("✓ Centrality analysis completed")
        return centrality_results

    def get_route_paths(self, route_id: str) -> List[List[str]]:
        """
        Get all possible paths for a specific route

        Args:
            route_id (str): Route ID to analyze

        Returns:
            List[List[str]]: List of paths (each path is a list of stop IDs)
        """
        if self.gtfs_processor.trips is None or self.gtfs_processor.stop_times is None:
            return []

        # Get all trips for this route
        route_trips = self.gtfs_processor.trips[
            self.gtfs_processor.trips["route_id"] == route_id
        ]["trip_id"].tolist()

        paths = []
        for trip_id in route_trips:
            trip_stops = self.gtfs_processor.stop_times[
                self.gtfs_processor.stop_times["trip_id"] == trip_id
            ].sort_values("stop_sequence")

            path = trip_stops["stop_id"].tolist()
            if path and path not in paths:
                paths.append(path)

        return paths

    def find_alternative_routes(
        self, start_stop: str, end_stop: str, max_routes: int = 3
    ) -> List[Tuple[List[str], float]]:
        """
        Find multiple alternative routes between two stops

        Args:
            start_stop (str): Starting stop ID
            end_stop (str): Destination stop ID
            max_routes (int): Maximum number of routes to find

        Returns:
            List[Tuple[List[str], float]]: List of (path, cost) tuples
        """
        if self.network_graph is None:
            self.build_network()

        routes = []
        graph_copy = self.network_graph.copy()

        for i in range(max_routes):
            try:
                path, cost = self.find_shortest_path(start_stop, end_stop)
                if path:
                    routes.append((path, cost))

                    # Remove edges from this path to find alternative routes
                    for j in range(len(path) - 1):
                        if graph_copy.has_edge(path[j], path[j + 1]):
                            graph_copy.remove_edge(path[j], path[j + 1])

                    # Update network graph temporarily
                    self.network_graph = graph_copy
                else:
                    break
            except nx.NetworkXNoPath:
                print(f"No more paths found after {i} routes.")
            except Exception as e:
                print(f"Error finding alternative route {i + 1}: {e}")
                break

        # Restore original graph
        self.network_graph = graph_copy
        return routes

    def get_network_statistics(self) -> dict:
        """
        Get comprehensive network statistics

        Returns:
            dict: Network statistics
        """
        if self.network_graph is None:
            self.build_network()

        G = self.network_graph

        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "is_connected": nx.is_connected(G),
            "num_connected_components": nx.number_connected_components(G),
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "network_density": nx.density(G),
            "average_clustering": nx.average_clustering(G),
        }

        if nx.is_connected(G):
            stats["diameter"] = nx.diameter(G)
            stats["average_shortest_path_length"] = nx.average_shortest_path_length(G)

        return stats
