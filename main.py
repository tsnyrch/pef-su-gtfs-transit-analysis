#!/usr/bin/env python3
"""
GTFS Transit Analysis - Main Runner Script

This script provides a command-line interface for running GTFS analysis.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.data_processing import GTFSProcessor
from src.prediction import DelayPredictor, DemandForecaster
from src.routing import TransitRouter
from src.visualization import TransitVisualizer

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="GTFS Transit Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data-path data/ --full-analysis
  python main.py --data-path data/ --network-only
  python main.py --data-path data/ --predict-delays
        """,
    )

    parser.add_argument(
        "--data-path",
        default="data/",
        help="Path to GTFS data directory (default: data/)",
    )

    parser.add_argument(
        "--output-path",
        default="outputs/",
        help="Path to output directory (default: outputs/)",
    )

    parser.add_argument(
        "--full-analysis", action="store_true", help="Run complete analysis pipeline"
    )

    parser.add_argument(
        "--network-only", action="store_true", help="Run only network analysis"
    )

    parser.add_argument(
        "--predict-delays", action="store_true", help="Run only delay prediction"
    )

    parser.add_argument(
        "--forecast-demand", action="store_true", help="Run only demand forecasting"
    )

    parser.add_argument(
        "--create-dashboard", action="store_true", help="Create interactive dashboard"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Validate paths
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        sys.exit(1)

    # Create output directory
    Path(args.output_path).mkdir(exist_ok=True)

    logger.info("🚊 Starting GTFS Transit Analysis")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")

    try:
        # Initialize GTFS processor
        logger.info("📂 Loading GTFS data...")
        gtfs = GTFSProcessor(args.data_path)
        data = gtfs.load_data()

        if not data:
            logger.error("No GTFS data loaded. Check data directory.")
            sys.exit(1)

        # Get basic statistics
        stats = gtfs.get_summary_statistics()
        logger.info(f"📊 Dataset summary: {len(data)} files loaded")
        for key, value in stats.items():
            if not isinstance(value, dict):
                logger.info(f"  {key}: {value:,}")

        # Initialize components based on arguments
        router = None
        delay_predictor = None
        demand_forecaster = None
        visualizer = None

        if args.network_only or args.full_analysis:
            logger.info("🗺️ Running network analysis...")
            router = TransitRouter(gtfs)

            # Analyze centrality
            centrality = router.analyze_centrality(["degree", "betweenness"])

            # Get network stats
            network_stats = router.get_network_statistics()
            logger.info(
                f"Network: {network_stats['num_nodes']} nodes, {network_stats['num_edges']} edges"
            )

            # Find important stops
            if "degree" in centrality:
                top_stops = sorted(
                    centrality["degree"].items(), key=lambda x: x[1], reverse=True
                )[:5]
                logger.info("Top 5 stops by degree centrality:")
                for i, (stop_id, score) in enumerate(top_stops, 1):
                    logger.info(f"  {i}. {stop_id}: {score}")

        if args.predict_delays or args.full_analysis:
            logger.info("🤖 Training delay prediction model...")
            delay_predictor = DelayPredictor(gtfs)

            # Prepare training data
            training_data = delay_predictor.prepare_training_data(simulate_delays=True)
            logger.info(f"Training data: {len(training_data):,} samples")

            # Train model
            results = delay_predictor.train_model(training_data)
            logger.info(
                f"Model performance - Test R²: {results['test_r2']:.3f}, MAE: {results['test_mae']:.2f} min"
            )

            # Demo prediction
            demo_scenario = {
                "hour": 8,
                "is_rush_hour": 1,
                "route_type": 3,
                "stop_sequence": 5,
                "weather_condition": 0,
            }
            predicted_delay = delay_predictor.predict_delay(demo_scenario)
            logger.info(
                f"Demo prediction (rush hour bus): {predicted_delay:.1f} min delay"
            )

        if args.forecast_demand or args.full_analysis:
            logger.info("📈 Running demand forecasting...")
            demand_forecaster = DemandForecaster(gtfs)

            # Simulate ridership data
            ridership_data = demand_forecaster.simulate_ridership_data(days=30)
            logger.info(f"Simulated ridership: {len(ridership_data):,} records")

            # Generate forecast
            forecast = demand_forecaster.forecast_ridership(forecast_days=7)
            total_forecast = (
                forecast.groupby("date")["forecasted_ridership"].sum().sum()
            )
            logger.info(f"7-day ridership forecast: {total_forecast:,.0f} passengers")

        if args.create_dashboard or args.full_analysis:
            logger.info("📊 Creating visualizations...")
            visualizer = TransitVisualizer(gtfs, router)

            # Create dashboard
            dashboard = visualizer.create_interactive_dashboard(
                delay_predictor=delay_predictor, demand_forecaster=demand_forecaster
            )

            # Save dashboard
            dashboard_path = os.path.join(args.output_path, "transit_dashboard.html")
            visualizer.save_dashboard_html(dashboard, dashboard_path)
            logger.info(f"Dashboard saved: {dashboard_path}")

            # Create network map if possible
            try:
                network_map = visualizer.plot_network_map(interactive=True)
                if network_map:
                    map_path = os.path.join(args.output_path, "network_map.html")
                    network_map.save(map_path)
                    logger.info(f"Network map saved: {map_path}")
            except Exception as e:
                logger.warning(f"Could not create network map: {e}")

        # Summary report
        logger.info("✅ Analysis complete!")

        if args.full_analysis:
            logger.info("\n📋 Summary Report:")
            logger.info("=" * 50)

            if stats:
                logger.info(f"Routes analyzed: {stats.get('num_routes', 'N/A')}")
                logger.info(f"Stops analyzed: {stats.get('num_stops', 'N/A')}")
                logger.info(f"Trips analyzed: {stats.get('num_trips', 'N/A')}")

            if router:
                network_stats = router.get_network_statistics()
                logger.info(
                    f"Network connectivity: {'Connected' if network_stats.get('is_connected') else 'Disconnected'}"
                )
                logger.info(
                    f"Network density: {network_stats.get('network_density', 0):.4f}"
                )

            if delay_predictor:
                logger.info(f"Delay model accuracy: {results['test_r2']:.3f} R²")

            if demand_forecaster:
                patterns = demand_forecaster.get_demand_patterns()
                peak_hour = max(patterns["hourly"].items(), key=lambda x: x[1])
                logger.info(
                    f"Peak ridership hour: {peak_hour[0]}:00 ({peak_hour[1]:.0f} avg passengers)"
                )

            logger.info(f"\nOutputs saved to: {args.output_path}")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
