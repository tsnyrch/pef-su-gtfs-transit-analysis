import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


class DelayPredictor:
    """
    Machine learning model for predicting transit delays
    """

    def __init__(self, gtfs_processor):
        """
        Initialize DelayPredictor

        Args:
            gtfs_processor: GTFSProcessor instance with loaded data
        """
        self.gtfs_processor = gtfs_processor
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False

    def prepare_training_data(self, simulate_delays: bool = True) -> pd.DataFrame:
        """
        Prepare training data for delay prediction

        Args:
            simulate_delays (bool): Whether to simulate delay data

        Returns:
            pd.DataFrame: Prepared training data
        """
        print("Preparing training data for delay prediction...")

        # Get feature data from GTFS processor
        features_df = self.gtfs_processor.create_features()

        if simulate_delays:
            # Simulate realistic delay patterns
            np.random.seed(42)
            n_records = len(features_df)

            # Base delay simulation
            base_delays = np.random.exponential(scale=2.0, size=n_records)

            # Add weather effects (simulate)
            weather_effect = np.random.choice(
                [0, 1, 2], size=n_records, p=[0.7, 0.2, 0.1]
            )
            weather_delays = weather_effect * np.random.exponential(
                scale=3.0, size=n_records
            )

            # Add rush hour effects
            rush_hour_multiplier = features_df["is_rush_hour"].fillna(0) * 1.5 + 1

            # Add route type effects
            route_type_effects = {
                0: 1.2,  # Tram
                1: 0.8,  # Subway
                2: 1.5,  # Rail
                3: 1.3,  # Bus
            }
            route_multiplier = (
                features_df["route_type"].map(route_type_effects).fillna(1.0)
            )

            # Combine all delay factors
            total_delays = (
                (base_delays + weather_delays) * rush_hour_multiplier * route_multiplier
            )

            # Add some negative delays (early arrivals)
            early_probability = 0.15
            early_mask = np.random.random(n_records) < early_probability
            total_delays[early_mask] = -np.random.exponential(
                scale=1.0, size=np.sum(early_mask)
            )

            features_df["delay_minutes"] = total_delays
            features_df["weather_condition"] = weather_effect

        # Select and engineer features for training
        training_features = []

        # Time-based features
        if "hour" in features_df.columns:
            training_features.extend(["hour", "is_rush_hour"])

        # Route and trip features
        categorical_features = ["route_type", "time_period", "route_type_name"]
        for feature in categorical_features:
            if feature in features_df.columns:
                training_features.append(feature)

        # Stop sequence features
        sequence_features = ["stop_sequence", "is_first_stop", "is_last_stop"]
        for feature in sequence_features:
            if feature in features_df.columns:
                training_features.append(feature)

        # Add simulated features
        if simulate_delays:
            training_features.append("weather_condition")

        # Prepare final training dataset
        training_data = features_df[training_features + ["delay_minutes"]].copy()
        training_data = training_data.dropna()

        print(
            f"✓ Prepared training data with {len(training_data)} records and {len(training_features)} features"
        )
        return training_data

    def train_model(
        self, training_data: pd.DataFrame, model_type: str = "random_forest"
    ) -> dict:
        """
        Train delay prediction model

        Args:
            training_data (pd.DataFrame): Training dataset
            model_type (str): Type of model to train

        Returns:
            dict: Training results and metrics
        """
        print(f"Training {model_type} model for delay prediction...")

        # Separate features and target
        target_column = "delay_minutes"
        feature_columns = [col for col in training_data.columns if col != target_column]

        X = training_data[feature_columns].copy()
        y = training_data[target_column].copy()

        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=["object"]).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        # Scale numerical features
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        self.feature_columns = feature_columns

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
        }

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring="neg_mean_absolute_error"
        )
        metrics["cv_mae"] = -cv_scores.mean()
        metrics["cv_mae_std"] = cv_scores.std()

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            metrics["feature_importance"] = feature_importance.to_dict("records")

        self.is_trained = True
        print("✓ Model training completed")

        return metrics

    def predict_delay(self, stop_data: dict) -> float:
        """
        Predict delay for a given stop/trip

        Args:
            stop_data (dict): Stop and trip information

        Returns:
            float: Predicted delay in minutes
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")

        # Prepare input data
        input_df = pd.DataFrame([stop_data])

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value

        # Select and order features
        input_features = input_df[self.feature_columns].copy()

        # Encode categorical variables
        categorical_columns = input_features.select_dtypes(include=["object"]).columns

        for col in categorical_columns:
            if col in self.label_encoders:
                try:
                    input_features[col] = self.label_encoders[col].transform(
                        input_features[col].astype(str)
                    )
                except ValueError:
                    # Handle unseen categories
                    input_features[col] = 0

        # Scale numerical features
        numerical_columns = input_features.select_dtypes(include=[np.number]).columns
        input_features[numerical_columns] = self.scaler.transform(
            input_features[numerical_columns]
        )

        # Make prediction
        prediction = self.model.predict(input_features)[0]
        return prediction


class DemandForecaster:
    """
    Class for forecasting ridership demand
    """

    def __init__(self, gtfs_processor):
        """
        Initialize DemandForecaster

        Args:
            gtfs_processor: GTFSProcessor instance with loaded data
        """
        self.gtfs_processor = gtfs_processor
        self.ridership_data = None
        self.model = None
        self.is_trained = False

    def simulate_ridership_data(self, days: int = 30) -> pd.DataFrame:
        """
        Simulate realistic ridership data

        Args:
            days (int): Number of days to simulate

        Returns:
            pd.DataFrame: Simulated ridership data
        """
        print(f"Simulating ridership data for {days} days...")

        np.random.seed(42)
        ridership_records = []

        # Get stops and routes
        stops = (
            self.gtfs_processor.stops["stop_id"].tolist()
            if self.gtfs_processor.stops is not None
            else []
        )
        routes = (
            self.gtfs_processor.routes["route_id"].tolist()
            if self.gtfs_processor.routes is not None
            else []
        )

        if not stops or not routes:
            print("Warning: No stops or routes found, creating dummy data")
            stops = [f"stop_{i}" for i in range(20)]
            routes = [f"route_{i}" for i in range(5)]

        # Generate data for each day
        base_date = datetime(2024, 1, 1)

        for day in range(days):
            current_date = base_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            is_holiday = day % 30 == 0  # Simulate holidays

            # Daily patterns
            for hour in range(5, 24):  # Service hours
                for stop_id in stops[: min(len(stops), 50)]:  # Limit for performance
                    # Base ridership with patterns
                    base_ridership = self._get_base_ridership(
                        hour, is_weekend, is_holiday
                    )

                    # Add stop-specific factors
                    stop_factor = np.random.uniform(0.5, 2.0)

                    # Add noise
                    noise = np.random.normal(0, base_ridership * 0.2)

                    ridership = max(0, base_ridership * stop_factor + noise)

                    ridership_records.append(
                        {
                            "date": current_date.date(),
                            "hour": hour,
                            "stop_id": stop_id,
                            "route_id": np.random.choice(routes),
                            "ridership": int(ridership),
                            "is_weekend": is_weekend,
                            "is_holiday": is_holiday,
                            "day_of_week": current_date.weekday(),
                            "month": current_date.month,
                        }
                    )

        ridership_df = pd.DataFrame(ridership_records)
        self.ridership_data = ridership_df

        print(f"✓ Simulated {len(ridership_df)} ridership records")
        return ridership_df

    def _get_base_ridership(
        self, hour: int, is_weekend: bool, is_holiday: bool
    ) -> float:
        """Get base ridership for given conditions"""
        # Morning rush (7-9 AM)
        if 7 <= hour <= 9:
            base = 100 if not is_weekend else 30
        # Evening rush (5-7 PM)
        elif 17 <= hour <= 19:
            base = 90 if not is_weekend else 40
        # Midday
        elif 10 <= hour <= 16:
            base = 50 if not is_weekend else 35
        # Evening
        elif 20 <= hour <= 22:
            base = 40
        # Early morning/late evening
        else:
            base = 15

        # Reduce for holidays
        if is_holiday:
            base *= 0.6

        return base

    def forecast_ridership(self, forecast_days: int = 7) -> pd.DataFrame:
        """
        Forecast ridership for upcoming days

        Args:
            forecast_days (int): Number of days to forecast

        Returns:
            pd.DataFrame: Forecasted ridership data
        """
        print(f"Forecasting ridership for {forecast_days} days...")

        if self.ridership_data is None:
            self.simulate_ridership_data()

        # Simple time series forecasting using historical patterns
        historical_data = self.ridership_data.copy()

        # Calculate average ridership by hour, day of week, stop
        patterns = (
            historical_data.groupby(["hour", "day_of_week", "stop_id"])
            .agg({"ridership": ["mean", "std"]})
            .reset_index()
        )

        patterns.columns = [
            "hour",
            "day_of_week",
            "stop_id",
            "avg_ridership",
            "std_ridership",
        ]
        patterns["std_ridership"] = patterns["std_ridership"].fillna(
            patterns["avg_ridership"] * 0.2
        )

        # Generate forecasts
        forecast_records = []
        last_date = historical_data["date"].max()

        for day in range(1, forecast_days + 1):
            forecast_date = last_date + timedelta(days=day)
            day_of_week = forecast_date.weekday()
            is_weekend = day_of_week >= 5

            for hour in range(5, 24):
                for stop_id in historical_data["stop_id"].unique():
                    # Find matching pattern
                    pattern = patterns[
                        (patterns["hour"] == hour)
                        & (patterns["day_of_week"] == day_of_week)
                        & (patterns["stop_id"] == stop_id)
                    ]

                    if len(pattern) > 0:
                        avg_ridership = pattern["avg_ridership"].iloc[0]
                        std_ridership = pattern["std_ridership"].iloc[0]

                        # Add some randomness to forecast
                        forecast_ridership = max(
                            0, np.random.normal(avg_ridership, std_ridership * 0.5)
                        )

                        forecast_records.append(
                            {
                                "date": forecast_date,
                                "hour": hour,
                                "stop_id": stop_id,
                                "forecasted_ridership": int(forecast_ridership),
                                "confidence_lower": max(
                                    0, int(forecast_ridership - std_ridership)
                                ),
                                "confidence_upper": int(
                                    forecast_ridership + std_ridership
                                ),
                                "is_weekend": is_weekend,
                                "day_of_week": day_of_week,
                            }
                        )

        forecast_df = pd.DataFrame(forecast_records)
        print(f"✓ Generated {len(forecast_df)} ridership forecasts")

        return forecast_df

    def get_demand_patterns(self) -> dict:
        """
        Analyze demand patterns from ridership data

        Returns:
            dict: Demand pattern analysis
        """
        if self.ridership_data is None:
            self.simulate_ridership_data()

        patterns = {}

        # Hourly patterns
        hourly_pattern = self.ridership_data.groupby("hour")["ridership"].mean()
        patterns["hourly"] = hourly_pattern.to_dict()

        # Day of week patterns
        dow_pattern = self.ridership_data.groupby("day_of_week")["ridership"].mean()
        patterns["day_of_week"] = dow_pattern.to_dict()

        # Weekend vs weekday
        weekend_pattern = self.ridership_data.groupby("is_weekend")["ridership"].mean()
        patterns["weekend_vs_weekday"] = weekend_pattern.to_dict()

        # Top stops by ridership
        top_stops = (
            self.ridership_data.groupby("stop_id")["ridership"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        patterns["top_stops"] = top_stops.to_dict()

        # Peak hours
        peak_hours = hourly_pattern.nlargest(3)
        patterns["peak_hours"] = peak_hours.to_dict()

        return patterns
