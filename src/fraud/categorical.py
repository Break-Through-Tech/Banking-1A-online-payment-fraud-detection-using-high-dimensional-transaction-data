"""
Leakage-safe categorical encoder for the IEEE-CIS Fraud Detection dataset.

This module is intended for models that require explicit numerical
representations of categorical features, such as linear models and neural
networks. Tree models with native categorical support may use the same raw
features without this encoder.

The encoder expects a merged transaction + identity pandas DataFrame.
All data-dependent mappings are learned during fit() and should therefore
be fitted on the chronological training split only.
"""

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder:
    """Encode IEEE-CIS categorical features using training-only mappings."""

    def __init__(self):
        # Rare-category thresholds selected during exploratory analysis.
        self.transaction_group_cutoffs = {
            "card2": 100,
            "card3": 50,
            "card5": 50,
            "addr1": 50,
        }

        self.identity_group_cutoffs = {
            "id_17": 50,
            "DeviceInfo": 50,
        }

        # High-cardinality features that use frequency encoding.
        self.frequency_cols = [
            "card1",
            "id_19",
            "id_20",
            "id_21",
            "id_25",
            "id_26",
            "id_33",
        ]

        # Features that use one-hot encoding after grouping.
        self.one_hot_cols = [
            "ProductCD",
            "card4",
            "card6_grouped",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "card2_grouped",
            "card3_grouped",
            "card5_grouped",
            "addr1_grouped",
            "id_12",
            "id_13",
            "id_14",
            "id_15",
            "id_16",
            "id_18",
            "id_22",
            "id_23",
            "id_24",
            "id_27",
            "id_28",
            "id_29",
            "id_32",
            "id_34",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
            "DeviceType",
            "id_17_grouped",
            "id_30_grouped",
            "id_31_grouped",
            "DeviceInfo_grouped",
        ]

        # Raw columns the input DataFrame must contain.
        self.required_raw_cols = [
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "P_emaildomain",
            "R_emaildomain",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
            "id_12",
            "id_13",
            "id_14",
            "id_15",
            "id_16",
            "id_17",
            "id_18",
            "id_19",
            "id_20",
            "id_21",
            "id_22",
            "id_23",
            "id_24",
            "id_25",
            "id_26",
            "id_27",
            "id_28",
            "id_29",
            "id_30",
            "id_31",
            "id_32",
            "id_33",
            "id_34",
            "id_35",
            "id_36",
            "id_37",
            "id_38",
            "DeviceType",
            "DeviceInfo",
        ]

        self._reset_state()

    def _reset_state(self):
        """Reset objects learned during fit()."""
        self.common_values_ = {}
        self.frequency_maps_ = {}
        self.one_hot_encoder_ = None
        self.is_fitted_ = False

    def _load_dataset(self, dataset):
        """Validate and copy an input DataFrame."""
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")

        data = dataset.copy()
        self._validate_columns(data)

        return data

    def _validate_columns(self, data):
        """Check that all required raw categorical columns are available."""
        missing_cols = [
            col
            for col in self.required_raw_cols
            if col not in data.columns
        ]

        if missing_cols:
            raise ValueError(
                "The dataset is missing required columns: "
                + ", ".join(missing_cols)
                + ". The encoder expects the merged transaction "
                  "and identity dataset."
            )

    @staticmethod
    def _normalize_series(series):
        """Convert categorical values to consistent strings with explicit missing."""
        return (
            series.astype("object")
            .where(series.notna(), "MISSING")
            .astype(str)
        )

    @staticmethod
    def _group_os(value):
        """Group operating-system versions into broader families."""
        if pd.isna(value):
            return "MISSING"

        value = str(value).lower()

        if "windows" in value:
            return "Windows"

        if "ios" in value:
            return "iOS"

        if "mac" in value:
            return "MacOS"

        if "android" in value:
            return "Android"

        if "linux" in value:
            return "Linux"

        return "Other"

    @staticmethod
    def _group_browser(value):
        """Group browser versions into broader browser families."""
        if pd.isna(value):
            return "MISSING"

        value = str(value).lower()

        if "android webview" in value:
            return "Android WebView"

        if "samsung" in value:
            return "Samsung"

        if "chromium" in value:
            return "Chromium"

        if "chrome" in value:
            return "Chrome"

        if "safari" in value:
            return "Safari"

        if "firefox" in value:
            return "Firefox"

        if "edge" in value:
            return "Edge"

        if (
            value == "ie"
            or value.startswith("ie ")
            or "internet explorer" in value
        ):
            return "Internet Explorer"

        if (
            "android browser" in value
            or "generic/android" in value
            or value == "android"
        ):
            return "Android Browser"

        if (
            "google search application" in value
            or value == "google"
        ):
            return "Google Search"

        if "silk" in value:
            return "Silk"

        if "opera" in value:
            return "Opera"

        return "Other"

    @staticmethod
    def _group_card6(value):
        """Keep common card6 categories and group rare values."""
        if pd.isna(value):
            return "MISSING"

        value = str(value).lower()

        if value in {"debit", "credit"}:
            return value

        return "OTHER"

    def _apply_fixed_groups(self, data):
        """Apply grouping rules that do not need to be learned."""
        data = data.copy()

        data["id_30_grouped"] = (
            data["id_30"].apply(self._group_os)
        )

        data["id_31_grouped"] = (
            data["id_31"].apply(self._group_browser)
        )

        data["card6_grouped"] = (
            data["card6"].apply(self._group_card6)
        )

        return data

    def fit(self, dataset):
        """Learn all data-dependent mappings from the training split."""
        self._reset_state()

        data = self._load_dataset(dataset)
        data = self._apply_fixed_groups(data)

        all_cutoffs = {
            **self.transaction_group_cutoffs,
            **self.identity_group_cutoffs,
        }

        # Learn which categories are common enough to keep.
        for col, cutoff in all_cutoffs.items():
            values = self._normalize_series(data[col])

            counts = values.value_counts()

            self.common_values_[col] = set(
                counts[counts >= cutoff].index
            )

            data[f"{col}_grouped"] = values.where(
                values.eq("MISSING")
                | values.isin(self.common_values_[col]),
                "OTHER",
            )

        # Learn category frequencies from training only.
        for col in self.frequency_cols:
            values = self._normalize_series(data[col])

            self.frequency_maps_[col] = (
                values
                .value_counts(normalize=True)
                .to_dict()
            )

        # Prepare one-hot categorical data.
        one_hot_data = data[self.one_hot_cols].copy()

        for col in one_hot_data.columns:
            one_hot_data[col] = self._normalize_series(
                one_hot_data[col]
            )

        # Learn one-hot categories from training only.
        self.one_hot_encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )

        self.one_hot_encoder_.fit(one_hot_data)

        self.is_fitted_ = True

        return self

    def transform(self, dataset):
        """Encode data using mappings learned during fit()."""
        if not self.is_fitted_:
            raise ValueError(
                "Encoder must be fitted on training data "
                "before transform() is called."
            )

        data = self._load_dataset(dataset)
        data = self._apply_fixed_groups(data)

        all_cutoffs = {
            **self.transaction_group_cutoffs,
            **self.identity_group_cutoffs,
        }

        # Apply rare-category rules learned during fit().
        for col in all_cutoffs:
            values = self._normalize_series(data[col])

            data[f"{col}_grouped"] = values.where(
                values.eq("MISSING")
                | values.isin(self.common_values_[col]),
                "OTHER",
            )

        # Apply training frequency mappings.
        frequency_data = pd.DataFrame(index=data.index)

        for col in self.frequency_cols:
            values = self._normalize_series(data[col])

            frequency_data[f"{col}_freq"] = (
                values
                .map(self.frequency_maps_[col])
                .fillna(0.0)
            )

        # Prepare one-hot data.
        one_hot_data = data[self.one_hot_cols].copy()

        for col in one_hot_data.columns:
            one_hot_data[col] = self._normalize_series(
                one_hot_data[col]
            )

        # Apply one-hot categories learned during training.
        one_hot_encoded = (
            self.one_hot_encoder_.transform(one_hot_data)
        )

        # Convert frequency features to sparse format.
        frequency_encoded = csr_matrix(
            frequency_data.to_numpy(dtype=float)
        )

        # Combine one-hot and frequency features.
        encoded_data = hstack(
            [one_hot_encoded, frequency_encoded],
            format="csr",
        )

        return encoded_data

    def fit_transform(self, dataset):
        """Fit on training data and return its encoded features."""
        return self.fit(dataset).transform(dataset)

    def get_feature_names(self):
        """Return the names of all encoded categorical features."""
        if not self.is_fitted_:
            raise ValueError(
                "Encoder must be fitted before "
                "getting feature names."
            )

        one_hot_names = list(
            self.one_hot_encoder_.get_feature_names_out(
                self.one_hot_cols
            )
        )

        frequency_names = [
            f"{col}_freq"
            for col in self.frequency_cols
        ]

        return one_hot_names + frequency_names

    def get_feature_names_out(self):
        """Scikit-learn-style alias for get_feature_names()."""
        return self.get_feature_names()