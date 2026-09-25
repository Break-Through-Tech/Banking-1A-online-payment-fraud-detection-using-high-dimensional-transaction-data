"""
Reusable categorical encoder for the IEEE-CIS Fraud Detection dataset.

Input can be either:
1. a pandas DataFrame, or
2. a path to a CSV file

The input dataset should contain the merged raw transaction and identity
categorical columns.
"""

from pathlib import Path

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
            "addr1": 50
        }

        self.identity_group_cutoffs = {
            "id_17": 50,
            "DeviceInfo": 50
        }

        # High-cardinality features that use frequency encoding.
        self.frequency_cols = [
            "card1",
            "id_19",
            "id_20",
            "id_21",
            "id_25",
            "id_26",
            "id_33"
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
            "DeviceInfo_grouped"
        ]

        # Raw columns the input dataset must contain.
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
            "DeviceInfo"
        ]

        # Objects learned only during fit().
        self.common_values_ = {}
        self.frequency_maps_ = {}
        self.one_hot_encoder_ = None
        self.is_fitted_ = False


    def _load_dataset(self, dataset):
        """Accept either a DataFrame or a CSV filepath."""

        if isinstance(dataset, pd.DataFrame):
            data = dataset.copy()

        elif isinstance(dataset, (str, Path)):
            data = pd.read_csv(dataset)

        else:
            raise TypeError(
                "dataset must be a pandas DataFrame or a CSV filepath."
            )

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

        if "samsung" in value:
            return "Samsung"

        if "chrome" in value and "chromium" not in value:
            return "Chrome"

        if "chromium" in value:
            return "Chromium"

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

        if "android webview" in value:
            return "Android WebView"

        return "Other"


    @staticmethod
    def _group_card6(value):
        """Keep common card6 categories and group rare values."""

        if pd.isna(value):
            return "MISSING"

        if value in ["debit", "credit"]:
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
        """
        Learn all data-dependent mappings from the training split.

        dataset may be:
        - a pandas DataFrame
        - a path to a CSV file
        """

        data = self._load_dataset(dataset)
        data = self._apply_fixed_groups(data)

        all_cutoffs = {
            **self.transaction_group_cutoffs,
            **self.identity_group_cutoffs
        }

        # Learn which categories are common enough to keep.
        for col, cutoff in all_cutoffs.items():

            counts = data[col].value_counts(dropna=True)

            self.common_values_[col] = set(
                counts[counts >= cutoff].index
            )

            data[f"{col}_grouped"] = data[col].apply(
                lambda x: (
                    "MISSING"
                    if pd.isna(x)
                    else str(x)
                    if x in self.common_values_[col]
                    else "OTHER"
                )
            )

        # Learn category frequencies from training only.
        for col in self.frequency_cols:

            values = (
                data[col]
                .astype("object")
                .where(data[col].notna(), "MISSING")
                .astype(str)
            )

            self.frequency_maps_[col] = (
                values
                .value_counts(normalize=True)
                .to_dict()
            )

        # Prepare one-hot categorical data.
        one_hot_data = data[self.one_hot_cols].copy()

        for col in one_hot_data.columns:

            one_hot_data[col] = (
                one_hot_data[col]
                .astype("object")
                .where(one_hot_data[col].notna(), "MISSING")
                .astype(str)
            )

        # Learn one-hot categories from training only.
        self.one_hot_encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )

        self.one_hot_encoder_.fit(one_hot_data)

        self.is_fitted_ = True

        return self


    def transform(self, dataset):
        """
        Encode a dataset using mappings learned during fit().

        This method does not learn or update any mappings.
        """

        if not self.is_fitted_:
            raise ValueError(
                "Encoder must be fitted on training data "
                "before transform() is called."
            )

        data = self._load_dataset(dataset)
        data = self._apply_fixed_groups(data)

        all_cutoffs = {
            **self.transaction_group_cutoffs,
            **self.identity_group_cutoffs
        }

        # Apply rare-category rules learned during fit().
        for col in all_cutoffs:

            data[f"{col}_grouped"] = data[col].apply(
                lambda x: (
                    "MISSING"
                    if pd.isna(x)
                    else str(x)
                    if x in self.common_values_[col]
                    else "OTHER"
                )
            )

        # Apply training frequency mappings.
        frequency_data = pd.DataFrame(index=data.index)

        for col in self.frequency_cols:

            values = (
                data[col]
                .astype("object")
                .where(data[col].notna(), "MISSING")
                .astype(str)
            )

            frequency_data[f"{col}_freq"] = (
                values
                .map(self.frequency_maps_[col])
                .fillna(0.0)
            )

        # Prepare one-hot data.
        one_hot_data = data[self.one_hot_cols].copy()

        for col in one_hot_data.columns:

            one_hot_data[col] = (
                one_hot_data[col]
                .astype("object")
                .where(one_hot_data[col].notna(), "MISSING")
                .astype(str)
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
            format="csr"
        )

        return encoded_data


    def fit_transform(self, dataset):
        """Fit on training data and return its encoded features."""

        self.fit(dataset)

        return self.transform(dataset)


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
