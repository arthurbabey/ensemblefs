from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from typing import Any

class DataProcessor:
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        columns_to_drop: Optional[List[str]] = None,
        drop_missing_values: bool = False,
        merge_key: Optional[str] = None,
        normalize: bool = True,
        target_column: str = "target",
    ) -> None:
        """
        Initialize the DataProcessor with specific parameters for preprocessing.

        Args:
            categorical_columns: List of column names to treat as categorical.
            columns_to_drop: List of column names to drop from the dataset.
            drop_missing_values: Flag to determine if missing values should be dropped.
            merge_key: Column name to use as a key when merging data with metadata.
            normalize: Flag to determine if numerical features should be normalized.
            target_column: Name of the target column in the dataset.
        """
        self.categorical_columns: Optional[List[str]] = categorical_columns
        self.columns_to_drop: Optional[List[str]] = columns_to_drop
        self.drop_missing_values: bool = drop_missing_values
        self.merge_key: Optional[str] = merge_key
        self.normalize: bool = normalize
        self.target_column: str = target_column
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def preprocess_data(
        self,
        data: Union[str, pd.DataFrame],
        index_col: Optional[str] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Load and preprocess data from a CSV file or DataFrame, with optional metadata merging.

        Args:
            data: Path to the CSV file or a pandas DataFrame.
            index_col: Column to set as index. Defaults to None.
            metadata: Path to the CSV file or DataFrame containing metadata. Defaults to None.

        Returns:
            The preprocessed data as a pandas DataFrame.
        """
        data_df = self._load_data(data, index_col)

        if metadata is not None:
            meta_df = self._load_data(metadata, index_col)
            data_df = self._merge_data_and_metadata(data_df, meta_df)

        for condition, method in [
            (self.columns_to_drop, self._drop_columns),
            (self.drop_missing_values, self._drop_missing_values),
            (self.categorical_columns, self._encode_categorical_variables),
            (self.normalize, self._scale_numerical_features),
            (self.target_column, self._rename_target_column),
        ]:
            if condition:
                data_df = method(data_df)
        return data_df

    def _load_data(
        self, data: Union[str, pd.DataFrame], index_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Helper method to load data and set the index if specified.

        Args:
            data: Path to the CSV file or a pandas DataFrame.
            index_col: Column to set as index. Defaults to None.

        Returns:
            The loaded pandas DataFrame with index set if specified.
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(
                "Input data must be a file path (str) or a pandas DataFrame"
            )

        if index_col is not None:
            df.set_index(index_col, inplace=True)
        return df

    def _merge_data_and_metadata(
        self, data_df: pd.DataFrame, meta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge the main data frame with metadata.

        Args:
            data_df: The main data DataFrame.
            meta_df: The metadata DataFrame.

        Returns:
            The merged DataFrame.
        """
        if not self.merge_key:
            raise ValueError("merge_key must be provided for merging data and metadata")
        return pd.merge(data_df, meta_df, on=self.merge_key)

    def _rename_target_column(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename the target column in the data frame to 'target'.

        Args:
            data_df: The data DataFrame to be modified.

        Returns:
            The DataFrame with the renamed target column.
        """
        data_df.rename(columns={self.target_column: "target"}, inplace=True)
        self.target_column = "target"
        return data_df

    def _drop_columns(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns from the data frame.

        Args:
            data_df: The data DataFrame to be modified.

        Returns:
            The DataFrame with specified columns dropped.
        """
        if self.columns_to_drop:
            data_df.drop(columns=self.columns_to_drop, inplace=True, errors="ignore")
        return data_df

    def _drop_missing_values(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop missing values by dropping rows with NaNs.

        Args:
            data_df: The data DataFrame with missing values.

        Returns:
            The DataFrame with missing values dropped.
        """
        return data_df.dropna()

    def _encode_categorical_variables(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding and store the mappings.

        Args:
            data_df: The data DataFrame with categorical columns.

        Returns:
            The DataFrame with categorical variables encoded.
        """
        if not self.categorical_columns:
            return data_df

        for col in self.categorical_columns:
            if col in data_df.columns:
                label_encoder = LabelEncoder()
                data_df[col] = label_encoder.fit_transform(data_df[col])
                self.label_encoders[col] = label_encoder
        return data_df

    def get_label_mapping(self, column_name: str) -> Dict[str, int]:
        """
        Retrieve the label encoding mapping for a specific column.

        Args:
            column_name: The column for which to get the label encoding mapping.

        Returns:
            A dictionary mapping original labels to encoded values.
        """
        if column_name in self.label_encoders:
            label_encoder = self.label_encoders[column_name]
            return dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        else:
            raise ValueError(f"No label encoder found for column: {column_name}")

    def _scale_numerical_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using standard scaling.

        Args:
            data_df: The data DataFrame with numerical columns.

        Returns:
            The DataFrame with numerical features scaled.
        """
        categorical_cols = self.categorical_columns if self.categorical_columns else []
        numerical_cols = [col for col in data_df.columns if col not in categorical_cols]
        scaler = StandardScaler()
        data_df[numerical_cols] = scaler.fit_transform(data_df[numerical_cols])
        return data_df

    def _filtered_time_dataset(
        self, data_df: pd.DataFrame, min_num_timepoints: int, clone_column: str
    ) -> pd.DataFrame:
        """
        Filter dataset to retain only clones with at least min_num_timepoints.

        Args:
            data_df: DataFrame containing the dataset.
            min_num_timepoints: Minimum number of time points required per clone.
            clone_column: Column name for the clone identifier.

        Returns:
            DataFrame with clones filtered based on time points.
        """
        filtered_df = data_df.groupby(clone_column).filter(
            lambda x: len(x) >= min_num_timepoints
        )
        return filtered_df.sort_values(clone_column)

    def _fill_nan(
        self,
        df: pd.DataFrame,
        method: str = "mean",
        **knn_kwargs: Any,          # forwarded only if method == "knn"
    ) -> pd.DataFrame:
        """
        Fill NaN values in *df* according to *method*.

        Parameters
        ----------
        df : pd.DataFrame
            The data whose missing values should be filled.
        method : {"mean", "knn"}, default "mean"
            Imputation strategy:
            - "mean" : column-wise mean for numeric, mode for categoricals.
            - "knn"  : KNNImputer for numeric, mode for categoricals.
        **knn_kwargs : Any
            Extra keyword arguments passed straight to
            ``sklearn.impute.KNNImputer`` when *method* == "knn".
            Example: ``n_neighbors=5, weights="distance"``.

        Returns
        -------
        pd.DataFrame
            A copy of *df* with NaNs imputed.
        """
        df = df.copy()  # avoid mutating the caller’s frame

        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(include="category").columns

        if method == "mean":
            # numeric
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "knn":
            # numeric via sklearn KNN
            if numeric_cols.empty:
                raise ValueError("KNN imputation requires at least one numeric column.")
            imputer = KNNImputer(**knn_kwargs)
            df[numeric_cols] = pd.DataFrame(
                imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols,
                index=df.index,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # categoricals: always use mode (most frequent)
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

        return df

    def flatten_time(
        self,
        data_df: pd.DataFrame,
        clone_column: str,
        time_column: str,
        time_dependent_columns: List[str],
        min_num_timepoints: Optional[int] = None,
        fill_nan_method: str = "mean",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Flatten dataset based on time-dependent columns, optionally filtering by minimum time points and filling NaNs.

        Args:
            data_df: DataFrame containing the dataset.
            clone_column: Column name for the clone identifier.
            time_column: Column name for the time variable.
            time_dependent_columns: List of columns that vary with time.
            min_num_timepoints: Optional minimum number of time points per clone for filtering.
            fill_nan_method: Method to fill NaN values. Defaults to "mean".

        Returns:
            DataFrame where time-dependent columns are pivoted and flattened by clone, with NaN values filled.
        """
        if min_num_timepoints is not None:
            data_df = self._filtered_time_dataset(
                data_df, min_num_timepoints, clone_column
            )

        flattened_data = []
        # Reverse mapping for TIMEPOINT
        mapping = {v: k for k, v in self.get_label_mapping("TIMEPOINT").items()}
        data_df["TIMEPOINT"] = data_df["TIMEPOINT"].map(mapping)

        for clone, clone_df in data_df.groupby(clone_column):
            melted_df = clone_df.melt(
                id_vars=[clone_column, time_column],
                value_vars=time_dependent_columns,
                var_name="VARIABLE",
                value_name="VALUE",
            )
            melted_df["time_var"] = (
                melted_df[time_column].astype(str) + "_" + melted_df["VARIABLE"]
            )
            pivoted_df = melted_df.pivot(
                index=clone_column, columns="time_var", values="VALUE"
            )
            flattened_data.append(pivoted_df)

        flattened_df = pd.concat(flattened_data)
        target_df = data_df[[clone_column, self.target_column]].drop_duplicates()
        flattened_df = flattened_df.reset_index()
        flattened_df = (
            pd.merge(flattened_df, target_df, on=clone_column)
            .set_index(clone_column)
            .sort_index()
        )
        flattened_df = flattened_df.dropna(subset=[self.target_column])
        flattened_df = self._fill_nan(flattened_df, fill_nan_method, **kwargs)
        return flattened_df
