import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    def __init__(
        self,
        categorical_columns=None,
        columns_to_drop=None,
        drop_missing_values=False,
        merge_key=None,
        normalize=True,
        target_column="target",
    ):
        """
        Initialize the DataProcessor with specific parameters for preprocessing.

        Args:
            categorical_columns (list of str): List of column names to treat as categorical.
            columns_to_drop (list of str): List of column names to drop from the dataset.
            drop_missing_values (bool): Flag to determine if missing values should be dropped.
            merge_key (str): Column name to use as a key when merging data with metadata.
            normalize (bool): Flag to determine if numerical features should be normalized.
            target_column (str): Name of the target column in the dataset.
        """
        self.categorical_columns = categorical_columns
        self.columns_to_drop = columns_to_drop
        self.drop_missing_values = drop_missing_values
        self.merge_key = merge_key
        self.normalize = normalize
        self.target_column = target_column

    def preprocess_data(self, data, index_col=None, metadata=None):
        """
        Load and preprocess data from a CSV file or DataFrame, with optional metadata merging.

        Args:
            data (str or DataFrame): Path to the CSV file or a pandas DataFrame.
            index_col (str, optional): Column to set as index. Defaults to None.
            metadata (str or DataFrame, optional): Path to the CSV file or DataFrame containing metadata. Defaults to None.

        Returns:
            DataFrame: The preprocessed data as a pandas DataFrame.
        """
        # Load the main data
        data_df = self._load_data(data, index_col)

        # If metadata is provided, load and merge it
        if metadata is not None:
            meta_df = self._load_data(metadata, index_col)
            data_df = self._merge_data_and_metadata(data_df, meta_df)

        for step in [
            (self.columns_to_drop, self._drop_columns),
            (self.drop_missing_values, self._drop_missing_values),
            (self.categorical_columns, self._encode_categorical_variables),
            (self.normalize, self._scale_numerical_features),
            (self.target_column, self._rename_target_column),
        ]:
            condition, method = step
            if condition:
                data_df = method(data_df)

        return data_df

    def _load_data(self, data, index_col=None):
        """
        Helper method to load data and set the index if specified.

        Args:
            data (str or DataFrame): Path to the CSV file or a pandas DataFrame.
            index_col (str, optional): Column to set as index. Defaults to None.

        Returns:
            DataFrame: The loaded pandas DataFrame with index set if specified.
        """
        # Check if 'data' is a file path (string) or a pandas DataFrame
        if isinstance(data, str):
            df = pd.read_csv(data)  # Load from CSV file if it's a string path
        elif isinstance(data, pd.DataFrame):
            df = data  # Use directly if it's already a DataFrame
        else:
            raise ValueError(
                "Input data must be a file path (str) or a pandas DataFrame"
            )

        # Set index column if specified
        if index_col is not None:
            df.set_index(index_col, inplace=True)

        return df

    def _merge_data_and_metadata(self, data_df, meta_df):
        """
        Merge the main data frame with metadata.

        Args:
            data_df (DataFrame): The main data DataFrame.
            meta_df (DataFrame): The metadata DataFrame.

        Returns:
            DataFrame: The merged DataFrame.
        """
        return pd.merge(data_df, meta_df, on=self.merge_key)

    def _rename_target_column(self, data_df):
        """
        Rename the target column in the data frame to 'target'.

        Args:
            data_df (DataFrame): The data DataFrame to be modified.

        Returns:
            DataFrame: The DataFrame with the renamed target column.
        """
        data_df.rename(columns={self.target_column: "target"}, inplace=True)
        self.target_column = "target"
        return data_df

    def _drop_columns(self, data_df):
        """
        Drop specified columns from the data frame.

        Args:
            data_df (DataFrame): The data DataFrame to be modified.

        Returns:
            DataFrame: The DataFrame with specified columns dropped.
        """
        data_df.drop(columns=self.columns_to_drop, inplace=True, errors="ignore")
        return data_df

    def _drop_missing_values(self, data_df):
        """
        Drop missing values by dropping rows with NaNs.

        Args:
            data_df (DataFrame): The data DataFrame with missing values.

        Returns:
            DataFrame: The DataFrame with missing values dropped.
        """
        return data_df.dropna()

    def _encode_categorical_variables(self, data_df):
        """
        Encode categorical variables using label encoding and store the mappings.

        Args:
            data_df (DataFrame): The data DataFrame with categorical columns.

        Returns:
            DataFrame: The DataFrame with categorical variables encoded.
        """
        if self.categorical_columns is None:
            return data_df
        else:
            self.label_encoders = (
                {}
            )  # Dictionary to store the label encoders for each column
            for col in self.categorical_columns:
                if col in data_df.columns:
                    label_encoder = LabelEncoder()
                    data_df[col] = label_encoder.fit_transform(data_df[col])
                    self.label_encoders[col] = (
                        label_encoder  # Store the label encoder for this column
                    )
            return data_df

    def get_label_mapping(self, column_name):
        """
        Retrieve the label encoding mapping for a specific column.

        Args:
            column_name (str): The column for which to get the label encoding mapping.

        Returns:
            dict: A dictionary mapping original labels to encoded values.
        """
        if column_name in self.label_encoders:
            label_encoder = self.label_encoders[column_name]
            return dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        else:
            raise ValueError(f"No label encoder found for column: {column_name}")

    def _scale_numerical_features(self, data_df):
        """
        Scale numerical features using standard scaling.

        Args:
            data_df (DataFrame): The data DataFrame with numerical columns.

        Returns:
            DataFrame: The DataFrame with numerical features scaled.
        """
        # Treat self.categorical_columns as an empty list if it is None
        categorical_cols = (
            self.categorical_columns if self.categorical_columns is not None else []
        )

        # Identify numerical columns
        numerical_cols = [col for col in data_df.columns if col not in categorical_cols]

        # Initialize the scaler and scale numerical features
        scaler = StandardScaler()
        data_df[numerical_cols] = scaler.fit_transform(data_df[numerical_cols])

        return data_df

    def _filtered_time_dataset(self, data_df, min_num_timepoints, clone_column):
        """
        Filter dataset to retain only clones with at least min_num_timepoints.

        Parameters:
        - data_df: DataFrame containing the dataset.
        - min_num_timepoints: Minimum number of time points required per clone.
        - clone_column: Column name for the clone identifier.

        Returns:
        - filtered_df: DataFrame with clones filtered based on time points.
        """
        # Filter clones based on the minimum number of time points
        filtered_df = data_df.groupby(clone_column).filter(
            lambda x: len(x) >= min_num_timepoints
        )
        return filtered_df.sort_values(clone_column)

    def _fill_nan(self, df, method="mean"):
        """
        Fill NaN values in the DataFrame based on the specified method.

        Parameters:
        - df: The DataFrame to fill NaN values in.
        - method: The strategy to use for filling NaNs. The "mean" method is implemented
                for both numeric (mean) and categorical (mode) columns.

        Returns:
        - df: The DataFrame with NaN values filled.
        """
        if method == "mean":
            # Fill NaN for numeric columns using the column mean
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # Fill NaN for categorical columns using the mode
            categorical_cols = df.select_dtypes(include="category").columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        else:
            raise ValueError(f"Unknown method: {method}")

        return df

    def flatten_time(
        self,
        data_df,
        clone_column,
        time_column,
        time_dependent_columns,
        min_num_timepoints=None,
        fill_nan_method="mean",
    ):
        """
        Flatten dataset based on time-dependent columns, optionally filtering by minimum time points and filling NaNs.

        Parameters:
        - data_df: DataFrame containing the dataset.
        - clone_column: Column name for the clone identifier.
        - time_column: Column name for the time variable.
        - time_dependent_columns: List of columns that vary with time.
        - min_num_timepoints: Optional minimum number of time points per clone for filtering.
        - fill_nan_method: Method to fill NaN values. Defaults to "mean".

        Returns:
        - flattened_df: DataFrame where time-dependent columns are pivoted and flattened by clone, with NaN values filled.
        """
        # Filter if minimum number of time points is specified
        if min_num_timepoints is not None:
            data_df = self._filtered_time_dataset(
                data_df, min_num_timepoints, clone_column
            )

        flattened_data = []

        # reverse mapping for TIMEPOINT
        mapping = {v: k for k, v in self.get_label_mapping("TIMEPOINT").items()}
        data_df["TIMEPOINT"] = data_df["TIMEPOINT"].map(mapping)

        # Process each unique clone
        for clone, clone_df in data_df.groupby(clone_column):
            # Reshape the clone-specific data from wide to long format
            melted_df = clone_df.melt(
                id_vars=[clone_column, time_column],
                value_vars=time_dependent_columns,
                var_name="VARIABLE",
                value_name="VALUE",
            )

            # Create a new column combining time and variable names
            melted_df["time_var"] = (
                melted_df[time_column].astype(str) + "_" + melted_df["VARIABLE"]
            )

            # Pivot the table so each time-variable combination becomes a column
            pivoted_df = melted_df.pivot(
                index=clone_column, columns="time_var", values="VALUE"
            )
            flattened_data.append(pivoted_df)

        # Concatenate the flattened data for all clones
        flattened_df = pd.concat(flattened_data)

        # Merge the target values back into the flattened DataFrame
        target_df = data_df[[clone_column, self.target_column]].drop_duplicates()
        flattened_df = flattened_df.reset_index()

        # Merge by clone_column and reset index
        flattened_df = (
            pd.merge(flattened_df, target_df, on=clone_column)
            .set_index(clone_column)
            .sort_index()
        )

        # Drop rows with missing target values
        flattened_df = flattened_df.dropna(subset=[self.target_column])

        # Fill NaN values using the specified method
        flattened_df = self._fill_nan(flattened_df, method=fill_nan_method)

        return flattened_df
