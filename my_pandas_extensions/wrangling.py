
# IMPORTS

import pandas as pd
import numpy as np
import janitor as jn
import pandas_flavor as pf
from typing import List, Optional

#---------------------------------------------------------------

# remove_outliers()
# required packages:
#   import pandas as pd
#   import numpy as np
#   import pandas_flavor as pf

@pf.register_dataframe_method
def remove_outliers(df, iqr_multiplier = 1.5, columns = None):
    """
    Identifies and replaces outliers with NaN in specified columns of a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to process.
    iqr_multiplier : float or int, optional
        A positive number used to define the interquartile range (IQR) multiplier for outlier detection. 
        Defaults to 1.5.
    columns : list, optional
        List of column names to process. If None, all numeric columns will be processed.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with outliers replaced by NaN in specified columns.

    Raises:
    -------
    TypeError
        If df is not a pandas DataFrame.
        If iqr_multiplier is not a positive float or integer.
        If columns is provided but is not a list.
    ValueError
        If any specified column is not in the DataFrame.
        If any specified column is not numeric.

    Examples:
    ---------
    
    remove_outliers(df, iqr_multiplier = 1.5, columns = ["price"])
    
    ---
    
    df.groupby("category_2").apply(
    lambda x: remove_outliers(x, columns=["total_price", "price"]))
    
    ---
    
    df.remove_outliers(iqr_multiplier = 1.5)
    
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if not isinstance(iqr_multiplier, (int, float)) or iqr_multiplier <= 0:
        raise TypeError("iqr_multiplier must be a positive number")

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    elif not isinstance(columns, list):
        raise TypeError("columns must be a list of column names")

    # Validate columns all at once
    invalid_columns = set(columns) - set(df.columns)
    if invalid_columns:
        raise ValueError(f"Columns {invalid_columns} not found in the DataFrame")
    
    non_numeric_columns = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_columns:
        raise ValueError(f"Columns {non_numeric_columns} are not numeric")

    # Calculate quantiles for all columns at once
    quantiles = df[columns].quantile([0.25, 0.75])
    
    for col in columns:
        q25, q75 = quantiles.loc[0.25, col], quantiles.loc[0.75, col]
        iqr = q75 - q25
        
        lower_limit = q25 - iqr_multiplier * iqr
        upper_limit = q75 + iqr_multiplier * iqr
        
        mask = (df[col] < lower_limit) | (df[col] > upper_limit)
        df.loc[mask, col] = np.nan
    
    return df

### For Series ###
# required packages:
#   import pandas as pd
#   import numpy as np
#   import pandas_flavor as pf

@pf.register_dataframe_method
def remove_outliers_series(self, iqr_multiplier = 1.5):
    """
    Identifies and removes outliers from a pandas Series.

    Parameters:
    -----------
    self : pd.Series
        The input Series to process.
    iqr_multiplier : float or int, optional
        A positive number used to define the interquartile range (IQR) multiplier for outlier detection. 
        Defaults to 1.5.

    Returns:
    --------
    pd.Series
        A Series with outliers replaced by NaN.
        
    Examples:
    ---------
    
    df["total_price"].remove_outliers()
    
    ---
    
    df[["total_price", "price"]] = df[["total_price", "price"]].apply(lambda x: x.remove_outliers())
    
    """
    q25 = self.quantile(0.25)
    q75 = self.quantile(0.75)
    iqr = q75 - q25
    
    lower_limit = q25 - iqr_multiplier * iqr
    upper_limit = q75 + iqr_multiplier * iqr
    
    return self.mask((self < lower_limit) | (self > upper_limit))

################################# END OF FUNCTION #################################

# wrg_cardinality_OnetoManyFeatures()
# required packages: 
#   import pandas as pd
#   import janitor as jn
#   import pandas_flavor as pf
#   from typing import List, Optional

@pf.register_dataframe_method
def wrg_cardinality_OnetoManyFeatures(
    df: pd.DataFrame,
    reduce_cardinality_cols: Optional[List[str]] = None,
    cardinality_query: str = "",
    one_to_many_features_categories: Optional[List[str]] = None,
    one_to_many_wide_index: str = ""
) -> pd.DataFrame:
    """
    Perform cardinality reduction and one-to-many feature expansion on a pandas DataFrame.

    This function has two main operations:
    1. Cardinality reduction: Reduces the number of unique values in specified columns.
    2. One-to-many feature expansion: Converts categorical columns into binary columns using one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame to be processed.
        reduce_cardinality_cols (Optional[List[str]], optional): List of column names for cardinality reduction. 
            If None, no cardinality reduction is performed. Defaults to None.
        cardinality_query (str, optional): Query string to filter rows for determining value sets in cardinality reduction. 
            Used only if reduce_cardinality_cols is not None. Defaults to an empty string.
        one_to_many_features_categories (Optional[List[str]], optional): List of column names for one-to-many feature expansion. 
            If None, no expansion is performed. Defaults to None.
        one_to_many_wide_index (str, optional): Name of the column to use as index when merging expanded features back to the original DataFrame. 
            Used only if one_to_many_features_categories is not None. Defaults to an empty string.

    Returns:
        pd.DataFrame: Processed DataFrame with reduced cardinality and/or expanded one-to-many features.

    Raises:
        ValueError: If reduce_cardinality_cols or one_to_many_features_categories is provided but is not a list.

    Notes:
        - For cardinality reduction, values not in the filtered set are replaced with "Other".
        - For one-to-many feature expansion, new binary columns are created with a prefix derived from the original column name.
        - The function assumes that all necessary columns exist in the input DataFrame.
        - The cardinality_query and one_to_many_wide_index parameters should be used carefully to ensure correct operation.

    Example:
    
    df_ = wrg_cardinality_OnetoManyFeatures(
        df = df,
        reduce_cardinality_cols = ["country_code", "email_provider"],
        cardinality_query = "sales >= 6",
        one_to_many_features_categories = ["tag", "made_purchase"],
        one_to_many_wide_index = "mailchimp_id"
    )
    
    """
    
    if not isinstance(reduce_cardinality_cols, (list, type(None))):
        raise ValueError("reduce_cardinality_cols must be a list or None")
    
    if not isinstance(one_to_many_features_categories, (list, type(None))):
        raise ValueError("one_to_many_features_categories must be a list or None")
    
    if reduce_cardinality_cols:
        keep_ = df.query(cardinality_query)
        value_sets = {col: set(keep_[col]) for col in reduce_cardinality_cols}
        
        for col, value_set in value_sets.items():
            df[col] = df[col].map(lambda x: x if x in value_set else "Other")


    if one_to_many_features_categories:
        df = df.reset_index(drop=True).reset_index().rename(columns={"index": "temp_id"})
        
        dfs_wide = []
        for col in one_to_many_features_categories:
            df_wide = pd.get_dummies(df[col], prefix=f"{col[0]}{col[len(col)//2]}{col[-1]}")
            df_wide = df_wide \
                .pipe(func = jn.clean_names)
            df_wide[one_to_many_wide_index] = df[one_to_many_wide_index]
            dfs_wide.append(df_wide)

        df = df.drop("temp_id", axis=1)

        for df_wide in dfs_wide:
            df = df.merge(df_wide, on=one_to_many_wide_index, how="left")

    return df

################################# END OF FUNCTION #################################
