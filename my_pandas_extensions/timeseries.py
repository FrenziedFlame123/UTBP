
# IMPORTS

import pandas as pd
import numpy as np
import pandas_flavor as pf

#---------------------------------------------------------------

# summarize_by_time()
# required packages: 
#   import pandas as pd
#   import numpy as np
#   import pandas_flavor as pf

@pf.register_dataframe_method
def summarize_by_time(
    df, 
    date_column, 
    value_column, 
    groups = None, 
    rule = "D", 
    agg_func = np.sum,
    kind = "timestamp",
    wide_format = True,
    fillna = None,
    *args,
    **kwargs):
    """
    Summarizes data over specified time intervals with optional grouping and aggregation.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to be summarized.
        
    date_column : str
        The name of the column in `df` that contains datetime or period information.
        
    value_column : str or list of str
        The column(s) in `df` that contain the values to be aggregated.
        
    groups : str or list of str, optional
        Column(s) by which to group the data before resampling. If None, no grouping is applied.
        
    rule : str, optional, default "D"
        The frequency string that defines the resampling rule. Examples include "D" for daily,
        "M" for monthly, "A" for annual, etc.
        
    agg_func : function, optional, default np.sum
        The aggregation function to apply to each group during resampling.
        
    kind : {'timestamp', 'period'}, optional, default "timestamp"
        The kind of index to return: either 'timestamp' or 'period'.
        
    wide_format : bool, optional, default True
        If True, returns a DataFrame in a wide format (with separate columns for each group).
        If False, returns the DataFrame in a long format.
        
    fillna : scalar, dict, or None, optional
        Value to use to fill missing values (NaN) after resampling. If None, no fill is applied.
        
    *args : tuple
        Additional positional arguments passed to the aggregation function.
        
    **kwargs : dict
        Additional keyword arguments passed to the aggregation function.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the summarized data, indexed by the resampled time periods.
        If `wide_format` is True, the DataFrame will have a multi-index if grouped, otherwise, 
        a single index corresponding to the resampled time periods.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
        If `value_column` is not a string or a list of strings.
    
    Notes
    -----
    - The function handles the resampling of the data based on the `rule` provided.
    - If `groups` is specified, the data is grouped by these columns before resampling.
    - The `agg_func` is applied to each `value_column` after resampling.
    - The returned DataFrame can be in a wide format with multiple columns if `wide_format` is True.
    - If `fillna` is provided, it will replace NaN values after resampling.

    Examples
    --------
    
    summarize_by_time(df, 'date', 'sales', groups=['store'], rule='M', agg_func=np.mean)
    
    ---
    
    df.summarize_by_time(
    date_column = "order_date",
    value_column = "total_price",
    groups = ["category_2"],
    rule = "D",
    kind = "period",
    agg_func = [np.sum, np.mean],
    wide_format = True,
    # fillna = np.nan
    )
    
    """
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' should be a pandas DataFrame.")
    
    # Convert value_column to list if it isn't already
    if not isinstance(value_column, list):
        value_column = [value_column]
        
    # Set the date column as the index
    if df.index.name != date_column:
        df = df.set_index(date_column)
    
    # Group if groups are provided, else operate on the entire DataFrame
    if groups:
        df = df.groupby(groups).resample(rule=rule, kind=kind)
    else:
        df = df.resample(rule=rule, kind=kind)
    
    # Create aggregation dictionary
    agg_dict = {col: agg_func for col in value_column}
    
    # Apply the aggregation function
    df = df.agg(agg_dict, *args, **kwargs)
    
    # Handle Pivot Wider if required
    if wide_format and groups:
        df = df.unstack(groups)
        if kind == "period":
            df.index = df.index.to_period()

    # Handle fillna if required
    if fillna is not None:
        df = df.fillna(fillna)
                
    return df

################################# END OF FUNCTION #################################