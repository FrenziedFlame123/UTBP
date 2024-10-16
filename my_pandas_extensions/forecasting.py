
# IMPORTS

import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from tqdm import tqdm
import pandas_flavor as pf
import matplotlib
from mizani.formatters import dollar_format
from plydata.cat_tools import cat_reorder
from plotnine import (
ggplot,
aes,
geom_line,
geom_ribbon,
facet_wrap,
scale_x_datetime,
scale_y_continuous,
scale_color_manual,
theme_minimal,
theme_matplotlib,
theme,
element_rect,
labs
)

#---------------------------------------------------------------

# prep_forecast_data_for_update()
# required packages:
#   import pandas as pd
#   import numpy as np
#   import pandas_flavor as pf

@pf.register_dataframe_method
def prep_forecast_data_for_update(df_prep, id_column_prep, date_column_prep):
    """
    Prepares forecast data for updating in a SQL database by renaming columns,
    validating required columns, and formatting date columns.

    Parameters:
    -----------
    df_prep : pandas.DataFrame
        The DataFrame containing forecast data that needs to be preprocessed.
        
    id_column_prep : str
        The name of the column in `df_prep` representing the unique identifier (ID) that will be renamed to "id".
        
    date_column_prep : str
        The name of the column in `df_prep` representing dates that will be renamed to "date".

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the required columns ["id", "date", "value", "prediction", "ci_low", "ci_high"] 
        and with the date column formatted as a timestamp, suitable for updating in a SQL database.

    Raises:
    -------
    Exception
        If `df_prep` does not contain all required columns.
        
    Examples:
    ---------
    
    prep_forecast_data_for_update(
        df_prep = df,
        id_column_prep = "category",
        date_column_prep = "order_date"
    )
    
    ---
    
    df.prep_forecast_data_for_update(
        id_column_prep = "frame",
        date_column_prep = "date"
    )
    
    """
    
    # Format the column names
    df_prep = df_prep.rename({id_column_prep: "id", date_column_prep: "date"}, axis=1)

    # Validate correct columns
    required_col_names = ["id", "date", "value", "prediction", "ci_low", "ci_high"]
    
    if not all(col in df_prep.columns for col in required_col_names):
        col_text = ", ".join(required_col_names)
        raise Exception(f"Columns must contain: {col_text}")
    
    df_prep = df_prep[required_col_names]
    
    # Check format for SQL Database
    df_prep["date"] = df_prep["date"].dt.to_timestamp()

    return df_prep

################################# END OF FUNCTION #################################

# arima_forecast()
# required packages: 
#   import numpy as np
#   from sktime.forecasting.arima import AutoARIMA
#   from tqdm import tqdm
#   import pandas_flavor as pf

@pf.register_dataframe_method
def arima_forecast(
    df,
    h,
    sp,
    alpha = 0.05,
    suppress_warnings = True,
    return_pred_int= True,
    *args,
    **kwargs
    ):
    """
    Forecast time series data using AutoARIMA models for each column in a DataFrame.
    
    This function applies AutoARIMA to each time series column in the input DataFrame,
    forecasts future values for a specified horizon, and returns a DataFrame with the 
    original values, predictions, and confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame where each column represents a different time series.
        
    h : int
        The forecast horizon, i.e., the number of periods into the future to forecast.
        
    sp : int
        The seasonal period for the AutoARIMA model.
        
    alpha : float, optional, default=0.05
        The significance level for the confidence intervals. A lower alpha results 
        in wider confidence intervals.
        
    suppress_warnings : bool, optional, default=True
        If True, warnings during the model fitting process are suppressed.
        
    return_pred_int : bool, optional, default=True
        If True, the function returns confidence intervals along with predictions.
        
    *args : tuple
        Additional positional arguments to pass to the AutoARIMA model.
        
    **kwargs : dict
        Additional keyword arguments to pass to the AutoARIMA model.
    
    Returns
    -------
    ret : pandas.DataFrame
        A DataFrame containing the original values, predictions, and confidence intervals 
        for each time series in the input DataFrame. The DataFrame is indexed by the 
        original index of `df`, and any level columns introduced during the process are dropped.
    
    Notes
    -----
    - The function iterates over each column in the input DataFrame, fits an AutoARIMA model 
      to the time series, and makes future forecasts.
    - Predictions are combined with the original data and the confidence intervals into a single 
      DataFrame, which is returned after resetting the index and removing unnecessary level columns.
    
    Example
    -------
    
    df.arima_forecast(h = 12, sp = 1)
    
    """
    
    # Checks
    if not isinstance(df, pd.DataFrame):
        raise Exception("'df' must be a pandas DataFrame.")
    
    if not isinstance(h, int):
        raise Exception("'h' must be an integer.")
    
    if not isinstance(sp, int):
        raise Exception("'sp' must be an integer.")
    
    # FOR LOOP
    
    model_results_dict = {}
    for col in tqdm(df.columns, mininterval = 0):
    
        # Series Extraction
        y = df[col]
        
        # Modeling
        forecaster = AutoARIMA(
        sp = sp,
        suppress_warnings = suppress_warnings,
        *args,
        **kwargs
        )
         
        forecaster.fit(y)
        
        # Predictions & Conf Intervals
        predictions, conf_int_df= forecaster.predict(
        fh = np.arange(1, h+1),
        return_pred_int = return_pred_int,
        alpha = alpha
        )
        
        # Combine into data frame
        ret = pd.concat([y, predictions, conf_int_df], axis = 1)
        ret.columns = ["value", "prediction", "ci_low", "ci_high"]
        
        # Update Dictionary
        model_results_dict[col] = ret
        
    # Stack Each Dict Element on Top of Each Other
    model_results_df = pd.concat(model_results_dict, axis = 0)
    
    # Handle Names
    nms = [*df.columns.names, *df.index.names]
    model_results_df.index.names = nms
    
    # Reset Index
    ret = model_results_df.reset_index()
    
    # Drop columns containing "level"
    cols_to_keep = ~ret.columns.str.startswith("level")
    
    ret = ret.loc[:, cols_to_keep]
    
    return ret

################################# END OF FUNCTION #################################

# ets_forecast()
# required packages: 
#   import numpy as np
#   from sktime.forecasting.ets import AutoETS
#   from tqdm import tqdm
#   import pandas_flavor as pf

@pf.register_dataframe_method
def ets_forecast(
    df,
    h,
    sp,
    suppress_warnings = True,
    *args,
    **kwargs
    ):
    
    # Checks
    if not isinstance(df, pd.DataFrame):
        raise Exception("'df' must be a pandas DataFrame.")
    
    if not isinstance(h, int):
        raise Exception("'h' must be an integer.")
    
    if not isinstance(sp, int):
        raise Exception("'sp' must be an integer.")
    
    # FOR LOOP
    
    model_results_dict = {}
    for col in tqdm(df.columns, mininterval = 0):
    
        # Series Extraction
        y = df[col]
        
        # Modeling
        forecaster = AutoETS(
        sp = sp,
        suppress_warnings = suppress_warnings,
        *args,
        **kwargs
        )
         
        forecaster.fit(y)
        
        # Predictions & Conf Intervals
        predictions = forecaster.predict(
        fh = np.arange(1, h+1)
        )
        
        # Combine into data frame
        ret = pd.concat([y, predictions], axis = 1)
        ret.columns = ["value", "prediction"]
        
        # Update Dictionary
        model_results_dict[col] = ret
        
    # Stack Each Dict Element on Top of Each Other
    model_results_df = pd.concat(model_results_dict, axis = 0)
    
    # Handle Names
    nms = [*df.columns.names, *df.index.names]
    model_results_df.index.names = nms
    
    # Reset Index
    ret = model_results_df.reset_index()
    
    # Drop columns containing "level"
    cols_to_keep = ~ret.columns.str.startswith("level")
    
    ret = ret.loc[:, cols_to_keep]
    
    return ret

################################# END OF FUNCTION #################################

# plot_forecast()
# required packages:
#   import pandas as pd
#   import numpy as np
#   import matplotlib
#   import pandas_flavor as pf
#   from mizani.formatters import dollar_format
#   from plydata.cat_tools import cat_reorder
#   from plotnine import (
#     ggplot, 
#     aes, 
#     geom_line, 
#     geom_ribbon, 
#     facet_wrap, 
#     scale_x_datetime, 
#     scale_y_continuous, 
#     scale_color_manual, 
#     theme_minimal, 
#     theme_matplotlib, 
#     theme, 
#     element_rect, 
#     labs
#     )

@pf.register_dataframe_method
def plot_forecast(
    df,
    id_column,
    date_column,
    value_column,
    prediction_column,
    ci_low = None,
    ci_high = None,
    ci_alpha = None,
    facet_wrap_ncol = 1,
    facet_scales = "free_y",
    datetime_labels = "%Y",
    datetime_breaks = "1 year",
    scale_colors = ["red", "blue"],
    categorical_reorder = False,
    categorical_reorder_function = np.mean,
    matplotlib_theme = False,
    matplotlib_style = None,
    theme_legend_position = "none",
    strip_background_fill = False,
    legend_background_fill = False,
    theme_subplots_adjust = {"wspace": 0.25},
    theme_figure_size = (16, 8),
    labs_title = "Forecast Plot", 
    labs_x = "", 
    labs_y = "Revenue",
    *args,
    **kwargs
    ):
    """
    Plots a forecast with optional confidence intervals using ggplot2-style syntax in Python.

    This function creates a customizable forecast plot that visualizes both actual values and predicted values
    over time. It supports faceting by an identifier column and allows various styling options, including
    confidence intervals, custom themes, and reordering of categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
        
    id_column : str
        The name of the column containing the identifier for faceting (e.g., different groups or categories).
        
    date_column : str
        The name of the column containing date or time information.
        
    value_column : str
        The name of the column containing actual values (e.g., observed data).
        
    prediction_column : str
        The name of the column containing predicted values (e.g., forecasted data).
        
    ci_low : str, optional
        The name of the column containing the lower bound of the confidence interval. Default is None.
        
    ci_high : str, optional
        The name of the column containing the upper bound of the confidence interval. Default is None.
        
    ci_alpha : float, optional
        The alpha (transparency) level for the confidence interval ribbon. Should be between 0 and 1. Default is None.
        
    facet_wrap_ncol : int, optional
        The number of columns to use in the faceted plot layout. Default is 1.
        
    facet_wrap_nrow : int, optional
        The number of rows to use in the faceted plot layout. Default is 2.
        
    facet_scales : str, optional
        Controls the scales for the facets. Options are "fixed", "free", "free_x", or "free_y". Default is "free_y".
        
    datetime_labels : str, optional
        The format for the date labels on the x-axis. Default is "%Y" (year).
        
    datetime_breaks : str, optional
        The interval for date breaks on the x-axis. Default is "1 year".
        
    scale_colors : list of str, optional
        Colors to use for the actual and predicted values. Default is ["red", "blue"].
        
    categorical_reorder : bool, optional
        If True, reorders the categories of the id_column based on a specified function. Default is False.
        
    categorical_reorder_function : callable, optional
        The function to apply when reordering categories. Default is np.mean.
        
    matplotlib_theme : bool, optional
        If True, applies a Matplotlib theme to the plot. Must be used in conjunction with `matplotlib_style`. Default is False.
        
    matplotlib_style : str, optional
        The name of the Matplotlib style to apply. Must be used in conjunction with `matplotlib_theme`. Default is None.
        
    theme_legend_position : str, optional
        The position of the legend in the plot. Default is "none".
        
    strip_background_fill : str or bool, optional
        The fill color for the strip background in faceted plots. If False, no fill is applied. Default is False.
        
    legend_background_fill : str or bool, optional
        The fill color for the legend background. If False, no fill is applied. Default is False.
        
    theme_subplots_adjust : dict, optional
        Parameters for adjusting subplot spacing. Default is {"wspace": 0.25}.
        
    theme_figure_size : tuple of int, optional
        The figure size of the plot, given as (width, height). Default is (16, 8).
        
    labs_title : str, optional
        The title of the plot. Default is "Forecast Plot".
        
    labs_x : str, optional
        The label for the x-axis. Default is "" (no label).
        
    labs_y : str, optional
        The label for the y-axis. Default is "Revenue".
        
    *args : tuple, optional
        Additional arguments passed to the `dollar_format` function for formatting y-axis labels.
        
    **kwargs : dict, optional
        Additional keyword arguments passed to the `dollar_format` function for formatting y-axis labels.

    Returns
    -------
    plotnine.ggplot
        A ggplot object representing the forecast plot.

    Raises
    ------
    ValueError
        If some but not all of `ci_low`, `ci_high`, or `ci_alpha` are provided.
    ValueError
        If only one of `matplotlib_theme` or `matplotlib_style` is provided.
    ValueError
        If `df` is not a pandas DataFrame.
    Exception
        If the `date_column` cannot be converted to a `datetime64[ns]` format.

    Notes
    -----
    - This function uses `plotnine` for plotting, which is a grammar of graphics implementation similar to ggplot2 in R.
    - If confidence intervals (`ci_low`, `ci_high`, `ci_alpha`) are provided, they are shown as a ribbon around the prediction line.
    - The `date_column` should ideally be in a datetime format; otherwise, the function will attempt to convert it.
    
    Examples:
    ------
    plot_forecast(
    df = arima_forecasst_df,
    id_column = "category_1",
    date_column = "order_date",
    value_column = "value",
    prediction_column = "prediction",
    ci_low = "ci_low",
    ci_high = "ci_high",
    ci_alpha = 0.1,
    prefix = "$",
    suffix = "",
    big_mark = ",",
    digits = 0)
    
    ---
    
    df.plot_forecast(
        id_column = "category_1",
        date_column = "order_date",
        value_column = "value",
        prediction_column = "prediction",
        ci_low = "ci_low",
        ci_high = "ci_high",
        ci_alpha = 0.1,
        matplotlib_theme = True,
        matplotlib_style = "dark_background",
        theme_legend_position = "right",
        strip_background_fill = "black",
        legend_background_fill = "gray",
        prefix = "$",
        suffix = "",
        big_mark = ",",
        digits = 0
        )
    
    Available Matplotlib styles:
    -------
    'Solarize_Light2',
    '_classic_test_patch',
    'bmh',
    'classic',
    'dark_background',
    'fast',
    'fivethirtyeight',
    'ggplot',
    'grayscale',
    'seaborn',
    'seaborn-bright',
    'seaborn-colorblind',
    'seaborn-dark',
    'seaborn-dark-palette',
    'seaborn-darkgrid',
    'seaborn-deep',
    'seaborn-muted',
    'seaborn-notebook',
    'seaborn-paper',
    'seaborn-pastel',
    'seaborn-poster',
    'seaborn-talk',
    'seaborn-ticks',
    'seaborn-white',
    'seaborn-whitegrid',
    'tableau-colorblind10'
    
    """
    
    # Checks
    if not all([ci_low, ci_high, ci_alpha]) and any([ci_low, ci_high, ci_alpha]):
        raise ValueError("You cannot individually use [ci_low, ci_high, ci_alpha]; provide all or none.")
    
    if bool(matplotlib_theme) != bool(matplotlib_style):
        raise ValueError("You cannot individually use [matplotlib_theme, matplotlib_style]; provide both or neither.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame")
    
    # Data Wrangling
    if ci_low and ci_high and ci_alpha is not None:
        df_prepped = df \
        .melt(
            value_vars = [value_column, prediction_column],
            id_vars = [id_column, date_column, ci_low, ci_high], ## columns to maintain
            value_name = "_value",
            var_name = "_variable"
        ) \
        .rename({"_value": "value", "_variable": "variable"}, axis = 1)
    else:
        df_prepped = df \
        .melt(
            value_vars = [value_column, prediction_column],
            id_vars = [id_column, date_column], ## columns to maintain
            value_name = "_value",
            var_name = "_variable"
        ) \
        .rename({"_value": "value", "_variable": "variable"}, axis = 1)
        
    # Categorical Conversion
    if categorical_reorder == True:
        df_prepped[id_column] = cat_reorder(
            c = df_prepped[id_column],
            x = df_prepped["value"],
            fun = categorical_reorder_function,
            ascending = False)
    
    # Check for period, convert to datetime64
    
    if df_prepped[date_column].dtype is not "datetime64[ns]":
        try:
            df_prepped[date_column] = df_prepped[date_column].dt.to_timestamp()
        except:
            pass
        try:
            df_prepped[date_column] = pd.to_datetime(df_prepped[date_column])
        except:
            pass
        try:
            df_prepped.assign(date_column = lambda x: x[date_column].astype("datetime64[ns]"))
        except:
            raise Exception("Could not auto-convert 'date_column' to datetime64.")
    
    # Preparing the plot
    canvas = ggplot(mapping = aes(x = date_column, y = "value", color = "variable"), data = df_prepped)
    gplot = canvas
    
    if ci_low and ci_high and ci_alpha is not None:
        gplot = canvas + geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = ci_alpha, color = None)
    
    gplot = gplot + geom_line()
    gplot = gplot + facet_wrap(id_column, ncol = facet_wrap_ncol, scales = facet_scales)
    gplot = gplot + scale_x_datetime(date_labels = datetime_labels, date_breaks = datetime_breaks)
    gplot = gplot + scale_y_continuous(labels = dollar_format(*args, **kwargs))
    gplot = gplot + scale_color_manual(values = scale_colors)
    
    if matplotlib_theme == True:
        matplotlib.pyplot.style.use(matplotlib_style)
        gplot = gplot + theme_matplotlib()
    else:
        gplot = gplot + theme_minimal()
    
    if isinstance(legend_background_fill, str):    
        if isinstance(strip_background_fill, str):
            gplot = gplot + theme(strip_background = element_rect(fill = strip_background_fill), legend_position = theme_legend_position, subplots_adjust = theme_subplots_adjust, figure_size = theme_figure_size, legend_background = element_rect(fill = legend_background_fill))
        else:
            gplot = gplot + theme(legend_position = theme_legend_position, subplots_adjust = theme_subplots_adjust, figure_size = theme_figure_size, legend_background = element_rect(fill = legend_background_fill))
    else:
        if isinstance(strip_background_fill, str):
            gplot = gplot + theme(strip_background = element_rect(fill = strip_background_fill), legend_position = theme_legend_position, subplots_adjust = theme_subplots_adjust, figure_size = theme_figure_size)
        else:
            gplot = gplot + theme(legend_position = theme_legend_position, subplots_adjust = theme_subplots_adjust, figure_size = theme_figure_size)  
    
    gplot = gplot + labs(title = labs_title, x = labs_x, y = labs_y)
    
    return gplot

################################# END OF FUNCTION #################################

