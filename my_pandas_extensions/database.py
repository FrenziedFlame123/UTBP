
# IMPORTS

import os
import pickle
import sqlalchemy as sql
import pandas as pd
import numpy as np
import pandas_flavor as pf
import inspect
from typing import List, Dict, Union, Optional
from sqlalchemy.types import String, Numeric

#---------------------------------------------------------------

# collect_data_merge()
# required packages: 
#   import sqlalchemy as sql
#   import pandas as pd
#   import inspect
#   from typing import List, Dict, Union

def collect_data_merge(
    conn_string: str = "",
    tables_to_merge: List[Dict[str, Union[str, List[str]]]] = None,
    data_dict: Dict[str, pd.DataFrame] = None,
    extract_from_dictionary: bool = False
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Connects to a specified database to retrieve tables or uses a provided dictionary of pandas DataFrames. 
    If connecting to a database, loads all tables into a dictionary of DataFrames, cleaning the data by 
    dropping any 'index' columns and replacing dots with underscores in column names.
    
    If no merging operations are specified, returns the dictionary of all DataFrames.
    If merging operations are provided, performs a series of merges as specified in the `tables_to_merge` 
    parameter, handling both single and multiple tables for left and right sides of each merge. 
    After completing all merge operations, returns the final merged DataFrame.

    Parameters
    ----------
    conn_string : str, optional
        Database connection string. If provided, the function will connect to the database and 
        load all tables into DataFrames. Defaults to an empty string.
        
    tables_to_merge : list of dict, optional
        A list of dictionaries specifying the merge operations to perform. Each dictionary should include:
        - 'left_table' (str or list): Name of the left table(s) for merging.
        - 'right_table' (str or list): Name of the right table(s) for merging.
        - 'left_on' (str): Column name in the left table for merging.
        - 'right_on' (str): Column name in the right table for merging.
        - 'merge_type' (str): Type of merge ('left', 'right', 'inner', or 'outer').
        - 'produced_table_name' (str): Name for the resulting merged table.
        Defaults to None.
        
    data_dict : dict of pd.DataFrame, optional
        Dictionary of DataFrames to use instead of connecting to a database. Each key is a table name 
        and each value is the corresponding DataFrame. Defaults to None.
        
    extract_from_dictionary : bool, optional
        If True, dynamically assigns each DataFrame in `data_dict` to a variable using `globals()`.

    Returns
    -------
    dict of pd.DataFrame or pd.DataFrame
        If no merging operations are specified, returns a dictionary where the keys are table names 
        and the values are DataFrames. 
        If merging operations are specified, returns the final merged DataFrame.

    Raises
    ------
    ValueError
        If both `conn_string` and `data_dict` are empty, as at least one data source must be provided.

    Examples
    --------
    # Example 1: Using a database connection
    
    df_dict = collect_data_merge(
        conn_string="sqlite:///directory",
        tables_to_merge=[
        {'left_table': 'table_x',
        'right_table': 'table_y',
        'left_on': 'product_id',
        'right_on': 'item_id',
        'merge_type': 'left',
        'produced_table_name': 'x_and_y_merged'},

        {'left_table': 'x_and_y_merged',
        'right_table': 'table_z',
        'left_on': 'item_id',
        'right_on': 'call_id',
        'merge_type': 'inner',
        'produced_table_name': 'final_merged'}
        ]
    )
            
    ---

    # Example 2: Using a provided dictionary of DataFrames
    
    my_data_dict = {'table_x': df_x, 'table_y': df_y, 'table_z': df_z}
    df_dict = collect_data_merge(
        data_dict=my_data_dict,
        extract_from_dictionary=True,
        tables_to_merge=[
        {'left_table': 'table_x',
        'right_table': 'table_y',
        'left_on': 'product_id',
        'right_on': 'item_id',
        'merge_type': 'left',
        'produced_table_name': 'x_and_y_merged'},

        {'left_table': 'x_and_y_merged',
        'right_table': 'table_z',
        'left_on': 'item_id',
        'right_on': 'call_id',
        'merge_type': 'inner',
        'produced_table_name': 'final_merged'}
        ]
    )
    """
    caller_globals = inspect.currentframe().f_back.f_globals

    if conn_string:
        engine = sql.create_engine(conn_string)
        with engine.connect() as conn:
            inspector = sql.inspect(conn)
            table_names = inspector.get_table_names()
            
            data_dict = {}
            for table in table_names:
                df = pd.read_sql(sql.text(f"SELECT * FROM {table}"), con=conn)
                df = df.drop(columns=['index'], errors='ignore')
                df.columns = df.columns.str.replace('.', '_')
                data_dict[table] = df

    elif data_dict is None:
        data_dict = {}
    
    if not tables_to_merge:
        if extract_from_dictionary:
            for k, v in data_dict.items():
                caller_globals[f"df_{k}"] = v
        return data_dict
    
    for merge_op in tables_to_merge:
        left_table, right_table = merge_op['left_table'], merge_op['right_table']
        left_on, right_on = merge_op['left_on'], merge_op['right_on']
        merge_type, produced_table_name = merge_op['merge_type'], merge_op['produced_table_name']
        
        left_df = data_dict[left_table] if isinstance(left_table, str) else pd.concat([data_dict[t] for t in left_table], axis=0)
        right_df = data_dict[right_table] if isinstance(right_table, str) else pd.concat([data_dict[t] for t in right_table], axis=0)
        
        data_dict[produced_table_name] = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=merge_type)
    
    if extract_from_dictionary:
        for k, v in data_dict.items():
            caller_globals[f"df_{k}"] = v
    
    return data_dict

################################# END OF FUNCTION #################################

# save_dataframes_to_folder()
# required packages:
#   import os
#   import pickle
#   import pandas as pd
#   from typing import Dict, Optional

def save_dataframes_to_folder(
    data_dict: Dict[str, pd.DataFrame],
    folder_path: str,
    file_format: str = 'csv',
    index: bool = False,
    compression: Optional[str] = None,
    excel_engine: str = 'openpyxl'
) -> None:
    """
    Save all DataFrames in a dictionary to a specified folder.

    Args:
    data_dict (Dict[str, pd.DataFrame]): Dictionary containing DataFrames to save.
    folder_path (str): Path to the folder where files will be saved.
    file_format (str): Format to save files in. Options: 'csv', 'parquet', 'xlsx', 'pkl', 'json'. Default is 'csv'.
    index (bool): Whether to save the index as a column. Default is False.
    compression (Optional[str]): Compression to use. 
                                 For CSV: None, 'gzip', 'bz2', 'zip', 'xz'
                                 For Parquet: None, 'snappy', 'gzip', 'brotli'
                                 For XLSX, PKL, JSON: Not applicable
    excel_engine (str): The engine to use for Excel writing. Default is 'openpyxl'.
    
    Example:
    --------
    
    save_dataframes_to_folder(df_dict, folder_path="./output_data", file_format="csv")
    
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    for name, df in data_dict.items():
        file_name = f"{name}.{file_format}"
        file_path = os.path.join(folder_path, file_name)

        if file_format.lower() == 'csv':
            df.to_csv(file_path, index=index, compression=compression)
        elif file_format.lower() == 'parquet':
            df.to_parquet(file_path, index=index, compression=compression)
        elif file_format.lower() == 'xlsx':
            df.to_excel(file_path, index=index, engine=excel_engine)
        elif file_format.lower() == 'pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(df, f)
        elif file_format.lower() == 'json':
            df.to_json(file_path, orient='records', indent=4)
        else:
            raise ValueError("Unsupported file format. Use 'csv', 'parquet', 'xlsx', 'pkl', or 'json'.")

        print(f"Saved {name} to {file_path}")

################################# END OF FUNCTION #################################

# write_forecast_to_database()
# required packages:
#   import sqlalchemy as sql
#   import pandas as pd
#   import numpy as np
#   import pandas_flavor as pf
#   from sqlalchemy.types import String, Numeric

@pf.register_dataframe_method
def write_forecast_to_database(
    df,
    id_column,
    date_column,
    conn_string,
    table_name,
    if_exists="fail",
    **kwargs
):
    """
    Writes a forecast DataFrame to a SQL database.

    This function prepares a DataFrame containing forecast data by renaming specified
    columns and ensuring the DataFrame has the required columns. It then writes the 
    prepared DataFrame to a specified table in a SQL database.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the forecast data to be written to the database.
    
    id_column : str
        The name of the column in `df` that contains the unique identifier for each entry.
        This column will be renamed to 'id' before writing to the database.
    
    date_column : str
        The name of the column in `df` that contains the date information. This column 
        will be renamed to 'date' and will be converted to a datetime type if necessary 
        before writing to the database.
    
    conn_string : str
        The connection string used to connect to the SQL database.
    
    table_name : str
        The name of the table in the database where the DataFrame will be written. If the 
        table does not exist, it will be created.
    
    if_exists : str, default="fail"
        What to do if the table already exists in the database:
        - 'fail': Raise a ValueError.
        - 'replace': Drop the table before inserting new values.
        - 'append': Insert new values into the existing table.
    
    **kwargs : dict
        Additional arguments to pass to `pandas.DataFrame.to_sql`, such as `chunksize`, 
        `method`, etc.

    Raises
    ------
    ValueError
        If `df` is not a pandas DataFrame, or if any of the `id_column`, `date_column`, 
        `conn_string`, or `table_name` parameters are not strings.
    
    Exception
        If `df` does not contain the required columns: 'id', 'date', 'value', 'prediction',
        'ci_low', 'ci_high'.

    Returns
    -------
    None
        This function does not return any value. It writes the DataFrame to the specified
        SQL database table.

    Notes
    -----
    - The function assumes that the forecast DataFrame has the following columns after renaming:
      'id', 'date', 'value', 'prediction', 'ci_low', and 'ci_high'.
    - The 'date' column is converted to a datetime format if it is not already in that format.
    - SQL data types are assigned to the columns to ensure correct storage in the database.
    
    Example:
    --------

    write_forecast_to_database(
    df = arima_forecast_df,
    id_column = "category_2",
    date_column = "order_date",
    conn_string = "sqlite:///00_database/bike_orders_database.sqlite",
    table_name = "forecast",
    if_exists = "replace")

    ---

    df.write_forecast_to_database(
    id_column = "category_2",
    date_column = "order_date",
    conn_string = "sqlite:///00_database/bike_orders_database.sqlite",
    table_name = "forecast",
    if_exists = "replace")

    """
    
    # Checks
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame.")
    
    if not all(isinstance(arg, str) for arg in [id_column, date_column, conn_string, table_name]):
        raise ValueError("All of the ['id_column', 'date_column', 'conn_string', 'table_name'] must be strings")

    def prep_forecast_data_for_update(df_prep, id_column_prep, date_column_prep):
        # Format the column names
        df_prep = df_prep.rename({id_column_prep: "id", date_column_prep: "date"}, axis=1)

        # Validate correct columns
        required_col_names = ["id", "date", "value", "prediction", "ci_low", "ci_high"]
        
        if not all(col in df_prep.columns for col in required_col_names):
            col_text = ", ".join(required_col_names)
            raise Exception(f"Columns must contain: {col_text}")
        
        df_prep = df_prep[required_col_names]
        
        # Check format for SQL Database
        if df_prep['date'].dtype != 'datetime64[ns]':
            try:
                df_prep['date'] = df_prep['date'].dt.to_timestamp()
            except:
                pass
            try:
                df_prep['date'] = pd.to_datetime(df_prep['date'])
            except:
                pass
            if df_prep['date'].dtype != 'datetime64[ns]':
                raise Exception("Could not auto-convert 'date' to datetime64.")

        return df_prep

    # Prepare the data for SQL insertion
    df = prep_forecast_data_for_update(
        df_prep=df,
        id_column_prep=id_column,
        date_column_prep=date_column
    )
 
    sql_dtype = {
        "id": String(),
        "date": String(),
        "value": Numeric(),
        "prediction": Numeric(),
        "ci_low": Numeric(),
        "ci_high": Numeric()
    }
    
    # Connect to Database
    
    engine = sql.create_engine(conn_string)
    
    with engine.connect() as conn:
        # Write the DataFrame to SQL
        df.to_sql(
            con=conn,
            name=table_name,
            if_exists=if_exists,
            dtype=sql_dtype,
            index=False,
            **kwargs
        )

################################# END OF FUNCTION #################################

# read_forecast_from_database()
# required packages:
#   import sqlalchemy as sql
#   import pandas as pd
#   import numpy as np

def read_forecast_from_database(
    conn_string,
    table_name,
    **kwargs
):
    """
    Reads forecast data from a specified table in a database and returns it as a pandas DataFrame.

    Parameters:
    -----------
    conn_string : str
        A database connection string compatible with SQLAlchemy.
    table_name : str
        The name of the table from which to read the forecast data.
    **kwargs : 
        Additional keyword arguments to pass to `pd.read_sql_query`, such as `columns` or `chunksize`.

    Returns:
    --------
    df : pandas.DataFrame
        A DataFrame containing the data read from the specified table, with the "date" column parsed as datetime.

    Notes:
    ------
    - The function assumes that the table contains a column named "date" which will be automatically parsed as datetime.
    - Ensure that the database connection string is correctly formatted and the table exists within the database.

    Example:
    --------
    
    read_forecast_from_database(
    conn_string = "sqlite:///00_database/bike_orders_database.sqlite",
    table_name = "forecast")
    
    """
    # Connect to Database
    
    engine = sql.create_engine(conn_string)
    
    with engine.connect() as conn:
        query = sql.text(f"SELECT * FROM {table_name}")
        
        # Read from table
        df = pd.read_sql_query(
            sql = query,
            con = conn,
            parse_dates = ["date"],
            **kwargs
        )

    return df

################################# END OF FUNCTION #################################