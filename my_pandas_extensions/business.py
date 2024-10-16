# IMPORTS

import pandas as pd
import numpy as np
import janitor as jn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

#---------------------------------------------------------------

# cost_table_for_growth()
# required packages: 
#   import pandas as pd
#   import numpy as np
#   import janitor as jn
#   import plotly.express as px
#   import plotly.graph_objects as go
#   from plotly.subplots import make_subplots
#   from IPython.display import display

def cost_table_for_growth(
    total_list_size: int = None,
    periodical_gain: float = None,
    periodical_loss: float = None,
    periodical_campaign_count: float = None,
    loss_by_campaign: float = None,
    customer_value: float = None,
    conversion_rate: float = None,
    reduce_loss_rate = None,
    period: int = 12,
    period_type: str = "Month",
    growth_type = "exponential",
    simulate : bool = False,
    growth_rate_interval: list = [],
    conversion_rate_interval: list = [],
    growth_rate_interval_count: int = 3,
    conversion_rate_interval_count: int = 3,
    **kwargs
):
    """
    Examples:
    ---------
    
    cost_table_for_growth(
        total_list_size = 100000,
        periodical_gain = 6000,
        periodical_loss = 2500,
        periodical_campaign_count = 5,
        loss_by_campaign = 500,
        customer_value = 2000,
        conversion_rate = 0.05,
        reduce_loss_rate = 0.30,
        period = 12,
        period_type = "Month",
        growth_type = "exponential"
    )
    
    ---
    
    cost_table_for_growth(
        total_list_size = 100000,
        periodical_gain = 6000,
        periodical_loss = 2500,
        periodical_campaign_count = 5,
        loss_by_campaign = 500,
        customer_value = 2000,
        period = 12,
        period_type = "Month",
        growth_type = "exponential",
        simulate = True,
        growth_rate_interval = [0.05, 0.3],
        conversion_rate_interval = [0.09, 0.23],
        growth_rate_interval_count = 4,
        conversion_rate_interval_count = 5
    )
    
    """
    
    if simulate is True:
        if conversion_rate is True:
            raise Exception("'conversion_rate' can be used only when 'simulate' = False")
        
    if simulate is False:
        
        tuple_ = (
        total_list_size, 
        periodical_gain,
        periodical_loss,
        periodical_campaign_count,
        loss_by_campaign,
        customer_value,
        conversion_rate,
        period,
        growth_type
        )
        
        if (growth_rate_interval != [] or conversion_rate_interval != []) or (growth_rate_interval_count != 3 or conversion_rate_interval_count != 3):
            raise Exception("""
                You cannot use the following if 'simulate' = False (default)
                [
                'growth_rate_interval',
                'conversion_rate_interval',
                'growth_rate_interval_count',
                'conversion_rate_interval_count'
                ]
                """)
        
        if all(tuple_) is not True:
            raise Exception("""
            Neither of the following can be left blank:
            [
            'total_list_size',
            'periodical_gain',
            'periodical_loss',
            'periodical_campaign_count',
            'loss_by_campaign',
            'customer_value',
            'conversion_rate',
            'period',
            'growth_type'
            ]
            """
            )
        
        if not (conversion_rate >= 0 and conversion_rate <= 1):
            raise ValueError("'conversion_rate' must be between 0 and 1.")
        
        if reduce_loss_rate:
            if not (reduce_loss_rate >= 0 and reduce_loss_rate <= 1):
                raise ValueError("'reduce_loss_rate' must be between 0 and 1.")
        
        loss_per_mail = loss_by_campaign / total_list_size
        lost_customers = total_list_size * loss_per_mail * periodical_campaign_count * conversion_rate
        lost_revenue_per_period = lost_customers * customer_value
        cost_no_growth = period * lost_revenue_per_period
        
        growth_rate = (periodical_gain - periodical_loss) / total_list_size
        

        table_df = pd.DataFrame({f"Period_({period_type})": np.arange(0, (period + 1))})
        table_df["ListSize_when_NoGrowth"] = np.repeat(total_list_size, period + 1)

        table_df = table_df \
            .assign(LostCustomers_when_NoGrowth = lambda x: x.ListSize_when_NoGrowth * loss_per_mail * periodical_campaign_count * conversion_rate) \
            .assign(Cost_when_NoGrowth = lambda x: x.LostCustomers_when_NoGrowth * customer_value)


        if growth_type == "exponential":
            table_df = table_df \
                .assign(ListSize_with_Growth = lambda x: x.ListSize_when_NoGrowth * ((1 + growth_rate) ** x[f"Period_({period_type})"]))

        if growth_type == "linear":
            table_df = table_df \
                .assign(ListSize_with_Growth = lambda x: x.ListSize_when_NoGrowth + (growth_rate * x[f"Period_({period_type})"]))


        table_df = table_df \
            .assign(LostCustomers_with_Growth = lambda x: x.ListSize_with_Growth * loss_per_mail * periodical_campaign_count * conversion_rate) \
            .assign(Cost_with_Growth = lambda x: x.LostCustomers_with_Growth * customer_value) \
            .assign(CostDifference_from_NoGrowth = lambda x: (x.Cost_with_Growth - x.Cost_when_NoGrowth)) \
            .assign(TotalCost_with_Growth = lambda x: x.Cost_with_Growth.cumsum())

        columns_to_round = ["Cost_when_NoGrowth", "Cost_with_Growth", "CostDifference_from_NoGrowth", "TotalCost_with_Growth", "LostCustomers_when_NoGrowth", "LostCustomers_with_Growth", "ListSize_when_NoGrowth", "ListSize_with_Growth"]
        table_df[columns_to_round] = table_df[columns_to_round].apply(lambda x: round(x, 0)).astype(int)
        table_df = table_df.reindex(columns = [f"Period_({period_type})", "Cost_when_NoGrowth", "Cost_with_Growth", "CostDifference_from_NoGrowth", "TotalCost_with_Growth", "LostCustomers_when_NoGrowth", "LostCustomers_with_Growth", "ListSize_when_NoGrowth", "ListSize_with_Growth"])



        fig_listsize = px.line(
            data_frame = table_df,
            x = f"Period_({period_type})",
            y = ["ListSize_when_NoGrowth", "ListSize_with_Growth"],
            color_discrete_map = {
                "ListSize_when_NoGrowth": "cyan",
                "ListSize_with_Growth": "orange"}
        ) \
            .add_hline(y = 0)
        
        fig_revenue = px.line(
            data_frame = table_df,
            x = f"Period_({period_type})",
            y = ["Cost_when_NoGrowth", "Cost_with_Growth"]
        ) \
            .add_hline(y = 0)

        fig_listsize = go.Figure(fig_listsize)
        fig_revenue = go.Figure(fig_revenue)

        fig = make_subplots(rows=1, cols=2, subplot_titles=('List Size', 'Revenue'))

        for trace in fig_listsize.data:
            fig.add_trace(trace, row=1, col=1)

        for trace in fig_revenue.data:
            fig.add_trace(trace, row=1, col=2)
        
        fig.show()


        ratio = table_df["Cost_with_Growth"].sum() / table_df["Cost_when_NoGrowth"].sum()
        percentage_increase = (ratio - 1) * 100
        compare_revenue = round(percentage_increase, 2)

        if reduce_loss_rate:
            print(f"""
            Loss of potential increase by {compare_revenue}%.\n
            -----------------------------------------------------\n
            Reducing the loss by {reduce_loss_rate * 100}% would save {int(round((table_df["Cost_when_NoGrowth"].sum() * reduce_loss_rate), 0))}.
            """)
        else:
            print(f"Loss of potential increase by {compare_revenue}%.")
        
        return(table_df)
        
    if simulate is True:
        
        tuple_ = (
        total_list_size, 
        periodical_gain,
        periodical_loss,
        periodical_campaign_count,
        loss_by_campaign,
        customer_value,
        period,
        growth_type,
        growth_rate_interval,
        conversion_rate_interval,
        growth_rate_interval_count,
        conversion_rate_interval_count
        )
        
        if all(tuple_) is not True:
            raise Exception("""
            Neither of the following can be left blank:
            [
            'total_list_size', 
            'periodical_gain',
            'periodical_loss',
            'periodical_campaign_count',
            'loss_by_campaign',
            'customer_value',
            'period',
            'growth_type',
            'growth_rate_interval',
            'conversion_rate_interval',
            'growth_rate_interval_count',
            'conversion_rate_interval_count'
            ]
            """
            )
        
        
        data_dict = {
            f"List_Growth_Rate_{period_type}": np.linspace(growth_rate_interval[0], growth_rate_interval[1], num = growth_rate_interval_count),
            "Conversion_Rate": np.linspace(conversion_rate_interval[0], conversion_rate_interval[1], num = conversion_rate_interval_count)}
        
        parameter_grid_df = jn.expand_grid(others = data_dict)
        
        def temporary_function(x, y):
            def mini_function(
                total_list_size2 = total_list_size,
                periodical_campaign_count2 = periodical_campaign_count,
                loss_by_campaign2 = loss_by_campaign,
                customer_value2 = customer_value,
                conversion_rate2 = x,
                growth_rate2 = y,
                period2 = period,
                period_type2 = period_type,
                growth_type2 = growth_type):
                
                    loss_per_mail2 = loss_by_campaign2 / total_list_size2
                    lost_customers2 = total_list_size2 * loss_per_mail2 * periodical_campaign_count2 * conversion_rate2
                    lost_revenue_per_period2 = lost_customers2 * customer_value2
                    cost_no_growth2 = period2 * lost_revenue_per_period2

                    table_df2 = pd.DataFrame({f"Period_({period_type2})": np.arange(0, (period2 + 1))})
                    table_df2["ListSize_when_NoGrowth"] = np.repeat(total_list_size2, period2 + 1)

                    table_df2 = table_df2 \
                        .assign(LostCustomers_when_NoGrowth = lambda x: x.ListSize_when_NoGrowth * loss_per_mail2 * periodical_campaign_count2 * conversion_rate2) \
                        .assign(Cost_when_NoGrowth = lambda x: x.LostCustomers_when_NoGrowth * customer_value2)
                    
                    if growth_type2 == "exponential":
                        table_df2 = table_df2 \
                            .assign(ListSize_with_Growth = lambda x: x.ListSize_when_NoGrowth * ((1 + growth_rate2) ** x[f"Period_({period_type2})"]))

                    if growth_type2 == "linear":
                        table_df2 = table_df2 \
                            .assign(ListSize_with_Growth = lambda x: x.ListSize_when_NoGrowth + (growth_rate2 * x[f"Period_({period_type2})"]))
                    
                    table_df2 = table_df2 \
                        .assign(LostCustomers_with_Growth = lambda x: x.ListSize_with_Growth * loss_per_mail2 * periodical_campaign_count2 * conversion_rate2) \
                        .assign(Cost_with_Growth = lambda x: x.LostCustomers_with_Growth * customer_value2) \

                    
                    table_df2 = table_df2[["Cost_when_NoGrowth", "Cost_with_Growth"]] \
                        .sum() \
                        .to_frame() \
                        .transpose()
                    
                    return table_df2

            table_df2 = mini_function()
        
            return table_df2
        
        summary_list = [temporary_function(x, y) for x, y in zip(parameter_grid_df["List_Growth_Rate_Month"], parameter_grid_df["Conversion_Rate"])]
        
        simulation_results_df = pd.concat(summary_list, axis = 0) \
            .reset_index() \
            .drop("index", axis = 1) \
            .merge(parameter_grid_df, left_index = True, right_index = True)
        
        simulation_results_df["Cost_when_NoGrowth"] = simulation_results_df["Cost_when_NoGrowth"].round(0).astype(int)
        simulation_results_df["Cost_with_Growth"] = simulation_results_df["Cost_with_Growth"].round(0).astype(int)
        simulation_results_df = simulation_results_df \
            .reindex(columns = ["List_Growth_Rate_Month", "Conversion_Rate", "Cost_with_Growth", "Cost_when_NoGrowth"])
        
        
        display(simulation_results_df)
        
        simulation_results_wide_df = simulation_results_df \
            .drop("Cost_when_NoGrowth", axis = 1) \
            .pivot(
                index = "List_Growth_Rate_Month",
                columns = "Conversion_Rate",
                values = "Cost_with_Growth"
            )
        
        display(simulation_results_wide_df)
        
        fig = px.imshow(
            simulation_results_wide_df,
            x = simulation_results_wide_df.columns,
            y = simulation_results_wide_df.index,
            origin = "lower",
            aspect = "auto",
            title = "Lead Cost Simulation",
            labels = dict(
                x = "Conversion Rate",
                y = f"Growth Rate ({period_type})",
                color = "Cost"
                ),
            **kwargs
            )
        
        fig.update_xaxes(
            tickvals=simulation_results_wide_df.columns,
            ticktext=[f"{x:.2f}" for x in simulation_results_wide_df.columns],
            tickmode='array'
        )

        fig.update_yaxes(
            tickvals=simulation_results_wide_df.index,
            ticktext=[f"{float(y):.4f}" for y in simulation_results_wide_df.index],
            tickmode='array'
        )

        fig.update_traces(xgap=0.5, ygap=0.5)
        
        return fig

################################# END OF FUNCTION #################################

# score_strategy_optimization()
# required packages: 
#   import pandas as pd
#   import numpy as np
#   import plotly.express as px

def make_strategy(
    df, 
    thresh = 0.99,
    id_column = "user_email",
    score_column = "Score_1",
    y_column = "made_purchase",
    for_marketing_team = False, 
    verbose = False
):
    
    # Ranking the leads
    leads_scored_small_df = df[[id_column, score_column, y_column]]

    leads_ranked_df = leads_scored_small_df \
        .sort_values(score_column, ascending = False) \
        .assign(rank = lambda x: np.arange(0, len(x[y_column])) + 1 ) \
        .assign(gain = lambda x: np.cumsum(x[y_column]) / np.sum(x[y_column]))
        
    # Make the Strategy
    strategy_df = leads_ranked_df \
        .assign(category = lambda x: np.where(x['gain'] <= thresh, "Hot-Lead", "Cold-Lead"))
    
    if for_marketing_team:
        strategy_df = df \
            .merge(
                right       = strategy_df[['category']],
                how         = 'left',
                left_index  = True,
                right_index = True
            )
    
    if verbose:
        print("Strategy Created.")
        
    return strategy_df


# 2.0 AGGREGATE THE LEAD STRATEGY RESULTS
#  aggregate_strategy_results()

def aggregate_strategy_results(strategy_df, y_column):
    
    results_df = strategy_df \
        .groupby('category') \
        .agg(
            count = (y_column, 'count'),
            sum_made_purchase = (y_column, 'sum')
        )
    
    return results_df


def strategy_calc_expected_value(
    results_df,
    list_size = 1e5,
    unsub_rate_per_periodic_campaign = 0.001,
    periodic_campaign_count = 5,
    avg_sales_per_period = 250000,
    avg_periodic_campaign_count = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    period = "month",
    verbose = False,
):
    
    # Confusion matrix calculations ----
    try:
        cold_lead_count = results_df['count']['Cold-Lead']
    except:
        cold_lead_count = 0

    try:
        hot_lead_count = results_df['count']['Hot-Lead']
    except:
        hot_lead_count = 0

    try:
        missed_purchases = results_df['sum_made_purchase']['Cold-Lead']
    except:
        missed_purchases = 0
        
    try:
        made_purchases = results_df['sum_made_purchase']['Hot-Lead']
    except:
        made_purchases = 0
        
    # Confusion matrix summaries
    
    total_count = (cold_lead_count + hot_lead_count)

    total_purchases = (missed_purchases + made_purchases)

    sample_factor = list_size / total_count

    sales_per_email_sent = avg_sales_per_period / avg_periodic_campaign_count
    
    # Preliminary Expected Value Calcuations
    
    # 3.1 [Savings] Cold That Are Not Targeted

    savings_cold_no_target = cold_lead_count * \
        periodic_campaign_count * unsub_rate_per_periodic_campaign * \
        customer_conversion_rate * avg_customer_value * \
        sample_factor
        
    # 3.2 [Cost] Missed Sales That Are Not Targeted

    missed_purchase_ratio = missed_purchases / (missed_purchases + made_purchases)

    cost_missed_purchases = sales_per_email_sent * periodic_campaign_count * missed_purchase_ratio
    
    # 3.3 [Cost] Hot Leads Targeted That Unsubscribe

    cost_hot_target_but_unsub = hot_lead_count * \
        periodic_campaign_count * unsub_rate_per_periodic_campaign * \
        customer_conversion_rate * avg_customer_value * \
        sample_factor
        
    # 3.4 [Savings] Sales Achieved

    made_purchase_ratio = made_purchases / (missed_purchases + made_purchases)

    savings_made_purchases = sales_per_email_sent * periodic_campaign_count * made_purchase_ratio
    
    # 4.2 Expected Monthly Value (Unrealized because of delayed nuture effect)

    ev = savings_made_purchases + \
        savings_cold_no_target - cost_missed_purchases

    # 4.3 Expected Monthly Savings (Unrealized until nurture takes effect)

    es = savings_cold_no_target - cost_missed_purchases


    # 4.4 Expected Saved Customers (Unrealized until nuture takes effect)

    esc = savings_cold_no_target / avg_customer_value
    
    if verbose:
        print(f"Expected Value: {'${:,.0f}'.format(ev)}")
        print(f"Expected Savings: {'${:,.0f}'.format(es)}")
        print(f"Sales by {period}: {'${:,.0f}'.format(savings_made_purchases)}")
        print(f"Saved Customers: {'{:,.0f}'.format(esc)}")
        
    return {
        'expected_value': ev,
        'expected_savings': es,
        f'sales_by_{period}': savings_made_purchases,
        'expected_customers_saved': esc
    }


# 4.0 OPTIMIZE THE THRESHOLD AND GENERATE A TABLE
#  strategy_create_thresh_table()

def strategy_create_thresh_table(
    leads_scored_df,
    thresh = np.linspace(0, 1, num=100),
    id_column = "user_email",
    score_column = "Score_1",
    y_column = "made_purchase",
    list_size = 1e5,
    unsub_rate_per_periodic_campaign = 0.005,
    periodic_campaign_count = 5,
    avg_sales_per_period = 250000,
    avg_periodic_campaign_count = 5,
    customer_conversion_rate = 0.05,
    avg_customer_value = 2000,
    period = "month",
    highlight_max = True,
    highlight_max_color = "yellow",
    verbose = False,
):
    thresh_df = pd.Series(thresh, name="threshold").to_frame()

    sim_results_list = [
        make_strategy(
            leads_scored_df,
            thresh=tup[0],
            id_column=id_column,
            score_column=score_column,
            y_column=y_column,
            verbose=verbose,
        )
        .pipe(aggregate_strategy_results, y_column=y_column)
        .pipe(
            strategy_calc_expected_value,
            list_size=list_size,
            unsub_rate_per_periodic_campaign=unsub_rate_per_periodic_campaign,
            periodic_campaign_count=periodic_campaign_count,
            avg_sales_per_period=avg_sales_per_period,
            avg_periodic_campaign_count=avg_periodic_campaign_count,
            customer_conversion_rate=customer_conversion_rate,
            avg_customer_value=avg_customer_value,
            period=period,
            verbose=verbose,
        )
        for tup in zip(thresh_df["threshold"])
    ]

    sim_results_df = pd.DataFrame(sim_results_list)

    thresh_optim_df = pd.concat([thresh_df, sim_results_df], axis=1)

    if highlight_max:
        thresh_optim_df = thresh_optim_df.style.highlight_max(
            color=highlight_max_color, axis=0
        )

    return thresh_optim_df


# 5.0 SELECT THE BEST THRESHOLD
#  def select_optimum_thresh()

def select_optimum_thresh(
    thresh_optim_df,
    optim_col = "expected_value",
    period = "month",
    periodic_sales_reduction_safeguard = 0.90,   # sales should not go below this threshold
    verbose = False
):
    
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
    
    # Find optim
    _filter_1 = thresh_optim_df[optim_col] == thresh_optim_df[optim_col].max()
    
    #Find safeguard
    _filter_2 = thresh_optim_df[f"sales_by_{period}"] >= periodic_sales_reduction_safeguard * thresh_optim_df[f"sales_by_{period}"].max()
    
    #Test if optim is in the safeguard
    if (all(_filter_1 + _filter_2 == _filter_2)):
        _filter = _filter_1
    else:
        _filter = _filter_2
    
    #Apply Filter
    thresh_selected = thresh_optim_df[_filter].head(1)
    
    # Values
    ret = thresh_selected["threshold"].values[0]
    
    if verbose:
        print(f"Optimal threshold: {ret}")
    
    return ret
    

def get_expected_value(thresh_optim_df, threshold, verbose = False):
    
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
    
    df = thresh_optim_df[thresh_optim_df["threshold"] >= threshold].head(1)
    
    if verbose:
        print("Expected Value Table:")
        print(df)
    
    return df


def plot_optim_thresh(
    thresh_optim_df,
    optim_col = "expected_value",
    periodic_sales_reduction_safeguard = 0.90,
    verbose = False
):
    
    # Handle styler object
    try:
        thresh_optim_df = thresh_optim_df.data
    except:
        thresh_optim_df = thresh_optim_df
    
    # Make the plot
    
    fig = px.line(
        thresh_optim_df,
        x = "threshold",
        y = "expected_value"
    )

    fig.add_hline(y = 0, line_color = "black")

    fig.add_vline(
        x = select_optimum_thresh(
            thresh_optim_df, 
            optim_col=optim_col, 
            periodic_sales_reduction_safeguard=periodic_sales_reduction_safeguard
            ),
        line_color = "red",
        line_dash = "dash"
    )
    
    if verbose:
        print("Plot created.")
    
    return fig


#*********
def score_strategy_optimization(
    leads_scored_df,
    id_column="",
    score_column="",
    y_column="",
    thresh=np.linspace(0, 1, num=100),
    optim_col="",
    periodic_sales_reduction_safeguard=0.90,
    for_marketing_team=True,
    list_size=1e5,
    unsub_rate_per_periodic_campaign=0.005,
    periodic_campaign_count=5,
    avg_sales_per_period=250000,
    avg_periodic_campaign_count=5,
    customer_conversion_rate=0.05,
    avg_customer_value=2000,
    period="month",
    highlight_max=True,
    highlight_max_color="yellow",
    verbose=False,
):
    """
    Optimizes a lead scoring strategy by selecting the optimal threshold for categorizing leads
    and calculating the expected outcomes based on the selected threshold.

    Parameters:
    ----------
    leads_scored_df : pandas.DataFrame
        The DataFrame containing the lead data, including lead IDs, scores, and outcomes.
    
    id_column : str, default=""
        The column in `leads_scored_df` representing the unique identifier for each lead (e.g., user email).

    score_column : str, default=""
        The column in `leads_scored_df` representing the score or probability that a lead will convert 
        into a positive outcome (e.g., purchase likelihood score).
    
    y_column : str, default=""
        The column in `leads_scored_df` representing the target outcome (e.g., whether the lead made a purchase).

    thresh : numpy array, default=np.linspace(0, 1, num=100)
        A range of threshold values to evaluate for categorizing leads into 'Hot-Lead' or 'Cold-Lead'.
    
    optim_col : str, default=""
        The column used to optimize the threshold selection. Typically, this will be 'expected_value' 
        or another metric reflecting business goals.
    
    periodic_sales_reduction_safeguard : float, default=0.90
        A safeguard to ensure that sales after the strategy is applied do not fall below this percentage 
        of the maximum possible sales.
    
    for_marketing_team : bool, default=True
        If True, merges the strategy DataFrame with the original lead DataFrame for easier analysis by 
        the marketing team.
    
    list_size : float, default=1e5
        The total number of leads or customers in the list, used to scale calculations for the entire list.
    
    unsub_rate_per_periodic_campaign : float, default=0.005
        The rate at which leads are expected to unsubscribe from periodic email campaigns.
    
    periodic_campaign_count : int, default=5
        The number of periodic campaigns that will be run, which impacts unsubscribes and conversions.
    
    avg_sales_per_period : float, default=250000
        The average periodic sales amount, used in calculating the value of targeting hot leads.
    
    avg_periodic_campaign_count : int, default=5
        The average number of campaigns run during the analysis period.
    
    customer_conversion_rate : float, default=0.05
        The expected conversion rate of leads into paying customers.
    
    avg_customer_value : float, default=2000
        The average value of a customer, used to calculate the overall impact of the campaign.
    
    period : str, default="month"
        The time period used in the analysis (e.g., "month", "year").

    highlight_max : bool, default=True
        Whether to highlight the maximum values in the optimization DataFrame.
    
    highlight_max_color : str, default="yellow"
        The color to use when highlighting the maximum values in the optimization DataFrame.
    
    verbose : bool, default=False
        If True, prints additional details during the function execution for debugging and tracking progress.

    Returns:
    -------
    dict :
        A dictionary with the following keys:
        
        - `lead_strategy_df`: pandas DataFrame containing the final strategy with leads categorized as 'Hot-Lead' or 'Cold-Lead' based on the optimal threshold.
        
        - `expected_value`: pandas DataFrame with the expected value and other calculations for the optimal threshold.
        
        - `thresh_optim_df`: pandas DataFrame or Styler containing the results of the threshold optimization process, including expected value, sales, and savings.
        
        - `thresh_plot`: Plotly figure showing the expected value across the threshold values, with the optimal threshold marked.

    Example:
    --------
    >>> optimization_results = score_strategy_optimization(
            leads_scored_df=leads_scored_df,
            id_column="user_email", 
            score_column="Score_1", 
            y_column="made_purchase",
            thresh=np.linspace(0, 1, num=100),
            optim_col="expected_value",
            periodic_sales_reduction_safeguard=0.90,
            for_marketing_team=True,
            list_size=1e5,
            unsub_rate_per_periodic_campaign=0.005,
            periodic_campaign_count=5,
            avg_sales_per_period=250000,
            avg_periodic_campaign_count=5,
            customer_conversion_rate=0.05,
            avg_customer_value=2000,
            period="month",
            highlight_max=True,
            highlight_max_color="yellow",
            verbose=False
        )
        
    >>> lead_strategy_df = optimization_results['lead_strategy_df']
    >>> expected_value = optimization_results['expected_value']
    >>> thresh_optim_df = optimization_results['thresh_optim_df']
    >>> thresh_plot = optimization_results['thresh_plot']

    Notes:
    ------
    - Ensure that the input DataFrame (`leads_scored_df`) contains the appropriate columns for `id_column`, 
      `score_column`, and `y_column`.
    - The function is modular and can be extended with additional parameters or business logic.
    - Setting `verbose=True` will provide useful debugging information if issues arise during execution.
    """
    
    # Lead strategy create thresh table
    thresh_optim_df = strategy_create_thresh_table(
        leads_scored_df,
        thresh=thresh,
        id_column=id_column,
        score_column=score_column,
        y_column=y_column,
        list_size=list_size,
        unsub_rate_per_periodic_campaign=unsub_rate_per_periodic_campaign,
        periodic_campaign_count=periodic_campaign_count,
        avg_sales_per_period=avg_sales_per_period,
        avg_periodic_campaign_count=avg_periodic_campaign_count,
        customer_conversion_rate=customer_conversion_rate,
        avg_customer_value=avg_customer_value,
        period=period,
        highlight_max=highlight_max,
        highlight_max_color=highlight_max_color,
        verbose=verbose,
    )

    # Lead select optimum thresh
    thresh_optim = select_optimum_thresh(
        thresh_optim_df,
        optim_col=optim_col,
        period=period,
        periodic_sales_reduction_safeguard=periodic_sales_reduction_safeguard,
        verbose=verbose,
    )

    # Expected value
    expected_value = get_expected_value(
        thresh_optim_df, threshold=thresh_optim, verbose=verbose
    )

    # Lead plot
    thresh_plot = plot_optim_thresh(
        thresh_optim_df,
        optim_col=optim_col,
        periodic_sales_reduction_safeguard=periodic_sales_reduction_safeguard,
        verbose=verbose,
    )

    # Recalculate Lead Strategy
    lead_strategy_df = make_strategy(
        leads_scored_df,
        thresh=thresh_optim,
        id_column=id_column,
        score_column=score_column,
        y_column=y_column,
        for_marketing_team=for_marketing_team,
        verbose=verbose,
    )

    # Dictionary for return
    ret = dict(
        lead_strategy_df=lead_strategy_df,
        expected_value=expected_value,
        thresh_optim_df=thresh_optim_df,
        thresh_plot=thresh_plot,
    )

    return ret

################################# END OF FUNCTION #################################
