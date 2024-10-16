
#  uvicorn backend_fastapi:app --reload --port 8000


import os
import pandas as pd
import numpy as np
from my_pandas_extensions.business import score_strategy_optimization
import h2o
import janitor as jn
import re
from my_pandas_extensions.database import collect_data_merge
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import json
import joblib
from my_pandas_extensions.modeling import ML_Util


collect_data_merge(conn_string = "sqlite:///example_data.sqlite", extract_from_dictionary = True)
products_df = df_Products
tags_df = df_Tags
transactions_df = df_Transactions
website_df = df_Website
products_df["product_id"] = products_df["product_id"].astype("int")
subscribers_df = df_Subscribers
subscribers_df["mailchimp_id"] = subscribers_df["mailchimp_id"].astype("int")
subscribers_df["member_rating"] = subscribers_df["member_rating"].astype("int")
subscribers_df["optin_time"] = subscribers_df["optin_time"].astype("datetime64")
tags_df["mailchimp_id"] = tags_df["mailchimp_id"].astype("int")
transactions_df["purchased_at"] = transactions_df["purchased_at"].astype("datetime64")
transactions_df["product_id"] = transactions_df["product_id"].astype("int")
website_df["date"] = website_df["date"].astype("datetime64")
website_df["pageviews"] = website_df["pageviews"].astype("int")
website_df["organicsearches"] = website_df["organicsearches"].astype("int")
website_df["sessions"] = website_df["sessions"].astype("int")
user_events_df = tags_df \
    .groupby("mailchimp_id") \
    .agg(
        {
            "tag": "count"
        }
    ) \
    .set_axis(["tag_count"], axis = 1) \
    .reset_index()
subscribers_joined_df = subscribers_df \
    .merge(
        right = user_events_df,
        left_on = "mailchimp_id",
        right_on = "mailchimp_id",
        how = "left") \
    .fillna(
        {
            "tag_count": 0
        }
    )
subscribers_joined_df["tag_count"] = subscribers_joined_df["tag_count"].astype("int")
made_purchase_df = transactions_df["user_email"].unique()
subscribers_joined_df["made_purchase"] = subscribers_df["user_email"].isin(made_purchase_df).astype("int")
leads_df = subscribers_joined_df
df = leads_df \
    .groupby("country_code") \
    .agg(
        sales = ("made_purchase", np.sum),
        prop_in_group = ("made_purchase", np.mean)
    ) \
    .assign(prop_overall = lambda x: x.sales / x.sales.sum()) \
    .assign(prop_cumsum = lambda x: x.prop_overall.cumsum()) \
    .reset_index() \
    .assign(country_code = lambda x: x.country_code.str.lower()) \
    .set_index("country_code") \
    .sort_values(by = "sales", ascending = False)
date_max = leads_df["optin_time"].max()
date_min = leads_df["optin_time"].min()
time_range = date_min - date_max
time_range.days
leads_df["optin_time"] - date_max
(leads_df["optin_time"] - date_max).dt.days
leads_df["optin_days"] = (leads_df["optin_time"] - date_max).dt.days
leads_df["user_email"]
"garrick.langworth@gmail.com".split("@")
"garrick.langworth@gmail.com".split("@")[1]
leads_df["email_provider"] = leads_df["user_email"].map(lambda x: x.split("@")[1])
leads_df["tag_count"] / abs(leads_df["optin_days"])
leads_df["tag_count_by_optin_day"] = leads_df["tag_count"] / abs(leads_df["optin_days"] - 1)
tags_wide_leads_df = tags_df \
    .assign(value = lambda x: 1) \
    .pivot(
        index = "mailchimp_id",
        columns = "tag",
        values = "value"
    ) \
    .fillna(value = 0) \
    .pipe(func = jn.clean_names)
tags_wide_leads_df.columns = tags_wide_leads_df.columns \
    .to_series() \
    .apply(func = lambda x: f"tag_{x}") \
    .to_list()
tags_wide_leads_df = tags_wide_leads_df.reset_index()
leads_tags_df = leads_df \
    .merge(
        right = tags_wide_leads_df,
        left_on = "mailchimp_id",
        right_on = "mailchimp_id",
        how = "left")
for col in leads_tags_df.columns:
    if re.match(pattern = "^tag", string = col):
        leads_tags_df[col] = leads_tags_df[col].fillna(0)
countries_to_keep = df.query("sales >= 6") \
    .index \
    .to_list()
leads_tags_df["country_code"] = leads_tags_df["country_code"] \
    .map(lambda x: x if x in countries_to_keep else "other")
leads_df = leads_tags_df



app = FastAPI()


@app.get("/")
async def main():
    content = """
    <body>
    <h1>Welcome to the UTBP Web App API.</h1>
    <p>Navigate to the <code>/docs</code> to see the API documentation.</p>
    </body>
    """
    
    return HTMLResponse(content = content)

@app.get("/get_dataset")
async def get_dataset():
    
    json = leads_df.to_json()

    return JSONResponse(json)

@app.post("/data")
async def data(request: Request):

    request_body = await request.body()

    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json)
        
    leads_json = leads_df.to_json()
    
    return JSONResponse(leads_json)


@app.post("/predict")
async def predict(request: Request):
    
    request_body = await request.body()
    
    data_json = json.loads(request_body)
    leads_df = pd.read_json(data_json)

    h2o.init()
    loaded_model = h2o.load_model("h2o_models\StackedEnsemble_BestOfFamily_4_AutoML_1_20241015_104849")

    output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")

    additional_model_info = joblib.load(os.path.join(output_dir_h2o, 'additional_model_info.pkl'))
    best_threshold = additional_model_info['threshold']
    venn_abers = additional_model_info['venn_abers']

    class CalibratedModelClassification:
        def __init__(self, model, calibrator):
            self.model = model
            self.calibrator = calibrator

        def predict_proba(self, data):
            if isinstance(data, pd.DataFrame):
                data_h2o = h2o.H2OFrame(data)
                for col in data.columns:
                    if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                        data_h2o[col] = data_h2o[col].asfactor()
            elif isinstance(data, h2o.H2OFrame):
                data_h2o = data
            else:
                raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

            preds = self.model.predict(data_h2o)
            preds_df = preds.as_data_frame()

            if 'p0' in preds_df.columns and 'p1' in preds_df.columns:
                p_raw = preds_df[['p0', 'p1']].values
            else:
                raise KeyError("Class probabilities not found in model output.")

            calibrated_probs = self.calibrator.predict_proba(p_raw)

            prob_lower = calibrated_probs[0]
            prob_upper = calibrated_probs[1]
            prob_mean = (prob_lower + prob_upper) / 2

            return prob_mean[:, 1]

        def predict_dataframe(self, data, include_data=True, threshold=0.5):
            if isinstance(data, pd.DataFrame):
                data_h2o = h2o.H2OFrame(data)
                for col in data.columns:
                    if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                        data_h2o[col] = data_h2o[col].asfactor()
            elif isinstance(data, h2o.H2OFrame):
                data_h2o = data
            else:
                raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

            preds = self.model.predict(data_h2o)
            preds_df = preds.as_data_frame()

            p0 = preds_df['p0'].values
            p1 = preds_df['p1'].values
            p_raw = np.column_stack((p0, p1))

            calibrated_probs = self.calibrator.predict_proba(p_raw)

            prob_lower = calibrated_probs[0]
            prob_upper = calibrated_probs[1]
            prob_mean = (prob_lower + prob_upper) / 2

            prob_sum = prob_mean.sum(axis=1).reshape(-1, 1)
            normalized_probs = prob_mean / prob_sum

            preds_df['Score_0'] = normalized_probs[:, 0]
            preds_df['Score_1'] = normalized_probs[:, 1]
            preds_df['Label'] = (preds_df['Score_1'] > threshold).astype(int)

            if include_data:
                return pd.concat([data.reset_index(drop=True), preds_df[['Score_0', 'Score_1', 'Label']]], axis=1)
            else:
                return preds_df[['Score_0', 'Score_1', 'Label']]

        def save(self, directory):
            self.model_path = h2o.save_model(model=self.model, path=directory, force=True)
            save_path = os.path.join(directory, "calibrated_model.pkl")
            joblib.dump({'calibrator': self.calibrator, 'model_path': self.model_path}, save_path)
            print(f"CalibratedModel saved to: {save_path}")

        @classmethod
        def load(cls, directory):
            load_path = os.path.join(directory, "calibrated_model.pkl")
            data = joblib.load(load_path)
            model = h2o.load_model(data['model_path'])
            calibrator = data['calibrator']
            return cls(model, calibrator)

    class CalibratedModelWithOptimizedThreshold:
        def __init__(self, calibrated_model, threshold=0.5):
            self.calibrated_model = calibrated_model
            self.threshold = threshold

        def predict_proba(self, h2o_test):
            return self.calibrated_model.predict_proba(h2o_test)
        
        def predict_dataframe(self, data, include_data=True):
            preds_df = self.calibrated_model.predict_dataframe(data, include_data=include_data, threshold=self.threshold)
            return preds_df

        def predict(self, h2o_test):
            y_pred_proba = self.predict_proba(h2o_test)
            return (y_pred_proba > self.threshold).astype(int)

        def save(self, path):
            joblib.dump(self, path)

        @staticmethod
        def load(path):
            return joblib.load(path)
        
    best_model_calibrated_h2o = CalibratedModelClassification(loaded_model, venn_abers)
    final_model = CalibratedModelWithOptimizedThreshold(best_model_calibrated_h2o, threshold=best_threshold)

    leads_scored_df = final_model.predict_dataframe(leads_df, include_data=True)
    leads_scored_df = leads_scored_df.reindex(columns = (["Score_0", "Score_1", "made_purchase"] + [col for col in leads_scored_df.columns if col not in ["Score_0", "Score_1", "made_purchase"]]))

    scores = leads_scored_df[["Score_1"]].to_json()
    
    return JSONResponse(scores)


@app.post("/calculate_strategy")
async def calculate_strategy(
    periodic_sales_reduction_safeguard:float=0.90,
    list_size:int=100000,
    unsub_rate_per_periodic_campaign:float=0.005,
    periodic_campaign_count:int=5,
    avg_sales_per_period:float=250000.0,
    avg_periodic_campaign_count:int=5,
    customer_conversion_rate:float=0.05,
    avg_customer_value:float=2000.0
):
    
    h2o.init()
    loaded_model = h2o.load_model("h2o_models\StackedEnsemble_BestOfFamily_4_AutoML_1_20241015_104849")

    output_dir_h2o = os.path.join(os.getcwd(), "h2o_models")

    additional_model_info = joblib.load(os.path.join(output_dir_h2o, 'additional_model_info.pkl'))
    best_threshold = additional_model_info['threshold']
    venn_abers = additional_model_info['venn_abers']

    class CalibratedModelClassification:
        def __init__(self, model, calibrator):
            self.model = model
            self.calibrator = calibrator

        def predict_proba(self, data):
            if isinstance(data, pd.DataFrame):
                data_h2o = h2o.H2OFrame(data)
                for col in data.columns:
                    if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                        data_h2o[col] = data_h2o[col].asfactor()
            elif isinstance(data, h2o.H2OFrame):
                data_h2o = data
            else:
                raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

            preds = self.model.predict(data_h2o)
            preds_df = preds.as_data_frame()

            if 'p0' in preds_df.columns and 'p1' in preds_df.columns:
                p_raw = preds_df[['p0', 'p1']].values
            else:
                raise KeyError("Class probabilities not found in model output.")

            calibrated_probs = self.calibrator.predict_proba(p_raw)

            prob_lower = calibrated_probs[0]
            prob_upper = calibrated_probs[1]
            prob_mean = (prob_lower + prob_upper) / 2

            return prob_mean[:, 1]

        def predict_dataframe(self, data, include_data=True, threshold=0.5):
            if isinstance(data, pd.DataFrame):
                data_h2o = h2o.H2OFrame(data)
                for col in data.columns:
                    if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
                        data_h2o[col] = data_h2o[col].asfactor()
            elif isinstance(data, h2o.H2OFrame):
                data_h2o = data
            else:
                raise ValueError("Input data must be a pandas DataFrame or an H2OFrame.")

            preds = self.model.predict(data_h2o)
            preds_df = preds.as_data_frame()

            p0 = preds_df['p0'].values
            p1 = preds_df['p1'].values
            p_raw = np.column_stack((p0, p1))

            calibrated_probs = self.calibrator.predict_proba(p_raw)

            prob_lower = calibrated_probs[0]
            prob_upper = calibrated_probs[1]
            prob_mean = (prob_lower + prob_upper) / 2

            prob_sum = prob_mean.sum(axis=1).reshape(-1, 1)
            normalized_probs = prob_mean / prob_sum

            preds_df['Score_0'] = normalized_probs[:, 0]
            preds_df['Score_1'] = normalized_probs[:, 1]
            preds_df['Label'] = (preds_df['Score_1'] > threshold).astype(int)

            if include_data:
                return pd.concat([data.reset_index(drop=True), preds_df[['Score_0', 'Score_1', 'Label']]], axis=1)
            else:
                return preds_df[['Score_0', 'Score_1', 'Label']]

        def save(self, directory):
            self.model_path = h2o.save_model(model=self.model, path=directory, force=True)
            save_path = os.path.join(directory, "calibrated_model.pkl")
            joblib.dump({'calibrator': self.calibrator, 'model_path': self.model_path}, save_path)
            print(f"CalibratedModel saved to: {save_path}")

        @classmethod
        def load(cls, directory):
            load_path = os.path.join(directory, "calibrated_model.pkl")
            data = joblib.load(load_path)
            model = h2o.load_model(data['model_path'])
            calibrator = data['calibrator']
            return cls(model, calibrator)

    class CalibratedModelWithOptimizedThreshold:
        def __init__(self, calibrated_model, threshold=0.5):
            self.calibrated_model = calibrated_model
            self.threshold = threshold

        def predict_proba(self, h2o_test):
            return self.calibrated_model.predict_proba(h2o_test)
        
        def predict_dataframe(self, data, include_data=True):
            preds_df = self.calibrated_model.predict_dataframe(data, include_data=include_data, threshold=self.threshold)
            return preds_df

        def predict(self, h2o_test):
            y_pred_proba = self.predict_proba(h2o_test)
            return (y_pred_proba > self.threshold).astype(int)

        def save(self, path):
            joblib.dump(self, path)

        @staticmethod
        def load(path):
            return joblib.load(path)
        
    best_model_calibrated_h2o = CalibratedModelClassification(loaded_model, venn_abers)
    final_model = CalibratedModelWithOptimizedThreshold(best_model_calibrated_h2o, threshold=best_threshold)

    leads_scored_df = final_model.predict_dataframe(leads_df, include_data=True)

    leads_scored_df = leads_scored_df.reindex(columns = (["Score_0", "Score_1", "made_purchase"] + [col for col in leads_scored_df.columns if col not in ["Score_0", "Score_1", "made_purchase"]]))
        
    
    optimization_results = score_strategy_optimization(
        leads_scored_df=leads_scored_df,
        id_column="user_email", 
        score_column="Score_1", 
        y_column="made_purchase",
        thresh=np.linspace(0, 1, num=100),
        optim_col="expected_value",
        periodic_sales_reduction_safeguard=periodic_sales_reduction_safeguard,
        for_marketing_team=True,
        list_size=list_size,
        unsub_rate_per_periodic_campaign=unsub_rate_per_periodic_campaign,
        periodic_campaign_count=periodic_campaign_count,
        avg_sales_per_period=avg_sales_per_period,
        avg_periodic_campaign_count=avg_periodic_campaign_count,
        customer_conversion_rate=customer_conversion_rate,
        avg_customer_value=avg_customer_value,
        period="month",
        highlight_max=False,
        highlight_max_color="green",
        verbose=False
    )
    
    
    results = {
        "lead_strategy": optimization_results["lead_strategy_df"].to_json(),
        "expected_value": optimization_results["expected_value"].to_json(),
        "thresh_optim_table": optimization_results["thresh_optim_df"].to_json()
    }
    
    return JSONResponse(results)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)