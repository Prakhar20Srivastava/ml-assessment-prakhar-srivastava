# B1. Problem Formulation

## (a)Formulate this as a machine learning problem. State clearly: what is the target variable, what are the candidate input features, and what type of ML problem is this? Justify your choice of problem type.

Target Variable (What you predict)
Number of items sold (or incremental uplift in items sold during the promotion period)
This is a continuous numeric value, which directly aligns with the business objective: maximizing sales volume.
Candidate Input Features (Predictors)
combine store-level, promotion-level, and contextual features:
1. Store Characteristics
   	Store size (sq. ft.)
   	Location type (urban / semi-urban / rural)
	Historical average sales
	Monthly footfall
	Inventory levels
2. Customer & Demographics
	Age distribution
	Income levels
	Purchase behavior patterns (avg basket size, frequency)
3. Competition & Market Context
	Competition density (number of nearby competitors)
	Local events / seasonality
	Regional demand trends
4. Promotion Details
	Promotion type (Flat Discount, BOGO, etc.) → categorical
	Discount % (if applicable)
	Duration of promotion
5. Temporal Features
	Month / season
	Holiday / festive period indicator
	Year-over-year trends
This problem can be best formulated as a:
Supervised Regression Problem. Justification as below

The target variable is continuous (items sold), which naturally fits regression.
You have historical labeled data: (store conditions + promotion type → items sold).
Regression allows to estimate expected sales for each promotion option at a given store.

How It Solves the Business Problem
Once trained:
For a given store and month, simulate all 5 promotions as input.
The model predicts expected items sold for each promotion.
Choose the promotion with the highest predicted sales.


## (b) The company currently measures performance using total sales revenue. Explain why using items sold (sales volume) is a more reliable target variable for this problem. What broader principle does this illustrate about target variable selection in real-world ML projects?

| Aspect                          | **Sales Revenue**                                          | **Items Sold (Volume)**                 |
| ------------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| What it measures                | Monetary outcome (price × quantity)                        | Pure demand (number of units sold)      |
| Sensitivity to promotions       | Highly affected by discounts, pricing, and offer structure | Reflects actual customer response       |
| Comparability across promotions | ❌ Not comparable (BOGO vs discount distort differently)    | ✅ Comparable across all promotion types |
| Signal clarity                  | Noisy (mixes price + demand effects)                       | Clean (isolates demand impact)          |
| Alignment with objective        | Indirect                                                   | Direct (maximize purchases)             |
| Risk                            | Misleading conclusions                                     | More reliable learning                  |
| Business interpretability       | Confounded                                                 | Intuitive                               |


| Principle                    | Explanation                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------- |
| Target–decision alignment    | Target should reflect the outcome directly impacted by the decision variable  |
| Avoid composite metrics      | Metrics like revenue combine multiple effects → harder to learn true patterns |
| Isolate causal impact        | Choose targets that best capture cause-effect (promotion → demand)            |
| Reduce noise                 | Cleaner targets improve model learning and generalization                     |
| Decompose complex objectives | Optimize demand first, then layer pricing/profit decisions                    |



## (c) A junior analyst suggests running one single global model across all 50 stores. Propose and justify an alternative modelling strategy that accounts for the fact that stores in different locations respond very differently to the same promotion.

A single global model is convenient, but it will average out important differences between urban, semi-urban, and rural stores. That usually leads to “good on average, poor for many” recommendations.

A better approach is to explicitly model heterogeneity in promotion response.

Recommended Strategy: Hierarchical / Segmented Modeling
Option 1: Cluster + Local Models (Practical & Effective)

Cluster stores based on similarity:
	Location type (urban / rural)	
	Footfall
	Customer demographics
	Sales patterns
Train separate models per cluster:
e.g., Urban model, Rural model, Premium stores model

Why this works

Stores within a cluster behave similarly → cleaner patterns
Avoids dilution of signals from very different store types
Improves prediction accuracy for each segment

Trade-off
Less data per model → risk of overfitting (manageable with proper tuning)



# B2. Data and EDA Strategy

## (a) Data Joining, Grain, and Aggregations

Integrate the four tables using common keys:

Transactions table (fact table)
Contains: transaction_id, store_id, date, items_sold, revenue

Store attributes table
Join on: store_id

Promotion details table
Join on: store_id + date (or promotion_id if available)

Calendar table
Join on: date

Grain of final dataset

One row = one store × one time period × one promotion

Typically: Store–Month–Promotion level (or Store–Week if more granular)
Aggregations before modelling

Since transactions are at a lower level, aggregate to the modeling grain:

Target aggregation
items_sold = SUM(items_sold)
revenue = SUM(revenue) (optional)

Behavioral metrics
avg_basket_size = AVG(items per transaction)
num_transactions = COUNT(transaction_id)

Footfall proxy
unique_customers = COUNT(DISTINCT customer_id) (if available)
Promotion features
Promotion type (categorical)
Discount %, duration
Calendar features
% weekends in month
festival_flag (binary or count)

Why this grain
Aligns with decision frequency (monthly promotions)
Reduces noise from transaction-level variability
Enables consistent comparison across stores


## (b) Exploratory Data Analysis (EDA)

1. Promotion vs Items Sold (Boxplot / Bar Chart)

What to check:

Distribution of items sold across promotion types

Insights: Which promotions perform better on average
Variability across promotions

Impact:
Helps validate signal strength
May inspire interaction features (promotion × store type)

2. Sales Trends Over Time (Line Chart)

What to check:

Monthly/weekly sales trends
Seasonality and spikes (festivals)

Insights:
Strong seasonal patterns
Trend shifts

Impact: Add time features (month, season, lag variables)

3. Store Segmentation Analysis (Scatter / Grouped Plots)

(e.g., footfall vs items sold, split by location)

What to check:

Differences between urban vs rural stores
Relationship between footfall and sales

Insights:
Heterogeneity across store types

Impact:
Justifies clustering or segmented models
Add interaction features

4. Correlation / Feature Importance (Heatmap)

What to check:

Correlation between numerical features and target

Insights:
Identify strong predictors (footfall, inventory, etc.)
Detect multicollinearity

Impact:
Feature selection
Remove redundant variables

5. Distribution Analysis (Histogram of target)

What to check:
Skewness in items sold

Insights:
Long tails / outliers

Impact:
Apply log transformation if needed
Choose robust models

## (c) Handling Promotion Imbalance

Problem
80% of data = no promotion

Model may:
Bias toward predicting non-promo scenarios
Underlearn promotion effects

Solutions
1. Resampling
	Oversample promotion cases
	Undersample non-promo cases
2. Weighting
	Assign higher weights to promotion rows during training
3. Stratified modeling
	Ensure balanced representation in train/validation splits
4. Feature design
	Explicit promotion_flag

Interaction terms (promotion × store features)

Key Idea: Ensure the model learns promotion impact, not just dominant “no-promo” patterns.


# B3. Model Evaluation and Deployment

## Part (a) — Train-test split, why random fails, and evaluation metrics

The fundamental problem with random splitting here is that this is time-series panel data. If the split is done randomly, the model could end up training on Month 30 data and being tested on Month 6 — which means it has effectively seen the future before being asked to predict it. This is data leakage, and it produces evaluation scores that look great on paper but fall apart the moment the model goes live. On top of that, promotions tend to follow seasonal patterns — December festive spikes, monsoon-driven footfall shifts in Indian retail — and a random shuffle tears through that structure completely, leaving the model with nothing meaningful to learn from.

The better approach is to split by time. With three years of data across 50 stores, a sensible breakdown would be:

- Training set: Months 1–28 (all 50 stores) — roughly 78% of the data
- Validation set: Months 29–32 — for tuning hyperparameters and selecting features
- Test set: Months 33–36 — kept completely untouched until final evaluation

This mirrors how the model will actually be used — it will always be making predictions about a future it hasn't seen, using only historical data. For a more rigorous test of how well it generalises, a walk-forward validation approach is worth considering: train on Months 1–N, predict Month N+1, then roll the window forward and repeat, averaging results across all iterations.

**Evaluation metrics and their business interpretation:**

| Metric | Formula | Business interpretation |
|--------|---------|------------------------|
| **RMSE on items sold** | √(mean squared error) | Penalises large errors more heavily, which matters here — a badly wrong promotion recommendation can cost hundreds of units. |
| **Promotion accuracy** | % of months where recommended promotion = actual best promotion | Straightforward to explain to a marketing team: how often did the model pick the right promotion? |
| **Regret / opportunity cost** | (Items sold under best promotion) − (Items sold under recommended promotion) | Perhaps the most business-relevant of all — it captures the actual cost of following a wrong recommendation in units lost. |
| **Lift over baseline** | Improvement over a naïve benchmark (e.g. "always run Flat Discount") | Answers the question: is the model actually better than just doing what has always been done? |

These metrics need to be read together rather than in isolation. A low RMSE with high regret, for example, suggests the model is good at predicting sales volumes but still tends to recommend the wrong promotion. And promotion accuracy on its own can be deceptive — a model that always recommends the most common promotion will score reasonably well without having learned anything useful.

---

## Part (b) — Feature importance to explain different recommendations

The key tool here is local feature importance, specifically SHAP values. Unlike global feature importance — which averages out what matters across all stores and all months — SHAP breaks down each individual prediction and shows exactly which features pushed the model toward a particular recommendation, and by how much.

The process would be to run SHAP on Store 12's December prediction and again on its March prediction, then compare the two. In December, the expectation is that the month indicator, the proportion of Gold and Silver loyalty members, and average basket value will emerge as the dominant drivers — and it makes intuitive sense. Loyal, higher-spending customers in a festive month are much more likely to respond to points accumulation than to a straightforward price cut. Come March, those same features become largely irrelevant. What takes over instead is footfall volume, local competition density, and price sensitivity signals — suggesting a different kind of customer walking through the door, one who is browsing rather than loyal, and who needs a visible discount to convert.

When presenting this to the marketing team, it is worth avoiding raw SHAP numbers entirely — they mean very little to someone who isn't familiar with the method. A more effective approach is to translate the findings into plain language: "In December, more than two-thirds of Store 12's customers are loyalty cardholders, and they respond to points offers at roughly 2.4 times the rate they respond to discounts during the festive season. In March, footfall climbs by around 34% but much of that growth comes from first-time or occasional visitors who have no loyalty history and are primarily motivated by price." That kind of framing connects the model's logic to something the marketing team already understands — customer behaviour.

---

## Part (c) — End-to-end deployment and monitoring

**Saving the model:**

Once trained, the model should be saved in a format suited to the framework being used — joblib works well for scikit-learn pipelines, while XGBoost and LightGBM have their own native binary formats. ONNX is a reasonable choice if portability across frameworks matters. The important thing is to save the entire pipeline, not just the model itself. The preprocessing steps — encoders, scalers, imputers — need to travel with it, along with a record of the feature names, their order, and key metadata like the training date and validation RMSE. All of this should sit in a versioned model registry, whether that is MLflow or something as simple as a versioned S3 bucket with a manifest file.

**Monthly inference pipeline:**

Each month, before recommendations go out, the pipeline should run through four steps:

1. Pull the latest store-level data from the data warehouse — footfall counts, last month's sales, the promotional calendar, any competitor activity flags
2. Run it through the same preprocessing pipeline used at training time, without re-fitting anything. Re-fitting scalers or encoders on new data is a subtle but serious mistake — it changes what the features mean and introduces drift
3. Generate predictions across all 50 stores using model.predict()
4. Write the outputs — store ID, recommended promotion, confidence score — to wherever the marketing team accesses them, whether that is a database table or a dashboard

**Monitoring and degradation detection:**

Three things are worth monitoring on an ongoing basis.

The first is data drift. Each month, the incoming feature distributions should be compared to what the model saw during training, using something like the Population Stability Index for continuous variables or a chi-squared test for categorical ones. If the profile of customers or the patterns of footfall start shifting significantly, the model's internal logic may no longer apply — even if nothing has visibly gone wrong yet.

The second is prediction drift. If the model starts recommending Loyalty Points Bonus for 80% of stores in a month when it historically recommended it for around 30%, that is a signal worth investigating. The recommendations themselves can be an early warning system.

The third is outcome monitoring, though this one is necessarily lagged. Once the actual sales results come in, they should be compared to what the model predicted, tracking rolling RMSE and regret over a three-month window. If RMSE climbs more than 20% above its training baseline, or if regret grows by more than 15%, that should trigger a retraining review.

**Retraining strategy:**

When retraining does become necessary, it is generally better to extend the training window rather than start from scratch. Adding recent months to the dataset — while keeping the temporal order intact — lets the model adapt to shifts in consumer behaviour, new competitors entering the market, or structural changes like post-pandemic footfall patterns, without throwing away years of useful historical signal.
