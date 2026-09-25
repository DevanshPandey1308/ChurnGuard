# ChurnGuard Power BI reporting layer

Power BI is the business reporting layer. The React app remains the operational
surface for single and batch scoring, analytics, explainability, and
methodology. The Power BI report should consume saved ChurnGuard scores and
global model-importance outputs; it must not recreate inference or training in
DAX.

## Prepared data contract

Run the exporter from the repository root with frozen `models/` artifacts and
the existing global SHAP artifact available. The raw-transaction mode below
builds the complete configured population and scores it directly through the
same shared inference helper used by FastAPI:

```powershell
.venv\Scripts\python.exe -m churnguard.reporting.powerbi_export `
  --raw-transactions "data[1].csv" `
  --output-dir artifacts/powerbi `
  --model-artifacts models `
  --shap-importance artifacts/shap/global_feature_importance.csv `
  --horizon-days 90 `
  --encoding latin1
```

This mode calls the existing `clean_transactions`, `build_snapshots`, and
`build_features` functions. Snapshot dates come from the frozen artifact's
training and validation dates plus the configured final-test date; the model's
artifact format does not currently serialize its final-test dates. It applies
the frozen train-only numeric medians and categorical schema, then calls the
same model scoring helper used by the API. Unseen category levels become
missing categories under the frozen pandas category schema, matching the
existing LightGBM preprocessing convention. The public API's request
validation remains strict. No labels are constructed.

The original `--input <flat-feature-csv>` mode remains available; it validates
the feature contract and calls `/batch_score`, so a running API is required
for that mode. Both modes join country/snapshot metadata and validate score
alignment. Neither mode fits a model or invents a score. Both write:

### `churnguard_customer_scores.csv`

| Column | Source / meaning |
|---|---|
| `CustomerID` | Scoring input / API output identifier |
| `snapshot_date` | Scoring input / API output snapshot date |
| `country` | Existing model input, joined from the scoring input |
| `churn_probability` | Calibrated probability returned by `/batch_score` |
| `predicted_90d_value` | Model-predicted future 90-day net spend returned by `/batch_score` |
| `risk_weighted_value` | Existing API output: churn probability × predicted 90-day value |
| `churn_rank` | Existing within-snapshot API ranking |
| `value_rank` | Existing within-snapshot API ranking |
| `risk_value_rank` | Existing within-snapshot risk-weighted ranking |

The API does not return a `risk_band` or define a high-risk cutoff. The export
does not invent either. Use the continuous churn probability distribution and
rank outputs until the business approves a threshold. Negative predicted net
spend and risk-weighted values are retained as model outputs. They are not
guaranteed recoverable revenue.

### `churnguard_global_feature_importance.csv`

This is a validated copy of the existing `artifacts/shap/global_feature_importance.csv`
with `feature`, `mean_abs_shap`, `mean_shap`, and `rank`. It is global
importance from the existing June held-out analysis, not batch-specific SHAP
and not causal impact. No customer-level explanations are exported.

Generated CSVs are placed under `artifacts/powerbi/`, which is ignored by git.
Re-run the exporter for each population and refresh the CSV sources in Power
BI. The full `data[1].csv` output generated in this run contains 8,968 customer
snapshots: March 1 (1,682), April 1 (2,134), May 1 (2,434), and June 1 (2,718)
2011. These are repeated customer snapshots rather than unique independent
customers. The raw file had 541,909 rows; canonical cleaning retained 401,604.

## Report model and measures

Load the two CSVs as separate tables named `ChurnGuard Scores` and
`Global Feature Importance`. Set `snapshot_date` to Date, `CustomerID` and rank
columns to Whole Number, `country`/`feature` to Text, and probability/importance
and value columns to Decimal Number. Do not create a relationship between the
two tables.

Create these measures on `ChurnGuard Scores`:

```DAX
Customers Scored = DISTINCTCOUNT('ChurnGuard Scores'[CustomerID])

Customer Snapshots Scored = COUNTROWS('ChurnGuard Scores')

Average Churn Probability = AVERAGE('ChurnGuard Scores'[churn_probability])

Predicted 90 Day Net Spend = SUM('ChurnGuard Scores'[predicted_90d_value])

Risk Weighted Value = SUM('ChurnGuard Scores'[risk_weighted_value])
```

When multiple snapshots are loaded, a customer can recur and the 90-day
prediction windows overlap. Label row counts as customer snapshots; filter to
one snapshot when interpreting summed predicted value or risk-weighted value.

## Three-page report specification

1. **Executive Risk Overview** — compact cards for Customers Scored, Average
   Churn Probability, Predicted 90-Day Net Spend, and Risk-Weighted Value; a
   churn-probability distribution; a churn-probability vs predicted-value
   scatter plot; and a snapshot-date slicer. Use a restrained light canvas,
   dark text, and one semantic risk accent. Do not add a high-risk count until a
   cutoff is approved.
2. **Customer Targeting** — a customer table with CustomerID, snapshot date,
   country, churn probability, predicted value, risk-weighted value, and the
   three API ranks. Sort by ascending `risk_value_rank`; include snapshot-date
   and country filters. Keep probability/value columns sortable and formatted.
3. **Model Insights** — a horizontal bar chart of `mean_abs_shap` by feature,
   ordered by rank, with `mean_shap` as a tooltip or companion table. State that
   global attribution is noncausal and was generated on the June holdout.

Use a clean light background, compact cards, restrained blue/teal accents,
subtle borders, and consistent typography to align with the React frontend.
Avoid retaining old RFM segment charts or sales-volume KPIs as ChurnGuard
metrics.

## PBIX inspection and remaining Desktop work

The repository copy is `Ecommerce_Dashboard.pbix`. Its embedded report layout
contains one page (`Page 1`) with 20 visuals referencing legacy entities
`df_cleaned`, `rfm_data`, `rfm_new_data`, `date`, and `Report Measures`. The
layout references old fields/measures including `R_Score`, `F_Score`,
`M_Score`, `Segment`, `TotalSales`, `SalesVolume`, and `ActiveCustomers`.
Power BI Desktop or a PBIX authoring tool was not available to edit and render
the copy in this environment; the PBIX is unchanged.

To finish the report manually in Power BI Desktop, work only on the ChurnGuard
copy:

1. Open `Ecommerce_Dashboard.pbix` and immediately use **Save As** to create
   `ChurnGuard_Customer_Intelligence.pbix` in the ChurnGuard repository.
2. Import the two generated CSVs from `artifacts/powerbi/` as the tables named
   above. Confirm column types and add the DAX measures above.
3. Replace the existing RFM page content with the three pages in the specified
   order. Remove RFM visuals, slicers, measures, and fields from the report;
   do not rename them to imply they are ChurnGuard outputs.
4. Apply the visual specification, configure sorting/filters, and verify that
   refresh reads the two ChurnGuard CSVs. Validate the report at normal and
   narrow canvas sizes before distributing it.

The original RFM project/dashboard is outside this repository and was not
opened or modified. The Power BI portion is prepared but not complete until the
ChurnGuard PBIX copy has been updated and visually verified in Desktop.
