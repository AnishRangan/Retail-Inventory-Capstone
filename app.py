
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# STREAMLIT DASHBOARD CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Inventory Capstone",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# DASHBOARD TITLE AND INTRODUCTION
# --------------------------------------------------

st.title("Retail Inventory Optimization — EDA Dashboard")

st.markdown("""
### Demand Forecasting Analysis

This dashboard presents exploratory data analysis of
historical retail sales observations.

Our objective is to understand sales patterns, examine
potential demand indicators, and evaluate the suitability
of the dataset for future forecasting and inventory
optimization models.
""")

# --------------------------------------------------
# LOAD CLEANED DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("demand_forecasting_cleaned.csv")

    # Convert the date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    return df

df = load_data()

# --------------------------------------------------
# OVERVIEW KPIs
# --------------------------------------------------

total_records = len(df)
mean_sales = df["Sales Quantity"].mean()
median_sales = df["Sales Quantity"].median()
std_sales = df["Sales Quantity"].std()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Observations", f"{total_records:,}")
col2.metric("Average Sales", f"{mean_sales:.2f}")
col3.metric("Median Sales", f"{median_sales:.2f}")
col4.metric("Standard Deviation", f"{std_sales:.2f}")

st.caption(
    "Sales statistics represent units per recorded observation, "
    "not actual daily storewide demand."
)

# --------------------------------------------------
# SALES DISTRIBUTION AND OUTLIER ANALYSIS
# --------------------------------------------------

st.divider()
st.header("1. Sales Distribution and Outlier Analysis")

st.markdown("""
This section examines how recorded sales quantities are
distributed and identifies potential statistical outliers.

These visualizations help assess sales variability before
developing demand forecasting models.
""")

# Create two columns for side-by-side visualizations
chart_col1, chart_col2 = st.columns(2)

# Histogram of sales quantities
with chart_col1:
    fig_hist = px.histogram(
        df,
        x="Sales Quantity",
        nbins=30,
        title="Distribution of Sales Quantities",
        labels={"Sales Quantity": "Units per Observation"}
    )

    fig_hist.update_layout(
        xaxis_title="Sales Quantity",
        yaxis_title="Number of Observations"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

# Box plot to identify statistical outliers
with chart_col2:
    fig_box = px.box(
        df,
        x="Sales Quantity",
        title="Sales Quantity — Box Plot",
        points="outliers"
    )

    fig_box.update_layout(
        xaxis_title="Sales Quantity"
    )

    st.plotly_chart(fig_box, use_container_width=True)

# Interpretation of EDA results
st.info("""
**EDA Finding:** Sales quantities are approximately uniformly
distributed between 1 and 499 units.

The mean (248.73) and median (249) are nearly identical,
indicating a roughly balanced distribution.

No outliers were identified using the overall 1.5 × IQR rule.
However, this does not rule out unusual sales within individual
products, stores, or time periods.
""")

# --------------------------------------------------
# DAILY SALES TRENDS AND OBSERVATION ANALYSIS
# --------------------------------------------------

st.divider()
st.header("2. Daily Sales Trends and Observation Counts")

st.markdown("""
This section analyzes recorded sales activity over time.

We compare total daily sales, the number of recorded observations,
and average sales per observation to determine whether changes
in total sales reflect actual purchasing patterns or differences
in the number of records collected.
""")

# Aggregate sales observations by date
daily_sales = (
    df.groupby("Date")
    .agg(
        Total_Sales=("Sales Quantity", "sum"),
        Observation_Count=("Sales Quantity", "count"),
        Average_Sales=("Sales Quantity", "mean")
    )
    .reset_index()
    .sort_values("Date")
)

# Chart 1: Total daily sales
fig_total = px.line(
    daily_sales,
    x="Date",
    y="Total_Sales",
    title="Total Recorded Sales by Day",
    labels={"Total_Sales": "Total Units Sold"}
)

st.plotly_chart(fig_total, use_container_width=True)

# Chart 2: Observation count
fig_count = px.line(
    daily_sales,
    x="Date",
    y="Observation_Count",
    title="Number of Sales Observations per Day",
    labels={"Observation_Count": "Number of Records"}
)

st.plotly_chart(fig_count, use_container_width=True)

# Chart 3: Average sales per observation
fig_average = px.line(
    daily_sales,
    x="Date",
    y="Average_Sales",
    title="Average Sales Quantity per Observation",
    labels={"Average_Sales": "Average Units per Record"}
)

st.plotly_chart(fig_average, use_container_width=True)

# Calculate correlations from the dataset
correlation = daily_sales[
    ["Total_Sales", "Observation_Count", "Average_Sales"]
].corr()

total_count_corr = correlation.loc[
    "Total_Sales", "Observation_Count"
]

# Display the main finding
st.metric(
    "Correlation: Daily Sales vs Observation Count",
    f"{total_count_corr:.3f}"
)

st.warning("""
**Important Data Limitation**

Daily total sales are strongly correlated with the number
of recorded observations.

This indicates that changes in daily totals are substantially
associated with how many product/store records are present.

Therefore, aggregate daily sales should not automatically
be interpreted as actual storewide customer demand.

We must account for this limitation before developing
demand forecasting models.
""")

# --------------------------------------------------
# CATEGORICAL DEMAND DRIVERS
# --------------------------------------------------

st.divider()
st.header("3. Categorical Demand Drivers")

st.markdown("""
This section explores potential relationships between
sales quantities and categorical indicators such as
promotions, seasonality, external factors, demand
trends, and customer segments.

These comparisons are descriptive and do not establish
causal relationships.
""")

# List of categorical variables
categorical_factors = [
    "Promotions",
    "Seasonality Factors",
    "External Factors",
    "Demand Trend",
    "Customer Segments"
]

# Interactive dropdown
selected_factor = st.selectbox(
    "Select a demand indicator:",
    categorical_factors
)

# Calculate summary statistics for the chosen variable
category_summary = (
    df.groupby(selected_factor)["Sales Quantity"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

# Interactive bar chart
fig_category = px.bar(
    category_summary,
    x=selected_factor,
    y="mean",
    title=f"Average Sales Quantity by {selected_factor}",
    labels={"mean": "Average Units per Observation"},
    hover_data=["count", "median", "std"]
)

fig_category.update_layout(
    yaxis_range=[0, category_summary["mean"].max() * 1.15]
)

st.plotly_chart(fig_category, use_container_width=True)

# Summary statistics table
st.subheader("Category Summary Statistics")
st.dataframe(
    category_summary.round(2),
    use_container_width=True,
    hide_index=True
)

st.info("""
**EDA Finding:** Average sales quantities are relatively
similar across promotions, seasonality, external factors,
demand trends, and customer segments.

These variables currently show limited descriptive
separation in sales quantities. Further validation will
be needed to determine whether they offer predictive value.
""")

# --------------------------------------------------
# PRICE VS SALES QUANTITY ANALYSIS
# --------------------------------------------------

st.divider()
st.header("4. Price vs. Sales Quantity")

st.markdown("""
This section examines whether product prices are associated
with differences in recorded sales quantities.

We use a scatter plot and Pearson correlation coefficient
to investigate the strength and direction of the linear
relationship between Price and Sales Quantity.
""")

# Calculate Pearson correlation
price_corr = df["Price"].corr(df["Sales Quantity"])

# Display correlation metric
st.metric(
    "Price vs. Sales Quantity Correlation",
    f"{price_corr:.3f}"
)

# Create interactive scatter plot
fig_price = px.scatter(
    df,
    x="Price",
    y="Sales Quantity",
    opacity=0.3,
    title="Price vs. Sales Quantity",
    hover_data=["Product ID", "Store ID", "Promotions"]
)

fig_price.update_layout(
    xaxis_title="Product Price",
    yaxis_title="Sales Quantity (Units per Observation)"
)

st.plotly_chart(fig_price, use_container_width=True)

st.info("""
**EDA Finding:** The Pearson correlation between
Price and Sales Quantity is approximately -0.006,
indicating virtually no linear relationship.

The scatter plot shows low and high sales quantities
across the full price range.

Price alone does not appear to be a strong linear
predictor of sales quantities in this dataset.
This finding does not rule out nonlinear relationships
or interactions with other variables.
""")

# --------------------------------------------------
# FORECASTING READINESS AND DATA LIMITATIONS
# --------------------------------------------------

st.divider()
st.header("5. Forecasting Readiness and Data Limitations")

st.markdown("""
Before developing demand forecasting models, we must
determine whether sufficient historical observations
exist for individual products and stores.

This section assesses the frequency of repeated
Product ID and Store ID combinations.
""")

# Count observations for each product-store pair
product_store_counts = (
    df.groupby(["Product ID", "Store ID"])
    .size()
    .reset_index(name="Observation_Count")
)

# Calculate coverage statistics
total_pairs = len(product_store_counts)
single_pairs = (product_store_counts["Observation_Count"] == 1).sum()
multiple_pairs = (product_store_counts["Observation_Count"] > 1).sum()
single_pct = (single_pairs / total_pairs) * 100

# Display summary KPIs
c1, c2, c3 = st.columns(3)

c1.metric("Unique Product-Store Pairs", f"{total_pairs:,}")
c2.metric("Pairs Observed Once", f"{single_pairs:,}")
c3.metric("Percentage Observed Once", f"{single_pct:.2f}%")

# Chart: Observation coverage
coverage_df = pd.DataFrame({
    "Observation Frequency": [
        "One Observation",
        "Multiple Observations"
    ],
    "Number of Pairs": [
        single_pairs,
        multiple_pairs
    ]
})

fig_coverage = px.bar(
    coverage_df,
    x="Observation Frequency",
    y="Number of Pairs",
    title="Product-Store Observation Coverage",
    text="Number of Pairs"
)

st.plotly_chart(fig_coverage, use_container_width=True)

# Show most frequently observed product-store pairs
st.subheader("Most Frequently Observed Product-Store Pairs")

st.dataframe(
    product_store_counts.sort_values(
        "Observation_Count",
        ascending=False
    ).head(10),
    use_container_width=True,
    hide_index=True
)

st.warning("""
**Forecasting Limitation**

Most Product ID and Store ID combinations appear only once
in the dataset.

This means there is insufficient repeated history for
conventional SKU-store-level time-series forecasting.

Although the dataset contains 365 dates, daily aggregate
sales are also strongly associated with observation counts.

Future work will evaluate alternative aggregation strategies,
cross-sectional prediction methods, and scenario-based
demand modeling before selecting the forecasting approach.
""")
