import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.set_page_config(page_title="📊 Enhanced CSV Dashboard", layout="wide")
st.title("📈 CSV Dashboard: Enhanced Visual Analytics")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric data for correlation heatmap.")
        return
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def plot_time_series(df):
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    if len(date_cols) > 0:
        time_col = st.selectbox("🕒 Choose Time Column", date_cols)
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        y_col = st.selectbox("📈 Choose Value Column", num_cols)
        fig = px.line(df.sort_values(by=time_col), x=time_col, y=y_col, title=f"{y_col} Over Time")
        st.plotly_chart(fig, use_container_width=True)

def plot_categorical_pie(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        st.warning("No categorical columns found for pie chart.")
        return

    cat = st.selectbox("📦 Select Category for Pie Chart", cat_cols)
    pie_data = df[cat].value_counts().nlargest(20).reset_index()
    pie_data.columns = [cat, "count"]

    fig = px.pie(pie_data, values="count", names=cat, title=f"Top 20 {cat} Distribution")
    st.plotly_chart(fig, use_container_width=True)

def explore_custom_plot(df):
    st.subheader("🎯 Custom Column Plot")

    if df.empty or df.shape[1] < 2:
        st.warning("Not enough columns to plot.")
        return

    # Detect numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Step 1: Choose Chart Type First
    chart_type = st.radio("Choose Chart Type", ["Scatter", "Bar"], key="chart_type_custom")

    # Step 2: Filter column options based on chart type
    if chart_type == "Scatter":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a scatter plot.")
            return
        x_options, y_options = numeric_cols, numeric_cols
    elif chart_type == "Bar":
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            st.warning("Need at least one categorical and one numeric column for a bar plot.")
            return
        x_options, y_options = categorical_cols, numeric_cols
    else:  # Line chart
        if not numeric_cols:
            st.warning("Need numeric data for a line plot.")
            return
        x_options = df.columns.tolist()  # could be time or ordered category
        y_options = numeric_cols

    # Step 3: Select X and Y axes
    x = st.selectbox("Select X-axis", x_options, key="x_axis")
    y = st.selectbox("Select Y-axis", y_options, key="y_axis")

    # Step 4: Optional Color Grouping
    color_col = st.selectbox("🎨 Optional Color/Group By Column", ["None"] + df.columns.tolist(), index=0, key="color_by")
    color_by = None if color_col == "None" else color_col

    # Step 5: Prepare Data and Plot
    try:
        df[y] = pd.to_numeric(df[y], errors='coerce')
    except:
        st.warning(f"Y-axis column `{y}` could not be converted to numeric. Plot might fail.")

    df_plot = df[[x, y] + ([color_by] if color_by else [])].dropna()

    try:
        if chart_type == "Scatter":
            fig = px.scatter(df_plot, x=x, y=y, color=color_by, title=f"{chart_type} plot of {y} vs {x}")
        elif chart_type == "Bar":
            fig = px.bar(df_plot, x=x, y=y, color=color_by, title=f"{chart_type} plot of {y} vs {x}")
        else:
            fig = px.line(df_plot, x=x, y=y, color=color_by, title=f"{chart_type} plot of {y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Could not generate the plot: {e}")

def calculate_financials(df):
    invoice_cols = [col for col in df.columns if "invoice" in col.lower()]
    if invoice_cols:
        invoice_col = invoice_cols[0]
        total_sales = df[invoice_col].fillna(0).sum()
        st.success(f"💰 Total Sales (from '{invoice_col}'): ₹{total_sales:,.2f}")
    else:
        total_sales = 0
        st.warning("⚠️ No column found with 'invoice' in its name to calculate Total Sales.")

    shipping_cols = [col for col in df.columns if "shipping" in col.lower() and pd.api.types.is_numeric_dtype(df[col])]
    if shipping_cols:
        total_shipping = df[shipping_cols].fillna(0).sum().sum()
        st.info(f"🚚 Total Shipping Amount (from columns: {', '.join(shipping_cols)}): ₹{total_shipping:,.2f}")
    else:
        total_shipping = 0
        st.warning("⚠️ No numeric column found with 'shipping' in its name to calculate Shipping Amount.")

    tax_cols = [col for col in df.columns if (("tax_amount" in col.lower()) or ("total tax amount" in col.lower())) and pd.api.types.is_numeric_dtype(df[col])]
    if tax_cols:
        total_tax = df[tax_cols].fillna(0).sum().sum()
        st.info(f"🧾 Total Tax Amount (from columns: {', '.join(tax_cols)}): ₹{total_tax:,.2f}")
    else:
        total_tax = 0
        st.warning("⚠️ No numeric column found with 'tax_amount' or 'total tax amount' in its name.")

    discount_cols = [col for col in df.columns if "discount" in col.lower() and pd.api.types.is_numeric_dtype(df[col])]
    if discount_cols:
        total_discount = df[discount_cols].fillna(0).sum().sum()
        st.info(f"🏷️ Total Discount Amount (from columns: {', '.join(discount_cols)}): ₹{total_discount:,.2f}")
    else:
        total_discount = 0
        st.info("🏷️ No numeric column found with 'discount' in its name to calculate Discounts.")

    try:
        profit = total_sales - (total_tax + total_shipping + total_discount)
        st.success(f"💵 Estimated Profit: ₹{profit:,.2f}")
    except Exception as e:
        st.warning(f"⚠️ Could not calculate profit: {e}")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df_original = df.copy()

        st.success("✅ CSV loaded successfully!")

        null_ratio = df.isnull().mean()
        dropped_cols = null_ratio[null_ratio > 0.95].index.tolist()
        if dropped_cols:
            df.drop(columns=dropped_cols, inplace=True)
            st.warning(f"⚠️ Dropped columns with more than 95% missing values: {', '.join(dropped_cols)}")
        else:
            st.success("✅ No columns dropped due to high missing values.")

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        if df.isnull().sum().sum() == 0:
            st.success("✅ All missing values filled using forward and backward fill.")
        else:
            st.warning("⚠️ Some missing values could not be filled.")

        with st.expander("🔍 Data Preview"):
            st.dataframe(df.head())

        st.markdown("---")
        st.header("💼 Financial Summary")
        calculate_financials(df)

        date_cols = [col for col in df.columns if col.strip().lower().endswith('date')]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=False, errors='coerce')

        with st.expander("📊 Summary Statistics"):
            st.dataframe(df.describe(include='all'))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔁 Correlation Heatmap")
            plot_correlation_heatmap(df)

        with col2:
            st.subheader("📦 Category Distribution (Pie Chart)")
            plot_categorical_pie(df)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🛠️ Custom Visualization")
            explore_custom_plot(df)

        with col4:
            st.subheader("🕒 Time-Series Analysis")
            plot_time_series(df)
       
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("👆 Please upload a CSV file to begin.")
