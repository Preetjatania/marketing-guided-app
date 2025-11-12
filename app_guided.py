import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import marketing_toolkit as mt

st.set_page_config(page_title="Marketing Analytics Workbench — Guided", layout="wide")

# =======================================================
# Helpers & Utilities
# =======================================================

def safe_run(section):
    """Decorator for error-handled Streamlit sections."""
    def wrapper(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"😅 Oops! Something went wrong while running **{section}**.")
                with st.expander("See technical details"):
                    st.code(traceback.format_exc())
        return inner
    return wrapper


@st.cache_data
def cached_bass_forecast(sales, ahead, repeat, k):
    if repeat:
        return mt.predict_bass_with_repeats(sales, k, ahead)
    return mt.predict_bass(sales, ahead)


@st.cache_data
def cached_kmeans_preview(df, numeric_cols, k_range):
    return mt.run_kmeans_elbow(df, numeric_cols, k_range)


# =======================================================
# Sidebar — Data Upload, Reset, and Samples
# =======================================================
with st.sidebar:
    st.header("🎯 Let's get started!")
    st.write("Upload your dataset, or grab one of our samples below 👇")

    if st.button("🔄 Reset App"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

src = st.radio("How would you like to provide data?",
               ["Upload file", "Paste a link (GitHub/Drive/CSV URL)", "Use a sample"],
               help="Upload a file, paste a URL, or use a built-in sample.")

df = None
if src == "Upload file":
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if up:
        if up.name.endswith(".csv"):
            df = pd.read_csv(up)
        else:
            df = pd.read_excel(up)

elif src == "Paste a link (GitHub/Drive/CSV URL)":
    url = st.text_input("Paste your file link here 👇",
                        help="Works with direct links to CSV/Excel files — e.g., GitHub raw URLs or shared Google Drive links.")
    if url:
        try:
            if url.endswith(".csv"):
                df = pd.read_csv(url)
            elif url.endswith((".xlsx", ".xls")):
                df = pd.read_excel(url)
            else:
                df = pd.read_csv(url)  # attempt CSV by default
            st.success("Loaded data successfully from link ✅")
        except Exception as e:
            st.error("⚠️ Could not read the file from that link. Make sure it’s a *direct* CSV/Excel link.")
            st.caption("If it’s a GitHub file, click **Raw** first, then copy that URL.")

elif src == "Use a sample":
    sample_choice = st.selectbox("Pick a sample dataset",
                                 ["None",
                                  "Sample Sales (Bass)",
                                  "Sample Customers (Segmentation)",
                                  "Sample Brand Ratings (Perceptual Map)"])
    if sample_choice == "Sample Sales (Bass)":
        df = mt.load_sample_sales().to_frame("Sales")
    elif sample_choice == "Sample Customers (Segmentation)":
        df = mt.load_sample_segmentation()
    elif sample_choice == "Sample Brand Ratings (Perceptual Map)":
        df = mt.load_sample_perceptual()


    df = None
    if src == "Upload file":
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if up:
            if up.name.endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
    else:
        sample_choice = st.selectbox("Pick a sample dataset",
                                     ["None",
                                      "Sample Sales (Bass)",
                                      "Sample Customers (Segmentation)",
                                      "Sample Brand Ratings (Perceptual Map)"])
        if sample_choice == "Sample Sales (Bass)":
            df = mt.load_sample_sales().to_frame("Sales")
        elif sample_choice == "Sample Customers (Segmentation)":
            df = mt.load_sample_segmentation()
        elif sample_choice == "Sample Brand Ratings (Perceptual Map)":
            df = mt.load_sample_perceptual()

    st.caption("Need a template?")
    st.download_button("📥 Sales Template",
                       "period,sales\n2024-01,120\n2024-02,135\n",
                       "template_sales.csv", "text/csv")
    st.download_button("📥 Segmentation Template",
                       "recency_days,frequency,monetary,visits\n10,4,250,6\n",
                       "template_segmentation.csv", "text/csv")

# =======================================================
# Main Title
# =======================================================
st.title("Hey there 👋 Welcome to your Marketing Analytics Workbench!")
st.markdown("Upload your data, and let's explore **sales forecasts**, **customer segments**, "
            "and **brand perceptions** — all in one friendly place.")

if df is not None:
    st.success(f"Data loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
else:
    st.info("⬅️ Start by uploading data or picking a sample on the left.")
    st.stop()

# =======================================================
# Tabs
# =======================================================
tab1, tab2, tab3 = st.tabs([
    "Forecast future sales",
    "Group customers into personas",
    "Map brand positions"
])

# =======================================================
# Tab 1 — Bass Diffusion
# =======================================================
with tab1:
    st.subheader("📈 Forecast future sales")
    st.write("Let’s uncover how your product might spread through the market over time.")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.warning("No numeric columns found! Please upload a dataset with sales numbers.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            sales_col = st.selectbox("Choose your sales column", num_cols)
        with col2:
            periods_ahead = st.number_input("Forecast periods ahead", 1, 240, 12, help="Number of future periods to predict.")

        repeat_toggle = st.checkbox("Include repeat purchases", value=False,
                                    help="When checked, includes customers who buy again after adoption.")
        repeat_rate = 0.5
        if repeat_toggle:
            repeat_rate = st.slider("Repeat purchase rate (k)", 0.0, 2.0, 0.5, 0.1)

        run_forecast = st.button("Run Forecast 🚀")

        @safe_run("Bass Diffusion Forecast")
        def do_forecast():
            sales = df[sales_col].dropna().astype(float).values
            p, q, M, fitted = mt.estimate_bass_params(sales)
            st.markdown(f"**Estimated parameters:** p={p:.4f}, q={q:.4f}, M={M:,.0f}")

            p, q, M, _ = cached_bass_forecast(sales, periods_ahead, repeat_toggle, repeat_rate)
            fig = mt.plot_bass_forecast(sales, periods_ahead, repeat_toggle, repeat_rate)
            st.pyplot(fig)

        if run_forecast:
            do_forecast()

# =======================================================
# Tab 2 — Segmentation
# =======================================================
with tab2:
    st.subheader("👥 Group customers into personas")
    st.write("Let’s find natural clusters of similar customers — these are your personas!")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]

    chosen = st.multiselect("Pick numeric columns for clustering", options=num_cols, default=num_cols[:3])

    if not chosen:
        st.info("Select at least two numeric columns to start.")
    else:
        k_min, k_max = st.slider("Range of clusters (for preview)", 2, 15, (2, 8))

        if st.button("Preview elbow chart"):
            costs = cached_kmeans_preview(df, chosen, (k_min, k_max))
            st.session_state["elbow_preview"] = costs
            fig = mt.plot_elbow_chart(costs)
            st.pyplot(fig)
            st.success("Nice! The 'elbow' point suggests your ideal K.")
            st.session_state["numeric_cols"] = chosen

        if "elbow_preview" in st.session_state:
            st.write("Pick your final K below 👇")
            k_final = st.number_input("Final K", min_value=k_min, max_value=k_max, value=k_min)
            if st.button("Create Personas ✨"):
                labels, _ = mt.run_kmeans(df, st.session_state["numeric_cols"], int(k_final))
                summary = mt.summarize_personas(df, labels, st.session_state["numeric_cols"])
                st.success("Personas created!")
                st.dataframe(summary)
                csv = summary.to_csv().encode("utf-8")
                st.download_button("Download personas CSV", csv, "personas.csv", "text/csv")

# =======================================================
# Tab 3 — Perceptual Maps
# =======================================================
with tab3:
    st.subheader("🗺️ Map brand positions")
    st.write("Visualize how brands relate to each other and which attributes drive perceptions.")

    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    brand_col = st.selectbox("Which column has brand names?", options=["— pick —"] + text_cols)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    attr_cols = st.multiselect("Pick attribute columns", options=[c for c in num_cols if c != brand_col], default=num_cols[:3])
    scale = st.checkbox("Standardize attributes", True)

    if brand_col != "— pick —" and attr_cols:
        if st.button("Create Perceptual Map 🪄"):
            coords, loadings, var = mt.attribute_rating_map(df[[brand_col] + attr_cols].dropna(), brand_col=brand_col, scale=scale)
            st.write(f"Explained variance — PC1: {var[0]:.1%}, PC2: {var[1]:.1%}")
            fig = mt.plot_attribute_rating_map(coords, loadings)
            st.pyplot(fig)
            csv = coords.to_csv().encode("utf-8")
            st.download_button("Download coordinates", csv, "brand_coords.csv", "text/csv")
    else:
        st.info("Pick a brand column and at least 2–3 numeric attributes.")

# =======================================================
# Footer
# =======================================================
st.markdown("---")
st.caption("Made with ❤️ by your friendly Marketing Analytics Workbench.")
