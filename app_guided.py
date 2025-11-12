import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Tuple

from marketing_toolkit import ConsumerSegmentation, BassDiffusion, PerceptualMaps

st.set_page_config(page_title="Marketing Analytics Workbench — Guided", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def detect_modules(df: pd.DataFrame) -> List[str]:
    suggestions = []
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]

    if len(num_cols) >= 1 and df.shape[0] >= 8:
        suggestions.append("Bass Diffusion (forecast adoption from sales over time)")
    if len(num_cols) >= 2 and df.shape[0] >= 50:
        suggestions.append("Consumer Segmentation (group customers into personas)")

    numeric = df.select_dtypes(include=[np.number])
    if len(text_cols) >= 1 and len(num_cols) >= 3:
        suggestions.append("Perceptual Map — Attribute Ratings (visualize brand positions)")
    if numeric.shape[0] == numeric.shape[1] and numeric.shape[0] >= 3:
        suggestions.append("Perceptual Map — Overall Similarity (from distance matrix)")
    return suggestions

def bass_confidence(n_periods: int) -> Tuple[str, str]:
    if n_periods < 6:
        return ("Low", "You have fewer than 6 periods — early data makes forecasts very uncertain.")
    elif n_periods < 10:
        return ("Moderate", "Around 6–9 periods — fit is possible, but treat results as directional.")
    elif n_periods < 18:
        return ("Good", "10–17 periods — usually enough to see adoption curve shape.")
    else:
        return ("High", "18+ periods — robust estimation for peak timing and market size.")

def make_persona_names(df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    Xz = (X - X.mean()) / X.std(ddof=0)
    out = df.copy()
    out['cluster'] = labels
    persona_names = {}
    for k in sorted(np.unique(labels)):
        sub = Xz[out['cluster'] == k]
        means = sub.mean().sort_values(ascending=False)
        top_feats = means.index[:2].tolist()
        name = " & ".join([f"High {f}" for f in top_feats]) if len(top_feats) >= 2 else (f"High {top_feats[0]}" if top_feats else f"Cluster {k}")
        persona_names[k] = name
    out['persona'] = out['cluster'].map(persona_names)
    return out

def explain_bass(p, q, m, sales_len):
    conf_label, conf_text = bass_confidence(sales_len)
    notes = []
    notes.append(f"Based on your data, innovators (p) ≈ {p:.3f} and social spread (q) ≈ {q:.3f}.")
    if q > p:
        notes.append("Word-of-mouth appears stronger than pure innovation — adoption may accelerate after an initial slow start.")
    else:
        notes.append("Early adoption is driven more by innovators than by social contagion.")
    notes.append(f"Estimated market potential (m) ≈ {m:,.0f} cumulative units.")
    notes.append(f"Confidence: **{conf_label}**. {conf_text}")
    return " ".join(notes)

def explain_segmentation(k, feature_cols):
    return f"We grouped customers into **{k} segments** using the features: {', '.join(feature_cols)}. Each segment shares similar patterns (e.g., spend, frequency). Use these to tailor messaging and offers."

def explain_ar(pc1, pc2):
    return f"PC1 explains **{pc1:.0%}** and PC2 explains **{pc2:.0%}** of variation. Brands to the right score higher on attributes that load positively on PC1; brands up top score higher on PC2 attributes."

# -----------------------------
# Example datasets
# -----------------------------
def example_sales():
    np.random.seed(0)
    t = np.arange(1, 13)
    p, q, m = 0.03, 0.4, 5000
    def _cum(tt, p, q, m):
        return m*(1-np.exp(-(p+q)*tt))/(1+(q/p)*np.exp(-(p+q)*tt))
    cum = _cum(t, p, q, m)
    sales = np.diff(np.r_[0.0, cum]) + np.random.normal(0, 3, size=len(t))
    return pd.DataFrame({'sales': np.maximum(sales, 0)})

def example_customers(n=500):
    np.random.seed(1)
    df = pd.DataFrame({
        'recency_days': np.random.gamma(5, 6, size=n),
        'frequency': np.random.poisson(3, size=n),
        'monetary': np.random.gamma(10, 20, size=n),
        'visits': np.random.poisson(5, size=n)
    })
    return df

def example_ar():
    return pd.DataFrame({
        'Brand':['A','B','C','D'],
        'Quality':[7,5,6,8],
        'Value':[5,8,6,4],
        'Style':[6,5,7,7],
        'Sustainability':[8,4,5,9]
    })

# -----------------------------
# UI — Guided Mode
# -----------------------------
st.title("Guided Mode — Marketing Analytics Workbench")
st.caption("No data science jargon. Just answers. Upload your data and we'll guide you.")

with st.sidebar:
    st.header("1) Load your data")
    src = st.radio("Choose a source", ["Upload file", "Paste URL (CSV/Excel/GitHub)"])
    df = None
    if src == "Upload file":
        up = st.file_uploader("Upload a CSV or Excel file", type=["csv","xlsx","xls"])
        if up is not None:
            name = up.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up, engine="openpyxl")
    else:
        url = st.text_input("Paste direct link (CSV/Excel) or GitHub file URL")
        if url:
            import re, requests
            def is_github_url(u): return "github.com" in u
            def convert_raw(u):
                m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", u)
                if m:
                    user, repo, branch, path = m.groups()
                    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
                return u
            if is_github_url(url):
                url = convert_raw(url)
            try:
                df = pd.read_csv(url)
            except Exception:
                df = pd.read_excel(url, engine="openpyxl")

    st.divider()
    st.header("Or try an example")
    demo_choice = st.selectbox("Load a sample dataset", ["None", "Sample Sales (12 months)", "Sample Customers", "Sample Brand Ratings"])
    if demo_choice == "Sample Sales (12 months)":
        df = example_sales()
    elif demo_choice == "Sample Customers":
        df = example_customers()
    elif demo_choice == "Sample Brand Ratings":
        df = example_ar()

    st.divider()
    if df is not None:
        st.success("Data loaded ✔")
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

st.markdown("### 2) What can we do with your data?")
if df is None:
    st.info("Load a dataset to see tailored suggestions. Or pick a sample in the sidebar.")
else:
    suggestions = detect_modules(df)
    if suggestions:
        for s in suggestions:
            st.markdown(f"- **{s}**")
    else:
        st.warning("We couldn't detect a clear fit. Try a sample or adjust your data.")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Forecast future sales", "Group customers into personas", "Map brand positions"])

    # ----------------- Tab 1: Bass Diffusion Wizard -----------------
    with tab1:
        st.subheader("Forecast future sales (no formulas required)")
        st.write("Tell us which column has your sales per period (e.g., monthly). We'll fit a curve and forecast ahead.")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.info("We need a numeric sales column. Try the 'Sample Sales (12 months)' dataset in the sidebar.")
        else:
            sales_col = st.selectbox("Sales column", options=num_cols)

            # Free-entry forecast horizon
            ahead = st.number_input(
                "Forecast periods ahead",
                min_value=1,
                value=12,
                step=1,
                help="Enter how many future periods to forecast (e.g., months)."
            )
            if ahead > 120:
                st.info("Heads up: forecasting more than 120 periods may be unreliable or slow.")

            run = st.button("Make a forecast", key="run_bass")
            if run:
                # Fit on history
                y = df[sales_col].astype(float).fillna(0).values
                bd = BassDiffusion()
                fit = bd.fit(y)

                st.success("Done! Here's what we found:")
                st.write(explain_bass(fit.p, fit.q, fit.m, len(y)))

                # Forecast future
                cum_future, sales_future = bd.forecast(fit, periods_ahead=int(ahead))
                t_hist = fit.t
                t_future = np.arange(t_hist[-1] + 1, t_hist[-1] + 1 + int(ahead))

                # CUMULATIVE (history + forecast)
                fig, ax = plt.subplots()
                ax.plot(t_hist, np.cumsum(fit.sales), marker='o', label='What actually happened')
                ax.plot(t_hist, fit.fitted_cum, label='Fitted cumulative')
                ax.plot(t_future, cum_future, linestyle='--', label='Forecast cumulative')
                ax.set_title('Total adopters over time')
                ax.set_xlabel('Time'); ax.set_ylabel('Cumulative sales'); ax.legend(); ax.grid(True)
                st.pyplot(fig)

                # PERIOD SALES (history + forecast)
                fig2, ax2 = plt.subplots()
                ax2.plot(t_hist, fit.sales, marker='o', label='Your sales')
                ax2.plot(t_hist, fit.fitted_sales, label='Fitted pattern')
                ax2.plot(t_future, sales_future, linestyle='--', label='Forecast sales')
                ax2.set_title('Sales per period')
                ax2.set_xlabel('Time'); ax2.set_ylabel('Sales'); ax2.legend(); ax2.grid(True)
                st.pyplot(fig2)

                # Table + download
                forecast_df = pd.DataFrame({
                    'period': t_future,
                    'forecast_sales': sales_future
                })
                st.markdown("**What next?** Use these values as your expected demand, and plan inventory/ads around the peak.")
                st.dataframe(forecast_df.head(20))
                st.download_button("Download forecast (CSV)", forecast_df.to_csv(index=False).encode('utf-8'), "forecast.csv", "text/csv")

# ----------------- Tab 2: Segmentation Wizard -----------------
with tab2:
    st.subheader("Group customers into personas")
    st.write("Pick numeric columns (e.g., recency, frequency, spend). We'll show an elbow preview first, then you pick K.")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    chosen = st.multiselect("Numeric columns", options=num_cols, default=num_cols[:4])

    if chosen:
        k_min, k_max = st.slider("Try groups from K=", 2, 15, (2, 8))

        # 1) Preview button: compute elbow/silhouette over the range, no personas yet
        preview = st.button("Preview elbow & silhouette", key="preview_k")
        if preview:
            seg_preview = ConsumerSegmentation(standardize=True)
            res_preview = seg_preview.fit_kmeans(df[chosen].dropna(), k_range=(k_min, k_max))

            st.success(f"Preview ready. Suggested K (based on our heuristic): **{res_preview.best_k}**")

            # Elbow
            fig1, ax1 = plt.subplots()
            ks = sorted(res_preview.inertias.keys())
            vals = [res_preview.inertias[k] for k in ks]
            ax1.plot(ks, vals, marker='o')
            ax1.set_title('Elbow preview (lower is better)')
            ax1.set_xlabel('K'); ax1.set_ylabel('Inertia'); ax1.grid(True)
            st.pyplot(fig1)

            # Silhouette
            fig2, ax2 = plt.subplots()
            ks2 = sorted(res_preview.silhouettes.keys())
            vals2 = [res_preview.silhouettes[k] for k in ks2]
            ax2.plot(ks2, vals2, marker='o')
            ax2.set_title('Silhouette preview (higher is better)')
            ax2.set_xlabel('K'); ax2.set_ylabel('Score'); ax2.grid(True)
            st.pyplot(fig2)

            st.markdown("**Pick the K you want to use** (you can follow the elbow ‘knee’ and the highest silhouette):")
            k_final = st.number_input(
                "Final K",
                min_value=int(k_min),
                max_value=int(k_max),
                value=int(res_preview.best_k),
                step=1,
                help="Choose how many personas you want."
            )

            # 2) Finalize button: run clustering at exactly K = k_final and produce personas
            if st.button("Create personas with this K", key="finalize_k"):
                seg_final = ConsumerSegmentation(standardize=True)
                res_final = seg_final.fit_kmeans(df[chosen].dropna(), k_range=(int(k_final), int(k_final)))

                personas_df = make_persona_names(
                    df.loc[df[chosen].dropna().index].copy(),
                    res_final.labels,
                    chosen
                )

                st.success(f"Personas created with K = {int(k_final)}.")
                st.write(explain_segmentation(int(k_final), chosen))

                counts = personas_df['persona'].value_counts().reset_index()
                counts.columns = ['persona', 'customers']
                st.markdown("**Personas found**")
                st.dataframe(counts)

                st.download_button(
                    "Download personas (CSV)",
                    personas_df.to_csv(index=False).encode('utf-8'),
                    "personas.csv",
                    "text/csv"
                )
        else:
            st.info("Click **Preview elbow & silhouette** to see the curves before choosing K.")
    else:
        st.info("We need at least two numeric columns to form personas.")


    # ----------------- Tab 3: Perceptual Map Wizard -----------------
    with tab3:
        st.subheader("Map where brands sit in customers' minds")
        st.write("Have a table with brand names and attribute ratings (like Quality, Value, Style)? We'll make a simple map.")
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        brand_col = st.selectbox("Which column is the brand name?", options=(["— pick —"] + text_cols))
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        attrs = st.multiselect("Which columns are the attribute ratings?", options=[c for c in numeric_cols if c != brand_col], default=numeric_cols[:3])
        scale = st.checkbox("Standardize attributes (recommended)", value=True)

        if brand_col != "— pick —" and attrs:
            if st.button("Create the map", key="run_ar_map"):
                pm = PerceptualMaps()
                ratings = df[[brand_col] + attrs].dropna()
                ar = pm.attribute_rating_map(ratings, brand_col=brand_col, scale=scale)
                st.success("Here's your map — hover labels guide interpretation.")

                st.write(explain_ar(ar.explained_variance_ratio[0], ar.explained_variance_ratio[1]))

                fig, ax = plt.subplots()
                ax.scatter(ar.coords_2d['PC1'], ar.coords_2d['PC2'])
                for brand, (x, y) in ar.coords_2d[['PC1','PC2']].iterrows():
                    ax.text(x, y, brand)
                for attr, (lx, ly) in ar.loadings[['PC1','PC2']].iterrows():
                    ax.arrow(0, 0, lx, ly, head_width=0.02, length_includes_head=True)
                    ax.text(lx*1.1, ly*1.1, attr)
                ax.axhline(0); ax.axvline(0); ax.grid(True)
                ax.set_title('Brand Map (attributes driving each axis)')
                st.pyplot(fig)

                st.download_button("Download brand coordinates (CSV)", ar.coords_2d.reset_index().rename(columns={'index':'Brand'}).to_csv(index=False).encode('utf-8'), "brand_coords.csv", "text/csv")
        else:
            st.info("Pick a brand column and at least 2–3 numeric attribute columns.")
