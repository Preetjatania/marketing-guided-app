"""
marketing_toolkit.py
--------------------
Unified analytics toolkit for marketing insights.

Includes:
1. Bass Diffusion (standard + repeated purchase)
2. Consumer Segmentation (KMeans + KPrototypes + elbow)
3. Perceptual Maps (Attribute Rating + Overall Similarity)
4. Built-in plotting helpers returning matplotlib figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# Optional dependency
try:
    from kmodes.kprototypes import KPrototypes
    HAVE_KPROTOTYPES = True
except Exception:
    HAVE_KPROTOTYPES = False


# ======================================================
# 1. BASS DIFFUSION
# ======================================================
def _A_hat(t, p, q, M):
    return M * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

def _N_hat(t, p, q, M):
    return _A_hat(t, p, q, M) - _A_hat(t - 1, p, q, M)

def estimate_bass_params(sales):
    """Estimate Bass parameters p,q,M by nonlinear least squares."""
    y = np.asarray(sales, dtype=float)
    T = len(y)
    def prediction_error(params):
        p, q, M = params
        pred = [_N_hat(t, p, q, M) for t in range(1, T + 1)]
        return y - pred
    A_t = np.sum(y)
    params0 = [0.01, 0.16, 3 * A_t]
    res = least_squares(prediction_error, params0, bounds=(0, np.inf))
    p, q, M = res.x
    fitted = np.array([_N_hat(t, p, q, M) for t in range(1, T + 1)])
    return p, q, M, fitted

def predict_bass(sales, periods_ahead=12):
    """Standard Bass forecast."""
    p, q, M, fitted = estimate_bass_params(sales)
    T = len(sales)
    t_future = np.arange(1, T + periods_ahead + 1)
    future = np.array([_N_hat(t, p, q, M) for t in t_future])
    return p, q, M, fitted, future[-periods_ahead:]

def predict_bass_with_repeats(sales, k=0.5, periods_ahead=12):
    """Bass model including repeat purchase rate k."""
    y = np.asarray(sales, dtype=float)
    T = len(y)
    def A_hat(t,p,q,M): return _A_hat(t,p,q,M)
    def N_hat(t,p,q,M): return _N_hat(t,p,q,M)
    def S_hat(t,p,q,M): return N_hat(t,p,q,M) + k * A_hat(t-1,p,q,M)
    def error(params):
        p,q,M = params
        pred = [S_hat(t,p,q,M) for t in range(1,T+1)]
        return y - pred
    S_t = np.sum(y)
    res = least_squares(error,[0.01,0.16,3*S_t],bounds=(0,np.inf))
    p,q,M = res.x
    future = [S_hat(t,p,q,M) for t in range(1,T+periods_ahead+1)]
    return p,q,M,np.array(future[-periods_ahead:])

def plot_bass_forecast(sales, periods_ahead=12, repeat_toggle=False, repeat_rate=0.5):
    """Return matplotlib figure comparing actual vs fitted vs forecast."""
    p,q,M,fitted = estimate_bass_params(sales)
    T = len(sales)
    t_hist = np.arange(1,T+1)
    if repeat_toggle:
        _,_,_,forecast = predict_bass_with_repeats(sales,repeat_rate,periods_ahead)
        label_fore = f"Forecast (repeat {repeat_rate})"
    else:
        _,_,_,forecast = predict_bass(sales,periods_ahead)
        label_fore = "Forecast"
    t_future = np.arange(T+1,T+periods_ahead+1)

    fig,ax = plt.subplots()
    ax.plot(t_hist,sales,marker='o',label='Actual sales')
    ax.plot(t_hist,fitted,label='Fitted Bass')
    ax.plot(t_future,forecast,'--',label=label_fore)
    ax.set_xlabel("Time")
    ax.set_ylabel("Sales")
    ax.set_title("Bass Diffusion Forecast")
    ax.legend()
    ax.grid(True)
    return fig


# ======================================================
# 2. SEGMENTATION
# ======================================================
def run_kmeans(df,numeric_cols,k):
    X = df[numeric_cols].dropna()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k,n_init=50,random_state=42)
    labels = km.fit_predict(Xs)
    return labels,km.inertia_

def run_kmeans_elbow(df,numeric_cols,k_range=(2,10)):
    X = df[numeric_cols].dropna()
    Xs = StandardScaler().fit_transform(X)
    costs={}
    for k in range(k_range[0],k_range[1]+1):
        km = KMeans(n_clusters=k,n_init=50,random_state=42)
        km.fit(Xs)
        costs[k]=km.inertia_
    return costs

def plot_elbow_chart(costs):
    fig,ax=plt.subplots()
    ks=list(costs.keys()); vals=list(costs.values())
    ax.plot(ks,vals,marker='o')
    ax.set_xlabel("K (clusters)")
    ax.set_ylabel("Inertia / Cost")
    ax.set_title("Elbow Chart")
    ax.grid(True)
    return fig

def run_kprototypes(df,numeric_cols,cat_cols,k):
    if not HAVE_KPROTOTYPES:
        raise ImportError("kmodes not installed. pip install kmodes")
    df_sub=df[numeric_cols+cat_cols].dropna()
    cat_idx=[df_sub.columns.get_loc(c) for c in cat_cols]
    model=KPrototypes(n_clusters=k,n_init=10,random_state=42)
    labels=model.fit_predict(df_sub,categorical=cat_idx)
    return labels,model.cost_

def summarize_personas(df,labels,cols):
    out=df.loc[df[cols].dropna().index].copy()
    out["persona"]=labels
    summary=(out.groupby("persona")[cols].mean()).round(2)
    return summary


# ======================================================
# 3. PERCEPTUAL MAPS
# ======================================================
def attribute_rating_map(df,brand_col=None,scale=True):
    if brand_col:
        brands=df[brand_col].astype(str)
        X=df.drop(columns=[brand_col])
    else:
        brands=df.index.astype(str)
        X=df.copy()
    if scale:
        X=(X-X.mean())/X.std(ddof=0)
    pca=PCA(n_components=2)
    coords=pca.fit_transform(X)
    loadings=pca.components_.T*np.sqrt(pca.explained_variance_)
    coords_df=pd.DataFrame(coords,index=brands,columns=["PC1","PC2"])
    load_df=pd.DataFrame(loadings,index=X.columns,columns=["PC1","PC2"])
    return coords_df,load_df,pca.explained_variance_ratio_

def plot_attribute_rating_map(coords_df,load_df,scale=1.0):
    fig,ax=plt.subplots()
    ax.scatter(coords_df["PC1"],coords_df["PC2"])
    for brand,(x,y) in coords_df.iterrows():
        ax.text(x,y,brand)
    for attr,(lx,ly) in load_df.iterrows():
        ax.arrow(0,0,lx*scale,ly*scale,head_width=0.03,color="red",length_includes_head=True)
        ax.text(lx*scale*1.1,ly*scale*1.1,attr,color="red")
    ax.axhline(0,color="gray",lw=1); ax.axvline(0,color="gray",lw=1)
    ax.set_title("Attribute Rating Perceptual Map (PCA)")
    ax.grid(True)
    return fig

def overall_similarity_map(dissimilarity,metric=False):
    if isinstance(dissimilarity,pd.DataFrame):
        labels=dissimilarity.index.astype(str)
        D=dissimilarity.values
    else:
        D=np.asarray(dissimilarity,float)
        labels=[f"Item {i}" for i in range(D.shape[0])]
    mds=MDS(n_components=2,dissimilarity="precomputed",metric=metric,random_state=42)
    coords=mds.fit_transform(D)
    return pd.DataFrame(coords,index=labels,columns=["Dim1","Dim2"])

def plot_overall_similarity_map(coords_df):
    fig,ax=plt.subplots()
    ax.scatter(coords_df["Dim1"],coords_df["Dim2"])
    for label,(x,y) in coords_df.iterrows():
        ax.text(x,y,label)
    ax.axhline(0,color="gray",lw=1); ax.axvline(0,color="gray",lw=1)
    ax.set_title("Overall Similarity Map (MDS)")
    ax.grid(True)
    return fig


# ======================================================
# 4. SAMPLE DATA LOADERS
# ======================================================
def load_sample_sales():
    url="https://raw.githubusercontent.com/zoutianxin1992/MarketingAnalyticsPython/main/Marketing%20Analytics%20in%20Python/Bass%20model/Dataset/3-2-2%20BassModelEstimatePQM2.csv"
    df=pd.read_csv(url)
    return df[df.columns[1]]

def load_sample_segmentation():
    url="https://raw.githubusercontent.com/zoutianxin1992/MarketingAnalyticsPython/main/Marketing%20Analytics%20in%20Python/Segmentation/Datasets/MallCustomersTwoVariables.csv"
    return pd.read_csv(url)

def load_sample_perceptual():
    url="https://raw.githubusercontent.com/zoutianxin1992/MarketingAnalyticsPython/main/Marketing%20Analytics%20in%20Python/perceptual%20map/5-2%20CarRating.csv"
    return pd.read_csv(url,index_col=0)
