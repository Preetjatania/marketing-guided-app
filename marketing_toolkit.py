
"""
marketing_toolkit.py
--------------------
Lightweight utilities for:
1) Consumer Segmentation (KMeans + elbow/silhouette, K-Prototypes for mixed data)
2) Product Diffusion (Bass model: fit & forecast)
3) Perceptual Maps (Attribute Ratings via PCA biplot, Overall Similarity via MDS)

Dependencies (install as needed):
    pip install numpy pandas scikit-learn matplotlib scipy kmodes

Author: your-name
License: MIT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.optimize import curve_fit

# Optional dependency for k-prototypes
try:
    from kmodes.kprototypes import KPrototypes
    _HAVE_KPROTO = True
except Exception:
    _HAVE_KPROTO = False


# ===========================
# Consumer Segmentation
# ===========================

@dataclass
class KMeansResult:
    best_k: int
    labels: np.ndarray
    model: KMeans
    inertias: Dict[int, float]
    silhouettes: Dict[int, float]


class ConsumerSegmentation:
    def __init__(self, standardize: bool = True, random_state: int = 42):
        self.standardize = standardize
        self.random_state = random_state
        self.scaler_: Optional[StandardScaler] = None

    def _prepare_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        return X

    def fit_kmeans(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        k_range: Tuple[int, int] = (2, 10),
        max_iter: int = 300,
        n_init: int = 10,
    ) -> KMeansResult:
        """
        Runs KMeans for K in [k_range[0], k_range[1]] and returns the best K
        by silhouette score (ties broken by lower inertia).
        """
        Xp = self._prepare_X(X)
        inertias, silhouettes = {}, {}
        best_k, best_score, best_model = None, -np.inf, None

        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=self.random_state)
            labels = km.fit_predict(Xp)
            inertia = float(km.inertia_)
            inertias[k] = inertia

            # Silhouette requires at least 2 clusters and avoids all-singleton
            if len(np.unique(labels)) > 1:
                sil = float(silhouette_score(Xp, labels))
                silhouettes[k] = sil
            else:
                silhouettes[k] = np.nan

            score = silhouettes[k]
            # Prefer higher silhouette; on tie, smaller inertia
            if (score > best_score) or (np.isclose(score, best_score) and inertia < inertias.get(best_k, np.inf)):
                best_score = score
                best_k = k
                best_model = km

        labels = best_model.labels_
        return KMeansResult(
            best_k=best_k,
            labels=labels,
            model=best_model,
            inertias=inertias,
            silhouettes=silhouettes,
        )

    def plot_elbow(self, result: KMeansResult):
        ks = sorted(result.inertias.keys())
        vals = [result.inertias[k] for k in ks]
        plt.figure()
        plt.plot(ks, vals, marker='o')
        plt.title('Elbow Chart (Inertia vs K)')
        plt.xlabel('K (clusters)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()

    def plot_silhouette_scores(self, result: KMeansResult):
        ks = sorted(result.silhouettes.keys())
        vals = [result.silhouettes[k] for k in ks]
        plt.figure()
        plt.plot(ks, vals, marker='o')
        plt.title('Silhouette Score vs K')
        plt.xlabel('K (clusters)')
        plt.ylabel('Mean Silhouette Score')
        plt.grid(True)
        plt.show()

    # -------- K-Prototypes (Mixed data) --------
    @staticmethod
    def _cat_indices(df: pd.DataFrame, categorical_cols: List[str]) -> List[int]:
        return [df.columns.get_loc(c) for c in categorical_cols]

    def fit_kprototypes(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        n_clusters: int,
        n_init: int = 5,
        max_iter: int = 100,
        gamma: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[KPrototypes]]:
        """
        Fits K-Prototypes for mixed-type data.
        Returns labels and model (None if kmodes not installed).
        """
        if not _HAVE_KPROTO:
            warnings.warn("kmodes is not installed. Run: pip install kmodes")
            return np.full(len(df), -1), None

        cat_idx = self._cat_indices(df, categorical_cols)
        data = df.values
        kproto = KPrototypes(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=self.random_state, gamma=gamma)
        labels = kproto.fit_predict(data, categorical=cat_idx)
        return labels, kproto


# ===========================
# Bass Diffusion Model
# ===========================

def _bass_cumulative(t, p, q, m):
    """Continuous-time Bass cumulative adoption function F(t)."""
    exp_term = np.exp(-(p + q) * t)
    return m * (1 - exp_term) / (1 + (q / p) * exp_term)

@dataclass
class BassFit:
    p: float
    q: float
    m: float
    t: np.ndarray
    sales: np.ndarray
    fitted_cum: np.ndarray
    fitted_sales: np.ndarray


class BassDiffusion:
    def __init__(self):
        pass

    def fit(self, sales: Union[pd.Series, np.ndarray]) -> BassFit:
        """
        Estimate Bass parameters (p, q, m) by fitting to cumulative sales.
        sales: array-like of sales per period (e.g., monthly). Missing periods should be filled beforehand.
        """
        y = np.asarray(sales, dtype=float)
        t = np.arange(1, len(y) + 1)  # 1..T
        cum = np.cumsum(y)

        # Sensible initial guesses
        m0 = max(cum[-1] * 1.2, cum[-1] + 1e-6)
        p0, q0 = 0.03, 0.38

        bounds = ((1e-6, 1e-6, cum[-1] + 1e-6), (1.0, 1.5, 1e6))
        params, _ = curve_fit(_bass_cumulative, t, cum, p0=[p0, q0, m0], bounds=bounds, maxfev=10000)
        p, q, m = params

        fitted_cum = _bass_cumulative(t, p, q, m)
        fitted_sales = np.diff(np.r_[0.0, fitted_cum])  # period sales = diff of cumulative

        return BassFit(p=float(p), q=float(q), m=float(m), t=t, sales=y, fitted_cum=fitted_cum, fitted_sales=fitted_sales)

    def forecast(self, fit: BassFit, periods_ahead: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return future cumulative adoption and period sales for next periods_ahead."""
        t_future = np.arange(fit.t[-1] + 1, fit.t[-1] + periods_ahead + 1)
        cum_future = _bass_cumulative(t_future, fit.p, fit.q, fit.m)
        full_cum = np.r_[fit.fitted_cum, cum_future]
        full_sales = np.diff(np.r_[0.0, full_cum])
        future_sales = full_sales[len(fit.sales):]
        return cum_future, future_sales

    def plot_fit(self, fit: BassFit):
        fig, ax = plt.subplots()
        ax.plot(fit.t, np.cumsum(fit.sales), label='Actual cumulative', marker='o')
        ax.plot(fit.t, fit.fitted_cum, label='Fitted cumulative')
        ax.set_title('Bass Model Fit (Cumulative)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Sales')
        ax.legend()
        ax.grid(True)
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.plot(fit.t, fit.sales, label='Actual sales', marker='o')
        ax2.plot(fit.t, fit.fitted_sales, label='Fitted sales')
        ax2.set_title('Bass Model Fit (Period Sales)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Sales')
        ax2.legend()
        ax2.grid(True)
        plt.show()


# ===========================
# Perceptual Maps
# ===========================

@dataclass
class ARMapResult:
    coords_2d: pd.DataFrame  # rows: brands, columns: ['PC1','PC2']
    loadings: pd.DataFrame   # rows: attributes, columns: ['PC1','PC2']
    explained_variance_ratio: np.ndarray
    pca: PCA

class PerceptualMaps:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def attribute_rating_map(
        self,
        ratings: pd.DataFrame,
        brand_col: Optional[str] = None,
        scale: bool = True
    ) -> ARMapResult:
        """
        ratings: DataFrame where rows are brands (or brand_col specifies an id) and columns are attribute ratings.
        Returns 2D PCA coordinates for brands and loadings as a biplot-friendly structure.
        """
        if brand_col is not None:
            brands = ratings[brand_col].astype(str).values
            X = ratings.drop(columns=[brand_col]).copy()
        else:
            brands = ratings.index.astype(str).values
            X = ratings.copy()

        if scale:
            X = (X - X.mean()) / X.std(ddof=0)

        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(X.values)
        coords_df = pd.DataFrame(coords, index=brands, columns=['PC1', 'PC2'])

        # Loadings scaled to components
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        load_df = pd.DataFrame(loadings, index=X.columns, columns=['PC1', 'PC2'])

        return ARMapResult(
            coords_2d=coords_df,
            loadings=load_df,
            explained_variance_ratio=pca.explained_variance_ratio_,
            pca=pca
        )

    def plot_ar_biplot(self, ar: ARMapResult, arrow_scale: float = 1.0):
        fig, ax = plt.subplots()
        ax.scatter(ar.coords_2d['PC1'], ar.coords_2d['PC2'])
        for brand, (x, y) in ar.coords_2d[['PC1', 'PC2']].iterrows():
            ax.text(x, y, brand)

        for attr, (lx, ly) in ar.loadings[['PC1','PC2']].iterrows():
            ax.arrow(0, 0, lx * arrow_scale, ly * arrow_scale, head_width=0.02, length_includes_head=True)
            ax.text(lx * arrow_scale * 1.1, ly * arrow_scale * 1.1, attr)

        ax.axhline(0, color='gray', linewidth=1)
        ax.axvline(0, color='gray', linewidth=1)
        ax.set_title('Perceptual Map (AR) - PCA Biplot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True)
        plt.show()

    def overall_similarity_map(
        self,
        dissimilarity: Union[pd.DataFrame, np.ndarray],
        metric: bool = False
    ) -> pd.DataFrame:
        """
        dissimilarity: square matrix (brands x brands). Lower = more similar.
        metric=False => non-metric MDS (typical for OS data).
        Returns a DataFrame of 2D coordinates with brand labels from the index/labels.
        """
        if isinstance(dissimilarity, pd.DataFrame):
            labels = dissimilarity.index.astype(str).tolist()
            D = dissimilarity.values
        else:
            D = np.asarray(dissimilarity, dtype=float)
            labels = [f"Item {i}" for i in range(D.shape[0])]

        if D.shape[0] != D.shape[1]:
            raise ValueError("Dissimilarity matrix must be square")

        mds = MDS(n_components=2, dissimilarity='precomputed', metric=metric, random_state=42)
        coords = mds.fit_transform(D)
        return pd.DataFrame(coords, index=labels, columns=['Dim1', 'Dim2'])

    def plot_os_map(self, coords_2d: pd.DataFrame):
        fig, ax = plt.subplots()
        ax.scatter(coords_2d['Dim1'], coords_2d['Dim2'])
        for label, (x, y) in coords_2d[['Dim1','Dim2']].iterrows():
            ax.text(x, y, label)
        ax.set_title('Perceptual Map (OS) - MDS')
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        ax.grid(True)
        plt.show()
