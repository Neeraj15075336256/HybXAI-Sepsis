"""
models/xgboost_branch.py
=========================
XGBoost static embedder: trains on the 43-dim static feature vector,
then uses random-projection leaf embeddings as input to the fusion gate.

The embedding approach (Shi et al., NeurIPS 2019-inspired) converts
XGBoost's discrete leaf assignments into a dense continuous vector:
  leaf_matrix (N, n_trees) × projection_matrix (n_trees, 128) → (N, 128)
followed by L2 normalisation.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score


# ── Default hyperparameters (match paper Table S-HP) ─────────
DEFAULT_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.05,
    colsample_bytree  = 0.8,
    subsample         = 0.8,
    min_child_weight  = 3,
    eval_metric       = 'logloss',
    verbosity         = 0,
    random_state      = 42,
)

EMBED_DIM = 128   # output dimension of random-projection embedding
SEED      = 42


class XGBoostEmbedder:
    """
    Wrapper around XGBClassifier that exposes:
      - fit()       — train the classifier
      - predict_proba()  — standard probability output
      - embed()     — returns the L2-normalised leaf embedding
    """

    def __init__(self, embed_dim: int = EMBED_DIM, **xgb_params):
        params = {**DEFAULT_PARAMS, **xgb_params}
        self.clf       = xgb.XGBClassifier(**params)
        self.embed_dim = embed_dim
        self.proj_     = None   # random projection matrix, set after fit

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            scale_pos_weight: float = None) -> 'XGBoostEmbedder':
        """
        Train the XGBoost classifier.

        Parameters
        ----------
        X, y       : training features and labels
        X_val, y_val : optional validation set for early-stopping eval
        scale_pos_weight : class imbalance weight (default: auto from y)
        """
        if scale_pos_weight is None:
            n_neg = (y == 0).sum(); n_pos = (y == 1).sum()
            scale_pos_weight = n_neg / (n_pos + 1e-8)
        self.clf.set_params(scale_pos_weight=scale_pos_weight)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.clf.fit(X, y, eval_set=eval_set, verbose=False)

        # Build random projection matrix from leaf shape
        leaf_sample = self._leaves(X[:1])
        n_trees = leaf_sample.shape[1]
        rng = np.random.default_rng(SEED)
        self.proj_ = rng.normal(0, 1.0 / np.sqrt(n_trees),
                                (n_trees, self.embed_dim)).astype(np.float32)

        if X_val is not None:
            val_auroc = roc_auc_score(y_val, self.clf.predict_proba(X_val)[:, 1])
            print(f"  XGBoost val AUROC: {val_auroc:.4f} | trees: {n_trees}")

        return self

    def _leaves(self, X: np.ndarray) -> np.ndarray:
        """Return leaf index matrix (N, n_trees) as float32."""
        return self.clf.get_booster().predict(
            xgb.DMatrix(X.astype(np.float32)), pred_leaf=True
        ).astype(np.float32)

    def embed(self, X: np.ndarray) -> np.ndarray:
        """
        Return L2-normalised leaf embedding, shape (N, embed_dim).
        Must call fit() first.
        """
        if self.proj_ is None:
            raise RuntimeError("Call fit() before embed()")
        leaf  = self._leaves(X)
        emb   = leaf @ self.proj_
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return (emb / norms).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.clf.feature_importances_


# ── Quick sanity check ────────────────────────────────────────
if __name__ == '__main__':
    N_TR, N_VA = 500, 100
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((N_TR, 43)).astype(np.float32)
    y_tr = rng.integers(0, 2, N_TR)
    X_va = rng.standard_normal((N_VA, 43)).astype(np.float32)
    y_va = rng.integers(0, 2, N_VA)

    embedder = XGBoostEmbedder(n_estimators=20)
    embedder.fit(X_tr, y_tr, X_va, y_va)
    emb = embedder.embed(X_va)
    print(f"Embedding shape: {emb.shape}  — norms: {np.linalg.norm(emb, axis=1).mean():.4f}")
    assert emb.shape == (N_VA, EMBED_DIM)
    print("XGBoost embedder OK")
