"""Phase 24: is the evaluation set drawn from the same waveform distribution as
train? Only allowed waveform features are used, and the discriminator is never
used to extract labels or to adapt the model - it only tells me whether a shift
exists and which feature families carry it."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import cv

D = cv.load()
X = np.vstack([D["Xtr"], D["Xte"]])
y = np.r_[np.zeros(len(D["Xtr"])), np.ones(len(D["Xte"]))]
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
auc = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      scoring="roc_auc")
print(f"train-vs-test AUC, all {X.shape[1]} features: {auc.mean():.4f} +- {auc.std():.4f}")
for g in "ABCDEFGHIJK":
    msk = D["gtag"] == g
    a = cross_val_score(m, X[:, msk], y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                        scoring="roc_auc")
    print(f"  group {g} ({msk.sum():3d} feats): AUC {a.mean():.4f}")
print("\n(AUC ~0.5 means no detectable shift; >0.65 would mean the evaluation"
      "\n records differ systematically and feature choice should be hardened.)")
