print(__doc__)
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from datetime import datetime
import sys

from scipy import interp
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc



'''
Trainining set과 Validation set의 차이가 77일이고 Validation set의 길이가 15일인 Time Series Cross Validation을 수행합니다.

각 Fold의 전체 Validation Set으로 예측한 뒤, Validation Set의 결항 데이터를 제거하여 평가합니다.
'''


# ohe = OneHotEncoder(sparse=False)
#
# ohe_wday = pd.DataFrame(ohe.fit_transform(df_afsnt["wday"].values.reshape(-1 ,1)), \
#                         columns = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"])
# ohe_airline = pd.DataFrame(ohe.fit_transform(df_afsnt["airline"].values.reshape(-1 ,1)))
# ohe_origin_dest = pd.DataFrame(ohe.fit_transform(df_afsnt['origin_dest'].values.reshape(-1 ,1)))

# X = df_afsnt[['year', 'wday', 'airline', 'is_flight_M', 'is_flight_T', 'is_flight_F', 'is_arrive', 'is_regular', 'month_sin',
#               'month_cos', 'is_weekend', 'time_sin', 'time_cos', 'origin_dest', 'distance', 'alt_origin', 'alt_dest']]
#
# X = X[X.columns[~X.columns.isin(["wday", "airline", "origin_dest"])]]
# X = pd.concat((X.reset_index(drop=True), ohe_wday), axis = 1)
# X = pd.concat((X.reset_index(drop=True), ohe_airline), axis = 1)
# X = pd.concat((X.reset_index(drop=True), ohe_origin_dest), axis = 1)

# y = df_afsnt['is_delay']

# 결항데이터 제거용
y_cancel = df_afsnt['is_cancel']
idx_cancel = y_cancel[y_cancel == True].index

# CV Class
class BigconTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        # fold마다 margin을 달리 잡는다.
        margin = 1050 * 77
        start = 0
        stop = 0
        for i in range(self.n_splits):
            stop = (i + 1) * k_fold_size
            mid = stop - 1050 * 15
            yield indices[start:mid - margin], indices[mid: stop]


random_state = 8282
cv = BigconTimeSeriesSplit(n_splits=5)
n_samples, n_features = X.shape


# Classification and ROC analysis
# Run classifier with cross-validation and plot ROC curves
def train_clf(model_name):
    if model_name == "randomforest" or model_name == "rf" or model_name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=random_state, n_jobs=-1)

    if model_name == "SVM" or model_name == "svm":
        clf = SVC()

    if mode_name == "logistic" or model_name == "LogisticRegression":
        clf = LogisticRegression()

    print("모델을 훈련합니다.")

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, val in cv.split(X, y):
        # Fold의 피팅 시작시간
        start_time = datetime.now()

        # 훈련
        probas_ = pd.DataFrame(
            {'idx_val': val, 'prob': clf.fit(X.iloc[train,], y[train]).predict_proba(X.iloc[val,])[:, 1]})

        # 종속변수의 결항데이터 제거
        val = list(np.setdiff1d(val, idx_cancel))
        probas_ = probas_[probas_.isin(val)['idx_val']]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[val], probas_.iloc[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

        # Fold의 피팅 소요시간
        time_elapsed = datetime.now() - start_time
        print('{0} Fold fitting time (hh:mm:ss.ms) {1}'.format(i + 1, time_elapsed))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def main():
    model_name = input("원하는 모델을 입력하세요.")
    train_clf(model_name)


if __name__ == "__main__":
    main()