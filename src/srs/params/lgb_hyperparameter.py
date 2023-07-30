import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import optuna
import warnings
import lightgbm as lgb
from imblearn.over_sampling import SMOTE # SMOTE
from sklearn.impute import KNNImputer # kNN Imputation
from sklearn.feature_selection import SelectKBest, f_classif# Feature Selection
# Data Encoder and Scaler
import category_encoders as encoders
from sklearn.preprocessing import LabelEncoder, RobustScaler
import pickle
warnings.simplefilter('ignore')

class CFG:
    '''設定値を格納'''
    num_boost_round = 926
    early_stopping_rounds = 98
    n_folds = 5 # 公差検証の分割数
    seed = 1234
    learning_rate = 0.01
    # light_gbm設定値
    boosting_type = "dart"
    verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
    
class Preprocessing:
    '''前処理を行うクラス'''
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.numerical_columns = train_df.drop(['Id', 'EJ', 'Class'], axis=1).columns
        self.features = pd.DataFrame(index=self.numerical_columns, columns=["F_value", "p_value"])
        
    def knn_imputer(self):
        # インスタンス生成
        imputer = KNNImputer(n_neighbors=5)
        
        # ローカル変数に値を格納
        temp_train_df = self.train_df
        temp_test_df = self.test_df
        
        # 訓練データに欠損値代入
        train_df_imputed = pd.DataFrame(imputer.fit_transform(temp_train_df[self.numerical_columns]), columns=self.numerical_columns)
        
        # テストデータに欠損値代入
        test_df_imputed = pd.DataFrame(imputer.transform(temp_test_df[self.numerical_columns]), columns=self.numerical_columns)

        # 元の訓練データも欠損値を補完したデータに置き換える
        temp_train_df = temp_train_df.drop(self.numerical_columns, axis=1)
        temp_train_df = pd.concat([temp_train_df, train_df_imputed], axis=1)

        # テストデータを欠損値を代入したデータに置き換える
        temp_test_df = temp_test_df.drop(self.numerical_columns, axis=1)
        temp_test_df = pd.concat([temp_test_df, test_df_imputed], axis=1)
        
        return temp_train_df, temp_test_df
    
    def clip_outliers(self):
        # ローカル変数に値を格納
        temp_train_df = self.train_df
        temp_test_df = self.test_df

        first_quartiles = temp_train_df[self.numerical_columns].quantile(0.25) # 第１四分位数
        third_quartiles = temp_train_df[self.numerical_columns].quantile(0.75) # 第３四分位数
        iqr = third_quartiles - first_quartiles # 四分位範囲

        lower_bound = first_quartiles - (iqr * 1.5) #外れ値の下限
        upper_bound = third_quartiles + (iqr * 1.5) #外れ値の上限

        # 訓練データとテストデータの両方に対して処理を行う
        for df in [temp_train_df, temp_test_df]:
            df[self.numerical_columns] = df[self.numerical_columns].clip(lower_bound, upper_bound, axis=1)

        return temp_train_df, temp_test_df
        
    def robust_scaler(self):
        # インスタンス生成
        scaler = RobustScaler()
        
        # ローカル変数に値を格納
        temp_train_df = self.train_df
        temp_test_df = self.test_df

        '''訓練データのスケーリング'''
        # インデックスを抽出
        index = temp_train_df.index
        # スケーリング
        scaler_train = scaler.fit_transform(temp_train_df[self.numerical_columns])
        scaled_train_df = pd.DataFrame(scaler_train, columns=self.numerical_columns)
        # インデックスを振りなおす
        scaled_train_df.index = index

        '''テストデータのスケーリング'''
        # インデックスを抽出
        index = temp_test_df.index
        # スケーリング
        scaler_test = scaler.fit_transform(temp_test_df[self.numerical_columns])
        scaled_test_df = pd.DataFrame(scaler_test, columns=self.numerical_columns)
        # インデックスを振りなおす
        scaled_test_df.index = index
        
        # 元の訓練データも欠損値を補完したデータに置き換える
        temp_train_df = temp_train_df.drop(self.numerical_columns, axis=1)
        temp_train_df = pd.concat([temp_train_df, scaled_train_df], axis=1)

        # テストデータを欠損値を代入したデータに置き換える
        temp_test_df = temp_test_df.drop(self.numerical_columns, axis=1)
        temp_test_df = pd.concat([temp_test_df, scaled_test_df], axis=1)
        
        return temp_train_df, temp_test_df
    
    def select_k_best(self, pvalue_upper_limit = 0.1, fscore_lower_limit = 5):
        # ローカル変数に値を格納
        temp_train_df = self.train_df
        temp_test_df = self.test_df
        
        # 訓練データを説明変数と目的変数に分割
        X_train = temp_train_df.drop(['Id', 'EJ', 'Class'], axis=1)
        y_train = temp_train_df['Class']
        # y_train.columns = ['Class']
        '''F値とp値を計算'''
        # インスタンス生成
        #     回帰: f_regression, mutual_info_regression
        #     分類: chi2, f_classif(分散分析のF値), mutual_info_classif
        # この時点ではkをもとの訓練データと同じにする
        fs = SelectKBest(score_func=f_classif, k=len(X_train.columns))
        # 特徴量選択
        X_selected = fs.fit_transform(X_train, y_train.values)

        '''選択したF値とp値と設定した閾値を用いて特徴量を選択'''
        new_features = [] # 選択された特徴量を格納
        drop_features = [] # 使わない特徴量を格納

        # F値が大きく、p値の小さい特徴量を選択
        for i in range(len(X_train.columns)):
            # F値とp値を格納
            self.features.loc[X_train.columns[i], "F_value"] = fs.scores_[i]
            self.features.loc[X_train.columns[i], "p_value"] = fs.pvalues_[i]
            
            if fs.pvalues_[i] <= pvalue_upper_limit and fs.scores_[i] >= fscore_lower_limit:
                new_features.append(X_train.columns[i])
            else:
                drop_features.append(X_train.columns[i])

        # 訓練データから選択した特徴量を抽出        
        X_selected_final = pd.DataFrame(X_selected)
        X_selected_final.columns = X_train.columns
        X_selected_final = X_selected_final[new_features]
        # print('=' * 30)
        # print('After the SelectKBest = {}'.format(X_selected_final.shape))
        # print('Drop-out Features = {}'.format(len(drop_features)))

        # 元のデータに反映
        # X_train = X_train.drop(drop_features, axis=1)
        temp_train_df = temp_train_df.drop(drop_features, axis=1)
        temp_test_df = temp_test_df.drop(drop_features, axis=1)
        
        self.features = self.features.loc[new_features, :] # 選択された特徴量だけをfeaturesに保存
        self.features = self.features.sort_values("F_value", ascending=False)# F値が大きい順にソート
        
        return temp_train_df, temp_test_df
        
def preprocessing_pipeline(train_df, test_df):
    # クラスのインスタンスを生成
    preprocessor = Preprocessing(train_df, test_df)
    
    # 各メソッドを順に実行
    # preprocessor.train_df, preprocessor.test_df = preprocessor.knn_imputer() # 欠損値代入
    # preprocessor.train_df, preprocessor.test_df = preprocessor.clip_outliers() # 外れ値除去
    # preprocessor.train_df, preprocessor.test_df = preprocessor.robust_scaler() # スケーリング
    preprocessor.train_df, preprocessor.test_df = preprocessor.select_k_best(pvalue_upper_limit = 0.1, fscore_lower_limit = 5) # 特徴量選択
    
    # print('selected features: \n{}'.format(preprocessor.features))

    # 最終的に処理されたデータフレームを返す
    return preprocessor.train_df, preprocessor.test_df

# 評価基準
def balanced_log_loss(y_true, y_pred):
    N = len(y_true)

    # Nc is the number of observations
    N_1 = np.sum(y_true == 1, axis=0)
    N_0 = np.sum(y_true == 0, axis=0)

    # In order to avoid the extremes of the log function, each predicted probability 𝑝 is replaced with max(min(𝑝,1−10−15),10−15)
    y_pred = np.maximum(np.minimum(y_pred, 1 - 1e-15), 1e-15)

    # balanced logarithmic loss
    loss_numerator = - (1/N_0) * np.sum((1 - y_true) * np.log(1-y_pred)) - (1/N_1) * np.sum(y_true * np.log(y_pred))

    return loss_numerator / 2

# Classの０，１の割合をそれぞれ計算
def calc_log_loss_weight(y_true):
    nc = np.bincount(y_true)
    w0, w1 = 1/(nc[0]/y_true.shape[0]), 1/(nc[1]/y_true.shape[0])
    return w0, w1

def objective(trial):
    # light-gbm設定値
    lgb_params = {
        # 探索するパラメータ
        'verbosity': -1, # 学習途中の情報を表示するかどうか
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        
        # 固定値
        "boosting_type": CFG.boosting_type,
        "objective": "binary",
        "learning_rate": CFG.learning_rate,
        "metric": "binary_logloss",
        'seed': CFG.seed,
        'n_jobs': -1, # -1でコア数をマックスで使う
        'is_unbalance':True, # 不均衡データの場合にTrueにする
    }
    
    scores = []
    # K-分割交差検証(層化抽出法)
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (train_index, valid_index) in enumerate(kfold.split(X_train, y_train)):
        # 訓練データを分割
        X_train_fold = X_train.iloc[train_index]
        y_train_fold = y_train.iloc[train_index]
        X_valid_fold = X_train.iloc[valid_index]
        y_valid_fold = y_train.iloc[valid_index]
        
        # 訓練データの重みを計算
        train_w0, train_w1 = calc_log_loss_weight(y_train_fold)
        # 検証データの重みを計算
        valid_w0, valid_w1 = calc_log_loss_weight(y_valid_fold)
        # 訓練データをlgb用に変換
        lgb_train = lgb.Dataset(X_train_fold, y_train_fold, weight=y_train_fold.map({0: train_w0, 1: train_w1}))
        # lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_features)
        # 検証データをlgb用に変換
        lgb_valid = lgb.Dataset(X_valid_fold, y_valid_fold, weight=y_valid_fold.map({0: valid_w0, 1: valid_w1}))
        # lgb_valid = lgb.Dataset(x_valid, y_valid, categorical_feature=categorical_features)
        
        model = lgb.train(
                    params = lgb_params,
                    train_set = lgb_train,
                    num_boost_round = CFG.num_boost_round,
                    valid_sets = [lgb_train, lgb_valid],
                    early_stopping_rounds = CFG.early_stopping_rounds,
                    verbose_eval = CFG.verbose_eval,
                    # 学習段階でbalanced_log_lossを使う場合はコメントアウト外す
                    # feval = lgb_metric,
                )
        # 予測
        preds = model.predict(X_valid_fold)
        # 予測値をラベルに変換
        # pred_labels = np.rint(preds)
        # 評価
        # val_score = balanced_log_loss(y_valid, pred_labels)
        val_score = balanced_log_loss(y_valid_fold, preds)
        
        scores.append(val_score)
    # クロスバリデーションの平均値を計算
    mean_score = np.mean(scores)
    
    return mean_score

if __name__ == '__main__':
    n_trials = 100 # ハイパーパラメータチューニングの試行回数
    BASE_DIR = 'data'
    train_df = pd.read_csv(f'{BASE_DIR}/train.csv')
    greeks_df = pd.read_csv(f'{BASE_DIR}/greeks.csv')
    test_df = pd.read_csv(f'{BASE_DIR}/test.csv')
    submission_df = pd.read_csv(f'{BASE_DIR}/sample_submission.csv')

    # 前処理
    train_df, test_df = preprocessing_pipeline(train_df, test_df)

    # 訓練データを説明変数と目的変数に分割
    X_train = train_df.drop(['Id', 'EJ', 'Class'], axis=1)
    y_train = train_df['Class']
    y_train.columns = ['Class']

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # 結果をpickleファイルに出力
    with open('src/srs/params/lgb_best_param.pkl', 'wb') as f:
        pickle.dump(study, f)