# kaggle
* コンペURL: <https://www.kaggle.com/competitions/icr-identify-age-related-conditions>
* チームでの戦い方とか: <https://www.takapy.work/entry/2020/12/22/225715> \
-> 読んだdiscussionとかnotebookはIssuesに共有します
* kaggleの日本人向けslack: <https://t.co/tylk3CY6nk>
* EDA: <https://www.kaggle.com/code/ayushs9020/understanding-the-competition-icr-eda>
* 決定木ベースモデル: <https://www.kaggle.com/code/xb12345/improve-based-on-icr-first-version>
* ハイパーパラメータチューニング: <https://www.kaggle.com/code/tauilabdelilah/icr-hyperparameter-tuning-optuna>
## やること
- [x] xgboostのハイパーパラメータチューニング
- [x] 特徴量選択(xgboostのfeature importance)
- [x] 前処理したデータで予測
- [x] preprocessed_xgboostで、waがやってた外れ値除去追加してLBスコアどれだけ変わるか
- [x] lightgbmのdartでハイパーパラメータチューニング（Optuna、特徴量はselect_k_bestで選ばれたもののみ使用、時間かかるから早めに）
- [ ] select_k_bestで選ばれた特徴量と選ばれなかった特徴量のヒストグラムを作ってどんな傾向があるか確認
- [ ] EDA.ipynbを参考に、相関の高い特徴量がselect_k_bestでどちらも選ばれてたらどちらか削除
- [x] preprocessed_xgboost.ipynbを、学習時にbalanced_log_lossを使う形に変更
- [ ] lightgbmのdart
- [ ] 決定木ベースのモデルで、特徴量選択した後、欠損値処理、スケーリングは行わずに学習させる
- [ ] Alphaを予測
- [ ] サンプリング方法
- [ ] Epsilon考慮するかどうか
- [ ] 予測が難しいデータ点をどう見分けるか
- [ ] アンサンブル
