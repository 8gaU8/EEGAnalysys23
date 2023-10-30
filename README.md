# EEGAnalysys23
- 23年度卒研解析スクリプト

## 実行
```
./batch.sh
```

## 構成
- `./scripts/scr01_load_ICA_epoching.py`
    - フィルタリング(1,50)
    - ICA(20コンポーネント、infomax)でICLabel
    - 各トリガーでエポッキング
- `./scripts/scr02_calc_average_erp.py`
    - `probe_tone`で平均計算
- `./scripts/script03_erp_amp_late.py`
    - 各被験者の条件ごと`probe_tone`での180ms, 300msのERPの潜時と振幅をDataFrameに
- `./eeg_utils.py`
    - ユーティリティ関数など
- `config.py`
    - 中間ファイルのパスなど
- `./patch_*.py`
    - 特殊ケースに対応するためのパッチ関数