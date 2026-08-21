# 面積ハカルくん

iPhone のカメラまたは写真ライブラリから画像を読み込み、基準線で縮尺を設定して、任意の対象領域の面積を測定する端末内処理型 Web アプリです。

## 公開URL

[面積ハカルくんを開く](https://satorumuro.github.io/hakarukun-web/)

## 主な機能

- 写真撮影・写真ライブラリからの読込み
- 画像のトリミング
- 2点の基準線と実寸（mm）による縮尺設定
- OpenCV.js の GrabCut による対象領域の自動抽出
- 最大連結成分の選択、穴埋め、境界の整形
- ブラシによる塗り足し・消去
- 面積（cm²）の再計算

写真と測定データはブラウザ内で処理され、サーバーには送信されません。測定値は参考値であり、精密測定、診断、安全性・品質などの重要な判断には使用しないでください。

## 体積ハカルくん（Google Colab試作版）

iPhoneで対象物の周囲を撮影した動画から、複数方向の輪郭を使って3D形状と体積（cm³ / mL）を推定する試作版も収録しています。専用の3D復元サーバーや有料契約は不要で、Google Colabの無料実行枠から試せます。無料枠の使用上限と実行環境の提供状況は変動し、保証されません。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatoruMuro/hakarukun-web/blob/main/colab/volume_hakarukun_colab.ipynb)

- [体積ハカルくんの案内ページ](https://satorumuro.github.io/hakarukun-web/volume/)
- [A4マーカーボードPDF](https://satorumuro.github.io/hakarukun-web/volume/volume-marker-board-a4.pdf)
- Notebook: `colab/volume_hakarukun_colab.ipynb`
- 計算モジュール: `colab/volume_pipeline.py`

マーカーボードをA4・100%で印刷し、対象物を中央に置いて斜め上から一周撮影します。Notebookを上から実行すると、推定体積、推定外形寸法、高さ別断面積、3Dプレビュー、GLB・STLモデルが得られます。動画は処理のためGoogle Colabの一時実行環境へアップロードされます。

輪郭確認画像にはフレーム番号と自動品質判定が表示されます。ボードや背景まで緑色になったフレームは自動除外され、番号を入力して追加の手動除外もできます。復元形状が設定した探索範囲の端に達した場合や、撮影の周回範囲が不足した場合は警告します。

この方式は複数方向の輪郭を重ねる視体積法です。見えない凹みは埋まった形となり、真の体積より大きく推定される場合があります。自動判定は明らかな輪郭失敗を減らす補助機能であり、精度を保証するものではありません。静止した不透明な小物で試し、既知体積の箱や円柱を3回以上測って、真値との差と測定間のばらつきを評価してください。

## ローカル実行

Node.js 22 以降が必要です。

```bash
npm ci
npm run dev
```

## GitHub Pages

`main` への push で GitHub Actions が静的ビルドを行い、GitHub Pages に公開します。
