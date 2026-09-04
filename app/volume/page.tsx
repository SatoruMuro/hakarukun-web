const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const colabUrl = "https://colab.research.google.com/github/SatoruMuro/hakarukun-web/blob/main/colab/volume_hakarukun_colab.ipynb";

export default function VolumePage() {
  return (
    <main className="volume-shell">
      <nav className="volume-nav" aria-label="ページ移動">
        <a className="volume-back" href={`${basePath}/`}>← 面積ハカルくんへ</a>
        <span className="prototype-badge">WEB・COLAB 試作版</span>
      </nav>

      <section className="volume-hero">
        <div className="volume-hero-copy">
          <p className="volume-overline">3Dモデル・動画から立体を測定</p>
          <h1>体積ハカルくん</h1>
          <p className="volume-lead">
            Scaniverseなどで作成した3Dモデルを切り出して測る方法と、
            iPhone動画からGoogle Colabで3D形状を推定する方法を無料で試せます。
          </p>
          <div className="volume-actions">
            <a className="volume-action" href={`${basePath}/volume/mesh/`}>3Dモデルから測る</a>
            <a className="volume-action secondary" href={colabUrl} target="_blank" rel="noreferrer">動画から作る（Colab）↗</a>
            <a className="volume-action secondary" href={`${basePath}/volume/volume-marker-board-a4.pdf`} target="_blank" rel="noreferrer">A4ボードを開く</a>
          </div>
        </div>
        <div className="volume-result-card" aria-label="出力例">
          <div>
            <p>OUTPUT</p>
            <strong>cm³ <span>= mL</span></strong>
          </div>
          <small>GLBの切断面と対象範囲を調整し、体積と推定外形寸法を端末内で計算します。</small>
        </div>
      </section>

      <section className="volume-section">
        <div className="volume-section-heading">
          <p>MESH WORKFLOW</p>
          <h2>3Dモデルなら3つの手順</h2>
        </div>
        <div className="volume-steps">
          <article className="volume-step-card">
            <span>1</span>
            <h3>無料アプリでスキャン</h3>
            <p>Scaniverseなどで対象物の周囲を撮影し、Gaussian SplatではなくMeshを作成します。</p>
          </article>
          <article className="volume-step-card">
            <span>2</span>
            <h3>GLBで書き出す</h3>
            <p>モデルのエクスポート形式はGLBを選びます。形状、実寸スケール、色情報を1ファイルで扱えます。</p>
          </article>
          <article className="volume-step-card">
            <span>3</span>
            <h3>机面と範囲を調整</h3>
            <p>黄色い切断面と緑の枠を対象物に合わせ、底面を閉じた推定体積を確認します。</p>
          </article>
        </div>
      </section>

      <aside className="volume-notice">
        <div>
          <h2>GLBとプライバシー</h2>
          <p>Mesh版ではGLBをブラウザ内だけで処理し、サーバーへ送信しません。動画版では計算のためGoogle Colabの一時実行環境へアップロードされます。</p>
        </div>
        <div>
          <h2>試作版の制限</h2>
          <ul>
            <li>Mesh版は、まず平面上に置いた静止物を対象にしています。</li>
            <li>スキャンの欠損、深い凹み、オーバーハングは誤差要因になります。</li>
            <li>動画版を使う場合はA4マーカーボードが必要です。</li>
            <li>精密測定や診断、安全性・品質の判断には使用できません。</li>
          </ul>
        </div>
      </aside>
    </main>
  );
}
