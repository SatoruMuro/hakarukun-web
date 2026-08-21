const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const colabUrl = "https://colab.research.google.com/github/SatoruMuro/hakarukun-web/blob/main/colab/volume_hakarukun_colab.ipynb";

export default function VolumePage() {
  return (
    <main className="volume-shell">
      <nav className="volume-nav" aria-label="ページ移動">
        <a className="volume-back" href={`${basePath}/`}>← 面積ハカルくんへ</a>
        <span className="prototype-badge">COLAB 試作版</span>
      </nav>

      <section className="volume-hero">
        <div className="volume-hero-copy">
          <p className="volume-overline">動画から立体を測定</p>
          <h1>体積ハカルくん</h1>
          <p className="volume-lead">
            iPhoneで対象物の周囲を一周撮影し、複数方向の輪郭から3D形状と体積を推定します。
            専用サーバーは不要で、計算はGoogle Colab上で実行します。
          </p>
          <div className="volume-actions">
            <a className="volume-action" href={colabUrl} target="_blank" rel="noreferrer">Google Colabで計算する ↗</a>
            <a className="volume-action secondary" href={`${basePath}/volume/volume-marker-board-a4.pdf`} target="_blank">A4マーカーボードを開く</a>
          </div>
        </div>
        <div className="volume-result-card" aria-label="出力例">
          <div>
            <p>OUTPUT</p>
            <strong>cm³ <span>= mL</span></strong>
          </div>
          <small>体積の数値、3Dプレビュー、GLB・STLモデルを出力します。</small>
        </div>
      </section>

      <section className="volume-section">
        <div className="volume-section-heading">
          <p>HOW TO USE</p>
          <h2>3つの手順で試せます</h2>
        </div>
        <div className="volume-steps">
          <article className="volume-step-card">
            <span>1</span>
            <h3>ボードを100%で印刷</h3>
            <p>A4・実際のサイズで印刷し、下部の確認線が定規で100 mmになることを確かめます。</p>
          </article>
          <article className="volume-step-card">
            <span>2</span>
            <h3>対象物の周囲を撮影</h3>
            <p>対象物を中央に置き、ボード全体を画面に残しながら、斜め上から10〜20秒で一周します。</p>
          </article>
          <article className="volume-step-card">
            <span>3</span>
            <h3>Colabで順番に実行</h3>
            <p>動画を選び、セルを上から実行します。輪郭を確認後、体積と3Dモデルをダウンロードできます。</p>
          </article>
        </div>
      </section>

      <aside className="volume-notice">
        <div>
          <h2>動画とプライバシー</h2>
          <p>面積測定とは異なり、選んだ動画は計算のためGoogle Colabの一時実行環境へアップロードされます。機密情報や個人を特定できる映像は使用しないでください。</p>
        </div>
        <div>
          <h2>試作版の制限</h2>
          <ul>
            <li>静止した不透明な小物を対象にしています。</li>
            <li>見えない凹みは埋まった形になり、体積を大きく推定する場合があります。</li>
            <li>精密測定や診断、安全性・品質の判断には使用できません。</li>
          </ul>
        </div>
      </aside>
    </main>
  );
}
