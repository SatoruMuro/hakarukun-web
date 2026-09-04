"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { measureTriangles, type Range2D, type Vector } from "./mesh-volume";

type ModelAnalysis = {
  scene: THREE.Group;
  triangles: Float32Array;
  bounds: THREE.Box3;
  planeNormal: Vector;
  planeOffset: number;
  basisU: Vector;
  basisV: Vector;
  projectedBounds: { minU: number; maxU: number; minV: number; maxV: number };
  maxHeight: number;
  orientationSign: 1 | -1;
  suggestedRange: Range2D;
  vertexCount: number;
  faceCount: number;
};

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const DEFAULT_MM_PER_UNIT = 1000;

function dot(a: Vector, b: Vector) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function normalized(value: Vector): Vector {
  const length = Math.hypot(value.x, value.y, value.z) || 1;
  return { x: value.x / length, y: value.y / length, z: value.z / length };
}

function cross(a: Vector, b: Vector): Vector {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function subtract(a: Vector, b: Vector): Vector {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function quantile(values: number[], ratio: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * Math.min(1, Math.max(0, ratio));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function weightedMedian(entries: { value: number; weight: number }[]) {
  if (!entries.length) return 0;
  entries.sort((a, b) => a.value - b.value);
  const total = entries.reduce((sum, entry) => sum + entry.weight, 0);
  let accumulated = 0;
  for (const entry of entries) {
    accumulated += entry.weight;
    if (accumulated >= total / 2) return entry.value;
  }
  return entries.at(-1)?.value ?? 0;
}

function vectorFromArray(values: Float32Array, offset: number): Vector {
  return { x: values[offset], y: values[offset + 1], z: values[offset + 2] };
}

function extractTriangles(root: THREE.Group) {
  root.updateMatrixWorld(true);
  const values: number[] = [];
  let vertexCount = 0;
  let faceCount = 0;
  const point = new THREE.Vector3();

  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    const geometry = object.geometry as THREE.BufferGeometry;
    const position = geometry.getAttribute("position");
    if (!position) return;
    const index = geometry.getIndex();
    vertexCount += position.count;
    const triangleCount = index ? Math.floor(index.count / 3) : Math.floor(position.count / 3);
    faceCount += triangleCount;
    for (let face = 0; face < triangleCount; face += 1) {
      for (let corner = 0; corner < 3; corner += 1) {
        const positionIndex = index ? index.getX(face * 3 + corner) : face * 3 + corner;
        point.fromBufferAttribute(position, positionIndex).applyMatrix4(object.matrixWorld);
        values.push(point.x, point.y, point.z);
      }
    }
  });

  return { triangles: new Float32Array(values), vertexCount, faceCount };
}

function analyzeScene(scene: THREE.Group): ModelAnalysis {
  const { triangles, vertexCount, faceCount } = extractTriangles(scene);
  if (!triangles.length) throw new Error("三角形メッシュが含まれていません。");

  const bounds = new THREE.Box3();
  const horizontalFaces: { value: number; weight: number }[] = [];
  let normalSum: Vector = { x: 0, y: 0, z: 0 };

  for (let index = 0; index < triangles.length; index += 9) {
    const a = vectorFromArray(triangles, index);
    const b = vectorFromArray(triangles, index + 3);
    const c = vectorFromArray(triangles, index + 6);
    bounds.expandByPoint(new THREE.Vector3(a.x, a.y, a.z));
    bounds.expandByPoint(new THREE.Vector3(b.x, b.y, b.z));
    bounds.expandByPoint(new THREE.Vector3(c.x, c.y, c.z));
    const rawNormal = cross(subtract(b, a), subtract(c, a));
    const doubleArea = Math.hypot(rawNormal.x, rawNormal.y, rawNormal.z);
    if (doubleArea <= 1e-12) continue;
    let faceNormal = {
      x: rawNormal.x / doubleArea,
      y: rawNormal.y / doubleArea,
      z: rawNormal.z / doubleArea,
    };
    if (Math.abs(faceNormal.y) < 0.75) continue;
    if (faceNormal.y < 0) faceNormal = { x: -faceNormal.x, y: -faceNormal.y, z: -faceNormal.z };
    const weight = doubleArea / 2;
    normalSum = {
      x: normalSum.x + faceNormal.x * weight,
      y: normalSum.y + faceNormal.y * weight,
      z: normalSum.z + faceNormal.z * weight,
    };
  }

  const planeNormal = normalized(normalSum.y > 0 ? normalSum : { x: 0, y: 1, z: 0 });
  for (let index = 0; index < triangles.length; index += 9) {
    const a = vectorFromArray(triangles, index);
    const b = vectorFromArray(triangles, index + 3);
    const c = vectorFromArray(triangles, index + 6);
    const rawNormal = cross(subtract(b, a), subtract(c, a));
    const doubleArea = Math.hypot(rawNormal.x, rawNormal.y, rawNormal.z);
    if (doubleArea <= 1e-12) continue;
    const alignment = Math.abs(dot(rawNormal, planeNormal)) / doubleArea;
    if (alignment < 0.9) continue;
    const center = { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3, z: (a.z + b.z + c.z) / 3 };
    horizontalFaces.push({ value: dot(center, planeNormal), weight: (doubleArea / 2) * alignment });
  }
  const planeOffset = weightedMedian(horizontalFaces);

  const reference = Math.abs(planeNormal.x) < 0.85 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 0, z: 1 };
  const basisU = normalized(subtract(reference, {
    x: planeNormal.x * dot(reference, planeNormal),
    y: planeNormal.y * dot(reference, planeNormal),
    z: planeNormal.z * dot(reference, planeNormal),
  }));
  const basisV = normalized(cross(planeNormal, basisU));

  const heights: number[] = [];
  const projectedU: number[] = [];
  const projectedV: number[] = [];
  for (let index = 0; index < triangles.length; index += 3) {
    const point = vectorFromArray(triangles, index);
    heights.push(dot(point, planeNormal) - planeOffset);
    projectedU.push(dot(point, basisU));
    projectedV.push(dot(point, basisV));
  }
  const maxHeight = Math.max(1e-5, quantile(heights, 0.995));
  const projectedBounds = {
    minU: quantile(projectedU, 0.001),
    maxU: quantile(projectedU, 0.999),
    minV: quantile(projectedV, 0.001),
    maxV: quantile(projectedV, 0.999),
  };

  const highU: number[] = [];
  const highV: number[] = [];
  for (let index = 0; index < heights.length; index += 1) {
    if (heights[index] > maxHeight * 0.78) {
      highU.push(projectedU[index]);
      highV.push(projectedV[index]);
    }
  }
  let centerU = highU.length ? quantile(highU, 0.5) : (projectedBounds.minU + projectedBounds.maxU) / 2;
  let centerV = highV.length ? quantile(highV, 0.5) : (projectedBounds.minV + projectedBounds.maxV) / 2;
  const distances = highU.map((value, index) => Math.hypot(value - centerU, highV[index] - centerV));
  const radius = Math.max(1e-6, quantile(distances, 0.82) * 1.35);
  const focusedU: number[] = [];
  const focusedV: number[] = [];
  for (let index = 0; index < highU.length; index += 1) {
    if (Math.hypot(highU[index] - centerU, highV[index] - centerV) <= radius) {
      focusedU.push(highU[index]);
      focusedV.push(highV[index]);
    }
  }
  const sceneSpanU = projectedBounds.maxU - projectedBounds.minU;
  const sceneSpanV = projectedBounds.maxV - projectedBounds.minV;
  const padding = Math.max(sceneSpanU, sceneSpanV) * 0.012;
  const minU = focusedU.length ? quantile(focusedU, 0.005) - padding : centerU - sceneSpanU * 0.18;
  const maxU = focusedU.length ? quantile(focusedU, 0.995) + padding : centerU + sceneSpanU * 0.18;
  const minV = focusedV.length ? quantile(focusedV, 0.005) - padding : centerV - sceneSpanV * 0.18;
  const maxV = focusedV.length ? quantile(focusedV, 0.995) + padding : centerV + sceneSpanV * 0.18;
  centerU = (minU + maxU) / 2;
  centerV = (minV + maxV) / 2;
  const suggestedRange = {
    centerU,
    centerV,
    sizeU: Math.max(maxU - minU, sceneSpanU * 0.08),
    sizeV: Math.max(maxV - minV, sceneSpanV * 0.08),
  };

  let orientationVote = 0;
  for (let index = 0; index < triangles.length; index += 9) {
    const a = vectorFromArray(triangles, index);
    const b = vectorFromArray(triangles, index + 3);
    const c = vectorFromArray(triangles, index + 6);
    const centerHeight = (dot(a, planeNormal) + dot(b, planeNormal) + dot(c, planeNormal)) / 3 - planeOffset;
    if (centerHeight < maxHeight * 0.65) continue;
    orientationVote += dot(cross(subtract(b, a), subtract(c, a)), planeNormal);
  }

  return {
    scene,
    triangles,
    bounds,
    planeNormal,
    planeOffset,
    basisU,
    basisV,
    projectedBounds,
    maxHeight,
    orientationSign: orientationVote >= 0 ? 1 : -1,
    suggestedRange,
    vertexCount,
    faceCount,
  };
}

function setMaterialClipping(material: THREE.Material, planes: THREE.Plane[]) {
  const standard = material as THREE.MeshStandardMaterial;
  standard.clippingPlanes = planes;
  standard.clipIntersection = false;
  standard.side = THREE.DoubleSide;
  standard.needsUpdate = true;
}

function MeshViewer({ analysis, cutOffset, range }: { analysis: ModelAnalysis; cutOffset: number; range: Range2D }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const displayRef = useRef<THREE.Group | null>(null);
  const materialsRef = useRef<THREE.Material[]>([]);
  const planeMeshRef = useRef<THREE.Mesh | null>(null);
  const cropLinesRef = useRef<THREE.LineSegments | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#e8e7e1");
    const camera = new THREE.PerspectiveCamera(38, 1, 0.0001, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.localClippingEnabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.touches.ONE = THREE.TOUCH.ROTATE;
    controls.touches.TWO = THREE.TOUCH.DOLLY_PAN;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x50625c, 2.1));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xdceeff, 1.2);
    fillLight.position.set(-4, 2, -3);
    scene.add(fillLight);

    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let frame = 0;
    const render = () => {
      frame = requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!scene || !camera || !controls) return;
    if (displayRef.current) scene.remove(displayRef.current);
    for (const material of materialsRef.current) material.dispose();
    materialsRef.current = [];

    const display = analysis.scene.clone(true);
    display.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      if (Array.isArray(object.material)) {
        object.material = object.material.map((material) => {
          const clone = material.clone();
          materialsRef.current.push(clone);
          return clone;
        });
      } else {
        const clone = object.material.clone();
        materialsRef.current.push(clone);
        object.material = clone;
      }
    });
    scene.add(display);
    displayRef.current = display;

    const normal = new THREE.Vector3(analysis.planeNormal.x, analysis.planeNormal.y, analysis.planeNormal.z);
    const u = new THREE.Vector3(analysis.basisU.x, analysis.basisU.y, analysis.basisU.z);
    const v = new THREE.Vector3(analysis.basisV.x, analysis.basisV.y, analysis.basisV.z);
    const initialRange = analysis.suggestedRange;
    const focus = normal.clone().multiplyScalar(analysis.planeOffset + analysis.maxHeight * 0.45)
      .addScaledVector(u, initialRange.centerU)
      .addScaledVector(v, initialRange.centerV);
    const radius = Math.max(initialRange.sizeU, initialRange.sizeV, analysis.maxHeight * 3, 0.03);
    camera.near = Math.max(radius / 1000, 0.00001);
    camera.far = Math.max(radius * 100, 10);
    camera.position.copy(focus).addScaledVector(normal, radius * 1.35).addScaledVector(u, radius * 1.05).addScaledVector(v, radius * 1.05);
    camera.updateProjectionMatrix();
    controls.target.copy(focus);
    controls.update();
  }, [analysis]); // The initial framing intentionally stays stable while sliders move.

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const normal = new THREE.Vector3(analysis.planeNormal.x, analysis.planeNormal.y, analysis.planeNormal.z);
    const u = new THREE.Vector3(analysis.basisU.x, analysis.basisU.y, analysis.basisU.z);
    const v = new THREE.Vector3(analysis.basisV.x, analysis.basisV.y, analysis.basisV.z);
    const cutPosition = analysis.planeOffset + cutOffset;
    const minU = range.centerU - range.sizeU / 2;
    const maxU = range.centerU + range.sizeU / 2;
    const minV = range.centerV - range.sizeV / 2;
    const maxV = range.centerV + range.sizeV / 2;
    const planes = [
      new THREE.Plane(normal.clone(), -cutPosition),
      new THREE.Plane(u.clone(), -minU),
      new THREE.Plane(u.clone().negate(), maxU),
      new THREE.Plane(v.clone(), -minV),
      new THREE.Plane(v.clone().negate(), maxV),
    ];
    for (const material of materialsRef.current) setMaterialClipping(material, planes);

    if (planeMeshRef.current) {
      scene.remove(planeMeshRef.current);
      planeMeshRef.current.geometry.dispose();
      (planeMeshRef.current.material as THREE.Material).dispose();
    }
    const planeGeometry = new THREE.PlaneGeometry(range.sizeU, range.sizeV);
    const planeMaterial = new THREE.MeshBasicMaterial({ color: 0xf6b841, transparent: true, opacity: 0.24, side: THREE.DoubleSide, depthWrite: false });
    const planeMesh = new THREE.Mesh(planeGeometry, planeMaterial);
    planeMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    planeMesh.position.copy(normal.clone().multiplyScalar(cutPosition)).addScaledVector(u, range.centerU).addScaledVector(v, range.centerV);
    planeMesh.renderOrder = 3;
    scene.add(planeMesh);
    planeMeshRef.current = planeMesh;

    if (cropLinesRef.current) {
      scene.remove(cropLinesRef.current);
      cropLinesRef.current.geometry.dispose();
      (cropLinesRef.current.material as THREE.Material).dispose();
    }
    const low = cutPosition;
    const high = cutPosition + Math.max(analysis.maxHeight - cutOffset, 0.005);
    const point = (uValue: number, vValue: number, height: number) => normal.clone().multiplyScalar(height).addScaledVector(u, uValue).addScaledVector(v, vValue);
    const corners = [
      point(minU, minV, low), point(maxU, minV, low), point(maxU, maxV, low), point(minU, maxV, low),
      point(minU, minV, high), point(maxU, minV, high), point(maxU, maxV, high), point(minU, maxV, high),
    ];
    const pairs = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const linePoints = pairs.flatMap(([a, b]) => [corners[a], corners[b]]);
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(linePoints);
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x08775f, transparent: true, opacity: 0.9 });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);
    cropLinesRef.current = lines;
  }, [analysis, cutOffset, range]);

  return <div ref={hostRef} className="mesh-viewer" aria-label="3Dモデル表示。1本指で回転、2本指で拡大と移動ができます。" />;
}

function RangeControl({ label, value, min, max, step, unit, onChange }: { label: string; value: number; min: number; max: number; step: number; unit: string; onChange(value: number): void }) {
  const safeMin = Math.min(min, max);
  const safeMax = Math.max(min, max);
  return (
    <label className="mesh-range-control">
      <span><b>{label}</b><output>{value.toFixed(1)} {unit}</output></span>
      <input type="range" min={safeMin} max={safeMax} step={step} value={Math.min(safeMax, Math.max(safeMin, value))} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

export default function MeshVolumePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [analysis, setAnalysis] = useState<ModelAnalysis | null>(null);
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("Scaniverseなどから書き出したGLBを選んでください。ファイルは端末外へ送信されません。");
  const [mmPerUnit, setMmPerUnit] = useState(DEFAULT_MM_PER_UNIT);
  const [cutOffset, setCutOffset] = useState(0.0005);
  const [range, setRange] = useState<Range2D | null>(null);

  const measurement = useMemo(() => analysis && range ? measureTriangles(analysis, cutOffset, range) : null, [analysis, cutOffset, range]);
  const volumeCm3 = measurement ? measurement.volume * mmPerUnit ** 3 / 1000 : 0;
  const areaCm2 = measurement ? measurement.projectedArea * mmPerUnit ** 2 / 100 : 0;
  const cropCoverage = measurement && range ? measurement.projectedArea / Math.max(range.sizeU * range.sizeV, 1e-12) : 0;
  const resultState = !measurement || measurement.usedFaces < 1 || volumeCm3 <= 0
    ? "invalid"
    : measurement.usedFaces < 10 || cropCoverage < 0.25 || cropCoverage > 1.08
      ? "warning"
      : "ready";

  async function loadFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".glb")) {
      setMessage("現在の試作版はGLBに対応しています。ScaniverseではGLBを選んで書き出してください。");
      return;
    }
    if (file.size > 250 * 1024 * 1024) {
      setMessage("ファイルが大きすぎます。250 MB以下になるよう、書き出し品質を1段階下げてください。");
      return;
    }
    setIsLoading(true);
    setMessage("3Dモデルと机面を解析しています…");
    try {
      const buffer = await file.arrayBuffer();
      const loaded = await new Promise<THREE.Group>((resolve, reject) => {
        new GLTFLoader().parse(buffer, "", (gltf) => resolve(gltf.scene), reject);
      });
      const nextAnalysis = analyzeScene(loaded);
      setAnalysis(nextAnalysis);
      setRange(nextAnalysis.suggestedRange);
      setCutOffset(Math.min(nextAnalysis.maxHeight * 0.08, 0.0005));
      setMmPerUnit(DEFAULT_MM_PER_UNIT);
      setFileName(file.name);
      setMessage("机面と対象範囲を自動設定しました。黄色い切断面と緑の枠を確認してください。");
    } catch (error) {
      console.error(error);
      setAnalysis(null);
      setRange(null);
      setMessage("GLBを読み込めませんでした。Meshとして書き出した別のGLBを試してください。");
    } finally {
      setIsLoading(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void loadFile(file);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) void loadFile(file);
  }

  function resetSuggestedRange() {
    if (!analysis) return;
    setRange(analysis.suggestedRange);
    setCutOffset(Math.min(analysis.maxHeight * 0.08, 0.0005));
    setMessage("机面と対象範囲を自動設定し直しました。");
  }

  const spanU = analysis ? analysis.projectedBounds.maxU - analysis.projectedBounds.minU : 1;
  const spanV = analysis ? analysis.projectedBounds.maxV - analysis.projectedBounds.minV : 1;
  const scale = Math.max(mmPerUnit, 1);

  return (
    <main className="mesh-shell">
      <nav className="volume-nav" aria-label="ページ移動">
        <a className="volume-back" href={`${basePath}/volume/`}>← 体積ハカルくんへ</a>
        <span className="prototype-badge mesh-badge">端末内処理・試作版</span>
      </nav>

      <header className="mesh-heading">
        <div>
          <p className="volume-overline">3D MODEL VOLUME</p>
          <h1>3Dモデルから体積を測る</h1>
          <p>机ごとスキャンされたGLBから対象範囲を切り出し、切断面を底面として体積を推定します。</p>
        </div>
        <button className="mesh-file-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={isLoading}>
          {analysis ? "別のGLBを選ぶ" : "GLBを選ぶ"}
        </button>
        <input ref={fileInputRef} className="visually-hidden" type="file" accept=".glb,model/gltf-binary" onChange={onFileChange} />
      </header>

      <section className="mesh-workspace">
        <div className="mesh-view-card">
          <div className="mesh-view-heading">
            <div>
              <p>MODEL PREVIEW</p>
              <h2>{fileName || "3Dモデルを読み込んでください"}</h2>
            </div>
            {analysis && <small>{analysis.vertexCount.toLocaleString()}頂点・{analysis.faceCount.toLocaleString()}面</small>}
          </div>
          <div className="mesh-stage" onDrop={onDrop} onDragOver={(event) => event.preventDefault()}>
            {analysis && range ? (
              <MeshViewer analysis={analysis} cutOffset={cutOffset} range={range} />
            ) : (
              <div className="mesh-empty">
                <div className="mesh-cube" aria-hidden="true"><i /><i /><i /></div>
                <h2>{isLoading ? "解析しています…" : "GLBをここにドロップ"}</h2>
                <p>またはボタンからScaniverseのMeshデータを選択します。</p>
                <button type="button" className="primary-button" onClick={() => fileInputRef.current?.click()} disabled={isLoading}>GLBを選ぶ</button>
              </div>
            )}
            {isLoading && <div className="processing-overlay"><div className="spinner" /><b>3Dモデルを解析中</b><small>大きなファイルでは少し時間がかかります</small></div>}
          </div>
          <p className="mesh-status"><span>●</span>{message}</p>
        </div>

        <aside className="mesh-controls">
          <section className={`mesh-result ${resultState}`}>
            <p>推定体積</p>
            <div><strong>{resultState === "invalid" ? "—" : volumeCm3.toFixed(2)}</strong><span>cm³</span></div>
            {measurement && resultState !== "invalid" ? (
              <>
                <small>推定外形 {(Math.max(measurement.width, measurement.depth) * scale).toFixed(1)} × {(Math.min(measurement.width, measurement.depth) * scale).toFixed(1)} × {(measurement.height * scale).toFixed(1)} mm</small>
                <small>投影面積 {areaCm2.toFixed(2)} cm²・使用面 {measurement.usedFaces.toLocaleString()}</small>
              </>
            ) : <small>対象物が枠内に入り、黄色い面が机面に重なるよう調整してください。</small>}
            <div className="mesh-quality">
              <i />
              <span>{resultState === "ready" ? "底面を閉じて試算できています" : resultState === "warning" ? "対象範囲を確認してください" : "まだ測定できません"}</span>
            </div>
          </section>

          {analysis && range ? (
            <>
              <section className="mesh-control-card">
                <div className="mesh-control-title"><span>1</span><div><p>CUT PLANE</p><h2>机面を切り離す</h2></div></div>
                <p className="mesh-control-copy">黄色い面を机と対象物の境目に合わせます。机が残る場合だけ少し上げます。</p>
                <RangeControl
                  label="机面からの高さ"
                  value={cutOffset * scale}
                  min={-2}
                  max={Math.max(3, analysis.maxHeight * scale * 0.92)}
                  step={0.1}
                  unit="mm"
                  onChange={(value) => setCutOffset(value / scale)}
                />
              </section>

              <section className="mesh-control-card">
                <div className="mesh-control-title"><span>2</span><div><p>CROP</p><h2>対象範囲を絞る</h2></div></div>
                <p className="mesh-control-copy">緑の枠に対象物だけが入り、机や周囲の物が入らないよう調整します。</p>
                <RangeControl label="横方向の中心" value={range.centerU * scale} min={analysis.projectedBounds.minU * scale} max={analysis.projectedBounds.maxU * scale} step={1} unit="mm" onChange={(value) => setRange({ ...range, centerU: value / scale })} />
                <RangeControl label="横幅" value={range.sizeU * scale} min={Math.max(5, spanU * scale * 0.03)} max={spanU * scale} step={1} unit="mm" onChange={(value) => setRange({ ...range, sizeU: value / scale })} />
                <RangeControl label="奥行きの中心" value={range.centerV * scale} min={analysis.projectedBounds.minV * scale} max={analysis.projectedBounds.maxV * scale} step={1} unit="mm" onChange={(value) => setRange({ ...range, centerV: value / scale })} />
                <RangeControl label="奥行き" value={range.sizeV * scale} min={Math.max(5, spanV * scale * 0.03)} max={spanV * scale} step={1} unit="mm" onChange={(value) => setRange({ ...range, sizeV: value / scale })} />
                <button type="button" className="secondary-button mesh-reset" onClick={resetSuggestedRange}>自動設定に戻す</button>
              </section>

              <section className="mesh-control-card">
                <div className="mesh-control-title"><span>3</span><div><p>SCALE</p><h2>実寸スケール</h2></div></div>
                <label className="mesh-number-field">
                  <span>モデル1単位の長さ</span>
                  <div><input type="number" min="0.001" step="1" value={mmPerUnit} onChange={(event) => setMmPerUnit(Math.max(0.001, Number(event.target.value) || DEFAULT_MM_PER_UNIT))} /><b>mm</b></div>
                </label>
                <p className="mesh-control-copy">GLBは通常1単位＝1 mなので1000 mmです。既知寸法と合わない場合のみ変更してください。</p>
              </section>
            </>
          ) : (
            <section className="mesh-control-card mesh-guide">
              <div className="mesh-control-title"><span>?</span><div><p>GUIDE</p><h2>今回の測定方法</h2></div></div>
              <ol>
                <li>Scaniverseで「Mesh」をGLB形式で書き出す</li>
                <li>この画面でGLBを選択する</li>
                <li>黄色い切断面と緑の対象範囲を調整する</li>
              </ol>
            </section>
          )}

          <p className="mesh-privacy"><b>端末内で完結</b><br />GLBの表示・切断・計算はブラウザ内で行い、ファイルをサーバーへ送信しません。</p>
        </aside>
      </section>

      <aside className="mesh-method-note">
        <div><b>現在の対象</b><span>机などの平面上に置いた、上から見て重なりの少ない物体</span></div>
        <div><b>計算方法</b><span>切断面より上の表面を積分し、切断面を底として閉じた体積を推定</span></div>
        <div><b>注意</b><span>穴・深い凹み・大きなオーバーハングがある形状では誤差が増えます</span></div>
      </aside>
    </main>
  );
}
