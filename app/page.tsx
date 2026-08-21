"use client";

import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type Point = { x: number; y: number };
type Rect = { x: number; y: number; width: number; height: number };
type InteractionMode = "calibrate" | "crop" | "roi" | "add" | "erase" | null;

interface CvMat {
  cols: number;
  data: Uint8Array;
  rows: number;
  delete(): void;
}

interface OpenCvRuntime {
  COLOR_RGBA2RGB: number;
  CV_8UC1: number;
  GC_FGD: number;
  GC_INIT_WITH_RECT: number;
  GC_PR_FGD: number;
  MORPH_CLOSE: number;
  MORPH_ELLIPSE: number;
  MORPH_OPEN: number;
  Mat: new (rows?: number, cols?: number, type?: number) => CvMat;
  Rect: new (x: number, y: number, width: number, height: number) => object;
  Size: new (width: number, height: number) => object;
  cvtColor(source: CvMat, destination: CvMat, code: number): void;
  getStructuringElement(shape: number, size: object): CvMat;
  grabCut(source: CvMat, mask: CvMat, rect: object, background: CvMat, foreground: CvMat, iterations: number, mode: number): void;
  imread(source: HTMLCanvasElement): CvMat;
  morphologyEx(source: CvMat, destination: CvMat, operation: number, kernel: CvMat): void;
  onRuntimeInitialized?: () => void;
}

const MAX_IMAGE_SIDE = 1000;
let openCvPromise: Promise<OpenCvRuntime> | null = null;

async function getOpenCv() {
  if (openCvPromise) return openCvPromise;
  openCvPromise = (async () => {
    const imported = await import("@techstark/opencv-js");
    const wrapped = imported as unknown as { default?: unknown };
    let candidate = wrapped.default ?? imported;
    if (candidate instanceof Promise) candidate = await candidate;
    const cv = candidate as OpenCvRuntime;
    if (cv.Mat) return cv;
    await new Promise<void>((resolve) => {
      cv.onRuntimeInitialized = resolve;
    });
    return cv;
  })();
  return openCvPromise;
}

function keepLargestComponentAndFillHoles(source: Uint8Array, width: number, height: number) {
  const total = width * height;
  const labels = new Int32Array(total);
  const queue = new Int32Array(total);
  let nextLabel = 0;
  let largestLabel = 0;
  let largestSize = 0;

  for (let start = 0; start < total; start += 1) {
    if (!source[start] || labels[start]) continue;
    nextLabel += 1;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    labels[start] = nextLabel;
    while (head < tail) {
      const index = queue[head++];
      const x = index % width;
      const y = Math.floor(index / width);
      for (let dy = -1; dy <= 1; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const neighbor = ny * width + nx;
          if (source[neighbor] && !labels[neighbor]) {
            labels[neighbor] = nextLabel;
            queue[tail++] = neighbor;
          }
        }
      }
    }
    if (tail > largestSize) {
      largestSize = tail;
      largestLabel = nextLabel;
    }
  }

  const result = new Uint8Array(total);
  if (!largestLabel) return result;
  for (let index = 0; index < total; index += 1) {
    if (labels[index] === largestLabel) result[index] = 1;
  }

  const outside = new Uint8Array(total);
  let head = 0;
  let tail = 0;
  const enqueueBackground = (index: number) => {
    if (!result[index] && !outside[index]) {
      outside[index] = 1;
      queue[tail++] = index;
    }
  };
  for (let x = 0; x < width; x += 1) {
    enqueueBackground(x);
    enqueueBackground((height - 1) * width + x);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueueBackground(y * width);
    enqueueBackground(y * width + width - 1);
  }
  while (head < tail) {
    const index = queue[head++];
    const x = index % width;
    const y = Math.floor(index / width);
    if (x > 0) enqueueBackground(index - 1);
    if (x + 1 < width) enqueueBackground(index + 1);
    if (y > 0) enqueueBackground(index - width);
    if (y + 1 < height) enqueueBackground(index + width);
  }
  for (let index = 0; index < total; index += 1) {
    if (!result[index] && !outside[index]) result[index] = 1;
  }
  return result;
}

function normalizedRect(start: Point, end: Point): Rect {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskRef = useRef<Uint8Array | null>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const libraryInputRef = useRef<HTMLInputElement>(null);
  const dragStartRef = useRef<Point | null>(null);
  const brushLastPointRef = useRef<Point | null>(null);

  const [imageName, setImageName] = useState("");
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [mode, setMode] = useState<InteractionMode>(null);
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([]);
  const [realLengthMm, setRealLengthMm] = useState(25);
  const [selectionRect, setSelectionRect] = useState<Rect | null>(null);
  const [selectionKind, setSelectionKind] = useState<"crop" | "roi" | null>(null);
  const [status, setStatus] = useState("写真を撮影するか、ライブラリから選択してください。");
  const [isPreparing, setIsPreparing] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [maskVersion, setMaskVersion] = useState(0);
  const [hasMask, setHasMask] = useState(false);
  const [brushSize, setBrushSize] = useState(18);
  const [brushCursor, setBrushCursor] = useState<Point | null>(null);
  const [areaPx, setAreaPx] = useState(0);

  const mmPerPx = useMemo(() => {
    if (calibrationPoints.length !== 2 || realLengthMm <= 0) return null;
    const px = distance(calibrationPoints[0], calibrationPoints[1]);
    return px > 0 ? realLengthMm / px : null;
  }, [calibrationPoints, realLengthMm]);

  const areaCm2 = mmPerPx && areaPx ? (areaPx * mmPerPx * mmPerPx) / 100 : 0;
  const currentStep = !imageSize.width ? 0 : !mmPerPx ? 1 : !hasMask ? 2 : 3;

  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    const source = sourceCanvasRef.current;
    if (!canvas || !source) return;

    if (canvas.width !== source.width || canvas.height !== source.height) {
      canvas.width = source.width;
      canvas.height = source.height;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0);

    if (maskCanvasRef.current) {
      ctx.drawImage(maskCanvasRef.current, 0, 0);
    }

    if (selectionRect) {
      ctx.save();
      ctx.fillStyle = selectionKind === "crop" ? "rgba(246, 184, 65, 0.13)" : "rgba(38, 198, 143, 0.13)";
      ctx.strokeStyle = selectionKind === "crop" ? "#f6b841" : "#26c68f";
      ctx.lineWidth = Math.max(3, canvas.width / 280);
      ctx.setLineDash([14, 9]);
      ctx.fillRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
      ctx.strokeRect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
      ctx.restore();
    }

    if (calibrationPoints.length) {
      ctx.save();
      ctx.strokeStyle = "#a96ff4";
      ctx.fillStyle = "#ffffff";
      ctx.lineWidth = Math.max(4, canvas.width / 220);
      ctx.shadowColor = "rgba(48, 26, 78, 0.35)";
      ctx.shadowBlur = 8;
      if (calibrationPoints.length === 2) {
        ctx.beginPath();
        ctx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
        ctx.lineTo(calibrationPoints[1].x, calibrationPoints[1].y);
        ctx.stroke();
      }
      for (const point of calibrationPoints) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, Math.max(8, canvas.width / 90), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    if (brushCursor && (mode === "add" || mode === "erase")) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(brushCursor.x, brushCursor.y, brushSize, 0, Math.PI * 2);
      ctx.fillStyle = mode === "add" ? "rgba(38, 198, 143, 0.14)" : "rgba(235, 79, 89, 0.12)";
      ctx.strokeStyle = mode === "add" ? "#26c68f" : "#eb4f59";
      ctx.lineWidth = Math.max(2, canvas.width / 420);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }, [brushCursor, brushSize, calibrationPoints, mode, selectionKind, selectionRect]);

  useEffect(() => {
    drawScene();
  }, [drawScene, imageSize, maskVersion]);

  function rebuildMaskOverlay(mask = maskRef.current) {
    if (!mask || !imageSize.width || !imageSize.height) {
      maskCanvasRef.current = null;
      return;
    }
    const overlay = document.createElement("canvas");
    overlay.width = imageSize.width;
    overlay.height = imageSize.height;
    const context = overlay.getContext("2d");
    if (!context) return;
    const pixels = context.createImageData(imageSize.width, imageSize.height);
    for (let index = 0; index < mask.length; index += 1) {
      if (!mask[index]) continue;
      const offset = index * 4;
      pixels.data[offset] = 12;
      pixels.data[offset + 1] = 205;
      pixels.data[offset + 2] = 139;
      pixels.data[offset + 3] = 116;
    }
    context.putImageData(pixels, 0, 0);
    maskCanvasRef.current = overlay;
    setMaskVersion((version) => version + 1);
  }

  function refreshMaskOverlayRegion(center: Point, radius: number) {
    const mask = maskRef.current;
    const overlay = maskCanvasRef.current;
    if (!mask || !overlay) return;
    const x0 = Math.max(0, Math.floor(center.x - radius - 2));
    const y0 = Math.max(0, Math.floor(center.y - radius - 2));
    const x1 = Math.min(imageSize.width, Math.ceil(center.x + radius + 2));
    const y1 = Math.min(imageSize.height, Math.ceil(center.y + radius + 2));
    const width = x1 - x0;
    const height = y1 - y0;
    if (width <= 0 || height <= 0) return;
    const context = overlay.getContext("2d");
    if (!context) return;
    const pixels = context.createImageData(width, height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const maskIndex = (y0 + y) * imageSize.width + x0 + x;
        if (!mask[maskIndex]) continue;
        const offset = (y * width + x) * 4;
        pixels.data[offset] = 12;
        pixels.data[offset + 1] = 205;
        pixels.data[offset + 2] = 139;
        pixels.data[offset + 3] = 116;
      }
    }
    context.putImageData(pixels, x0, y0);
    setMaskVersion((version) => version + 1);
  }

  function recalculateArea() {
    const mask = maskRef.current;
    if (!mask) {
      setAreaPx(0);
      return;
    }
    let count = 0;
    for (const value of mask) count += value ? 1 : 0;
    setAreaPx(count);
  }

  function paintMaskAt(point: Point, value: 0 | 1) {
    const mask = maskRef.current;
    if (!mask) return;
    const radiusSquared = brushSize * brushSize;
    const minX = Math.max(0, Math.floor(point.x - brushSize));
    const maxX = Math.min(imageSize.width - 1, Math.ceil(point.x + brushSize));
    const minY = Math.max(0, Math.floor(point.y - brushSize));
    const maxY = Math.min(imageSize.height - 1, Math.ceil(point.y + brushSize));
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const dx = x - point.x;
        const dy = y - point.y;
        if (dx * dx + dy * dy <= radiusSquared) mask[y * imageSize.width + x] = value;
      }
    }
    refreshMaskOverlayRegion(point, brushSize);
  }

  function paintMaskLine(from: Point, to: Point, value: 0 | 1) {
    const length = Math.max(1, Math.ceil(distance(from, to)));
    const interval = Math.max(1, Math.floor(brushSize / 3));
    const steps = Math.max(1, Math.ceil(length / interval));
    for (let step = 1; step <= steps; step += 1) {
      const ratio = step / steps;
      paintMaskAt({ x: from.x + (to.x - from.x) * ratio, y: from.y + (to.y - from.y) * ratio }, value);
    }
  }

  async function runGrabCut(roi: Rect) {
    const source = sourceCanvasRef.current;
    if (!source || !mmPerPx || isSegmenting) return;
    const x = Math.max(0, Math.floor(roi.x));
    const y = Math.max(0, Math.floor(roi.y));
    const width = Math.min(source.width - x, Math.max(2, Math.round(roi.width)));
    const height = Math.min(source.height - y, Math.max(2, Math.round(roi.height)));
    if (width < 10 || height < 10) {
      setStatus("測定対象全体とその周囲を少し含むように、もう少し広く囲んでください。");
      return;
    }

    setIsSegmenting(true);
    setStatus("対象領域を自動抽出しています。初回は少し時間がかかります…");
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    const roiCanvas = document.createElement("canvas");
    roiCanvas.width = width;
    roiCanvas.height = height;
    roiCanvas.getContext("2d")?.drawImage(source, x, y, width, height, 0, 0, width, height);

    let src: CvMat | null = null;
    let grabMask: CvMat | null = null;
    let binary: CvMat | null = null;
    let background: CvMat | null = null;
    let foreground: CvMat | null = null;
    let kernel: CvMat | null = null;
    try {
      const cv = await getOpenCv();
      src = cv.imread(roiCanvas);
      cv.cvtColor(src, src, cv.COLOR_RGBA2RGB);
      grabMask = new cv.Mat();
      background = new cv.Mat();
      foreground = new cv.Mat();
      const marginX = Math.max(1, Math.round(width * 0.05));
      const marginY = Math.max(1, Math.round(height * 0.05));
      const rect = new cv.Rect(marginX, marginY, Math.max(2, width - marginX * 2), Math.max(2, height - marginY * 2));
      cv.grabCut(src, grabMask, rect, background, foreground, 5, cv.GC_INIT_WITH_RECT);

      binary = new cv.Mat(height, width, cv.CV_8UC1);
      for (let index = 0; index < grabMask.data.length; index += 1) {
        const label = grabMask.data[index];
        binary.data[index] = label === cv.GC_FGD || label === cv.GC_PR_FGD ? 255 : 0;
      }
      kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9, 9));
      cv.morphologyEx(binary, binary, cv.MORPH_CLOSE, kernel);
      cv.morphologyEx(binary, binary, cv.MORPH_OPEN, kernel);

      const roiMask = keepLargestComponentAndFillHoles(binary.data, width, height);
      let detected = 0;
      for (const value of roiMask) detected += value ? 1 : 0;
      if (!detected) throw new Error("対象領域を抽出できませんでした。");

      const fullMask = new Uint8Array(imageSize.width * imageSize.height);
      for (let row = 0; row < height; row += 1) {
        fullMask.set(roiMask.subarray(row * width, (row + 1) * width), (y + row) * imageSize.width + x);
      }
      maskRef.current = fullMask;
      setHasMask(true);
      setSelectionRect(null);
      setSelectionKind(null);
      setMode(null);
      rebuildMaskOverlay(fullMask);
      setAreaPx(detected);
      setStatus("自動抽出が完了しました。緑の範囲を確認し、必要なら修正してください。");
    } catch (error) {
      console.error(error);
      setStatus("自動抽出に失敗しました。測定対象の周囲を少し広めに囲み直してください。");
    } finally {
      kernel?.delete();
      binary?.delete();
      foreground?.delete();
      background?.delete();
      grabMask?.delete();
      src?.delete();
      setIsSegmenting(false);
    }
  }

  async function loadImage(file: File) {
    setIsPreparing(true);
    setStatus("写真を準備しています…");
    try {
      let source: CanvasImageSource;
      let sourceWidth: number;
      let sourceHeight: number;

      if ("createImageBitmap" in window) {
        const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
        source = bitmap;
        sourceWidth = bitmap.width;
        sourceHeight = bitmap.height;
      } else {
        const objectUrl = URL.createObjectURL(file);
        const image = await new Promise<HTMLImageElement>((resolve, reject) => {
          const element = new Image();
          element.onload = () => resolve(element);
          element.onerror = reject;
          element.src = objectUrl;
        });
        source = image;
        sourceWidth = image.naturalWidth;
        sourceHeight = image.naturalHeight;
        URL.revokeObjectURL(objectUrl);
      }

      const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(sourceWidth, sourceHeight));
      const width = Math.max(1, Math.round(sourceWidth * scale));
      const height = Math.max(1, Math.round(sourceHeight * scale));
      const prepared = document.createElement("canvas");
      prepared.width = width;
      prepared.height = height;
      const preparedContext = prepared.getContext("2d", { willReadFrequently: true });
      if (!preparedContext) throw new Error("画像を読み込めませんでした。");
      preparedContext.drawImage(source, 0, 0, width, height);

      sourceCanvasRef.current = prepared;
      maskRef.current = null;
      maskCanvasRef.current = null;
      setHasMask(false);
      setImageName(file.name || "撮影した写真");
      setImageSize({ width, height });
      setCalibrationPoints([]);
      setSelectionRect(null);
      setSelectionKind(null);
      setAreaPx(0);
      setMode("calibrate");
      setStatus("写真内の基準物の両端を、順番に2か所タップしてください。");
    } catch (error) {
      console.error(error);
      setStatus("写真を読み込めませんでした。別の写真を選んでください。");
    } finally {
      setIsPreparing(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void loadImage(file);
    event.target.value = "";
  }

  function pointFromEvent(event: ReactPointerEvent<HTMLCanvasElement>): Point {
    const canvas = event.currentTarget;
    const bounds = canvas.getBoundingClientRect();
    return {
      x: Math.min(canvas.width, Math.max(0, ((event.clientX - bounds.left) / bounds.width) * canvas.width)),
      y: Math.min(canvas.height, Math.max(0, ((event.clientY - bounds.top) / bounds.height) * canvas.height)),
    };
  }

  function onCanvasPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!sourceCanvasRef.current || !mode) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = pointFromEvent(event);

    if (mode === "calibrate") {
      setCalibrationPoints((previous) => {
        const next = previous.length >= 2 ? [point] : [...previous, point];
        if (next.length === 1) {
          setStatus("もう一方の端をタップしてください。");
        } else {
          setMode(null);
          setStatus("基準線を設定しました。実際の長さを確認してください。");
        }
        return next;
      });
      return;
    }

    if (mode === "crop" || mode === "roi") {
      dragStartRef.current = point;
      setSelectionKind(mode);
      setSelectionRect({ x: point.x, y: point.y, width: 0, height: 0 });
      return;
    }

    if ((mode === "add" || mode === "erase") && maskRef.current) {
      brushLastPointRef.current = point;
      setBrushCursor(point);
      paintMaskAt(point, mode === "add" ? 1 : 0);
    }
  }

  function onCanvasPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const point = pointFromEvent(event);
    if (mode === "add" || mode === "erase") {
      setBrushCursor(point);
      if (brushLastPointRef.current && event.currentTarget.hasPointerCapture(event.pointerId)) {
        paintMaskLine(brushLastPointRef.current, point, mode === "add" ? 1 : 0);
        brushLastPointRef.current = point;
      }
      return;
    }
    if (!dragStartRef.current || (mode !== "crop" && mode !== "roi")) return;
    setSelectionRect(normalizedRect(dragStartRef.current, point));
  }

  function onCanvasPointerUp(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (mode === "add" || mode === "erase") {
      brushLastPointRef.current = null;
      recalculateArea();
      setStatus(mode === "add" ? "塗り足した範囲を反映しました。" : "消した範囲を反映しました。");
      return;
    }
    if (!dragStartRef.current || (mode !== "crop" && mode !== "roi")) return;
    const rect = normalizedRect(dragStartRef.current, pointFromEvent(event));
    const completedMode = mode;
    dragStartRef.current = null;
    if (rect.width < 8 || rect.height < 8) {
      setSelectionRect(null);
      setSelectionKind(null);
      setStatus("範囲が小さすぎます。もう一度ドラッグしてください。");
      return;
    }
    setSelectionRect(rect);
    setMode(null);
    setStatus(
      completedMode === "crop"
        ? "この範囲で切り出してよければ「トリミングを適用」を押してください。"
        : "測定対象の範囲を設定しました。自動抽出を開始します…",
    );
    if (completedMode === "roi") void runGrabCut(rect);
  }

  function beginCalibration() {
    setCalibrationPoints([]);
    setSelectionRect(null);
    setSelectionKind(null);
    setMode("calibrate");
    setStatus("基準物の両端を、順番に2か所タップしてください。");
  }

  function beginCrop() {
    setSelectionRect(null);
    setSelectionKind("crop");
    setMode("crop");
    setStatus("残したい範囲を指でドラッグしてください。");
  }

  function applyCrop() {
    const source = sourceCanvasRef.current;
    if (!source || !selectionRect || selectionKind !== "crop") return;
    const x = Math.max(0, Math.round(selectionRect.x));
    const y = Math.max(0, Math.round(selectionRect.y));
    const width = Math.min(source.width - x, Math.round(selectionRect.width));
    const height = Math.min(source.height - y, Math.round(selectionRect.height));
    if (width < 8 || height < 8) return;

    const cropped = document.createElement("canvas");
    cropped.width = width;
    cropped.height = height;
    cropped.getContext("2d")?.drawImage(source, x, y, width, height, 0, 0, width, height);
    sourceCanvasRef.current = cropped;
    maskRef.current = null;
    maskCanvasRef.current = null;
    setHasMask(false);
    setImageSize({ width, height });
    setSelectionRect(null);
    setSelectionKind(null);
    setCalibrationPoints([]);
    setAreaPx(0);
    setMode("calibrate");
    setStatus("トリミングしました。基準物の両端を2か所タップしてください。");
  }

  function cancelSelection() {
    setSelectionRect(null);
    setSelectionKind(null);
    setMode(null);
    setStatus(mmPerPx ? "測定対象を囲む準備ができました。" : "基準線を設定してください。");
  }

  function beginRoi() {
    if (!mmPerPx) return;
    setSelectionRect(null);
    setSelectionKind("roi");
    setMode("roi");
    setStatus("測定対象全体と、その周囲を少し含めてドラッグしてください。");
  }

  function beginRefine(nextMode: "add" | "erase") {
    if (!maskRef.current) return;
    setSelectionRect(null);
    setSelectionKind(null);
    setMode(nextMode);
    setStatus(nextMode === "add" ? "対象領域に追加したい部分を指で塗ってください。" : "対象領域から除外したい部分を指でなぞってください。");
  }

  function finishRefine() {
    setMode(null);
    setBrushCursor(null);
    brushLastPointRef.current = null;
    recalculateArea();
    setStatus("測定結果を確定しました。");
  }

  function restartSegmentation() {
    maskRef.current = null;
    maskCanvasRef.current = null;
    setHasMask(false);
    setMaskVersion((version) => version + 1);
    setAreaPx(0);
    beginRoi();
  }

  function resetAll() {
    sourceCanvasRef.current = null;
    maskRef.current = null;
    maskCanvasRef.current = null;
    setHasMask(false);
    setImageName("");
    setImageSize({ width: 0, height: 0 });
    setCalibrationPoints([]);
    setSelectionRect(null);
    setSelectionKind(null);
    setAreaPx(0);
    setMaskVersion((version) => version + 1);
    setMode(null);
    setStatus("写真を撮影するか、ライブラリから選択してください。");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">H</span>
          <div>
            <p className="brand-kicker">画像面積測定</p>
            <h1>面積ハカルくん</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <span className="local-badge"><i aria-hidden="true" /> 写真は端末内で処理</span>
          <a className="volume-link" href="./volume/">動画から体積を測る <span aria-hidden="true">→</span></a>
        </div>
      </header>

      <section className="workflow" aria-label="測定手順">
        {["写真", "基準", "対象", "結果"].map((label, index) => (
          <div className={`workflow-step ${index === currentStep ? "current" : ""} ${index < currentStep ? "done" : ""}`} key={label}>
            <span>{index < currentStep ? "✓" : index + 1}</span>
            <small>{label}</small>
          </div>
        ))}
      </section>

      <div className="workspace-grid">
        <section className="image-panel" aria-label="画像操作エリア">
          <div className="image-panel-heading">
            <div>
              <p className="eyebrow">測定画像</p>
              <h2>{imageName || "写真を準備してください"}</h2>
            </div>
            {imageSize.width > 0 && (
              <button className="text-button" type="button" onClick={resetAll}>写真を変更</button>
            )}
          </div>

          <div className={`canvas-stage ${mode ? "is-editing" : ""}`}>
            {imageSize.width ? (
              <canvas
                ref={canvasRef}
                aria-label="測定する写真"
                onPointerDown={onCanvasPointerDown}
                onPointerMove={onCanvasPointerMove}
                onPointerUp={onCanvasPointerUp}
                onPointerCancel={onCanvasPointerUp}
              />
            ) : (
              <div className="empty-state">
                <div className="camera-glyph" aria-hidden="true"><span /></div>
                <h2>測定対象と基準物を一緒に撮影</h2>
                <p>定規などの基準物を測定対象と同じ平面に置き、できるだけ真上から撮影してください。</p>
                <div className="photo-actions">
                  <button className="primary-button" type="button" onClick={() => cameraInputRef.current?.click()} disabled={isPreparing}>
                    {isPreparing ? "準備中…" : "写真を撮る"}
                  </button>
                  <button className="secondary-button" type="button" onClick={() => libraryInputRef.current?.click()} disabled={isPreparing}>
                    ライブラリから選ぶ
                  </button>
                </div>
              </div>
            )}
            {isSegmenting && (
              <div className="processing-overlay" aria-live="polite">
                <span className="spinner" aria-hidden="true" />
                <b>対象領域を抽出しています</b>
                <small>画面を閉じずにお待ちください</small>
              </div>
            )}
          </div>

          <p className="status-message" role="status"><span aria-hidden="true">●</span>{status}</p>

          <input ref={cameraInputRef} className="visually-hidden" type="file" accept="image/*" capture="environment" onChange={onFileChange} />
          <input ref={libraryInputRef} className="visually-hidden" type="file" accept="image/*" onChange={onFileChange} />
        </section>

        <aside className="control-panel">
          {!imageSize.width ? (
            <div className="guide-card">
              <span className="guide-number">1</span>
              <h2>撮影のポイント</h2>
              <ul>
                <li>測定対象と基準物を同じ平面に置く</li>
                <li>影や反射を避けて明るく撮る</li>
                <li>カメラを測定面に対して正対させる</li>
              </ul>
            </div>
          ) : (
            <>
              <div className="control-card">
                <div className="control-title-row">
                  <span className={`control-number ${mmPerPx ? "complete" : ""}`}>{mmPerPx ? "✓" : "1"}</span>
                  <div><p>基準設定</p><h2>実寸に換算する</h2></div>
                </div>
                <label className="number-field">
                  <span>基準線の実際の長さ</span>
                  <div><input type="number" min="0.1" step="0.1" value={realLengthMm} onChange={(event) => setRealLengthMm(Number(event.target.value))} /><b>mm</b></div>
                </label>
                <button className="secondary-button full" type="button" onClick={beginCalibration}>基準線を引き直す</button>
                {mmPerPx && <p className="conversion-value">1 px = {mmPerPx.toFixed(4)} mm</p>}
              </div>

              <div className={`control-card ${!mmPerPx ? "disabled-card" : ""}`}>
                <div className="control-title-row">
                  <span className={`control-number ${hasMask ? "complete" : ""}`}>{hasMask ? "✓" : "2"}</span>
                  <div><p>対象指定</p><h2>測定範囲を囲む</h2></div>
                </div>
                <p className="control-copy">測定対象の外側を少し含めて囲みます。</p>
                <button className="primary-button full" type="button" onClick={hasMask ? restartSegmentation : beginRoi} disabled={!mmPerPx || isSegmenting}>
                  {hasMask ? "測定対象を囲み直す" : isSegmenting ? "自動抽出中…" : "測定対象を囲む"}
                </button>
              </div>

              {hasMask && (
                <div className="result-card">
                  <p className="result-kicker">測定結果</p>
                  <div className="area-value"><strong>{areaCm2.toFixed(1)}</strong><span>cm²</span></div>
                  <small>{areaPx.toLocaleString("ja-JP")} px を測定</small>
                  <div className="refine-row">
                    <button className={mode === "add" ? "active" : ""} type="button" onClick={() => beginRefine("add")}><i className="add-dot" />塗り足す</button>
                    <button className={mode === "erase" ? "active erase" : ""} type="button" onClick={() => beginRefine("erase")}><i className="erase-dot" />消す</button>
                  </div>
                  <label className="brush-control">
                    <span>ブラシサイズ <b>{brushSize}px</b></span>
                    <input type="range" min="3" max="80" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
                  </label>
                  <button className="confirm-button" type="button" onClick={finishRefine}>この面積で確定</button>
                </div>
              )}

              <div className="compact-actions">
                {!hasMask && <button className="text-button" type="button" onClick={beginCrop}>トリミング</button>}
                {selectionKind === "crop" && selectionRect && <button className="text-button accent" type="button" onClick={applyCrop}>トリミングを適用</button>}
                {selectionRect && <button className="text-button" type="button" onClick={cancelSelection}>選択を取消</button>}
              </div>
            </>
          )}

          <p className="privacy-note"><span aria-hidden="true">⌁</span><b>プライバシー</b><br />写真はサーバーへ送信されず、この端末のブラウザ内だけで処理されます。</p>
        </aside>
      </div>

      <footer>本ツールの測定値は参考値です。精密測定、診断、安全性・品質などの重要な判断には使用しないでください。</footer>
    </main>
  );
}
