export type Vector = { x: number; y: number; z: number };
export type Range2D = { centerU: number; centerV: number; sizeU: number; sizeV: number };

export type MeasurementInput = {
  triangles: Float32Array;
  planeNormal: Vector;
  planeOffset: number;
  basisU: Vector;
  basisV: Vector;
  maxHeight: number;
  orientationSign: 1 | -1;
};

export type Measurement = {
  volume: number;
  projectedArea: number;
  usedFaces: number;
  width: number;
  depth: number;
  height: number;
};

function dot(a: Vector, b: Vector) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
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

function vectorFromArray(values: Float32Array, offset: number): Vector {
  return { x: values[offset], y: values[offset + 1], z: values[offset + 2] };
}

function clipPolygon(points: Vector[], normal: Vector, constant: number) {
  if (!points.length) return points;
  const result: Vector[] = [];
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const previous = points[(index + points.length - 1) % points.length];
    const currentDistance = dot(current, normal) + constant;
    const previousDistance = dot(previous, normal) + constant;
    const currentInside = currentDistance >= 0;
    const previousInside = previousDistance >= 0;
    if (currentInside !== previousInside) {
      const ratio = previousDistance / (previousDistance - currentDistance);
      result.push({
        x: previous.x + (current.x - previous.x) * ratio,
        y: previous.y + (current.y - previous.y) * ratio,
        z: previous.z + (current.z - previous.z) * ratio,
      });
    }
    if (currentInside) result.push(current);
  }
  return result;
}

export function measureTriangles(input: MeasurementInput, cutOffset: number, range: Range2D): Measurement {
  const { triangles, planeNormal, planeOffset, basisU, basisV, orientationSign } = input;
  const cutPosition = planeOffset + cutOffset;
  const minU = range.centerU - range.sizeU / 2;
  const maxU = range.centerU + range.sizeU / 2;
  const minV = range.centerV - range.sizeV / 2;
  const maxV = range.centerV + range.sizeV / 2;
  const planes = [
    { normal: planeNormal, constant: -cutPosition },
    { normal: basisU, constant: -minU },
    { normal: { x: -basisU.x, y: -basisU.y, z: -basisU.z }, constant: maxU },
    { normal: basisV, constant: -minV },
    { normal: { x: -basisV.x, y: -basisV.y, z: -basisV.z }, constant: maxV },
  ];

  let volume = 0;
  let projectedArea = 0;
  let usedFaces = 0;
  let minMeasuredU = Number.POSITIVE_INFINITY;
  let maxMeasuredU = Number.NEGATIVE_INFINITY;
  let minMeasuredV = Number.POSITIVE_INFINITY;
  let maxMeasuredV = Number.NEGATIVE_INFINITY;
  let maxHeight = 0;

  for (let index = 0; index < triangles.length; index += 9) {
    let polygon = [
      vectorFromArray(triangles, index),
      vectorFromArray(triangles, index + 3),
      vectorFromArray(triangles, index + 6),
    ];
    for (const plane of planes) {
      polygon = clipPolygon(polygon, plane.normal, plane.constant);
      if (polygon.length < 3) break;
    }
    if (polygon.length < 3) continue;

    for (let corner = 1; corner + 1 < polygon.length; corner += 1) {
      const a = polygon[0];
      const b = polygon[corner];
      const c = polygon[corner + 1];
      const projected = (dot(cross(subtract(b, a), subtract(c, a)), planeNormal) / 2) * orientationSign;
      if (projected <= 1e-12) continue;
      const heights = [dot(a, planeNormal) - cutPosition, dot(b, planeNormal) - cutPosition, dot(c, planeNormal) - cutPosition];
      const meanHeight = (heights[0] + heights[1] + heights[2]) / 3;
      if (meanHeight <= 0) continue;
      volume += projected * meanHeight;
      projectedArea += projected;
      usedFaces += 1;
      for (const point of [a, b, c]) {
        const height = dot(point, planeNormal) - cutPosition;
        if (height < Math.max(input.maxHeight * 0.08, 1e-5)) continue;
        const u = dot(point, basisU);
        const v = dot(point, basisV);
        minMeasuredU = Math.min(minMeasuredU, u);
        maxMeasuredU = Math.max(maxMeasuredU, u);
        minMeasuredV = Math.min(minMeasuredV, v);
        maxMeasuredV = Math.max(maxMeasuredV, v);
        maxHeight = Math.max(maxHeight, height);
      }
    }
  }

  return {
    volume,
    projectedArea,
    usedFaces,
    width: Number.isFinite(minMeasuredU) ? maxMeasuredU - minMeasuredU : 0,
    depth: Number.isFinite(minMeasuredV) ? maxMeasuredV - minMeasuredV : 0,
    height: maxHeight,
  };
}
