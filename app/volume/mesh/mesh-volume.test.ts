import assert from "node:assert/strict";
import test from "node:test";

import { measureTriangles, type MeasurementInput } from "./mesh-volume.ts";

const topOfBox = new Float32Array([
  -0.05, 0.02, -0.03,
  -0.05, 0.02, 0.03,
  0.05, 0.02, 0.03,
  -0.05, 0.02, -0.03,
  0.05, 0.02, 0.03,
  0.05, 0.02, -0.03,
]);

const input: MeasurementInput = {
  triangles: topOfBox,
  planeNormal: { x: 0, y: 1, z: 0 },
  planeOffset: 0,
  basisU: { x: 1, y: 0, z: 0 },
  basisV: { x: 0, y: 0, z: 1 },
  maxHeight: 0.02,
  orientationSign: 1,
};

test("100 × 60 × 20 mmの直方体を120 cm³と計算する", () => {
  const result = measureTriangles(input, 0, { centerU: 0, centerV: 0, sizeU: 0.1, sizeV: 0.06 });
  assert.ok(Math.abs(result.volume * 1_000_000 - 120) < 0.0001);
  assert.ok(Math.abs(result.width - 0.1) < 0.000001);
  assert.ok(Math.abs(result.depth - 0.06) < 0.000001);
  assert.ok(Math.abs(result.height - 0.02) < 0.000001);
});

test("切断面を1 mm上げると高さ19 mmとして計算する", () => {
  const result = measureTriangles(input, 0.001, { centerU: 0, centerV: 0, sizeU: 0.1, sizeV: 0.06 });
  assert.ok(Math.abs(result.volume * 1_000_000 - 114) < 0.0001);
});

test("対象範囲を半分にすると体積も半分になる", () => {
  const result = measureTriangles(input, 0, { centerU: -0.025, centerV: 0, sizeU: 0.05, sizeV: 0.06 });
  assert.ok(Math.abs(result.volume * 1_000_000 - 60) < 0.0001);
});

test("反転した面方向を補正できる", () => {
  const inverted = new Float32Array(topOfBox);
  for (let index = 0; index < inverted.length; index += 9) {
    for (let axis = 0; axis < 3; axis += 1) {
      const temporary = inverted[index + 3 + axis];
      inverted[index + 3 + axis] = inverted[index + 6 + axis];
      inverted[index + 6 + axis] = temporary;
    }
  }
  const result = measureTriangles({ ...input, triangles: inverted, orientationSign: -1 }, 0, { centerU: 0, centerV: 0, sizeU: 0.1, sizeV: 0.06 });
  assert.ok(Math.abs(result.volume * 1_000_000 - 120) < 0.0001);
});
