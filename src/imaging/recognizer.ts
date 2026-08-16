import { createWorker, PSM } from "tesseract.js";
import {
  reconcileWithSudoku,
  type CellPrediction,
  type OcrCandidate,
} from "./ocrPostprocess";

export type RecognitionResult = {
  values: number[];
  confidence: number[];
  normalizedImage: string;
  ignoredAnnotations: number;
};

type ProgressCallback = (progress: number, message: string) => void;
type OpenCvRuntime = typeof import("@techstark/opencv-js");
type OpenCvMat = { delete: () => void; data32S: Int32Array };

type Component = {
  pixels: number[];
  area: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

type GlyphVariant = {
  canvas: HTMLCanvasElement;
  geometryConfidence: number;
  heightRatio: number;
  centerOffset: number;
  quality: number;
};

type GlyphCell = GlyphVariant & {
  index: number;
  alternatives: GlyphVariant[];
};

const BOARD_SIZE = 900;
const GLYPH_SIZE = 96;
// False fixed clues are more harmful than omissions because imported clues are
// locked. Only very strong, clearly separated candidates are auto-populated.
const MIN_AUTO_CONFIDENCE = 0.72;

function centerSquare(source: HTMLCanvasElement): HTMLCanvasElement {
  const output = document.createElement("canvas");
  output.width = BOARD_SIZE;
  output.height = BOARD_SIZE;
  const context = output.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvasを初期化できませんでした。");
  const side = Math.min(source.width, source.height);
  const x = (source.width - side) / 2;
  const y = (source.height - side) / 2;
  context.drawImage(source, x, y, side, side, 0, 0, BOARD_SIZE, BOARD_SIZE);
  return output;
}

function orderCorners(points: Array<{ x: number; y: number }>) {
  const bySum = [...points].sort((a, b) => a.x + a.y - (b.x + b.y));
  const byDiff = [...points].sort((a, b) => a.y - a.x - (b.y - b.x));
  return [bySum[0], byDiff[0], bySum[3], byDiff[3]];
}

async function getOpenCv(): Promise<OpenCvRuntime> {
  const imported = await import("@techstark/opencv-js");
  const candidate: unknown = (imported as unknown as { default?: unknown }).default ?? imported;
  let cv = candidate as OpenCvRuntime;
  if (typeof (candidate as { then?: unknown })?.then === "function") {
    cv = await (candidate as Promise<OpenCvRuntime>);
  }
  if (cv?.Mat) return cv;
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(
      () => reject(new Error("OpenCVの読み込みがタイムアウトしました。")),
      15000,
    );
    (cv as OpenCvRuntime & { onRuntimeInitialized: () => void }).onRuntimeInitialized = () => {
      window.clearTimeout(timeout);
      resolve();
    };
  });
  return cv;
}

async function normalizeBoard(source: HTMLCanvasElement): Promise<HTMLCanvasElement> {
  try {
    const cv = await getOpenCv();
    const input = cv.imread(source);
    const gray = new cv.Mat();
    const blurred = new cv.Mat();
    const binary = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    let best: OpenCvMat | null = null;
    let bestArea = 0;

    try {
      cv.cvtColor(input, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, blurred, new cv.Size(7, 7), 0, 0, cv.BORDER_DEFAULT);
      cv.adaptiveThreshold(
        blurred,
        binary,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        21,
        7,
      );
      cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      for (let index = 0; index < contours.size(); index += 1) {
        const contour = contours.get(index);
        const perimeter = cv.arcLength(contour, true);
        const approximate = new cv.Mat();
        cv.approxPolyDP(contour, approximate, perimeter * 0.025, true);
        const area = Math.abs(cv.contourArea(approximate));
        if (
          approximate.rows === 4 &&
          cv.isContourConvex(approximate) &&
          area > bestArea &&
          area > input.rows * input.cols * 0.12
        ) {
          best?.delete();
          best = approximate.clone();
          bestArea = area;
        }
        approximate.delete();
        contour.delete();
      }

      if (!best) return centerSquare(source);

      const raw = best.data32S;
      const points = orderCorners(
        Array.from({ length: 4 }, (_, index) => ({
          x: raw[index * 2],
          y: raw[index * 2 + 1],
        })),
      );
      const sourcePoints = cv.matFromArray(
        4,
        1,
        cv.CV_32FC2,
        points.flatMap((point) => [point.x, point.y]),
      );
      const destinationPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0,
        0,
        BOARD_SIZE,
        0,
        BOARD_SIZE,
        BOARD_SIZE,
        0,
        BOARD_SIZE,
      ]);
      const transform = cv.getPerspectiveTransform(sourcePoints, destinationPoints);
      const warped = new cv.Mat();
      cv.warpPerspective(
        input,
        warped,
        transform,
        new cv.Size(BOARD_SIZE, BOARD_SIZE),
        cv.INTER_LINEAR,
        cv.BORDER_CONSTANT,
        new cv.Scalar(255, 255, 255, 255),
      );

      const output = document.createElement("canvas");
      output.width = BOARD_SIZE;
      output.height = BOARD_SIZE;
      cv.imshow(output, warped);
      sourcePoints.delete();
      destinationPoints.delete();
      transform.delete();
      warped.delete();
      best.delete();
      return output;
    } finally {
      input.delete();
      gray.delete();
      blurred.delete();
      binary.delete();
      contours.delete();
      hierarchy.delete();
    }
  } catch (error) {
    console.warn("盤面の自動検出に失敗したため、中央の正方形を使用します。", error);
    return centerSquare(source);
  }
}

function grayscaleOf(canvas: HTMLCanvasElement) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("補正画像を読み取れませんでした。");
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let pixel = 0, offset = 0; pixel < gray.length; pixel += 1, offset += 4) {
    gray[pixel] = Math.round(
      image.data[offset] * 0.299 +
      image.data[offset + 1] * 0.587 +
      image.data[offset + 2] * 0.114,
    );
  }
  return gray;
}

/**
 * Removes slow illumination changes before thresholding. A large local mean
 * represents the paper/background; subtracting it keeps printed strokes while
 * flattening phone/camera shadows across the board.
 */
function normalizeIllumination(
  gray: Uint8Array,
  width: number,
  height: number,
) {
  const stride = width + 1;
  const integral = new Uint32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      rowSum += gray[y * width + x];
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + rowSum;
    }
  }

  const radius = Math.max(24, Math.round(Math.min(width, height) / 28));
  const output = new Uint8Array(gray.length);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width - 1, x + radius);
      const sum =
        integral[(bottom + 1) * stride + right + 1] -
        integral[top * stride + right + 1] -
        integral[(bottom + 1) * stride + left] +
        integral[top * stride + left];
      const area = (right - left + 1) * (bottom - top + 1);
      const localBackground = sum / area;
      const corrected = 238 + (gray[y * width + x] - localBackground) * 1.55;
      output[y * width + x] = Math.max(0, Math.min(255, Math.round(corrected)));
    }
  }
  return output;
}

function adaptiveBinary(gray: Uint8Array, width: number, height: number) {
  const integral = new Uint32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      rowSum += gray[y * width + x];
      integral[(y + 1) * (width + 1) + x + 1] =
        integral[y * (width + 1) + x + 1] + rowSum;
    }
  }

  const binary = new Uint8Array(width * height);
  const radius = 18;
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width - 1, x + radius);
      const sum =
        integral[(bottom + 1) * (width + 1) + right + 1] -
        integral[top * (width + 1) + right + 1] -
        integral[(bottom + 1) * (width + 1) + left] +
        integral[top * (width + 1) + left];
      const area = (right - left + 1) * (bottom - top + 1);
      const mean = sum / area;
      const value = gray[y * width + x];
      binary[y * width + x] = value < mean - 8 && value < 235 ? 1 : 0;
    }
  }
  return binary;
}

function axisScore(
  binary: Uint8Array,
  width: number,
  height: number,
  position: number,
  axis: "x" | "y",
) {
  let score = 0;
  for (let band = -2; band <= 2; band += 1) {
    const coordinate = position + band;
    if (coordinate < 0 || coordinate >= (axis === "x" ? width : height)) continue;
    const crossSize = axis === "x" ? height : width;
    for (let cross = 0; cross < crossSize; cross += 2) {
      const index = axis === "x"
        ? cross * width + coordinate
        : coordinate * width + cross;
      score += binary[index];
    }
  }
  return score;
}

function detectGridLines(
  binary: Uint8Array,
  width: number,
  height: number,
  axis: "x" | "y",
) {
  const size = axis === "x" ? width : height;
  const crossSize = axis === "x" ? height : width;
  const lines = [0];

  for (let lineIndex = 1; lineIndex < 9; lineIndex += 1) {
    const expected = Math.round((size * lineIndex) / 9);
    // Perspective normalization already places lines near equal ninths. Keep the
    // projection search local so a bold clue cannot be mistaken for a grid line.
    const radius = Math.round(size / 90);
    let bestPosition = expected;
    let bestScore = -1;
    for (
      let position = Math.max(1, expected - radius);
      position <= Math.min(size - 2, expected + radius);
      position += 1
    ) {
      const score = axisScore(binary, width, height, position, axis);
      if (score > bestScore) {
        bestScore = score;
        bestPosition = position;
      }
    }
    lines.push(bestScore >= crossSize * 0.12 ? bestPosition : expected);
  }
  lines.push(size);

  const gaps = lines.slice(1).map((line, index) => line - lines[index]);
  const expectedGap = size / 9;
  const valid = gaps.every((gap) => gap > expectedGap * 0.58 && gap < expectedGap * 1.42);
  return valid ? lines : Array.from({ length: 10 }, (_, index) => Math.round((size * index) / 9));
}

function otsuThreshold(values: Uint8Array) {
  const histogram = new Uint32Array(256);
  values.forEach((value) => {
    histogram[value] += 1;
  });
  const total = values.length;
  let sum = 0;
  for (let value = 0; value < 256; value += 1) sum += value * histogram[value];

  let backgroundWeight = 0;
  let backgroundSum = 0;
  let bestVariance = -1;
  let threshold = 127;

  for (let value = 0; value < 256; value += 1) {
    backgroundWeight += histogram[value];
    if (!backgroundWeight) continue;
    const foregroundWeight = total - backgroundWeight;
    if (!foregroundWeight) break;
    backgroundSum += value * histogram[value];
    const backgroundMean = backgroundSum / backgroundWeight;
    const foregroundMean = (sum - backgroundSum) / foregroundWeight;
    const variance =
      backgroundWeight *
      foregroundWeight *
      (backgroundMean - foregroundMean) *
      (backgroundMean - foregroundMean);
    if (variance > bestVariance) {
      bestVariance = variance;
      threshold = value;
    }
  }
  return Math.min(220, Math.max(45, threshold));
}

function connectedComponents(binary: Uint8Array, width: number, height: number) {
  const visited = new Uint8Array(binary.length);
  const queue = new Int32Array(binary.length);
  const components: Component[] = [];
  const neighbors = [
    [-1, -1], [0, -1], [1, -1],
    [-1, 0], [1, 0],
    [-1, 1], [0, 1], [1, 1],
  ];

  for (let start = 0; start < binary.length; start += 1) {
    if (!binary[start] || visited[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;
    const component: Component = {
      pixels: [],
      area: 0,
      minX: width,
      minY: height,
      maxX: 0,
      maxY: 0,
    };

    while (head < tail) {
      const current = queue[head++];
      const x = current % width;
      const y = Math.floor(current / width);
      component.pixels.push(current);
      component.area += 1;
      component.minX = Math.min(component.minX, x);
      component.minY = Math.min(component.minY, y);
      component.maxX = Math.max(component.maxX, x);
      component.maxY = Math.max(component.maxY, y);

      for (const [dx, dy] of neighbors) {
        const nextX = x + dx;
        const nextY = y + dy;
        if (nextX < 0 || nextY < 0 || nextX >= width || nextY >= height) continue;
        const next = nextY * width + nextX;
        if (binary[next] && !visited[next]) {
          visited[next] = 1;
          queue[tail++] = next;
        }
      }
    }
    components.push(component);
  }

  return components;
}

function createGlyphCanvas(
  binary: Uint8Array,
  width: number,
  height: number,
  components: Component[],
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
) {
  const source = document.createElement("canvas");
  source.width = width;
  source.height = height;
  const sourceContext = source.getContext("2d");
  if (!sourceContext) throw new Error("文字画像を作成できませんでした。");
  const image = sourceContext.createImageData(width, height);
  image.data.fill(255);
  for (const component of components) {
    for (const pixel of component.pixels) {
      const offset = pixel * 4;
      image.data[offset] = 0;
      image.data[offset + 1] = 0;
      image.data[offset + 2] = 0;
      image.data[offset + 3] = 255;
    }
  }
  sourceContext.putImageData(image, 0, 0);

  const glyph = document.createElement("canvas");
  glyph.width = GLYPH_SIZE;
  glyph.height = GLYPH_SIZE;
  const context = glyph.getContext("2d");
  if (!context) throw new Error("認識画像を作成できませんでした。");
  context.fillStyle = "white";
  context.fillRect(0, 0, GLYPH_SIZE, GLYPH_SIZE);
  context.imageSmoothingEnabled = true;
  const contentWidth = bounds.maxX - bounds.minX + 1;
  const contentHeight = bounds.maxY - bounds.minY + 1;
  const scale = Math.min(66 / contentWidth, 66 / contentHeight);
  const targetWidth = Math.max(1, Math.round(contentWidth * scale));
  const targetHeight = Math.max(1, Math.round(contentHeight * scale));
  const targetX = Math.round((GLYPH_SIZE - targetWidth) / 2);
  const targetY = Math.round((GLYPH_SIZE - targetHeight) / 2);
  context.drawImage(
    source,
    bounds.minX,
    bounds.minY,
    contentWidth,
    contentHeight,
    targetX,
    targetY,
    targetWidth,
    targetHeight,
  );
  return glyph;
}

function binaryAtThreshold(values: Uint8Array, threshold: number) {
  const binary = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    binary[index] = values[index] < threshold ? 1 : 0;
  }
  return binary;
}

function adaptiveCellBinary(values: Uint8Array, width: number, height: number) {
  const stride = width + 1;
  const integral = new Uint32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      rowSum += values[y * width + x];
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + rowSum;
    }
  }

  const radius = Math.max(8, Math.round(Math.min(width, height) * 0.2));
  const binary = new Uint8Array(values.length);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width - 1, x + radius);
      const sum =
        integral[(bottom + 1) * stride + right + 1] -
        integral[top * stride + right + 1] -
        integral[(bottom + 1) * stride + left] +
        integral[top * stride + left];
      const area = (right - left + 1) * (bottom - top + 1);
      binary[y * width + x] = values[y * width + x] < sum / area - 10 ? 1 : 0;
    }
  }
  return binary;
}

function analyzeCellBinary(binary: Uint8Array, width: number, height: number) {
  const minimumArea = Math.max(18, Math.round(binary.length * 0.004));
  const components = connectedComponents(binary, width, height)
    .filter((component) => {
      const componentWidth = component.maxX - component.minX + 1;
      const componentHeight = component.maxY - component.minY + 1;
      const looksLikeHorizontalLine =
        componentWidth > width * 0.72 && componentHeight < height * 0.16;
      const looksLikeVerticalLine =
        componentHeight > height * 0.72 && componentWidth < width * 0.16;
      const coversMostOfCell = component.area > binary.length * 0.42;
      const touchesCropEdge =
        component.minX <= 1 ||
        component.minY <= 1 ||
        component.maxX >= width - 2 ||
        component.maxY >= height - 2;
      return (
        !looksLikeHorizontalLine &&
        !looksLikeVerticalLine &&
        !coversMostOfCell &&
        !touchesCropEdge &&
        component.area >= minimumArea
      );
    })
    .sort((a, b) => b.area - a.area);

  if (!components.length) return { kind: "blank" as const };

  // Sudoku digits are single connected shapes in the supported printed fonts.
  // Keeping the dominant component prevents distant shadow/noise fragments from
  // expanding the glyph bounds and shifting it away from the cell centre.
  const kept = [components[0]];
  const bounds = components[0];
  const contentWidth = bounds.maxX - bounds.minX + 1;
  const contentHeight = bounds.maxY - bounds.minY + 1;
  const widthRatio = contentWidth / width;
  const heightRatio = contentHeight / height;
  const areaRatio = bounds.area / binary.length;
  const centerX = (bounds.minX + bounds.maxX + 1) / 2 / width;
  const centerY = (bounds.minY + bounds.maxY + 1) / 2 / height;
  const centerOffset = Math.hypot(centerX - 0.5, centerY - 0.5);
  const fillRatio = bounds.area / (contentWidth * contentHeight);
  const liesInCorner =
    (centerX < 0.38 || centerX > 0.62) &&
    (centerY < 0.38 || centerY > 0.62);
  const likelyAnnotation =
    liesInCorner &&
    heightRatio < 0.48 &&
    widthRatio < 0.48;

  if (likelyAnnotation) return { kind: "annotation" as const };
  const plausibleDigit =
    heightRatio >= 0.36 &&
    heightRatio <= 0.86 &&
    widthRatio >= 0.1 &&
    widthRatio <= 0.72 &&
    areaRatio >= 0.012 &&
    areaRatio <= 0.28 &&
    fillRatio >= 0.09 &&
    fillRatio <= 0.72 &&
    centerOffset <= 0.3;
  if (!plausibleDigit) return { kind: "blank" as const };

  const heightScore = 1 - Math.min(1, Math.abs(heightRatio - 0.62) / 0.5);
  const centerScore = 1 - Math.min(1, centerOffset / 0.3);
  const areaScore = 1 - Math.min(1, Math.abs(areaRatio - 0.09) / 0.2);
  const quality = heightScore * 0.42 + centerScore * 0.43 + areaScore * 0.15;

  return {
    kind: "digit" as const,
    variant: {
      canvas: createGlyphCanvas(binary, width, height, kept, bounds),
      geometryConfidence: Math.min(0.98, Math.max(0.62, 0.6 + quality * 0.38)),
      heightRatio,
      centerOffset,
      quality,
    } satisfies GlyphVariant,
  };
}

function preprocessCell(
  gray: Uint8Array,
  boardWidth: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
) {
  // A generous inset is intentional: after perspective correction all clues
  // sit near the centre, while residual grid strokes stay at the cell edges.
  const insetX = Math.max(7, Math.round((x1 - x0) * 0.12));
  const insetY = Math.max(7, Math.round((y1 - y0) * 0.12));
  const left = x0 + insetX;
  const top = y0 + insetY;
  const width = Math.max(1, x1 - x0 - insetX * 2);
  const height = Math.max(1, y1 - y0 - insetY * 2);
  const crop = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      crop[y * width + x] = gray[(top + y) * boardWidth + left + x];
    }
  }

  const otsu = otsuThreshold(crop);
  const binaries = [
    binaryAtThreshold(crop, Math.min(150, otsu)),
    binaryAtThreshold(crop, otsu),
    adaptiveCellBinary(crop, width, height),
  ];
  const analyzed = binaries.map((binary) => analyzeCellBinary(binary, width, height));
  const digitVariants = analyzed
    .flatMap((result) => (result.kind === "digit" ? [result.variant] : []))
    .sort((a, b) => b.quality - a.quality);

  // A single threshold is easily fooled by shadows or print halftones. Require
  // two independent threshold images to produce the same centred shape.
  let bestConsensus = 0;
  for (let leftIndex = 0; leftIndex < digitVariants.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < digitVariants.length; rightIndex += 1) {
      bestConsensus = Math.max(
        bestConsensus,
        glyphSimilarity(
          glyphInk(digitVariants[leftIndex].canvas),
          glyphInk(digitVariants[rightIndex].canvas),
        ),
      );
    }
  }

  if (digitVariants.length >= 2 && bestConsensus >= 0.64) {
    const variants = digitVariants.slice(0, 3);
    variants[0].geometryConfidence *= Math.min(1, 0.9 + bestConsensus * 0.1);
    return {
      kind: "digit" as const,
      primary: variants[0],
      alternatives: variants.slice(1),
    };
  }
  if (analyzed.filter((result) => result.kind === "annotation").length >= 2) {
    return { kind: "annotation" as const };
  }
  return { kind: "blank" as const };
}

function extractGlyphCells(board: HTMLCanvasElement) {
  const gray = normalizeIllumination(
    grayscaleOf(board),
    board.width,
    board.height,
  );
  const binary = adaptiveBinary(gray, board.width, board.height);
  const verticalLines = detectGridLines(binary, board.width, board.height, "x");
  const horizontalLines = detectGridLines(binary, board.width, board.height, "y");
  const glyphs: GlyphCell[] = [];
  let ignoredAnnotations = 0;

  for (let row = 0; row < 9; row += 1) {
    for (let column = 0; column < 9; column += 1) {
      const processed = preprocessCell(
        gray,
        board.width,
        verticalLines[column],
        horizontalLines[row],
        verticalLines[column + 1],
        horizontalLines[row + 1],
      );
      if (processed.kind === "annotation") {
        ignoredAnnotations += 1;
      } else if (processed.kind === "digit") {
        glyphs.push({
          index: row * 9 + column,
          ...processed.primary,
          alternatives: processed.alternatives,
        });
      }
    }
  }

  return { glyphs, ignoredAnnotations };
}

function createSparseOcrCanvas(board: HTMLCanvasElement) {
  const gray = normalizeIllumination(
    grayscaleOf(board),
    board.width,
    board.height,
  );
  const binary = adaptiveBinary(gray, board.width, board.height);
  const verticalLines = detectGridLines(binary, board.width, board.height, "x");
  const horizontalLines = detectGridLines(binary, board.width, board.height, "y");
  const output = document.createElement("canvas");
  output.width = board.width;
  output.height = board.height;
  const context = output.getContext("2d");
  if (!context) throw new Error("盤面OCR画像を作成できませんでした。");
  const image = context.createImageData(board.width, board.height);
  image.data.fill(255);

  const nearLine = (coordinate: number, lines: number[]) =>
    lines.some((line) => Math.abs(coordinate - line) <= 10);
  for (let y = 0; y < board.height; y += 1) {
    if (nearLine(y, horizontalLines)) continue;
    for (let x = 0; x < board.width; x += 1) {
      if (!binary[y * board.width + x] || nearLine(x, verticalLines)) continue;
      const offset = (y * board.width + x) * 4;
      image.data[offset] = 0;
      image.data[offset + 1] = 0;
      image.data[offset + 2] = 0;
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  return output;
}

function collectSparsePredictions(
  result: Awaited<ReturnType<Awaited<ReturnType<typeof createWorker>>["recognize"]>>,
  width: number,
  height: number,
  allowedIndexes: Set<number>,
) {
  const predictions: CellPrediction[] = [];
  const words =
    result.data.blocks?.flatMap((block) =>
      block.paragraphs.flatMap((paragraph) =>
        paragraph.lines.flatMap((line) => line.words),
      ),
    ) ?? [];

  for (const word of words) {
    const digit = Number(word.text.replace(/[^1-9]/g, "")[0]);
    if (!digit || word.confidence < 55) continue;
    const boxWidth = word.bbox.x1 - word.bbox.x0;
    const boxHeight = word.bbox.y1 - word.bbox.y0;
    // Printed clues occupy roughly half a cell. This removes small A/B labels
    // even when a digits-only OCR pass tries to coerce them into a number.
    if (boxHeight < height / 24 || boxWidth > width / 7) continue;
    const centerX = (word.bbox.x0 + word.bbox.x1) / 2;
    const centerY = (word.bbox.y0 + word.bbox.y1) / 2;
    const column = Math.max(0, Math.min(8, Math.floor((centerX / width) * 9)));
    const row = Math.max(0, Math.min(8, Math.floor((centerY / height) * 9)));
    const index = row * 9 + column;
    if (!allowedIndexes.has(index)) continue;
    predictions.push({
      index,
      candidates: [{
        digit,
        confidence: Math.max(0.4, Math.min(0.68, (word.confidence / 100) * 0.72)),
      }],
    });
  }
  return predictions;
}

function mergePredictions(target: CellPrediction[], additions: CellPrediction[]) {
  const byIndex = new Map(target.map((prediction) => [prediction.index, prediction]));
  for (const addition of additions) {
    const existing = byIndex.get(addition.index);
    if (existing) {
      existing.candidates = mergeCandidates(existing.candidates, addition.candidates);
    } else {
      target.push(addition);
      byIndex.set(addition.index, addition);
    }
  }
}

function collectCandidates(
  result: Awaited<ReturnType<Awaited<ReturnType<typeof createWorker>>["recognize"]>>,
  glyph: GlyphVariant,
) {
  const candidates = new Map<number, number>();
  const text = result.data.text.trim().toUpperCase().replace(/[^A-Z1-9]/g, "");
  const primaryCharacter = text[0] ?? "";
  const primaryConfidence = result.data.confidence / 100;

  const add = (candidateText: string, candidateConfidence: number) => {
    const character = candidateText.trim()[0] ?? "";
    const digit = Number(character);
    if (!digit || digit > 9) return;
    const weighted = Math.max(
      0,
      Math.min(1, candidateConfidence * glyph.geometryConfidence),
    );
    candidates.set(digit, Math.max(candidates.get(digit) ?? 0, weighted));
  };

  if (/^[1-9]$/.test(primaryCharacter)) {
    add(primaryCharacter, primaryConfidence);
  } else if (/^[A-Z]$/.test(primaryCharacter)) {
    const isSmallOffCentreAnnotation =
      glyph.heightRatio < 0.45 && glyph.centerOffset > 0.22;
    if ((primaryCharacter === "A" || primaryCharacter === "B") && isSmallOffCentreAnnotation) {
      return { candidates: [] as OcrCandidate[], annotation: true };
    }

    // Bold printed Sudoku digits are often returned as visually similar letters
    // when the whitelist also needs to preserve A/B annotations.
    const digitLookalikes: Record<string, number> = {
      I: 1,
      L: 1,
      G: 6,
      C: 6,
      B: 8,
      O: 8,
      Q: 9,
      P: 9,
    };
    const correctedDigit = digitLookalikes[primaryCharacter];
    if (correctedDigit) {
      add(String(correctedDigit), primaryConfidence * 0.9);
    }
  }

  const words =
    result.data.blocks?.flatMap((block) =>
      block.paragraphs.flatMap((paragraph) =>
        paragraph.lines.flatMap((line) => line.words),
      ),
    ) ?? [];
  for (const word of words) {
    add(word.text, word.confidence / 100);
    for (const choice of word.choices ?? []) {
      add(choice.text, choice.confidence / 100);
    }
  }

  return {
    candidates: [...candidates.entries()]
    .map(([digit, confidence]) => ({ digit, confidence }))
    .filter((candidate) => candidate.confidence >= 0.3)
    .sort((a, b) => b.confidence - a.confidence),
    annotation: false,
  };
}

function mergeCandidates(...groups: OcrCandidate[][]) {
  const merged = new Map<number, number>();
  for (const group of groups) {
    for (const candidate of group) {
      merged.set(
        candidate.digit,
        Math.max(merged.get(candidate.digit) ?? 0, candidate.confidence),
      );
    }
  }
  return [...merged.entries()]
    .map(([digit, confidence]) => ({ digit, confidence }))
    .sort((a, b) => b.confidence - a.confidence);
}

function glyphInk(canvas: HTMLCanvasElement) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return new Float32Array();
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const ink = new Float32Array(canvas.width * canvas.height);
  for (let index = 0; index < ink.length; index += 1) {
    ink[index] = (255 - pixels[index * 4]) / 255;
  }
  return ink;
}

function glyphSimilarity(left: Float32Array, right: Float32Array) {
  if (!left.length || left.length !== right.length) return 0;
  let best = 0;
  for (let shiftY = -2; shiftY <= 2; shiftY += 2) {
    for (let shiftX = -2; shiftX <= 2; shiftX += 2) {
      let dot = 0;
      let leftNorm = 0;
      let rightNorm = 0;
      for (let y = 2; y < GLYPH_SIZE - 2; y += 1) {
        for (let x = 2; x < GLYPH_SIZE - 2; x += 1) {
          const shiftedX = x + shiftX;
          const shiftedY = y + shiftY;
          if (
            shiftedX < 0 ||
            shiftedX >= GLYPH_SIZE ||
            shiftedY < 0 ||
            shiftedY >= GLYPH_SIZE
          ) continue;
          const leftValue = left[y * GLYPH_SIZE + x];
          const rightValue = right[shiftedY * GLYPH_SIZE + shiftedX];
          dot += leftValue * rightValue;
          leftNorm += leftValue * leftValue;
          rightNorm += rightValue * rightValue;
        }
      }
      if (leftNorm && rightNorm) {
        best = Math.max(best, dot / Math.sqrt(leftNorm * rightNorm));
      }
    }
  }
  return best;
}

function augmentWithRecognizedGlyphs(
  glyphs: GlyphCell[],
  predictions: CellPrediction[],
) {
  const glyphByIndex = new Map(glyphs.map((glyph) => [glyph.index, glyph]));
  const templates = predictions
    .filter((prediction) => prediction.candidates[0]?.confidence >= 0.58)
    .flatMap((prediction) => {
      const glyph = glyphByIndex.get(prediction.index);
      if (!glyph) return [];
      return [{
        index: prediction.index,
        digit: prediction.candidates[0].digit,
        inks: [glyph, ...glyph.alternatives].map((variant) => glyphInk(variant.canvas)),
      }];
    });

  const additions: CellPrediction[] = [];
  for (const glyph of glyphs) {
    const targets = [glyph, ...glyph.alternatives]
      .map((variant) => glyphInk(variant.canvas));
    const scores = new Map<number, number>();
    for (const template of templates) {
      // Do not let a dubious OCR result reinforce itself.
      if (template.index === glyph.index) continue;
      let bestSimilarity = 0;
      for (const target of targets) {
        for (const templateInk of template.inks) {
          bestSimilarity = Math.max(
            bestSimilarity,
            glyphSimilarity(target, templateInk),
          );
        }
      }
      scores.set(
        template.digit,
        Math.max(scores.get(template.digit) ?? 0, bestSimilarity),
      );
    }

    const candidates = [...scores.entries()]
      .filter(([, similarity]) => similarity >= 0.76)
      .sort((left, right) => right[1] - left[1])
      .slice(0, 3)
      .map(([digit, similarity]) => ({
        digit,
        confidence: Math.min(0.72, similarity * 0.78),
      }));
    if (!candidates.length) continue;
    additions.push({
      index: glyph.index,
      candidates,
    });
  }
  mergePredictions(predictions, additions);
}

export async function recognizeSudoku(
  source: HTMLCanvasElement,
  onProgress: ProgressCallback,
): Promise<RecognitionResult> {
  onProgress(0.04, "盤面の四隅を探しています");
  const normalized = await normalizeBoard(source);
  onProgress(0.12, "罫線と81マスを分離しています");
  const extracted = extractGlyphCells(normalized);
  const { glyphs } = extracted;
  let { ignoredAnnotations } = extracted;

  if (!glyphs.length) {
    return {
      values: Array(81).fill(0),
      confidence: Array(81).fill(0),
      normalizedImage: normalized.toDataURL("image/jpeg", 0.9),
      ignoredAnnotations,
    };
  }

  onProgress(0.18, "数字認識モデルを準備しています");
  let recognizingCells = false;
  const worker = await createWorker(
    "eng",
    undefined,
    {
      logger: ({ progress, status }) => {
        if (!recognizingCells) {
          onProgress(
            0.18 + progress * 0.08,
            status === "loading tesseract core"
              ? "認識エンジンを読み込んでいます"
              : "数字認識モデルを準備しています",
          );
        }
      },
    },
    {
      load_system_dawg: "0",
      load_freq_dawg: "0",
    },
  );

  const predictions: CellPrediction[] = [];
  try {
    // A single board-wide pass is much faster than running several Tesseract
    // variants for every occupied cell. It also gives us strong exemplars from
    // the same printed page for the template comparison below.
    await worker.setParameters({
      tessedit_char_whitelist: "123456789",
      tessedit_pageseg_mode: PSM.SPARSE_TEXT,
      user_defined_dpi: "300",
    });
    recognizingCells = true;
    onProgress(0.27, "盤面全体から数字を読み取っています");
    const sparseCanvas = createSparseOcrCanvas(normalized);
    const sparseResult = await worker.recognize(
      sparseCanvas,
      {},
      { blocks: true, text: true },
    );
    const allowedIndexes = new Set(glyphs.map((glyph) => glyph.index));
    mergePredictions(
      predictions,
      collectSparsePredictions(
        sparseResult,
        sparseCanvas.width,
        sparseCanvas.height,
        allowedIndexes,
      ),
    );

    await worker.setParameters({
      tessedit_char_whitelist: "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
      tessedit_pageseg_mode: PSM.SINGLE_CHAR,
    });

    const strongSparseIndexes = new Set(
      predictions
        .filter((prediction) => (prediction.candidates[0]?.confidence ?? 0) >= 0.58)
        .map((prediction) => prediction.index),
    );
    const glyphsToInspect = glyphs.filter(
      (glyph) => !strongSparseIndexes.has(glyph.index),
    );
    const unresolvedGlyphs: GlyphCell[] = [];

    for (let position = 0; position < glyphsToInspect.length; position += 1) {
      const glyph = glyphsToInspect[position];
      onProgress(
        0.34 + (position / Math.max(1, glyphsToInspect.length)) * 0.38,
        `${position + 1} / ${glyphsToInspect.length} マスを個別確認しています`,
      );
      const result = await worker.recognize(
        glyph.canvas,
        {},
        { blocks: true, text: true },
      );
      const reading = collectCandidates(result, glyph);
      if (reading.annotation) ignoredAnnotations += 1;
      if (reading.candidates.length) {
        mergePredictions(predictions, [{
          index: glyph.index,
          candidates: reading.candidates,
        }]);
      } else if (!reading.annotation) {
        unresolvedGlyphs.push(glyph);
      }
    }

    if (unresolvedGlyphs.length) {
      // A mixed digit/letter whitelist can make Tesseract return nothing for
      // heavy typefaces (for example 8≈B and 6≈G). A digits-only retry recovers
      // those clues after small corner annotations have already been removed.
      await worker.setParameters({
        tessedit_char_whitelist: "123456789",
        tessedit_pageseg_mode: PSM.SINGLE_CHAR,
      });
      for (let position = 0; position < unresolvedGlyphs.length; position += 1) {
        const glyph = unresolvedGlyphs[position];
        onProgress(
          0.74 + (position / unresolvedGlyphs.length) * 0.14,
          `${position + 1} / ${unresolvedGlyphs.length} マスを数字専用で再確認しています`,
        );
        const retryResult = await worker.recognize(
          glyph.canvas,
          {},
          { blocks: true, text: true },
        );
        const retryReading = collectCandidates(retryResult, glyph);
        if (retryReading.candidates.length) {
          mergePredictions(predictions, [{
            index: glyph.index,
            candidates: retryReading.candidates.map((candidate) => ({
              ...candidate,
              confidence: candidate.confidence * 0.82,
            })),
          }]);
        }
      }
    }

    // Alternative threshold images are compared in-memory with trusted digits
    // from this exact page. This recovers shadowed 2/7/8 clues without another
    // expensive OCR request for every variant.
    onProgress(0.9, "同じ紙面の数字と照合しています");
    augmentWithRecognizedGlyphs(glyphs, predictions);

    onProgress(0.96, "高信頼度の数字を数独ルールで確認しています");
    const highPrecisionPredictions = predictions.filter((prediction) => {
      const [best, second] = prediction.candidates;
      return (
        (best?.confidence ?? 0) >= MIN_AUTO_CONFIDENCE &&
        (!second || best.confidence - second.confidence >= 0.045)
      );
    });
    const reconciled = reconcileWithSudoku(highPrecisionPredictions);
    onProgress(
      1,
      ignoredAnnotations
        ? `読み取り完了（小さな注記${ignoredAnnotations}件を除外）`
        : "読み取りが完了しました",
    );
    return {
      ...reconciled,
      normalizedImage: normalized.toDataURL("image/jpeg", 0.9),
      ignoredAnnotations,
    };
  } finally {
    await worker.terminate();
  }
}
