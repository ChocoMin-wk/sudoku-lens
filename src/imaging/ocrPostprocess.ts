import { analyzePuzzle, boardFromDigits, conflictIndexes } from "../domain/sudoku";

export type OcrCandidate = {
  digit: number;
  confidence: number;
};

export type CellPrediction = {
  index: number;
  candidates: OcrCandidate[];
};

export type ReconciledOcr = {
  values: number[];
  confidence: number[];
};

function hasConflicts(values: number[]) {
  return conflictIndexes(boardFromDigits(values)).size > 0;
}

export function reconcileWithSudoku(predictions: CellPrediction[]): ReconciledOcr {
  const values = Array(81).fill(0) as number[];
  const confidence = Array(81).fill(0) as number[];

  for (const prediction of predictions) {
    const best = prediction.candidates[0];
    if (!best) continue;
    values[prediction.index] = best.digit;
    confidence[prediction.index] = best.confidence;
  }

  const byIndex = new Map(predictions.map((prediction) => [prediction.index, prediction]));

  while (hasConflicts(values)) {
    const conflicts = [...conflictIndexes(boardFromDigits(values))];
    const weakest = conflicts.sort((a, b) => confidence[a] - confidence[b])[0];
    const prediction = byIndex.get(weakest);
    let replaced = false;

    for (const alternative of prediction?.candidates.slice(1) ?? []) {
      const previous = values[weakest];
      values[weakest] = alternative.digit;
      if (!hasConflicts(values)) {
        confidence[weakest] = Math.min(alternative.confidence, 0.64);
        replaced = true;
        break;
      }
      values[weakest] = previous;
    }

    if (!replaced) {
      values[weakest] = 0;
      confidence[weakest] = 0;
    }
  }

  let analysis = analyzePuzzle(values);
  if (analysis.solutionCount === 0) {
    const suspects = predictions
      .filter((prediction) => values[prediction.index] !== 0)
      .sort((a, b) => confidence[a.index] - confidence[b.index]);

    for (const suspect of suspects) {
      values[suspect.index] = 0;
      confidence[suspect.index] = 0;
      analysis = analyzePuzzle(values);
      if (analysis.solutionCount > 0) break;
    }
  }

  return { values, confidence };
}
