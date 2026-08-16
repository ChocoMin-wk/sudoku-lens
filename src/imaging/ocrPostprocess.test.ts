import { describe, expect, it } from "vitest";
import { SAMPLE_PUZZLE } from "../domain/sudoku";
import { reconcileWithSudoku, type CellPrediction } from "./ocrPostprocess";

function predictionsFromPuzzle(puzzle: string): CellPrediction[] {
  return [...puzzle]
    .map(Number)
    .map((digit, index) => ({ index, digit }))
    .filter(({ digit }) => digit !== 0)
    .map(({ digit, index }) => ({ index, candidates: [{ digit, confidence: 0.95 }] }));
}

describe("OCR Sudoku reconciliation", () => {
  it("keeps a correct recognized puzzle unchanged", () => {
    const result = reconcileWithSudoku(predictionsFromPuzzle(SAMPLE_PUZZLE));
    expect(result.values.join("")).toBe(SAMPLE_PUZZLE);
  });

  it("repairs a low-confidence conflicting clue from the unique solution", () => {
    const predictions = predictionsFromPuzzle(SAMPLE_PUZZLE);
    const changed = predictions.find((prediction) => prediction.index === 1)!;
    changed.candidates = [{ digit: 5, confidence: 0.25 }];

    const result = reconcileWithSudoku(predictions);
    expect(result.values[1]).toBe(3);
    expect(result.confidence[1]).toBeLessThan(0.5);
  });
});

