import { describe, expect, it } from "vitest";
import { SAMPLE_PUZZLE } from "../domain/sudoku";
import { reconcileWithSudoku, type CellPrediction } from "./ocrPostprocess";

const SHADOWED_PHOTO_PUZZLE = [
  "670030000",
  "002100500",
  "008000670",
  "010080037",
  "006000000",
  "000040020",
  "000609002",
  "000003080",
  "000500000",
].join("");

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

  it("removes a conflicting clue without inventing a fixed digit from the solution", () => {
    const predictions = predictionsFromPuzzle(SAMPLE_PUZZLE);
    const changed = predictions.find((prediction) => prediction.index === 1)!;
    changed.candidates = [{ digit: 5, confidence: 0.25 }];

    const result = reconcileWithSudoku(predictions);
    expect(result.values[1]).toBe(0);
    expect(result.confidence[1]).toBe(0);
  });

  it("keeps the shadowed photographed puzzle while excluding A/B annotations", () => {
    const result = reconcileWithSudoku(predictionsFromPuzzle(SHADOWED_PHOTO_PUZZLE));
    expect(result.values.join("")).toBe(SHADOWED_PHOTO_PUZZLE);
  });
});
