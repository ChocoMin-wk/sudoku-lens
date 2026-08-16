import { describe, expect, it } from "vitest";
import {
  SAMPLE_PUZZLE,
  analyzePuzzle,
  boardFromString,
  conflictIndexes,
  noteIsSet,
  peerIndexes,
  toggleNote,
} from "./sudoku";

describe("sudoku domain", () => {
  it("detects a unique solution for the sample", () => {
    const analysis = analyzePuzzle([...SAMPLE_PUZZLE].map(Number));
    expect(analysis.solutionCount).toBe(1);
    expect(analysis.solution).toHaveLength(81);
    expect(analysis.solution?.slice(0, 9).join("")).toBe("534678912");
  });

  it("marks every duplicate in a row", () => {
    const board = boardFromString(`550000000${"0".repeat(72)}`);
    expect([...conflictIndexes(board)].sort()).toEqual([0, 1]);
  });

  it("returns the twenty peers of a cell", () => {
    expect(new Set(peerIndexes(40)).size).toBe(20);
  });

  it("toggles candidate notes", () => {
    const notes = toggleNote(0, 7);
    expect(noteIsSet(notes, 7)).toBe(true);
    expect(noteIsSet(toggleNote(notes, 7), 7)).toBe(false);
  });
});
