import { describe, expect, it } from "vitest";
import { boardFromString, toggleNote } from "../domain/sudoku";
import { decodeCookieGame, encodeCookieGame, type SavedGame } from "./gameStorage";

describe("Cookie game persistence", () => {
  it("round-trips the current puzzle in a cookie-sized payload", () => {
    const board = boardFromString(
      "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    );
    board[2].value = 4;
    board[3].notes = toggleNote(toggleNote(0, 1), 9);
    board[3].confidence = 0.47;
    const game: SavedGame = {
      board,
      solution: Array.from({ length: 81 }, (_, index) => (index % 9) + 1),
      startedAt: 1_700_000_000_000,
      selected: 3,
      noteMode: true,
      undoStack: [board],
      redoStack: [],
      message: "入力内容は自動保存されています",
      wrongIndexes: [2, 8],
      updatedAt: 1_700_000_001_234,
    };

    const encoded = encodeCookieGame(game);
    const restored = decodeCookieGame(encoded);

    expect(encoded.length).toBeLessThan(3500);
    expect(restored?.board).toEqual(board);
    expect(restored?.solution).toEqual(game.solution);
    expect(restored?.selected).toBe(3);
    expect(restored?.noteMode).toBe(true);
    expect(restored?.wrongIndexes).toEqual([2, 8]);
    expect(restored?.undoStack).toEqual([]);
  });
});
