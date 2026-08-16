import type { Board, Digit } from "../domain/sudoku";

export const STORAGE_KEY = "sudoku-lens:game:v2";
export const LEGACY_STORAGE_KEY = "sudoku-lens:game:v1";
export const COOKIE_KEY = "sudoku_lens_game_v2";

export type SavedGame = {
  board: Board;
  solution: number[] | null;
  startedAt: number;
  selected: number | null;
  noteMode: boolean;
  undoStack: Board[];
  redoStack: Board[];
  message: string;
  wrongIndexes: number[];
  updatedAt: number;
};

type CompactGame = {
  g: string;
  v: string;
  n: string;
  c: string;
  s: string;
  t: string;
  p: string;
  o: 0 | 1;
  m: string;
  w: string;
  u: string;
};

const digitString = (values: Array<number | null>) =>
  values.map((value) => value ?? 0).join("");

function encodeBoard(board: Board): Pick<CompactGame, "g" | "v" | "n" | "c"> {
  return {
    g: digitString(board.map((cell) => cell.given)),
    v: digitString(board.map((cell) => cell.value)),
    n: board.map((cell) => cell.notes.toString(36).padStart(2, "0")).join(""),
    c: board
      .map((cell) => Math.round((cell.confidence ?? 0) * 100).toString(36).padStart(2, "0"))
      .join(""),
  };
}

function validDigit(character: string): Digit | null {
  const digit = Number(character);
  return digit >= 1 && digit <= 9 ? (digit as Digit) : null;
}

function decodeBoard(compact: Pick<CompactGame, "g" | "v" | "n" | "c">): Board | null {
  if (compact.g.length !== 81 || compact.v.length !== 81 || compact.n.length !== 162) {
    return null;
  }
  return Array.from({ length: 81 }, (_, index) => {
    const confidenceRaw = compact.c.slice(index * 2, index * 2 + 2);
    const confidence = Number.parseInt(confidenceRaw, 36) / 100;
    return {
      given: validDigit(compact.g[index]),
      value: validDigit(compact.v[index]),
      notes: Number.parseInt(compact.n.slice(index * 2, index * 2 + 2), 36) || 0,
      ...(Number.isFinite(confidence) && confidence > 0 ? { confidence } : {}),
    };
  });
}

export function encodeCookieGame(game: SavedGame) {
  const compact: CompactGame = {
    ...encodeBoard(game.board),
    s: game.solution ? digitString(game.solution) : "",
    t: game.startedAt.toString(36),
    p: game.selected === null ? "" : game.selected.toString(36),
    o: game.noteMode ? 1 : 0,
    m: game.message.slice(0, 80),
    w: game.wrongIndexes.map((index) => index.toString(36)).join("."),
    u: game.updatedAt.toString(36),
  };
  return encodeURIComponent(JSON.stringify(compact));
}

export function decodeCookieGame(encoded: string): SavedGame | null {
  try {
    const compact = JSON.parse(decodeURIComponent(encoded)) as CompactGame;
    const board = decodeBoard(compact);
    if (!board) return null;
    const solution = compact.s.length === 81 ? [...compact.s].map(Number) : null;
    const selected = compact.p ? Number.parseInt(compact.p, 36) : null;
    return {
      board,
      solution,
      startedAt: Number.parseInt(compact.t, 36),
      selected: selected !== null && selected >= 0 && selected < 81 ? selected : null,
      noteMode: compact.o === 1,
      undoStack: [],
      redoStack: [],
      message: compact.m || "前回の続きから再開しました",
      wrongIndexes: compact.w
        ? compact.w.split(".").map((value) => Number.parseInt(value, 36)).filter((index) => index < 81)
        : [],
      updatedAt: Number.parseInt(compact.u, 36) || 0,
    };
  } catch {
    return null;
  }
}

function normalizeSavedGame(candidate: Partial<SavedGame> | null): SavedGame | null {
  if (!candidate?.board || candidate.board.length !== 81) return null;
  const firstEmpty = candidate.board.findIndex((cell) => !cell.given && !cell.value);
  return {
    board: candidate.board,
    solution: candidate.solution?.length === 81 ? candidate.solution : null,
    startedAt: candidate.startedAt ?? Date.now(),
    selected: candidate.selected ?? (firstEmpty >= 0 ? firstEmpty : null),
    noteMode: candidate.noteMode ?? false,
    undoStack: candidate.undoStack ?? [],
    redoStack: candidate.redoStack ?? [],
    message: candidate.message ?? "前回の続きから再開しました",
    wrongIndexes: candidate.wrongIndexes ?? [],
    updatedAt: candidate.updatedAt ?? 0,
  };
}

function readCookieValue(cookieText: string) {
  const prefix = `${COOKIE_KEY}=`;
  return cookieText
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix))
    ?.slice(prefix.length);
}

export function loadSavedGame(): SavedGame | null {
  let local: SavedGame | null = null;
  try {
    const current = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as SavedGame | null;
    const legacy = JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY) ?? "null") as SavedGame | null;
    local = normalizeSavedGame(current ?? legacy);
  } catch {
    local = null;
  }

  const cookieValue = readCookieValue(document.cookie);
  const cookie = cookieValue ? decodeCookieGame(cookieValue) : null;
  if (!local) return cookie;
  if (!cookie) return local;
  return local.updatedAt >= cookie.updatedAt ? local : cookie;
}

export function saveGame(game: SavedGame) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(game));
  } catch {
    // Cookie保存は引き続き試す。
  }

  try {
    const secure = location.protocol === "https:" ? "; Secure" : "";
    document.cookie = `${COOKIE_KEY}=${encodeCookieGame(game)}; Max-Age=31536000; Path=/; SameSite=Lax${secure}`;
  } catch {
    // Cookieが無効な環境でもアプリ操作は継続する。
  }
}
