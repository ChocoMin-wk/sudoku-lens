export type Digit = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

export type Cell = {
  given: Digit | null;
  value: Digit | null;
  notes: number;
  confidence?: number;
};

export type Board = Cell[];

export const SAMPLE_PUZZLE =
  "530070000600195000098000060800060003400803001700020006060000280000419005000080079";

export function boardFromDigits(values: readonly number[], asGivens = true): Board {
  return Array.from({ length: 81 }, (_, index) => {
    const raw = values[index] ?? 0;
    const digit = raw >= 1 && raw <= 9 ? (raw as Digit) : null;
    return {
      given: asGivens ? digit : null,
      value: asGivens ? null : digit,
      notes: 0,
    };
  });
}

export function boardFromString(puzzle: string): Board {
  return boardFromDigits([...puzzle].map((character) => Number(character)));
}

export function valuesOf(board: Board): number[] {
  return board.map((cell) => cell.given ?? cell.value ?? 0);
}

export function cloneBoard(board: Board): Board {
  return board.map((cell) => ({ ...cell }));
}

export function peerIndexes(index: number): number[] {
  const row = Math.floor(index / 9);
  const column = index % 9;
  const boxRow = Math.floor(row / 3) * 3;
  const boxColumn = Math.floor(column / 3) * 3;
  const peers = new Set<number>();

  for (let position = 0; position < 9; position += 1) {
    peers.add(row * 9 + position);
    peers.add(position * 9 + column);
  }

  for (let y = boxRow; y < boxRow + 3; y += 1) {
    for (let x = boxColumn; x < boxColumn + 3; x += 1) {
      peers.add(y * 9 + x);
    }
  }

  peers.delete(index);
  return [...peers];
}

export function conflictIndexes(board: Board): Set<number> {
  const values = valuesOf(board);
  const conflicts = new Set<number>();
  const groups: number[][] = [];

  for (let row = 0; row < 9; row += 1) {
    groups.push(Array.from({ length: 9 }, (_, column) => row * 9 + column));
  }
  for (let column = 0; column < 9; column += 1) {
    groups.push(Array.from({ length: 9 }, (_, row) => row * 9 + column));
  }
  for (let boxRow = 0; boxRow < 3; boxRow += 1) {
    for (let boxColumn = 0; boxColumn < 3; boxColumn += 1) {
      const group: number[] = [];
      for (let y = 0; y < 3; y += 1) {
        for (let x = 0; x < 3; x += 1) {
          group.push((boxRow * 3 + y) * 9 + boxColumn * 3 + x);
        }
      }
      groups.push(group);
    }
  }

  for (const group of groups) {
    const seen = new Map<number, number[]>();
    for (const index of group) {
      const value = values[index];
      if (!value) continue;
      const positions = seen.get(value) ?? [];
      positions.push(index);
      seen.set(value, positions);
    }
    for (const positions of seen.values()) {
      if (positions.length > 1) positions.forEach((index) => conflicts.add(index));
    }
  }

  return conflicts;
}

export function isComplete(board: Board): boolean {
  return valuesOf(board).every((value) => value !== 0);
}

export function noteIsSet(notes: number, digit: Digit): boolean {
  return (notes & (1 << digit)) !== 0;
}

export function toggleNote(notes: number, digit: Digit): number {
  return notes ^ (1 << digit);
}

function allowed(values: number[], index: number, digit: number): boolean {
  const row = Math.floor(index / 9);
  const column = index % 9;
  const boxRow = Math.floor(row / 3) * 3;
  const boxColumn = Math.floor(column / 3) * 3;

  for (let position = 0; position < 9; position += 1) {
    if (values[row * 9 + position] === digit) return false;
    if (values[position * 9 + column] === digit) return false;
  }
  for (let y = boxRow; y < boxRow + 3; y += 1) {
    for (let x = boxColumn; x < boxColumn + 3; x += 1) {
      if (values[y * 9 + x] === digit) return false;
    }
  }
  return true;
}

function candidates(values: number[], index: number): number[] {
  const result: number[] = [];
  for (let digit = 1; digit <= 9; digit += 1) {
    if (allowed(values, index, digit)) result.push(digit);
  }
  return result;
}

export type PuzzleAnalysis = {
  solutionCount: 0 | 1 | 2;
  solution: number[] | null;
};

export function analyzePuzzle(input: readonly number[]): PuzzleAnalysis {
  const values = Array.from({ length: 81 }, (_, index) => input[index] ?? 0);
  const initialBoard = boardFromDigits(values, true);
  if (conflictIndexes(initialBoard).size > 0) {
    return { solutionCount: 0, solution: null };
  }

  let solutionCount = 0;
  let firstSolution: number[] | null = null;

  const search = () => {
    if (solutionCount >= 2) return;
    let target = -1;
    let targetCandidates: number[] = [];

    for (let index = 0; index < 81; index += 1) {
      if (values[index] !== 0) continue;
      const options = candidates(values, index);
      if (options.length === 0) return;
      if (target === -1 || options.length < targetCandidates.length) {
        target = index;
        targetCandidates = options;
        if (options.length === 1) break;
      }
    }

    if (target === -1) {
      solutionCount += 1;
      if (!firstSolution) firstSolution = [...values];
      return;
    }

    for (const digit of targetCandidates) {
      values[target] = digit;
      search();
      values[target] = 0;
      if (solutionCount >= 2) return;
    }
  };

  search();
  return {
    solutionCount: Math.min(solutionCount, 2) as 0 | 1 | 2,
    solution: firstSolution,
  };
}

