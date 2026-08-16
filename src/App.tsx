import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  SAMPLE_PUZZLE,
  analyzePuzzle,
  boardFromDigits,
  boardFromString,
  cloneBoard,
  conflictIndexes,
  isComplete,
  noteIsSet,
  peerIndexes,
  toggleNote,
  valuesOf,
  type Board,
  type Digit,
} from "./domain/sudoku";
import { recognizeSudoku, type RecognitionResult } from "./imaging/recognizer";
import "./styles.css";

const STORAGE_KEY = "sudoku-lens:game:v1";

type SavedGame = {
  board: Board;
  solution: number[] | null;
  startedAt: number;
};

function loadGame(): SavedGame {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as SavedGame | null;
    if (saved?.board?.length === 81) return saved;
  } catch {
    // 壊れた端末内データは例題で置き換える。
  }
  const board = boardFromString(SAMPLE_PUZZLE);
  return {
    board,
    solution: analyzePuzzle(valuesOf(board)).solution,
    startedAt: Date.now(),
  };
}

function formatTime(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function ScanDialog({
  onClose,
  onImport,
}: {
  onClose: () => void;
  onImport: (result: RecognitionResult) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [phase, setPhase] = useState<"source" | "recognizing" | "review">("source");
  const [cameraActive, setCameraActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [reviewValues, setReviewValues] = useState<number[]>(Array(81).fill(0));
  const [error, setError] = useState("");

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraActive(false);
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const startCamera = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch {
      setError("カメラを開始できませんでした。権限を確認するか、画像を選択してください。");
    }
  };

  const runRecognition = async (canvas: HTMLCanvasElement) => {
    stopCamera();
    setPhase("recognizing");
    setError("");
    try {
      const recognized = await recognizeSudoku(canvas, (nextProgress, message) => {
        setProgress(nextProgress);
        setProgressText(message);
      });
      setResult(recognized);
      setReviewValues(recognized.values);
      setPhase("review");
    } catch (recognitionError) {
      console.error(recognitionError);
      setError("画像を読み取れませんでした。盤面全体が正面から写る画像で、もう一度お試しください。");
      setPhase("source");
    }
  };

  const capture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    void runRecognition(canvas);
  };

  const loadFile = (file: File) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const scale = Math.min(1, 2200 / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);
      canvas.getContext("2d")?.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      void runRecognition(canvas);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      setError("画像ファイルを開けませんでした。");
    };
    image.src = url;
  };

  const confirmImport = () => {
    const analysis = analyzePuzzle(reviewValues);
    if (analysis.solutionCount === 0) {
      setError("この配置には解がありません。赤や黄色のマスを確認してください。");
      return;
    }
    if (analysis.solutionCount === 2) {
      setError("解が複数あります。読み漏らした数字がないか確認してください。");
      return;
    }
    onImport({
      values: reviewValues,
      confidence: result?.confidence ?? Array(81).fill(1),
      normalizedImage: result?.normalizedImage ?? "",
      ignoredAnnotations: result?.ignoredAnnotations ?? 0,
    });
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="scan-dialog" role="dialog" aria-modal="true" aria-labelledby="scan-title">
        <header className="dialog-header">
          <div>
            <p className="eyebrow">SCAN A PUZZLE</p>
            <h2 id="scan-title">写真から問題を取り込む</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="閉じる">×</button>
        </header>

        {phase === "source" && (
          <div className="scan-source">
            <div className="camera-frame">
              {cameraActive ? (
                <video ref={videoRef} muted playsInline aria-label="カメラ映像" />
              ) : (
                <div className="camera-placeholder">
                  <span className="camera-mark">▣</span>
                  <p>枠いっぱいに盤面を写します</p>
                  <small>影を避け、なるべく正面から撮影してください</small>
                </div>
              )}
              <div className="focus-frame" aria-hidden="true" />
            </div>
            {error && <p className="dialog-error" role="alert">{error}</p>}
            <div className="scan-actions">
              {cameraActive ? (
                <button className="primary-button" onClick={capture}>撮影して読み取る</button>
              ) : (
                <button className="primary-button" onClick={() => void startCamera()}>カメラを起動</button>
              )}
              <label className="secondary-button file-button">
                画像を選択
                <input
                  type="file"
                  accept="image/*"
                  capture="environment"
                  onChange={(event) => event.target.files?.[0] && loadFile(event.target.files[0])}
                />
              </label>
            </div>
          </div>
        )}

        {phase === "recognizing" && (
          <div className="recognition-progress" aria-live="polite">
            <div className="scan-animation" aria-hidden="true"><span /></div>
            <h3>{progressText || "画像を解析しています"}</h3>
            <div className="progress-track"><span style={{ width: `${Math.round(progress * 100)}%` }} /></div>
            <p>{Math.round(progress * 100)}% 画像は端末の外へ送信されません</p>
          </div>
        )}

        {phase === "review" && result && (
          <div className="review-layout">
            <div className="review-photo-wrap">
              <img src={result.normalizedImage} alt="補正した数独盤面" className="review-photo" />
              <p>自動補正した画像</p>
            </div>
            <div className="review-panel">
              <p className="review-help">
                認識結果を確認してください。信頼度の低いマスは黄色で表示しています。
                {result.ignoredAnnotations > 0 && ` 数字以外の小さな注記を${result.ignoredAnnotations}件除外しました。`}
              </p>
              <div className="review-grid" aria-label="認識結果の81マス">
                {reviewValues.map((value, index) => (
                  <input
                    key={index}
                    aria-label={`${Math.floor(index / 9) + 1}行${(index % 9) + 1}列`}
                    className={(result.confidence[index] ?? 0) < 0.72 && value ? "low-confidence" : ""}
                    inputMode="numeric"
                    maxLength={1}
                    value={value || ""}
                    onChange={(event) => {
                      const nextValue = Number(event.target.value.replace(/[^1-9]/g, "").slice(-1)) || 0;
                      setReviewValues((current) => current.map((item, cellIndex) => cellIndex === index ? nextValue : item));
                      setError("");
                    }}
                  />
                ))}
              </div>
              {error && <p className="dialog-error" role="alert">{error}</p>}
              <div className="scan-actions">
                <button className="secondary-button" onClick={() => setPhase("source")}>撮り直す</button>
                <button className="primary-button" onClick={confirmImport}>この問題で始める</button>
              </div>
            </div>
          </div>
        )}
        <canvas ref={canvasRef} hidden />
      </section>
    </div>
  );
}

export default function App() {
  const initial = useMemo(() => loadGame(), []);
  const [board, setBoard] = useState<Board>(initial.board);
  const [solution, setSolution] = useState<number[] | null>(initial.solution);
  const [startedAt, setStartedAt] = useState(initial.startedAt);
  const [selected, setSelected] = useState<number | null>(() => {
    const firstEmpty = initial.board.findIndex((cell) => !cell.given);
    return firstEmpty >= 0 ? firstEmpty : null;
  });
  const [noteMode, setNoteMode] = useState(false);
  const [undoStack, setUndoStack] = useState<Board[]>([]);
  const [redoStack, setRedoStack] = useState<Board[]>([]);
  const [message, setMessage] = useState("マスを選び、数字を入力してください");
  const [wrongIndexes, setWrongIndexes] = useState<Set<number>>(new Set());
  const [showScanner, setShowScanner] = useState(false);
  const [elapsed, setElapsed] = useState(() => Math.floor((Date.now() - initial.startedAt) / 1000));

  const conflicts = useMemo(() => conflictIndexes(board), [board]);
  const filled = useMemo(() => valuesOf(board).filter(Boolean).length, [board]);

  useEffect(() => {
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [startedAt]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ board, solution, startedAt } satisfies SavedGame));
  }, [board, solution, startedAt]);

  useEffect(() => {
    if ("serviceWorker" in navigator && import.meta.env.PROD) {
      void navigator.serviceWorker.register(`${import.meta.env.BASE_URL}service-worker.js`);
    }
  }, []);

  const commit = useCallback((nextBoard: Board) => {
    setUndoStack((stack) => [...stack.slice(-49), cloneBoard(board)]);
    setRedoStack([]);
    setBoard(nextBoard);
    setWrongIndexes(new Set());
  }, [board]);

  const enterDigit = useCallback((digit: Digit) => {
    if (selected === null || board[selected].given) return;
    const next = cloneBoard(board);
    const cell = next[selected];
    if (noteMode) {
      cell.notes = toggleNote(cell.notes, digit);
    } else {
      cell.value = cell.value === digit ? null : digit;
    }
    commit(next);
    setMessage("入力内容は自動保存されています");
  }, [board, commit, noteMode, selected]);

  const eraseValue = useCallback(() => {
    if (selected === null || board[selected].given) return;
    const next = cloneBoard(board);
    next[selected].value = null;
    commit(next);
  }, [board, commit, selected]);

  const clearNotes = useCallback(() => {
    if (selected === null || board[selected].given || board[selected].notes === 0) return;
    const next = cloneBoard(board);
    next[selected].notes = 0;
    commit(next);
    setMessage("選択中のマスのメモを消しました");
  }, [board, commit, selected]);

  const moveSelection = useCallback((rowOffset: number, columnOffset: number) => {
    const current = selected ?? 0;
    const row = Math.max(0, Math.min(8, Math.floor(current / 9) + rowOffset));
    const column = Math.max(0, Math.min(8, (current % 9) + columnOffset));
    setSelected(row * 9 + column);
  }, [selected]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (showScanner) return;
      if (/^[1-9]$/.test(event.key)) enterDigit(Number(event.key) as Digit);
      else if (event.key === "Backspace" || event.key === "Delete" || event.key === "0") eraseValue();
      else if (event.key.toLowerCase() === "n") setNoteMode((mode) => !mode);
      else if (event.key === "ArrowUp") moveSelection(-1, 0);
      else if (event.key === "ArrowDown") moveSelection(1, 0);
      else if (event.key === "ArrowLeft") moveSelection(0, -1);
      else if (event.key === "ArrowRight") moveSelection(0, 1);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [enterDigit, eraseValue, moveSelection, showScanner]);

  const resetGame = (nextBoard: Board) => {
    const analysis = analyzePuzzle(valuesOf(nextBoard));
    setBoard(nextBoard);
    setSolution(analysis.solutionCount === 1 ? analysis.solution : null);
    setStartedAt(Date.now());
    setElapsed(0);
    setUndoStack([]);
    setRedoStack([]);
    setWrongIndexes(new Set());
    const firstEmpty = nextBoard.findIndex((cell) => !cell.given);
    setSelected(firstEmpty >= 0 ? firstEmpty : 0);
    setMessage(analysis.solutionCount === 1 ? "新しい問題を始めました" : "自由入力盤を開きました");
  };

  const undo = () => {
    const previous = undoStack.at(-1);
    if (!previous) return;
    setRedoStack((stack) => [...stack, cloneBoard(board)]);
    setBoard(previous);
    setUndoStack((stack) => stack.slice(0, -1));
    setWrongIndexes(new Set());
  };

  const redo = () => {
    const next = redoStack.at(-1);
    if (!next) return;
    setUndoStack((stack) => [...stack, cloneBoard(board)]);
    setBoard(next);
    setRedoStack((stack) => stack.slice(0, -1));
    setWrongIndexes(new Set());
  };

  const verify = () => {
    if (conflicts.size > 0) {
      setMessage(`重複しているマスが${conflicts.size}個あります`);
      return;
    }
    if (!isComplete(board)) {
      setMessage(`あと${81 - filled}マスです。空欄を埋めてから検算できます`);
      return;
    }
    if (solution) {
      const wrong = new Set<number>();
      valuesOf(board).forEach((value, index) => {
        if (value !== solution[index]) wrong.add(index);
      });
      setWrongIndexes(wrong);
      setMessage(wrong.size ? `${wrong.size}マスを見直してください` : `完成です！ ${formatTime(elapsed)}で解けました`);
      return;
    }
    setMessage("完成です！ 行・列・ブロックのすべてが正しい配置です");
  };

  const selectedValue = selected === null ? 0 : (board[selected].given ?? board[selected].value ?? 0);
  const selectedPeers = useMemo(() => new Set(selected === null ? [] : peerIndexes(selected)), [selected]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="./" aria-label="数独レンズ ホーム">
          <span className="brand-grid" aria-hidden="true">#</span>
          <span><strong>数独レンズ</strong><small>SUDOKU LENS</small></span>
        </a>
        <div className="header-actions">
          <button className="text-button" onClick={() => resetGame(boardFromString(SAMPLE_PUZZLE))}>例題</button>
          <button className="text-button" onClick={() => resetGame(boardFromDigits(Array(81).fill(0)))}>空盤面</button>
          <button className="scan-button" onClick={() => setShowScanner(true)}><span>▣</span> 写真から読み取る</button>
        </div>
      </header>

      <div className="workspace">
        <section className="game-column" aria-labelledby="game-heading">
          <div className="game-heading-row">
            <div>
              <p className="eyebrow">PLAYING BOARD</p>
              <h1 id="game-heading">今日の一問</h1>
            </div>
            <div className="game-stats" aria-label="進行状況">
              <span><small>TIME</small>{formatTime(elapsed)}</span>
              <span><small>FILLED</small>{filled}<b>/ 81</b></span>
            </div>
          </div>

          <div className="board-frame">
            <div className="sudoku-board" role="grid" aria-label="数独盤面">
              {board.map((cell, index) => {
                const value = cell.given ?? cell.value;
                const isSelected = selected === index;
                const isRelated = selectedPeers.has(index);
                const isSame = Boolean(value && selectedValue === value);
                const classes = [
                  "sudoku-cell",
                  cell.given ? "given" : "editable",
                  value ? "has-value" : "",
                  cell.notes ? "has-notes" : "",
                  isSelected ? "selected" : "",
                  isRelated ? "related" : "",
                  isSame ? "same-value" : "",
                  conflicts.has(index) ? "conflict" : "",
                  wrongIndexes.has(index) ? "wrong" : "",
                ].filter(Boolean).join(" ");
                return (
                  <button
                    key={index}
                    className={classes}
                    role="gridcell"
                    aria-selected={isSelected}
                    aria-label={`${Math.floor(index / 9) + 1}行${(index % 9) + 1}列${value ? `、${value}` : "、空欄"}`}
                    onClick={() => setSelected(index)}
                  >
                    {cell.notes !== 0 && (
                      <span className="notes-grid" aria-hidden="true">
                        {Array.from({ length: 9 }, (_, noteIndex) => {
                          const digit = (noteIndex + 1) as Digit;
                          return <i key={digit}>{noteIsSet(cell.notes, digit) ? digit : ""}</i>;
                        })}
                      </span>
                    )}
                    {value && <span className="cell-value">{value}</span>}
                    {conflicts.has(index) && <span className="conflict-dot" aria-hidden="true">!</span>}
                  </button>
                );
              })}
            </div>
          </div>

        </section>

        <aside className="controls-panel" aria-label="入力パネル">
          <div className="mode-switch" role="group" aria-label="入力モード">
            <button className={!noteMode ? "active" : ""} onClick={() => setNoteMode(false)}>数字</button>
            <button className={noteMode ? "active" : ""} onClick={() => setNoteMode(true)}>メモ <kbd>N</kbd></button>
          </div>

          <div className="number-pad" aria-label="数字入力">
            {Array.from({ length: 9 }, (_, index) => (index + 1) as Digit).map((digit) => (
              <button key={digit} onClick={() => enterDigit(digit)}>{digit}</button>
            ))}
          </div>

          <div className="clear-actions">
            <button className="erase-button" onClick={eraseValue}><span>⌫</span> 数字クリア</button>
            <button className="erase-button" onClick={clearNotes}><span>×</span> メモクリア</button>
          </div>

          <div className="history-actions">
            <button onClick={undo} disabled={!undoStack.length}>↶<small>戻す</small></button>
            <button onClick={redo} disabled={!redoStack.length}>↷<small>やり直す</small></button>
          </div>

          <button className="verify-button" onClick={verify}><span>✓</span> 検算する</button>

          <div className="legend">
            <p><span className="legend-given" />問題の数字</p>
            <p><span className="legend-entry" />入力した数字</p>
            <p><span className="legend-warning" />重複・要確認</p>
          </div>
        </aside>

        <div className={`status-message ${conflicts.size ? "has-warning" : ""}`} aria-live="polite">
          <span aria-hidden="true">{conflicts.size ? "!" : "✓"}</span>
          <p>{conflicts.size ? `${conflicts.size}マスで同じ数字が重複しています` : message}</p>
        </div>
      </div>

      <footer>
        <p>盤面と写真はこの端末内だけで処理されます。</p>
        <p>自動保存済み</p>
      </footer>

      {showScanner && (
        <ScanDialog
          onClose={() => setShowScanner(false)}
          onImport={(recognized) => {
            const imported = boardFromDigits(recognized.values).map((cell, index) => ({
              ...cell,
              confidence: recognized.confidence[index],
            }));
            resetGame(imported);
            setMessage("写真から取り込んだ問題を始めました");
            setShowScanner(false);
          }}
        />
      )}
    </main>
  );
}
