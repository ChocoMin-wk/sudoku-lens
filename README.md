# 数独レンズ

写真から問題を取り込める、ブラウザだけで動く数独アプリです。

## 機能

- 通常入力と小さな候補数字のメモ
- 行・列・3×3ブロックの重複警告
- 一意解判定と完成時の検算
- カメラ撮影または画像ファイルからの問題取り込み
- OpenCV.jsによる盤面補正とTesseract.jsによる数字認識
- OCR結果の確認・手修正
- 端末内への自動保存、戻す・やり直す
- PWAとGitHub Pagesへの自動デプロイ

撮影画像と盤面データは外部サーバーへ送信しません。

## 開発

```bash
npm install
npm run dev
npm test
npm run build
```

## GitHub Pages

リポジトリの **Settings → Pages → Source** で **GitHub Actions** を選択してください。
`main`ブランチへのpushで、`.github/workflows/deploy-pages.yml`から公開されます。
ViteのベースパスはGitHub Actions上でリポジトリ名から自動設定されます。

