# 動画タブ譜抽出ツール (Video Tab Extractor)

YouTubeなどの動画から、演奏に合わせて表示されるタブ譜（楽譜）部分を抽出し、1つのPDFファイルとして保存するツールです。

## 主な機能

- **動画ダウンロード:** `yt-dlp`がサポートする任意のURLから動画をダウンロードします。
- **`yt-dlp`自動更新:** 起動時に`yt-dlp.exe`を自動で最新版に更新します。
- **楽譜エリアの指定:**
    - **自動検出:** 楽譜の背景色をクリックするだけで、楽譜エリア（ROI）を自動で検出します。
    - **手動選択:** マウスのドラッグ＆ドロップで正確な範囲を指定することも可能です。
- **ユニークフレーム抽出:** 楽譜が切り替わった（変化した）瞬間のみを画像として賢く抽出します。
- **PDF生成:** 抽出した楽譜画像を、動画タイトル付きの印刷しやすいA4サイズのPDFとして作成します。

## 必要なもの

- Python 3.x
- `yt-dlp.exe`
- `ffmpeg.exe` (yt-dlpが最高の画質・音質でダウンロードするために推奨)
- `requirements.txt` に記載されたPythonライブラリ

## セットアップ

1.  **リポジトリをクローンまたはダウンロードします。**

2.  **Pythonライブラリをインストールします。**
    ```sh
    pip install -r requirements.txt
    ```

3.  **`yt-dlp.exe` を配置します。**
    [yt-dlpの公式リリースページ](https://github.com/yt-dlp/yt-dlp/releases/latest) から最新の `yt-dlp.exe` をダウンロードし、`tab_extractor_v2.py` と同じフォルダに置いてください。

4.  **(推奨) `ffmpeg.exe` を配置します。**
    [FFmpegの公式サイト](https://ffmpeg.org/download.html) からダウンロードし、`ffmpeg.exe` を同じフォルダに置いてください。

## 使い方

基本的なコマンドは以下の通りです。

```sh
python tab_extractor_v2.py [動画のURL]
```

### コマンドラインオプション

- `url` (必須): 処理したい動画のURL。
- `--auto_roi`: 楽譜エリアの自動検出モードを有効にします。
- `--color_tolerance <数値>`: 背景色を検出する際の色の許容範囲（デフォルト: 30）。
- `--skip_seconds <数値>`: フレームの変化をチェックする間隔（秒）（デフォルト: 1.5）。
- `--skip-update`: 起動時の`yt-dlp`自動更新をスキップします。

### 実行例

**手動で楽譜範囲を選択する場合:**
```sh
python tab_extractor_v2.py "https://www.youtube.com/watch?v=..."
```

**自動で楽譜範囲を検出し、色の許容範囲を40に設定する場合:**
```sh
python tab_extractor_v2.py "https://www.youtube.com/watch?v=..." --auto_roi --color_tolerance 40
```
