import os
import subprocess
import argparse
import sys
import cv2
import numpy as np
import shutil
import re
from PIL import Image, ImageDraw, ImageFont

def get_script_path():
    """実行中のスクリプトの絶対パスを取得します。"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def run_yt_dlp_update(yt_dlp_path):
    """yt-dlp.exe -U を実行して自己更新を試みます。"""
    print("--- yt-dlp の更新を確認しています... ---")
    if not os.path.exists(yt_dlp_path):
        print(f"エラー: {yt_dlp_path} が見つかりません。")
        print("yt-dlp.exeをダウンロードして、このプログラムと同じフォルダに置いてください。")
        return False
    
    try:
        command = [yt_dlp_path, "-U"]
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        output = result.stdout.strip()
        errors = result.stderr.strip()
        if output:
            print(output)
        if errors:
            print(errors)

        print("--- 更新チェック完了 ---")
        return True

    except subprocess.CalledProcessError as e:
        print("yt-dlpの更新中にエラーが発生しました。")
        print(e.stdout)
        print(e.stderr)
        print("--- 更新チェック完了 (エラー) ---")
        return False
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        print("--- 更新チェック完了 (エラー) ---")
        return False

class UnifiedSelector:
    """シークバーとマウス操作を統合したUIを管理するクラス。"""
    def __init__(self, video_path, window_title):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"動画ファイルを開けません: {video_path}")

        self.window_name = window_title
        self.original_frame = None
        self.display_frame = None
        self.scale = 1.0
        self.current_pos_msec = -1

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_msec = int((frame_count / fps) * 1000) if fps > 0 else 0

        self.roi_rect = []
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.picked_color = []

    def _update_frame_and_reset(self, trackbar_value_msec):
        if abs(trackbar_value_msec - self.current_pos_msec) < 100 and self.original_frame is not None:
            return
        self.current_pos_msec = trackbar_value_msec
        self.cap.set(cv2.CAP_PROP_POS_MSEC, trackbar_value_msec)
        ret, frame = self.cap.read()
        if ret:
            self.original_frame = frame
            self.roi_rect = []
            self.picked_color = []
            self.drawing = False
            self._resize_for_display()
            self._update_display()

    def _resize_for_display(self):
        if self.original_frame is None: return
        original_height, original_width = self.original_frame.shape[:2]
        max_display_width = 1280
        if original_width > max_display_width:
            self.scale = max_display_width / original_width
            display_width = max_display_width
            display_height = int(original_height * self.scale)
            self.display_frame = cv2.resize(self.original_frame, (display_width, display_height))
        else:
            self.display_frame = self.original_frame.copy()
            self.scale = 1.0

    def _update_display(self):
        if self.display_frame is None: return
        frame_to_show = self.display_frame.copy()
        if self.roi_rect:
            cv2.rectangle(frame_to_show, (self.roi_rect[0], self.roi_rect[1]), 
                          (self.roi_rect[0] + self.roi_rect[2], self.roi_rect[1] + self.roi_rect[3]), 
                          (0, 255, 0), 2)
        cv2.imshow(self.window_name, frame_to_show)

    def _roi_mouse_callback(self, event, x, y, flags, param):
        h, w = self.display_frame.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.roi_rect = []
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_frame = self.display_frame.copy()
                cv2.rectangle(temp_frame, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = max(0, self.ix), max(0, self.iy)
            x2, y2 = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            self.roi_rect = [min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)]
            self._update_display()

    def _color_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = int(x / self.scale), int(y / self.scale)
            self.picked_color = self.original_frame[orig_y, orig_x].tolist()
            print(f"背景色として {self.picked_color} を選択しました。Enterで確定してください。")

    def select_roi(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if self.duration_msec > 0:
            cv2.createTrackbar('Time (ms)', self.window_name, 0, self.duration_msec, self._update_frame_and_reset)
            cv2.setTrackbarPos('Time (ms)', self.window_name, 1500)
        cv2.setMouseCallback(self.window_name, self._roi_mouse_callback)
        self._update_frame_and_reset(1500)

        print("シークバーでフレームを選び、マウスでROIをドラッグ後、Enterで確定。")
        final_roi = None
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13:
                if self.roi_rect and self.roi_rect[2] > 0 and self.roi_rect[3] > 0:
                    final_roi = [int(c / self.scale) for c in self.roi_rect]
                    break
                else:
                    print("ROIが選択されていません。")
            elif key == 27:
                final_roi = None
                break
        
        if final_roi is None:
            cv2.destroyAllWindows()
            return None, None

        print("ROIを確定しました。次に、その範囲内で背景色をクリックしてください。")
        cv2.setMouseCallback(self.window_name, self._color_mouse_callback)
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13:
                if self.picked_color:
                    break
                else:
                    print("色が選択されていません。")
            elif key == 27:
                self.picked_color = []
                break

        cv2.destroyAllWindows()
        return final_roi, self.picked_color

    def pick_color(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if self.duration_msec > 0:
            cv2.createTrackbar('Time (ms)', self.window_name, 0, self.duration_msec, self._update_frame_and_reset)
            cv2.setTrackbarPos('Time (ms)', self.window_name, 1500)
        cv2.setMouseCallback(self.window_name, self._color_mouse_callback)
        self._update_frame_and_reset(1500)

        print("シークバーでフレームを選び、背景色をクリック後、Enterで確定。")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 13:
                if self.picked_color:
                    break
                else:
                    print("色が選択されていません。")
            elif key == 27:
                self.picked_color = []
                break
        cv2.destroyAllWindows()
        return self.picked_color, self.original_frame

    def __del__(self):
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()

def select_roi_manually(video_path):
    """統合UIを使い、手動でROIと背景色を選択させる。"""
    try:
        selector = UnifiedSelector(video_path, "Select ROI & Color")
        roi, color = selector.select_roi()
        return roi, color
    except (IOError, cv2.error) as e:
        print(f"UIの初期化中にエラー: {e}")
        return None, None

def find_roi(video_path, color_tolerance=30):
    """統合UIを使い、背景色クリックによる自動ROI検出を行う。"""
    try:
        selector = UnifiedSelector(video_path, "Find ROI: Click background, use seek bar, then press Enter")
        picked_color, frame = selector.pick_color()
    except (IOError, cv2.error) as e:
        print(f"UIの初期化中にエラー: {e}")
        return None, None, None

    if not picked_color or frame is None:
        print("色選択がキャンセルされました。")
        return None, None, None

    color = np.array(picked_color, dtype=np.uint8)
    lower_bound = np.clip(color.astype(int) - color_tolerance, 0, 255).astype(np.uint8)
    upper_bound = np.clip(color.astype(int) + color_tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    original_height, original_width = frame.shape[:2]
    kernel_size = int(min(original_height, original_height) * 0.05)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("ROI検出エラー: 指定された色の範囲で領域を見つけられませんでした。")
        return None, None, picked_color

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    final_roi = (x, y, w, h)
    print(f"ROIを確定しました: x={x}, y={y}, w={w}, h={h}")
    return final_roi, frame, picked_color

def confirm_roi(frame, roi):
    """検出されたROIをユーザーに表示し、確認を求める。"""
    if frame is None or roi is None: return False

    x, y, w, h = roi
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)

    original_height, original_width = frame_copy.shape[:2]
    max_display_width = 1280
    if original_width > max_display_width:
        scale = max_display_width / original_width
        display_width = max_display_width
        display_height = int(original_height * scale)
        display_frame = cv2.resize(frame_copy, (display_width, display_height))
    else:
        display_frame = frame_copy

    window_name = "Confirm Auto ROI: Press Enter to accept, Esc to cancel."
    cv2.imshow(window_name, display_frame)
    
    print("自動検出された範囲を確認してください。Enterで確定、Escでキャンセルします。")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    return key == 13

def download_video(video_url, yt_dlp_path, download_dir="video_downloads", cookies_browser=None):
    """yt-dlp.exeを使用して最高品質の動画をダウンロードします。"""
    script_dir = get_script_path()
    full_download_dir = os.path.join(script_dir, download_dir)
    preferred_encoding = 'cp932' if sys.platform == 'win32' else 'utf-8'

    if not os.path.exists(yt_dlp_path):
        print(f"エラー: yt-dlp.exe が見つかりません: {yt_dlp_path}")
        return None, None

    try:
        print("動画のタイトルを取得しています...")
        
        base_command = [yt_dlp_path]
        if cookies_browser:
            base_command.extend(["--cookies-from-browser", cookies_browser])
        
        if getattr(sys, 'frozen', False):
            # PyInstallerで固められた場合、ffmpegは一時ディレクトリに展開される
            ffmpeg_path = os.path.join(sys._MEIPASS, "ffmpeg.exe")
            if os.path.exists(ffmpeg_path):
                base_command.extend(["--ffmpeg-location", ffmpeg_path])

        title_command = base_command + ["--get-title", "--skip-download", video_url]
        title_result = subprocess.run(title_command, check=True, capture_output=True, text=True, encoding=preferred_encoding, errors='ignore')
        video_title = title_result.stdout.strip()
        
        sanitized_title = re.sub(r'[\\/*?"<>|]', '_', video_title)
        output_filename = f"{sanitized_title}.mp4"
        
        os.makedirs(full_download_dir, exist_ok=True)
        output_path = os.path.join(full_download_dir, output_filename)

        if os.path.exists(output_path):
            print(f"動画は既に存在します: {output_path}")
            print("ダウンロードをスキップします。")
            return output_path, video_title

        print(f"動画をダウンロードしています: {video_title}")
        download_format = "bestvideo+bestaudio/best"
        download_command = base_command + [
            "-f", download_format,
            "--merge-output-format", "mp4",
            "-o", output_path,
            video_url
        ]
        subprocess.run(download_command, check=True, encoding=preferred_encoding, errors='ignore')
        print("動画のダウンロードが完了しました。")
        print(f"保存先: {output_path}")
        return output_path, video_title

    except subprocess.CalledProcessError as e:
        print("yt-dlpの実行中にエラーが発生しました。")
        if e.stdout:
            print("--- yt-dlpからのメッセージ ---")
            print(e.stdout)
        if e.stderr:
            print("--- yt-dlpからのエラー ---")
            print(e.stderr)
        print(f"コマンド '{' '.join(e.cmd)}' は終了コード {e.returncode} で失敗しました。")
        return None, None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None, None

def extract_unique_frames(video_path, roi, skip_seconds=1.5, output_dir="output_frames", threshold=2.0, max_threshold=50.0):
    """ROI内の変化を監視し、ユニークなフレームを画像として保存する。"""
    print(f"ユニークなフレームを抽出しています... (約{skip_seconds}秒ごとにチェック)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けません: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_skip = int(fps * skip_seconds)
    if frames_to_skip < 1:
        frames_to_skip = 1

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    x, y, w, h = roi
    last_frame_roi_gray = None
    saved_count = 0
    
    frame_number = int(fps * 1.5)
    if frame_number >= total_frames:
        frame_number = 0

    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_roi = frame[y:y+h, x:x+w]
        current_frame_roi_gray = cv2.cvtColor(current_frame_roi, cv2.COLOR_BGR2GRAY)

        is_unique = False
        mean_diff_str = ""

        if last_frame_roi_gray is None:
            is_unique = True
            mean_diff_str = "N/A (最初のフレーム)"
        else:
            diff = cv2.absdiff(last_frame_roi_gray, current_frame_roi_gray)
            mean_diff = np.mean(diff)
            if threshold < mean_diff < max_threshold:
                is_unique = True
                mean_diff_str = f"{mean_diff:.2f}"
            elif mean_diff >= max_threshold:
                print(f"無視しました: 差分が大きすぎます (>{max_threshold}) (フレーム番号: {frame_number})")

        if is_unique:
            if np.std(current_frame_roi_gray) < 2.0:
                print(f"無視しました: 単色フレームです (フレーム番号: {frame_number})")
                if last_frame_roi_gray is None:
                    pass
                frame_number += frames_to_skip
                continue

            saved_count += 1
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(filename, current_frame_roi)
            print(f"ユニークなフレームを保存しました: {filename} (フレーム番号: {frame_number}, 差分: {mean_diff_str})")
            last_frame_roi_gray = current_frame_roi_gray
        
        frame_number += frames_to_skip

    cap.release()
    print(f"抽出完了。合計 {saved_count} 枚のユニークな画像を保存しました。")
    return output_dir

def review_and_select_frames(image_folder):
    """抽出されたフレームを、擬似スクロールバー付きのウィンドウで確認・選択させる。"""
    print("\n--- フレーム確認モード ---")
    print("マウスホイール: スクロール | クリック: 除外/採用トグル | Enter: 確定 | Esc: キャンセル")
    
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
    if not image_files:
        print("確認するフレームがありません。")
        return []

    rejected_folder = os.path.join(image_folder, "rejected")
    if os.path.exists(rejected_folder):
        for f_name in os.listdir(rejected_folder):
            shutil.move(os.path.join(rejected_folder, f_name), os.path.join(image_folder, f_name))
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])

    # --- Layout Calculation ---
    CANVAS_WIDTH = 1280
    VIEWPORT_HEIGHT = 900
    PADDING = 10

    image_dims = []
    for img_path in image_files:
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            
            new_w, new_h = w, h
            if w > CANVAS_WIDTH - (2 * PADDING):
                scale = (CANVAS_WIDTH - (2 * PADDING)) / w
                new_w = int(w * scale)
                new_h = int(h * scale)
            
            image_dims.append((new_w, new_h))
        except Exception as e:
            print(f"警告: 画像ファイルの読み込みに失敗しました {img_path}: {e}")

    total_canvas_height = sum(h for w, h in image_dims) + (len(image_dims) + 1) * PADDING
    full_canvas = np.full((total_canvas_height, CANVAS_WIDTH, 3), 240, dtype=np.uint8)

    # --- Draw full canvas ---
    y_offset = PADDING
    image_boundaries = []
    for i, img_path in enumerate(image_files):
        if i >= len(image_dims): break
        img_w, img_h = image_dims[i]
        x_offset = (CANVAS_WIDTH - img_w) // 2
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        resized_img = cv2.resize(img, (img_w, img_h))
        full_canvas[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = resized_img
        image_boundaries.append((x_offset, y_offset, x_offset + img_w, y_offset + img_h))
        y_offset += img_h + PADDING

    # --- UI State and Callbacks ---
    state = {
        'image_files': image_files,
        'discarded_flags': [False] * len(image_files),
        'full_canvas': full_canvas,
        'image_boundaries': image_boundaries,
        'scroll_y': 0,
        'scroll_max': max(0, total_canvas_height - VIEWPORT_HEIGHT)
    }

    window_name = "Review Frames (Scroll with Wheel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, VIEWPORT_HEIGHT)

    def redraw_viewport():
        y = state['scroll_y']
        viewport = state['full_canvas'][y:y+VIEWPORT_HEIGHT, 0:CANVAS_WIDTH]
        cv2.imshow(window_name, viewport)

    def on_scroll(y_pos):
        state['scroll_y'] = y_pos
        redraw_viewport()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            absolute_y = y + state['scroll_y']
            for i, (x1, y1, x2, y2) in enumerate(param['image_boundaries']):
                if y1 <= absolute_y < y2 and x1 <= x < x2:
                    param['discarded_flags'][i] = not param['discarded_flags'][i]
                    # Redraw the X on the full canvas
                    img_w, img_h = image_dims[i]
                    x_offset, y_offset = x1, y1
                    if param['discarded_flags'][i]:
                        pt1, pt2 = (x_offset, y_offset), (x_offset + img_w, y_offset + img_h)
                        cv2.line(param['full_canvas'], pt1, pt2, (0, 0, 255), 3)
                        pt1, pt2 = (x_offset + img_w, y_offset), (x_offset, y_offset + img_h)
                        cv2.line(param['full_canvas'], pt1, pt2, (0, 0, 255), 3)
                    else:
                        img = cv2.imread(param['image_files'][i])
                        if img is not None:
                            resized_img = cv2.resize(img, (img_w, img_h))
                            param['full_canvas'][y_offset:y_offset+img_h, x_offset:x_offset+img_w] = resized_img
                    redraw_viewport()
                    break
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            scroll_amount = 100
            new_scroll_y = state['scroll_y']
            if flags > 0:
                new_scroll_y -= scroll_amount
            else:
                new_scroll_y += scroll_amount
            
            new_scroll_y = max(0, min(state['scroll_max'], new_scroll_y))
            if new_scroll_y != state['scroll_y']:
                state['scroll_y'] = new_scroll_y
                cv2.setTrackbarPos('Scroll', window_name, new_scroll_y)
                # The trackbar will call on_scroll, which calls redraw_viewport

    if state['scroll_max'] > 0:
        cv2.createTrackbar('Scroll', window_name, 0, state['scroll_max'], on_scroll)
    
    cv2.setMouseCallback(window_name, on_mouse, state)
    redraw_viewport()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13: break
        if key == 27: 
            print("確認がキャンセルされました。すべてのフレームが採用されます。")
            cv2.destroyAllWindows()
            return image_files
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("確認がキャンセルされました。すべてのフレームが採用されます。")
            cv2.destroyAllWindows()
            return image_files

    cv2.destroyAllWindows()

    os.makedirs(rejected_folder, exist_ok=True)
    approved_files = []
    for i, file_path in enumerate(state['image_files']):
        if state['discarded_flags'][i]:
            shutil.move(file_path, os.path.join(rejected_folder, os.path.basename(file_path)))
            print(f"除外しました: {os.path.basename(file_path)}")
        else:
            approved_files.append(file_path)
            
    print(f"\n{len(approved_files)}枚のフレームをPDF化します。")
    return approved_files

def create_pdf_from_images(image_files, pdf_path, title="", background_color=None):
    """A4用紙にタイトルを描画し、複数の画像を縦に並べてPDFを作成する。"""
    print(f"楽譜形式のPDFを作成しています: {pdf_path}")

    if not image_files:
        print("エラー: PDFに変換する画像が見つかりません。")
        return

    A4_WIDTH_PX, A4_HEIGHT_PX = 2480, 3508
    MARGIN_PX = 150
    GAP_PX = 50
    printable_width = A4_WIDTH_PX - (2 * MARGIN_PX)

    page_color = 'white'
    if background_color:
        page_color = (background_color[2], background_color[1], background_color[0])

    pages = []
    current_page = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), page_color)
    y_offset = MARGIN_PX

    if title:
        draw = ImageDraw.Draw(current_page)
        font_size = 80
        font_name = "meiryo.ttc" if sys.platform == "win32" else "Hiragino Sans GB.ttc"
        
        try:
            font = ImageFont.truetype(font_name, size=font_size)
        except IOError:
            print(f"警告: フォント '{font_name}' が見つかりません。代替フォントで描画します。")
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except IOError:
                print("警告: 'arial.ttf' も見つかりません。デフォルトフォントを使用します。")
                font = ImageFont.load_default()

        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(title, font=font)

        x_pos = (A4_WIDTH_PX - text_width) / 2
        draw.text((x_pos, y_offset), title, fill="black", font=font)
        y_offset += text_height + GAP_PX * 2

    for img_file in image_files:
        img = Image.open(img_file)
        
        aspect_ratio = img.height / img.width
        new_height = int(printable_width * aspect_ratio)
        resized_img = img.resize((printable_width, new_height), Image.LANCZOS)

        if y_offset + resized_img.height > A4_HEIGHT_PX - MARGIN_PX:
            pages.append(current_page)
            current_page = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), page_color)
            y_offset = MARGIN_PX

        current_page.paste(resized_img, (MARGIN_PX, y_offset))
        y_offset += resized_img.height + GAP_PX

    pages.append(current_page)

    if len(pages) > 1:
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    else:
        pages[0].save(pdf_path)

    print("PDFの作成が完了しました。")

def clean_title(title):
    """動画タイトルから曲名らしき部分を抽出する。"""
    # Remove content in brackets (【】, [], ())
    cleaned = re.sub(r'【.*?】|\[.*?\]|\(.*?\)', '', title)
    
    # Take the part before the first hyphen
    if ' - ' in cleaned:
        cleaned = cleaned.split(' - ', 1)[0]

    # Remove common keywords
    keywords = [
        'TAB', 'Guitar', 'Fingerstyle', 'Acoustic', 'Solo', 'Cover', 
        'ソロギター', 'アコギ', 'フィンガースタイル', 'カバー', 'タブ譜', '弾いてみた'
    ]
    for key in keywords:
        cleaned = re.sub(key, '', cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()
    return cleaned if cleaned else title

def start_processing(source, final_title, skip_seconds=1.5, auto_roi=False, color_tolerance=30, max_threshold=50.0, skip_update=False, cookies_from_browser=None):
    """
    動画のダウンロードからPDF作成までのメインロジック。
    GUIや他のスクリプトから呼び出すことを想定。
    """
    script_dir = get_script_path()
    yt_dlp_path = os.path.join(script_dir, "yt-dlp.exe")
    
    downloaded_file = None
    video_title = ""

    is_url = source.lower().startswith('http')

    if is_url:
        if not skip_update:
            if not run_yt_dlp_update(yt_dlp_path):
                print("\n警告: yt-dlpの更新に失敗しました。現在のバージョンで処理を続行します。")
        
        downloaded_file, video_title = download_video(source, yt_dlp_path, cookies_browser=cookies_from_browser)
    else:
        if not os.path.exists(source):
            print(f"エラー: 指定されたファイルが見つかりません: {source}")
            return # sys.exit(1) から変更
        downloaded_file = source
        video_title = os.path.splitext(os.path.basename(downloaded_file))[0]
        print(f"ローカルファイルを処理します: {downloaded_file}")

    if not downloaded_file:
        print("\n動画の取得に失敗したため、処理を中断しました。")
        return

    # ROI 選択
    roi = None
    background_color = None
    if auto_roi:
        potential_roi, roi_frame, background_color = find_roi(downloaded_file, color_tolerance=color_tolerance)
        if potential_roi:
            if confirm_roi(roi_frame, potential_roi):
                roi = potential_roi
            else:
                print("自動検出ROIがキャンセルされました。")
        
        if not roi:
            print("\n自動検出に失敗したか、キャンセルされました。手動選択に切り替えます。")
            roi, background_color = select_roi_manually(downloaded_file)
    else:
        roi, background_color = select_roi_manually(downloaded_file)

    if not roi:
        print("ROIが決定されなかったため、処理を中断します。")
        return

    # フレーム抽出とPDF作成
    print(f"使用するROI: {roi}")
    frame_folder = extract_unique_frames(downloaded_file, roi, skip_seconds=skip_seconds, max_threshold=max_threshold)
    if frame_folder and os.listdir(frame_folder):
        approved_files = review_and_select_frames(frame_folder)
        if approved_files:
            sanitized_pdf_title = re.sub(r'[\\/:*?"<>|]', '_', final_title)
            pdf_output_path = os.path.join(get_script_path(), f"{sanitized_pdf_title}.pdf")
            create_pdf_from_images(approved_files, pdf_output_path, title=final_title, background_color=background_color)
        else:
            print("採用されたフレームがなかったため、PDFは作成されませんでした。")
    else:
        print("ユニークなフレームが見つからなかったため、PDFは作成されませんでした。")
    
    print("\n処理が完了しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="楽譜動画をダウンロードしてPDFに変換するツール v13 (リファクタリング版)")
    parser.add_argument("source", help="処理する動画のURLまたはローカルファイルパス")
    parser.add_argument("--skip_seconds", type=float, default=1.5, help="フレームをチェックする間隔（秒）。")
    parser.add_argument("--auto_roi", action='store_true', help="ROI（楽譜エリア）の自動検出を試みます。")
    parser.add_argument("--color_tolerance", type=int, default=30, help="背景色検出の色の許容範囲。")
    parser.add_argument("--max_threshold", type=float, default=50.0, help="差分がこれより大きいフレームを除外します（フェードアウト対策）。")
    parser.add_argument("--skip-update", action='store_true', help="起動時のyt-dlp自動更新をスキップします。")
    parser.add_argument("--cookies-from-browser", nargs='?', const='firefox', default=None, 
                        help="ブラウザのクッキーを使って認証します。例: --cookies-from-browser chrome (デフォルト: firefox)")

    args = parser.parse_args()

    # --- CLI実行時のタイトル決定ロジック ---
    final_title = ""
    # ローカルファイルの場合は、まずファイル名からタイトルを推測
    if not args.source.lower().startswith('http'):
        video_title = os.path.splitext(os.path.basename(args.source))[0]
        final_title = clean_title(video_title)
    else:
        # URLの場合は、ダウンロード前にタイトルを取得してみる
        try:
            print("動画タイトルを取得しています...")
            yt_dlp_path = os.path.join(get_script_path(), "yt-dlp.exe")
            title_command = [yt_dlp_path, "--get-title", "--skip-download", args.source]
            if args.cookies_from_browser:
                title_command.extend(["--cookies-from-browser", args.cookies_from_browser])
            
            title_result = subprocess.run(title_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            video_title = title_result.stdout.strip()
            final_title = clean_title(video_title)
        except Exception as e:
            print(f"タイトルの事前取得に失敗しました: {e}")
            final_title = "downloaded_video"

    # ユーザーに最終確認
    prompt = (
        f"\n提案されたPDFタイトル: '{final_title}'\n"
        f"このままでよければEnterキーを押してください。変更する場合は新しいタイトルを入力してください: ")
    try:
        user_input = input(prompt)
    except EOFError:
        user_input = ''
    
    if user_input.strip():
        final_title = user_input.strip()
    # --- タイトル決定ロジックここまで ---

    start_processing(
        source=args.source,
        final_title=final_title,
        skip_seconds=args.skip_seconds,
        auto_roi=args.auto_roi,
        color_tolerance=args.color_tolerance,
        max_threshold=args.max_threshold,
        skip_update=args.skip_update,
        cookies_from_browser=args.cookies_from_browser
    )
