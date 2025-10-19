
import os
import subprocess
import argparse
import sys
import cv2
import numpy as np
import shutil
import re
from PIL import Image

# --- Global variables for mouse callback ---
drawing = False
ix, iy = -1, -1
roi_rect = []
picked_color = []

def get_script_path():
    """実行中のスクリプトの絶対パスを取得します。"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def draw_rectangle(event, x, y, flags, param):
    """Mouse callback function to draw rectangle."""
    global ix, iy, drawing, roi_rect
    frame = param['frame'].copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        roi_rect = [min(ix, x), min(iy, y), abs(ix - x), abs(iy - y)]
    cv2.imshow(param['window_name'], frame)

def pick_color_callback(event, x, y, flags, param):
    """背景色を選択するためのマウスコールバック"""
    global picked_color
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param.get('scale', 1.0)
        original_frame = param['original_frame']
        # 縮小表示された座標を元の座標に変換
        orig_x, orig_y = int(x / scale), int(y / scale)
        picked_color = original_frame[orig_y, orig_x].tolist()
        print(f"背景色として {picked_color} を選択しました。")

def select_roi_manually(video_path):
    """動画フレームをリサイズして表示し、手動でROIを選択させる。"""
    global roi_rect
    roi_rect = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けません: {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print("エラー: 動画からフレームを読み込めません。")
        cap.release()
        return None

    original_height, original_width = frame.shape[:2]
    max_display_width = 1280
    scale = 1.0

    if original_width > max_display_width:
        scale = max_display_width / original_width
        display_width = max_display_width
        display_height = int(original_height * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))
    else:
        display_frame = frame

    window_name = "Select Score Area: Drag mouse, then press Enter."
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, draw_rectangle, {'frame': display_frame, 'window_name': window_name})

    print("楽譜の範囲をマウスでドラッグし、Enterキーを押して確定してください。")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter key
            if roi_rect and roi_rect[2] > 0 and roi_rect[3] > 0:
                original_roi = [int(c / scale) for c in roi_rect]
                roi_rect = original_roi
                break
            else:
                print("範囲が選択されていません。ドラッグして範囲を選択してください。")
        elif key == 27:  # Escape key
            print("選択がキャンセルされました。")
            roi_rect = []
            break

    cap.release()
    cv2.destroyAllWindows()
    return roi_rect

def find_roi(video_path, color_tolerance=30):
    """
    v7: User-assisted color detection with UI fixes.
    Returns: (roi, frame) or (None, None)
    """
    global picked_color
    picked_color = []
    print(f"楽譜エリア(ROI)を自動検出しています (v7: User-Assisted Color Detection, Tolerance: {color_tolerance})...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けません。")
        return None, None

    ret, frame = cap.read()
    if not ret:
        cap.release()
        print("エラー: フレームを読み込めません。")
        return None, None

    # ウィンドウ縮小表示ロジック
    original_height, original_width = frame.shape[:2]
    max_display_width = 1280
    scale = 1.0
    if original_width > max_display_width:
        scale = max_display_width / original_width
        display_width = max_display_width
        display_height = int(original_height * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))
    else:
        display_frame = frame

    window_name = "Auto ROI: Click on the sheet music background to pick a color."
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, pick_color_callback, {'frame': display_frame, 'original_frame': frame, 'scale': scale})
    print("ウィンドウ上で、楽譜の背景（紙の色）をクリックしてください。Escキーでキャンセル。")

    while not picked_color:
        if cv2.waitKey(1) & 0xFF == 27:
            print("色選択がキャンセルされました。")
            cv2.destroyAllWindows()
            cap.release()
            return None, None
    
    cv2.destroyAllWindows()

    # --- マスク作成と輪郭検出 ---
    color = np.array(picked_color, dtype=np.uint8)
    lower_bound = np.clip(color.astype(int) - color_tolerance, 0, 255).astype(np.uint8)
    upper_bound = np.clip(color.astype(int) + color_tolerance, 0, 255).astype(np.uint8)

    mask = cv2.inRange(frame, lower_bound, upper_bound)

    kernel_size = int(min(original_width, original_height) * 0.05) # フレームサイズに応じたカーネル
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        cap.release()
        print("ROI検出エラー: 指定された色の範囲で領域を見つけられませんでした。")
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    cap.release()
    final_roi = (x, y, w, h)
    print(f"ROIを確定しました: x={x}, y={y}, w={w}, h={h}")
    return final_roi, frame

def confirm_roi(frame, roi):
    """検出されたROIをユーザーに表示し、確認を求める。"""
    if frame is None or roi is None: return False

    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    original_height, original_width = frame.shape[:2]
    max_display_width = 1280
    if original_width > max_display_width:
        scale = max_display_width / original_width
        display_width = max_display_width
        display_height = int(original_height * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))
    else:
        display_frame = frame

    window_name = "Confirm Auto ROI: Press Enter to accept, Esc to cancel."
    cv2.imshow(window_name, display_frame)
    
    print("自動検出された範囲を確認してください。Enterで確定、Escでキャンセルします。")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    return key == 13 # Enterキーが押された場合のみTrue

def download_video(video_url, download_dir="video_downloads"):
    """
    yt-dlp.exeを使用して最高品質の動画をダウンロードします。
    """
    script_dir = get_script_path()
    yt_dlp_path = os.path.join(script_dir, "yt-dlp.exe")
    full_download_dir = os.path.join(script_dir, download_dir)
    preferred_encoding = 'cp932' if sys.platform == 'win32' else 'utf-8'

    if not os.path.exists(yt_dlp_path):
        print(f"エラー: yt-dlp.exe が見つかりません: {yt_dlp_path}")
        return None, None

    try:
        print("動画のタイトルを取得しています...")
        title_command = [yt_dlp_path, "--get-title", "--skip-download", video_url]
        title_result = subprocess.run(title_command, check=True, capture_output=True, text=True, encoding=preferred_encoding)
        video_title = title_result.stdout.strip()
        
        sanitized_title = re.sub(r'[\\/*?"<>|]', '_', video_title)
        output_filename = f"{sanitized_title}.mp4"
        
        os.makedirs(full_download_dir, exist_ok=True)
        output_path = os.path.join(full_download_dir, output_filename)

        if os.path.exists(output_path):
            print(f"動画は既に存在します: {output_path}")
            print("ダウンロードをスキップします。")
            return output_path, sanitized_title

        print(f"動画をダウンロードしています: {video_title}")
        download_format = "bestvideo+bestaudio/best"
        download_command = [
            yt_dlp_path,
            "-f", download_format,
            "-o", output_path,
            video_url
        ]
        subprocess.run(download_command, check=True, capture_output=True, text=True, encoding=preferred_encoding)
        print("動画のダウンロードが完了しました。")
        print(f"保存先: {output_path}")
        return output_path, sanitized_title

    except subprocess.CalledProcessError as e:
        print("yt-dlpの実行中にエラーが発生しました。")
        print(f"エラー詳細: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None, None

def extract_unique_frames(video_path, roi, skip_seconds=1.5, output_dir="output_frames", threshold=2.0):
    """
    ROI内の変化を監視し、ユニークなフレームを画像として保存する。
    """
    print(f"ユニークなフレームを抽出しています... (約{skip_seconds}秒ごとにチェック)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けません。")
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
    frame_number = 0
    mean_diff = 0.0

    while frame_number < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_roi = frame[y:y+h, x:x+w]
        current_frame_roi_gray = cv2.cvtColor(current_frame_roi, cv2.COLOR_BGR2GRAY)

        if last_frame_roi_gray is None:
            is_unique = True
            mean_diff = 0.0
        else:
            diff = cv2.absdiff(last_frame_roi_gray, current_frame_roi_gray)
            mean_diff = np.mean(diff)
            is_unique = mean_diff > threshold

        if is_unique:
            saved_count += 1
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(filename, current_frame_roi)
            print(f"ユニークなフレームを保存しました: {filename} (フレーム番号: {frame_number}, 差分: {mean_diff:.2f})")
            last_frame_roi_gray = current_frame_roi_gray
        
        frame_number += frames_to_skip

    cap.release()
    print(f"抽出完了。合計 {saved_count} 枚のユニークな画像を保存しました。")
    return output_dir

def create_pdf_from_images(image_folder, pdf_path):
    """
    A4用紙にマージンを設定し、複数の画像を縦に並べてPDFを作成する。
    """
    print(f"楽譜形式のPDFを作成しています: {pdf_path}")
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])

    if not image_files:
        print("エラー: PDFに変換する画像が見つかりません。")
        return

    A4_WIDTH_PX, A4_HEIGHT_PX = 2480, 3508
    MARGIN_PX = 150
    GAP_PX = 50
    printable_width = A4_WIDTH_PX - (2 * MARGIN_PX)

    pages = []
    current_page = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), 'white')
    y_offset = MARGIN_PX

    for img_file in image_files:
        img = Image.open(img_file)
        
        aspect_ratio = img.height / img.width
        new_height = int(printable_width * aspect_ratio)
        resized_img = img.resize((printable_width, new_height), Image.LANCZOS)

        if y_offset + resized_img.height > A4_HEIGHT_PX - MARGIN_PX:
            pages.append(current_page)
            current_page = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), 'white')
            y_offset = MARGIN_PX

        current_page.paste(resized_img, (MARGIN_PX, y_offset))
        y_offset += resized_img.height + GAP_PX

    pages.append(current_page)

    if len(pages) > 1:
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    else:
        pages[0].save(pdf_path)

    print("PDFの作成が完了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="楽譜動画をダウンロードしてPDFに変換するツール v2")
    parser.add_argument("url", help="ダウンロードする動画のURL")
    parser.add_argument("--skip_seconds", type=float, default=1.5, help="フレームをチェックする間隔（秒）。値を大きくすると速くなりますが、変化を見逃す可能性があります。")
    parser.add_argument("--auto_roi", action='store_true', help="ROI（楽譜エリア）の自動検出を試みます。")
    parser.add_argument("--color_tolerance", type=int, default=30, help="背景色検出の色の許容範囲。")

    args = parser.parse_args()
    
    downloaded_file, video_title = download_video(args.url)
    
    if downloaded_file:
        roi = None
        if args.auto_roi:
            potential_roi, roi_frame = find_roi(downloaded_file, color_tolerance=args.color_tolerance)
            if potential_roi:
                if confirm_roi(roi_frame, potential_roi):
                    roi = potential_roi
                else:
                    print("自動検出ROIがキャンセルされました。")
            else:
                print("\n自動検出に失敗したため、手動選択に切り替えます。")
                roi = select_roi_manually(downloaded_file)
        else:
            roi = select_roi_manually(downloaded_file)

        if roi:
            print(f"使用するROI: {roi}")
            frame_folder = extract_unique_frames(downloaded_file, roi, skip_seconds=args.skip_seconds)
            if frame_folder and os.listdir(frame_folder):
                print(f"\nフレームは '{frame_folder}' フォルダに保存されました。")
                pdf_output_path = os.path.join(get_script_path(), f"{video_title}.pdf")
                create_pdf_from_images(frame_folder, pdf_output_path)
            else:
                print("ユニークなフレームが見つからなかったため、PDFは作成されませんでした。")
        else:
            print("ROIが決定されなかったため、処理を中断します。")
    else:
        print("\n処理を中断しました。")
