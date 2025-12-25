import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, simpledialog
import subprocess
import threading
import os
import sys
import io
import queue
import tab_extractor_v2 # リファクタリングしたスクリプトをインポート

# 標準出力をリダイレクトするためのクラス
class QueueIO(io.TextIOBase):
    def __init__(self, q):
        self.q = q
    def write(self, s):
        self.q.put(s)
        return len(s)
    def flush(self):
        pass

class TabExtractorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Tab Extractor GUI")
        master.geometry("800x600")
        
        # --- スタイル ---
        style = ttk.Style()
        style.theme_use('clam')

        # --- メインフレーム ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 入力セクション ---
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Video URL or File Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.source_var = tk.StringVar()
        self.source_entry = ttk.Entry(input_frame, textvariable=self.source_var, width=60)
        self.source_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = ttk.Button(input_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        # --- オプションセクション ---
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        # 左側オプション
        left_options_frame = ttk.Frame(options_frame)
        left_options_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.auto_roi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_options_frame, text="Auto ROI Detection", variable=self.auto_roi_var).pack(anchor=tk.W)

        self.skip_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_options_frame, text="Skip yt-dlp Update", variable=self.skip_update_var).pack(anchor=tk.W)

        # Cookie オプション
        cookie_frame = ttk.Frame(left_options_frame)
        cookie_frame.pack(anchor=tk.W, pady=5)
        self.use_cookies_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cookie_frame, text="Use Cookies from Browser:", variable=self.use_cookies_var, command=self.toggle_cookie_browser_entry).pack(side=tk.LEFT)
        self.cookie_browser_var = tk.StringVar(value="firefox")
        self.cookie_browser_entry = ttk.Entry(cookie_frame, textvariable=self.cookie_browser_var, width=15, state=tk.DISABLED)
        self.cookie_browser_entry.pack(side=tk.LEFT, padx=5)

        # 右側オプション
        right_options_frame = ttk.Frame(options_frame)
        right_options_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        ttk.Label(right_options_frame, text="Skip Seconds:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.skip_seconds_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(right_options_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.skip_seconds_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(right_options_frame, text="Color Tolerance:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.color_tolerance_var = tk.IntVar(value=30)
        ttk.Spinbox(right_options_frame, from_=1, to=100, textvariable=self.color_tolerance_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(right_options_frame, text="Max Threshold:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.max_threshold_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(right_options_frame, from_=1.0, to=200.0, textvariable=self.max_threshold_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=2)

        # --- 実行ボタン ---
        self.run_button = ttk.Button(main_frame, text="Run Extraction", command=self.run_extraction)
        self.run_button.pack(pady=10)

        # --- 出力コンソール ---
        console_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.console_output = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, bg="#282c34", fg="#abb2bf", insertbackground="white", state='disabled')
        self.console_output.pack(fill=tk.BOTH, expand=True)
        
        # 出力リダイレクト設定
        self.log_queue = queue.Queue()
        self.queue_io = QueueIO(self.log_queue)
        self.master.after(100, self.process_log_queue)

    def process_log_queue(self):
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self.log_message(line)
        self.master.after(100, self.process_log_queue)

    def log_message(self, message):
        self.console_output.configure(state='normal')
        self.console_output.insert(tk.END, message)
        self.console_output.see(tk.END)
        self.console_output.configure(state='disabled')

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=(("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*"))
        )
        if filename:
            self.source_var.set(filename)

    def toggle_cookie_browser_entry(self):
        if self.use_cookies_var.get():
            self.cookie_browser_entry.config(state=tk.NORMAL)
        else:
            self.cookie_browser_entry.config(state=tk.DISABLED)

    def run_extraction(self):
        self.run_button.config(state=tk.DISABLED)
        self.console_output.configure(state='normal')
        self.console_output.delete('1.0', tk.END)
        self.console_output.configure(state='disabled')
        
        thread = threading.Thread(target=self.run_extraction_thread)
        thread.daemon = True
        thread.start()
        
    def run_extraction_thread(self):
        # 標準出力をリダイレクト
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = self.queue_io
        sys.stderr = self.queue_io

        try:
            source = self.source_var.get()
            if not source:
                print("エラー: 動画のURLまたはファイルパスを指定してください。")
                return

            is_url = source.lower().startswith('http')
            video_title = ""

            # タイトルを取得
            if is_url:
                try:
                    print("動画タイトルを取得しています...")
                    yt_dlp_path = os.path.join(tab_extractor_v2.get_script_path(), "yt-dlp.exe")
                    if not os.path.exists(yt_dlp_path):
                         print(f"エラー: yt-dlp.exe が見つかりません: {yt_dlp_path}")
                         return

                    title_command = [yt_dlp_path, "--get-title", "--skip-download", source]
                    if self.use_cookies_var.get() and self.cookie_browser_var.get():
                        title_command.extend(["--cookies-from-browser", self.cookie_browser_var.get()])
                    
                    title_result = subprocess.run(title_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    video_title = title_result.stdout.strip()
                except Exception as e:
                    print(f"タイトルの事前取得に失敗しました: {e}")
                    video_title = "downloaded_video"
            else:
                video_title = os.path.splitext(os.path.basename(source))[0]

            suggested_title = tab_extractor_v2.clean_title(video_title)

            # GUIスレッドでダイアログを表示
            final_title = self.ask_for_title(suggested_title)
            
            if not final_title:
                print("タイトルが入力されなかったため、処理を中断しました。")
                return
            
            # メイン処理を呼び出し
            tab_extractor_v2.start_processing(
                source=source,
                final_title=final_title,
                skip_seconds=self.skip_seconds_var.get(),
                auto_roi=self.auto_roi_var.get(),
                color_tolerance=self.color_tolerance_var.get(),
                max_threshold=self.max_threshold_var.get(),
                skip_update=self.skip_update_var.get(),
                cookies_from_browser=self.cookie_browser_var.get() if self.use_cookies_var.get() else None
            )

        except Exception as e:
            print(f"\n--- 予期せぬエラーが発生しました: {e} ---")
        finally:
            # 標準出力を元に戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # 実行ボタンを有効に戻す
            self.master.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def ask_for_title(self, suggested_title):
        # simpledialogはGUIスレッドで実行する必要がある
        result_queue = queue.Queue()
        self.master.after(0, lambda: result_queue.put(
            simpledialog.askstring(
                "PDF Title",
                "以下のタイトルでPDFを作成します。必要に応じて修正してください:",
                initialvalue=suggested_title
            )
        ))
        return result_queue.get()


if __name__ == "__main__":
    # PyInstallerでの実行に対応
    if getattr(sys, 'frozen', False):
        # EXE実行時の特別な初期化があればここに追加
        pass
    
    root = tk.Tk()
    gui = TabExtractorGUI(root)
    root.mainloop()
