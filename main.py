from pathlib import Path
import tkinter as tk
import numpy as np
import librosa
import joblib
from tkinter import ttk, filedialog, messagebox
from ttkbootstrap import Style
from PIL import Image, ImageTk
from audio_visualizer import AudioVisualizer

class EnhancedGenderClassifierApp:
    def upload(self):
        """上传音频文件"""
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3")]
        )
        if file_path:
            try:
                self.current_file = file_path
                self.predict_btn.config(state=tk.NORMAL)
                self.status_label.config(text="音频文件已加载")
                # 更新频谱显示
                self.update_spectrum(file_path)
            except Exception as e:
                self.status_label.config(text=f"加载失败：{str(e)}")
                messagebox.showerror("错误", f"无法加载音频文件：{str(e)}")
                self.current_file = None
                self.predict_btn.config(state=tk.DISABLED)

    def predict(self):
        """预测音频性别"""
        if self.current_file:
            try:
                features = self.extract_features(self.current_file)
                scaled_features = self.scaler.transform([features])
                prob = self.model.predict_proba(scaled_features)
                confidence = np.max(prob)
                
                # 显示对应性别的图标
                if np.argmax(prob) == 0:
                    self.result_image.config(image=self.female_img)
                    gender = "女性"
                else:
                    self.result_image.config(image=self.male_img)
                    gender = "男性"
                
                # 显示置信度文本
                self.result_text.config(text=f"识别结果：{gender}（置信度：{confidence:.2%}）")
                
            except Exception as e:
                self.status_label.config(text=f"识别失败：{str(e)}")
                self.result_image.config(image='')
                self.result_text.config(text='')

    def load_icon(self, path, size):
        """加载并缩放图标"""
        try:
            img = Image.open(path)
            img = img.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图标：{str(e)}")
            return None

    def update_spectrum(self, file_path):
        """更新频谱显示"""
        try:
            result = self.visualizer.visualize_audio(file_path)
            if isinstance(result, str):
                self.status_label.config(text=f"频谱分析失败：{result}")
            else:
                self.status_label.config(text="频谱分析完成")
        except Exception as e:
            self.status_label.config(text=f"频谱分析失败：{str(e)}")

    def extract_features(self, file_path):
        """提取MFCC特征"""
        try:
            # 使用resampling优化加载速度
            y, sr = librosa.load(file_path, duration=3, sr=22050)
            # 使用较小的hop_length加快处理速度
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
            return np.mean(mfcc.T, axis=0)
        except Exception as e:
            messagebox.showerror("错误", f"特征提取失败：{str(e)}")
            raise

    def __init__(self, master):
        """初始化应用"""
        try:
            self.master = master
            style = Style(theme='flatly')
            master = style.master
            
            # 获取当前脚本所在目录
            self.base_dir = Path(__file__).parent
            
            # 预加载性别图标
            self.male_img = self.load_icon(self.base_dir / "assets" / "male.png", (120, 120))
            self.female_img = self.load_icon(self.base_dir / "assets" / "female.png", (120, 120))
            self.upload_icon = self.load_icon(self.base_dir / "assets" / "upload_icon.png", (24, 24))

            # 配置窗口
            master.title("AI语音性别识别系统")
            master.geometry("1200x800")
            master.resizable(True, True)
            
            # 加载模型
            try:
                self.model = joblib.load(self.base_dir / "models" / "gender_model.pkl")
                self.scaler = joblib.load(self.base_dir / "models" / "scaler.pkl")
            except Exception as e:
                messagebox.showerror("错误", f"无法加载模型：{str(e)}\n请先运行model_train.py训练模型")
                raise
            
            # 创建界面
            self.create_widgets()
            
            # 初始化变量
            self.current_file = None
            
        except Exception as e:
            messagebox.showerror("错误", f"程序初始化失败：{str(e)}")
            raise

    def create_widgets(self):
        """创建界面元素"""
        # 创建左右分栏
        self.paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧性别识别部分
        left_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, weight=1)
        
        # 右侧频谱显示部分
        right_frame = ttk.Frame(self.paned)
        self.paned.add(right_frame, weight=2)
        
        # 添加频谱图标题
        spectrum_label = ttk.Label(right_frame, text="音频频谱分析", 
                                 font=('Microsoft YaHei', 14, 'bold'))
        spectrum_label.pack(pady=5)

        # 创建音频可视化组件
        self.visualizer = AudioVisualizer(right_frame, figsize=(10, 8))

        # 左侧主容器框架
        main_frame = ttk.Frame(left_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 标题标签
        title_font = ('Microsoft YaHei', 16, 'bold')
        self.title_label = ttk.Label(main_frame, text="语音性别识别系统", 
                                   font=title_font, foreground='#2c3e50')
        self.title_label.pack(pady=15)

        # 上传按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        # 上传按钮
        self.upload_btn = ttk.Button(btn_frame, text=" 上传音频", 
                                   image=self.upload_icon, compound=tk.LEFT,
                                   command=self.upload, style='TButton')
        self.upload_btn.grid(row=0, column=0, padx=5)

        # 状态显示
        self.status_label = ttk.Label(btn_frame, text="等待上传音频...", 
                                    foreground='#3498db', font=('Microsoft YaHei', 12))
        self.status_label.grid(row=0, column=1, padx=10)

        # 识别按钮
        self.predict_btn = ttk.Button(main_frame, text="开始识别", 
                                    state=tk.DISABLED, command=self.predict,
                                    style='primary.TButton')
        self.predict_btn.pack(pady=15)

        # 结果展示框
        result_frame = ttk.LabelFrame(main_frame, text="识别结果", 
                                    padding=(20, 10))
        result_frame.pack(fill=tk.X, pady=20)
        
        # 创建图片显示区域
        self.result_image = ttk.Label(result_frame)
        self.result_image.pack(pady=10)
        self.result_text = ttk.Label(result_frame, 
                                   font=('Microsoft YaHei', 12, 'bold'), 
                                   foreground='#27ae60')
        self.result_text.pack(pady=5)

def main():
    """程序入口"""
    try:
        root = tk.Tk()
        app = EnhancedGenderClassifierApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("错误", f"程序启动失败：{str(e)}")
        raise

if __name__ == '__main__':
    main()
