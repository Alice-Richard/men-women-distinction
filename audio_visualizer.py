import tkinter as tk
from tkinter import ttk
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

class AudioVisualizer:
    def __init__(self, master, figsize=(10, 6)):
        """初始化音频可视化器"""
        self.master = master
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=figsize)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar.pack_forget()  # 初始时隐藏进度条
        
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.clear_plots()

    def clear_plots(self):
        """清空图表"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("波形图", fontsize=12)
        self.ax2.set_title("频谱图", fontsize=12)
        self.ax1.set_xlabel("时间（秒）", fontsize=10)
        self.ax1.set_ylabel("振幅", fontsize=10)
        self.ax2.set_xlabel("时间（秒）", fontsize=10)
        self.ax2.set_ylabel("频率（Hz）", fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_var.set(value)
        if value >= 100:
            self.progress_bar.pack_forget()
        else:
            self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

    def _process_audio(self, file_path):
        """在后台线程中处理音频"""
        try:
            self.update_progress(10)
            # 加载音频文件
            y, sr = librosa.load(file_path, duration=10)  # 最多加载10秒
            self.update_progress(40)
            
            # 在主线程中更新UI
            self.master.after(0, self._update_plots, y, sr)
            return True
            
        except Exception as e:
            self.master.after(0, lambda: self.update_progress(100))
            return str(e)

    def _update_plots(self, y, sr):
        """在主线程中更新图表"""
        try:
            # 清空旧图表
            self.clear_plots()
            self.update_progress(60)
            
            # 绘制波形图
            librosa.display.waveshow(y, sr=sr, ax=self.ax1)
            self.update_progress(80)
            
            # 计算和绘制频谱图
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=self.ax2)
            self.fig.colorbar(img, ax=self.ax2, format="%+2.f dB")
            
            # 调整布局
            self.fig.tight_layout()
            self.canvas.draw()
            self.update_progress(100)
            
        except Exception as e:
            self.update_progress(100)
            return str(e)

    def visualize_audio(self, file_path):
        """可视化音频文件"""
        self.update_progress(0)
        self.executor.submit(self._process_audio, file_path)
