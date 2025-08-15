import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
from pathlib import Path

def extract_features(folder_path, label):
    """从音频文件提取MFCC特征"""
    features = []
    labels = []
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"警告: 文件夹 {folder_path} 不存在")
        return np.array([]), np.array([])
        
    for filename in folder_path.glob("*.mp3"):
        try:
            # 使用优化的参数加载音频
            y, sr = librosa.load(filename, duration=3, sr=22050)
            # 使用较小的hop_length加快处理速度
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            features.append(mfcc_scaled)
            labels.append(label)
            print(f"已处理: {filename.name}")
        except Exception as e:
            print(f"处理{filename}时出错: {str(e)}")
    
    return np.array(features), np.array(labels)

def train_model():
    """训练性别识别模型"""
    print("开始提取特征...")
    base_dir = Path(__file__).parent
      # 提取特征
    female_features, female_labels = extract_features(base_dir / "data" / "female", 0)
    male_features, male_labels = extract_features(base_dir / "data" / "male", 1)
    
    if len(female_features) == 0 or len(male_features) == 0:
        print("错误: 没有找到足够的音频文件来训练模型")
        return

    # 合并数据
    X = np.concatenate((female_features, male_features))
    y = np.concatenate((female_labels, male_labels))    # 保存特征
    print(f"保存特征数据，形状: {X.shape}")
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    np.save(models_dir / "X.npy", X)
    np.save(models_dir / "y.npy", y)

    print("特征提取完成，开始训练模型...")

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 训练模型
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    print("开始训练模型...")
    model.fit(X_train, y_train)

    # 评估模型
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.2%}")
    print(f"测试集准确率: {test_acc:.2%}")    # 保存模型和缩放器
    print("保存模型...")
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "gender_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print("模型训练完成并保存")

if __name__ == "__main__":
    train_model()
