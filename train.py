import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'F:\UI\yolo11\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')
    model.train(data=r'F:\UI\yolo11\ultralytics-main\datasets\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=False,
                project='runs/train',
                name='exp',
                )








