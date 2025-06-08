import cv2
import os

img_dir = "output/scanet/new/scene0000_00/composite"
output_video = "output/scanet/new/scene0000_00/composite.mp4"

# 获取所有图片并按文件名排序
images = sorted([img for img in os.listdir(img_dir) if img.endswith(".png")])

# 读取第一张图片获取尺寸
frame = cv2.imread(os.path.join(img_dir, images[0]))
height, width, _ = frame.shape

# 设置视频编码
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 或 'MP4V' 生成 .mp4
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

for img_name in images:
    img_path = os.path.join(img_dir, img_name)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()