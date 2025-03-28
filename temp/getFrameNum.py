import cv2

# 打开视频文件
video_path = '../temp.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in the video: {total_frames}")

# 释放视频文件
cap.release()