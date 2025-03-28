import cv2

# 打开视频文件
video_path = "../video.mp4"  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计算新的尺寸，使其宽高比为1:1
# 选择宽和高中的较大值作为新的尺寸
new_size = max(width, height)

# 创建输出视频
output_path = "output_video_1_1.mp4"  # 输出视频路径
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率
out = cv2.VideoWriter(output_path, fourcc, fps, (new_size, new_size))

# 逐帧处理
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧的大小
    resized_frame = cv2.resize(frame, (new_size, new_size))

    # 写入输出视频
    out.write(resized_frame)

# 释放资源
cap.release()
out.release()
print("Resized video saved as", output_path)