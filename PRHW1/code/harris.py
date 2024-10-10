import cv2
import numpy as np

def harris_corner_detection(imagine, threshold=0.45, k=0.04, window_size=3):
    # Step 0: 高斯平滑
    image = cv2.GaussianBlur(imagine, (5, 5), 0)  # 使用 5x5 的高斯核进行平滑

    # Step 1: 使用 Sobel 算子计算梯度
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # para 2 = -1 时表示自动选择深度
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: 计算像素点邻域内梯度和的协方差矩阵（结构矩阵）
    Ixx = np.multiply(dx, dx)
    Ixy = np.multiply(dx, dy)
    Iyy = np.multiply(dy, dy)

    # Step 3: 计算像素点的角点响应函数，Harris 响应函数
    height, width = image.shape
    offset = window_size // 2  # offset 设置为窗口半径
    corner_response = np.zeros((height, width), dtype=np.float32)  # 初始化，用于存储每个像素点的角点响应函数值

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_Ixx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]  # 切片列表得到子列表
            window_Ixy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            window_Iyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            Sxx = np.sum(window_Ixx)
            Sxy = np.sum(window_Ixy)
            Syy = np.sum(window_Iyy)

            # Harris 响应
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            corner_response[y, x] = det - k * (trace ** 2)

    # Step 4: 阈值处理
    # 将所有值放缩到0-255
    corner_response_normalized = cv2.normalize(corner_response, None,
                                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 生成一个二值掩码，响应值超过阈值的设置为255
    _, corner_mask = cv2.threshold(corner_response_normalized,
                                   threshold * corner_response_normalized.max(), 255, 0)


    # Step 5: 非极大值抑制
    # corner_mask = cv2.dilate(corner_mask, None)  # 膨胀，增加角点响应区域
    # 尝试之后，感觉不应该使用膨胀操作
    # 膨胀操作的作用：增强角点响应，连接邻近角点，抵抗噪声，NMS的准备（膨胀后更容易找到局部最大值，膨胀会减少小的响应峰）
    corner_points = np.where(corner_mask == 255)  # 定位角点

    return list(zip(*corner_points[::-1]))


def main():
    # 加载图像
    image = cv2.imread('images/sudoku.png', cv2.IMREAD_GRAYSCALE)  # 读入图像并转换为灰度图像

    # 检测角点
    keypoints = harris_corner_detection(image)

    # 绘制点
    image_with_keypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in keypoints:
        cv2.circle(image_with_keypoints, point, 1, (0, 0, 255), -1)

    # 存储图像
    cv2.imwrite('results/sudoku_keypoints.png', image_with_keypoints)

if __name__=="__main__":
    main()