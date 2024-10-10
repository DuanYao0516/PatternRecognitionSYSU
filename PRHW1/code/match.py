import cv2
import numpy as np
import os


def harris_corner_detection(image, k=0.04, threshold=0.01):
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 使用 5x5 的高斯核进行平滑

    # Convert the image to grayscale 前面调用者已经做过灰度化
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
    # 使用sobel算子计算梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate elements of the Harris matrix
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Calculate sums of the elements in the windows
    height, width = gray.shape
    window_size = 3  # Size of the window for summation
    offset = window_size // 2
    corner_response = np.zeros_like(gray, dtype=np.float32)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Calculate the sums of the elements in the window
            Sxx = np.sum(Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Syy = np.sum(Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1])
            Sxy = np.sum(Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1])

            # Calculate the determinant and trace of the Harris matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            # Calculate the corner response using the Harris measure
            corner_response[y, x] = det - k * (trace ** 2)

    # Threshold the corner response to find corners
    corners = np.zeros_like(corner_response)
    corners[corner_response > threshold * np.max(corner_response)] = 255

    # Find coordinates of the corners
    corner_coords = np.argwhere(corners == 255)

    # Create keypoints from corner coordinates
    keypoints = [cv2.KeyPoint(x=float(coord[1]), y=float(coord[0]), size=3) for coord in corner_coords]

    return keypoints

def cvshow(name, img):
    # 展示图像，关闭后继续运行
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_kp(image):
    # 提取 sift 描述子
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Using Harris corner detection to get keypoints
    harris_kp = harris_corner_detection(gray_image)
    # Compute SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    _, des = sift.compute(gray_image, harris_kp)
    # kp_image = cv2.drawKeypoints(gray_image, harris_kp, None)
    kp_image = cv2.drawKeypoints(image, harris_kp, None)
    return kp_image, harris_kp, des


def hog_kp(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute Harris corner detection to get keypoints
    harris_kp = harris_corner_detection(gray_image)

    # Initialize HOG Descriptor
    winSize = (8, 8)  # Define the size of the window for HOG descriptor
    blockSize = (8, 8)  # Define the size of the block for HOG descriptor
    blockStride = (4, 4)  # Define the stride of the block for HOG descriptor
    cellSize = (4, 4)  # Define the size of the cell for HOG descriptor
    nbins = 9  # Define the number of bins for HOG descriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # Compute HOG descriptors
    descriptors = []
    max_descriptor_length = 0
    for kp in harris_kp:
        x, y = kp.pt
        x, y = int(x), int(y)
        patch = gray_image[max(0, y - 8):min(y + 8, gray_image.shape[0]), max(0, x - 8):min(x + 8, gray_image.shape[1])]

        # Check if the patch size is sufficient for HOG computation
        if patch.shape[0] >= winSize[1] and patch.shape[1] >= winSize[0]:
            hog_desc = hog.compute(patch)
            descriptors.append(hog_desc.flatten())
            max_descriptor_length = max(max_descriptor_length, len(hog_desc))

    # Zero-pad or truncate descriptors to ensure uniform length
    padded_descriptors = []
    for desc in descriptors:
        if len(desc) < max_descriptor_length:
            padded_desc = np.pad(desc, (0, max_descriptor_length - len(desc)), mode='constant')
        else:
            padded_desc = desc[:max_descriptor_length]
        padded_descriptors.append(padded_desc)

    return harris_kp, np.array(padded_descriptors)


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 返回可视化两张图匹配点结果
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis


# 全景拼接
def siftimg_rightlignment(img_right, img_left):
    _, kp1, des1 = sift_kp(img_right)
    _, kp2, des2 = sift_kp(img_left)
    goodMatch = get_good_match(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)  #  用RANSAC选择最优的四组配对点，再计算H矩阵。H为3*3矩阵
        
        # 将图片右进行视角变换，result是变换后图片
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        # cvshow('result_medium', result)
        # 将图片左传入result图片最左端
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result


def hogimg_rightlignment(img_right, img_left):
    kp1, des1 = hog_kp(img_right)
    kp2, des2 = hog_kp(img_left)
    goodMatch = get_good_match(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)

        # 将图片右进行视角变换，result是变换后图片
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        cv2.imwrite('results/result_medium.png', result)
        # 将图片左传入result图片最左端
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result


def crop_image(image):
    # 用于裁剪图像右侧的无灰度值区域，这个函数暂时没用到
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the leftmost column with non-zero pixel values
    leftmost_column = np.argmax(np.any(gray_image > 0, axis=0))

    # Find the rightmost column with non-zero pixel values
    rightmost_column = np.argmax(np.any(gray_image[:, ::-1] > 0, axis=0))

    # Calculate the width of the region with non-zero pixel values
    width = rightmost_column if rightmost_column > 0 else image.shape[1]

    # Crop the image to the region with non-zero pixel values
    cropped_image = image[:, leftmost_column:width]

    return cropped_image


def task1():
    # 两幅图像 SIFT + HOG 全景拼接

    # 读取拼接图片（注意图片左右的放置）
    # 是对右边的图形做变换
    img_right = cv2.imread(r'images/uttower2.jpg')
    img_left = cv2.imread(r'images/uttower1.jpg')
    # Paths for saving keypoints images
    keypoints_left_path = 'results/uttower1_keypoints.jpg'
    keypoints_right_path = 'results/uttower2_keypoints.jpg'
    # Paths for saving matches image
    matches_image_path = 'results/uttower_match.png'
    # Paths for saving stitching result
    sift_stitching_result_path = 'results/uttower_stitching_sift.png'
    # Paths for saving stitching result
    hog_stitching_result_path = 'results/uttower_stitching_hog.png'

    kpimg_right, kp1, des1 = sift_kp(img_right)
    kpimg_left, kp2, des2 = sift_kp(img_left)

    # 同时显示原图和关键点检测后的图
    cvshow('img_left', np.hstack((img_left, kpimg_left)))
    cvshow('img_right', np.hstack((img_right, kpimg_right)))
    cv2.imwrite(keypoints_left_path, kpimg_left)
    cv2.imwrite(keypoints_right_path, kpimg_right)

    goodMatch = get_good_match(des1, des2)

    all_goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch, None, flags=2)

    # goodmatch_img自己设置前多少个goodMatch[:10]
    goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch[:10], None, flags=2)

    cvshow('Keypoint Matches1', all_goodmatch_img)
    cvshow('Keypoint Matches2', goodmatch_img)
    cv2.imwrite(matches_image_path, all_goodmatch_img)

    # 把图片拼接成全景图 sift
    result = siftimg_rightlignment(img_right, img_left)
    cvshow('result', result)
    cv2.imwrite(sift_stitching_result_path, result)

    # 把图片拼接成全景图 hog
    result = hogimg_rightlignment(img_right, img_left)
    cvshow('result', result)
    cv2.imwrite(hog_stitching_result_path, result)

def task2():
    # 多幅图像全景图拼接
    result_path = 'results/yosemite_stitching.png'

    img_4 = cv2.imread(r'images/yosemite4.jpg')
    img_3 = cv2.imread(r'images/yosemite3.jpg')
    img_2 = cv2.imread(r'images/yosemite2.jpg')
    img_1 = cv2.imread(r'images/yosemite1.jpg')

    # result1 = siftimg_rightlignment(img_4, img_3)
    # cvshow('result1', result1)
    # result2 = siftimg_rightlignment(img_2, img_1)
    # cvshow('result2', result2)
    # result = siftimg_rightlignment(result1, result2)
    # cvshow('result', result)

    result1 = siftimg_rightlignment(img_4, img_3)
    cvshow('result1', result1)
    result2 = siftimg_rightlignment(result1, img_2)
    cvshow('result2', result2)
    result = siftimg_rightlignment(result2, img_1)
    # result = crop_image(result)
    cvshow('result', result)

    cv2.imwrite(result_path, result)


if __name__=="__main__":
    # task1()
    task2()
