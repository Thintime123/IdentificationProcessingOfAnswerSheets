import numpy as np
import cv2  # opencv    读取图像默认为BGR
import matplotlib.pyplot as plt  # matplotlib显示图像默认为RGB
import copy

def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 变换后对应坐标位置
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)  # 计算齐次变换矩阵：cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 透视变换：（将输入矩形乘以（齐次变换矩阵），得到输出矩阵）
    return warped


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # if method == "top-to-bottom" or method == "bottom-to-top":
    if method == "top-to-bottom" or method == "left-to right":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def reoutline(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘检测（使用 Canny 算法）
    edges = cv2.Canny(gray, 50, 150)
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓（假设外围矩形是最大的轮廓）
    largest_contour = max(contours, key=cv2.contourArea)
    # 创建一个与图像大小相同的全黑图像
    mask = np.zeros_like(image)
    # 在黑色图像上绘制最大的轮廓为白色填充区域
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)
    # 将原始图像与掩码图像进行按位与操作，去除外围矩形轮廓
    result = cv2.bitwise_and(image, mask)
    return result

def main():
    ANSWER_KEY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 2}

    # 图像预处理
    image = cv2.imread(r"./res/img/answer_sheet_3.png")  # 1
    image_s = copy.deepcopy(image)

    contours_img = image.copy()  # 4
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波-去除噪音    # 2
    edged = cv2.Canny(blurred, 50, 250)  # Canny算子边缘检测      # 3
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 3)  # 画出轮廓（答题卡）   4

    # 提取答题卡并进行透视变化
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓大小进行排序
        for c in cnts:  # 遍历每一个轮廓
            peri = cv2.arcLength(c, True)  # 计算轮廓的长度
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 找出轮廓的多边形拟合曲线
            if len(approx) == 4:  # 找到的轮廓是四边形（对应四个顶点）
                docCnt = approx
                break

    warped = four_point_transform(gray, docCnt.reshape(4, 2))  # 透视变换（齐次变换矩阵）
    warped1 = warped.copy()  # 5
    thresh = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # 0：表示系统自动判断；THRESH_OTSU：自适应阈值设置
    ###############################
    thresh_Contours = thresh.copy()  # 6

    # cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到每一个圆圈轮廓
    reoutline_img = reoutline(image_s)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找到每一个圆圈轮廓

    cv2.drawContours(thresh_Contours, cnts, -1, (0, 0, 255), 3)  # 画出所有轮廓   # 6
    ###################################################################
    # 提取答题卡中所有的有效选项（圆圈）
    questionCnts = []  # 提取每个选项的轮廓
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)  # 获取轮廓的尺寸
        ar = w / float(h)  # 计算比例
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:  # 自定义设置大小（根据实际情况）
            questionCnts.append(c)
    questionCnts = sort_contours(questionCnts, method="top-to-bottom")[0]  # 按照从上到下对所有的选项进行排序

    # print(questionCnts)

    correct = 0
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):  # 每排有5个选项
        cnts = sort_contours(questionCnts[i:i + 5])[0]  # 对每一排进行排序
        bubbled = None
        for (j, c) in enumerate(cnts):  # 遍历每一排对应的五个结果
            mask = np.zeros(thresh.shape, dtype="uint8")  # 使用mask来判断结果（全黑：0）表示涂写答案正确
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # -1表示填充
            # cv_show('mask', mask)		# 展示每个选项
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # mask=mask表示要提取的区域（可选参数）
            total = cv2.countNonZero(mask)  # 通过计算非零点数量来算是否选择这个答案
            if bubbled is None or total > bubbled[0]:  # 记录最大数
                bubbled = (total, j)
        color = (0, 0, 255)  # 对比正确答案
        k = ANSWER_KEY[q]  # 第q题的答案
        # 判断正确
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        cv2.drawContours(warped, [cnts[q]], -1, color, 8)  # 画出轮廓   # 7
    ###################################################################
    # 展示结果
    score = (correct / 5.0) * 100  # 计算总得分
    # print("[INFO] score: {:.2f}%".format(score))

    cv2.putText(warped, "{:.2f}%".format(score), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取的格式是BGR，Matplotlib是RGB
    contours_img = cv2.cvtColor(contours_img, cv2.COLOR_BGR2RGB)  # 4

    plt.subplot(241), plt.imshow(image), plt.axis('off'), plt.title('image')
    plt.subplot(242), plt.imshow(blurred), plt.axis('off'), plt.title('cv2.GaussianBlur')
    plt.subplot(243), plt.imshow(edged), plt.axis('off'), plt.title('cv2.Canny')
    plt.subplot(244), plt.imshow(contours_img), plt.axis('off'), plt.title('cv2.findContours')
    plt.subplot(245), plt.imshow(warped1), plt.axis('off'), plt.title('cv2.warpPerspective')
    plt.subplot(246), plt.imshow(thresh_Contours), plt.axis('off'), plt.title('cv2.findContours')
    plt.subplot(247), plt.imshow(warped), plt.axis('off'), plt.title('cv2.warpPerspective')
    plt.show()

    # timg = cv2.imread('./res/img/answer_sheet_3.png')
    # timg = reoutline(timg)
    # # ret, thresh2 = cv2.threshold(timg, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('origin', timg)
    # # cv2.imshow('after', thresh2)
    # cv2.waitKey(0)
    # cv2.destroyWindow()



if __name__ == "__main__":
    main()
