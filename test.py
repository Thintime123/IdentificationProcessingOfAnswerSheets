# -*- coding:utf-8 -*-
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import random

ANSWER_KEY_SCORE = {}

ANSWER_KEY = {0: "A", 1: "B", 2: "C", 3: "D"}

sheet_num = 1
path = ''
col = 3
row = (sheet_num + col - 1) // col
cnt = 0
problem_num = 60
scores = []

score_dict = {'0': 0, '20': 0, '40': 0, '60': 0, '80': 0, '100': 0}


def prework():
    number = [i for i in range(60)]
    answer = [random.randint(0, 4) for i in range(60)]
    global ANSWER_KEY_SCORE
    ANSWER_KEY_SCORE = dict(zip(number, answer))


def column_chart():
    plt.figure()
    x = [0, 20, 40, 60, 80, 100]
    y = []
    for key in score_dict:
        y.append(score_dict[key])

    plt.bar(x, y, 10)
    plt.title(r'Bar chart for data analysis')
    plt.xlabel('score')
    plt.ylabel('number')
    # plt.show()


def Pie_chart():
    plt.figure()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    label_list = ["[80, 100]", "[60, 80)", "[20, 60)", "[0, 20)"]  # 各部分标签
    perct1 = int((score_dict['80'] + score_dict['100']) / cnt * 100)
    perct2 = int(score_dict['60'] / cnt * 100)
    perct3 = int(score_dict['20'] + score_dict['40'] / cnt * 100)
    perct4 = int(score_dict['0'] / cnt * 100)
    size = [perct1, perct2, perct3, perct4]  # 各部分大小

    color = ["red", "green", "blue", "yellow"]  # 各部分颜色
    explode = [0.05, 0, 0, 0]  # 各部分突出值

    # 优化饼状图
    for per in size:
        ind = size.index(per)
        if per == 0:
            label_list.pop(ind)
            size.pop(ind)
            color.pop(ind)
            explode.pop(ind)

    patches, l_text, p_text = plt.pie(size, explode=explode, colors=color, labels=label_list, labeldistance=1.2,
                                      autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.6)
    plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
    plt.legend()


def Data_analysis_file():
    score_sum = sum(int(key) * value for key, value in score_dict.items())
    with open('Data_analysis_file.txt', 'w') as file:
        file.write(f'The number of students taking the exam: {sheet_num}\n')
        file.write(f'average score: {score_sum / sheet_num:.2f}\n')


def main():
    # print(f'正在识别第{cnt}份答题卡：')
    print(f'The answer sheet({cnt}) is being recognized.')

    # 加载一个图片到opencv中
    img = cv.imread(path)
    # cv.imshow("origin", img)
    # 将多个图像显示在同一个窗口中
    # plt.subplot(4, 3, 1), plt.imshow(img), plt.axis('off'), plt.title('origin')

    # 转化成灰度图片，降低计算复杂度（RGB）
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    # plt.subplot(4, 3, 2), plt.imshow(gray, cmap='gray'), plt.axis('off'), plt.title('gray')

    # 高斯模糊，使图像柔和平滑，便于边缘检测
    gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)
    # cv.imshow("gaussian", gaussian_bulr)
    # plt.subplot(4, 3, 3), plt.imshow(gaussian_bulr, cmap='gray'), plt.axis('off'), plt.title('gaussian_bulr')

    # 边缘检测,灰度值小于75这个值的会被丢弃，大于200参这个值会被当成边缘，在中间的部分，自动检测（像素梯度幅度）
    edged = cv.Canny(gaussian_bulr, 75, 200)
    # cv.imshow("edged", edged)
    # plt.subplot(4, 3, 4), plt.imshow(edged), plt.axis('off'), plt.title('edged')

    # 寻找轮廓，第二个参数表示只检测外围的轮廓（轮廓检索模式），第三个参数压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标（轮廓的近似）
    # 返回轮廓，图像的拓扑信息（[Next, Previous, First_Child, Parent]）
    cts, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 给轮廓加标记，便于我们在原图里面观察，注意必须是原图才能画出红色，灰度图是没有颜色的
    cv.drawContours(img, cts, -1, (0, 0, 255), 3)
    # cv.imshow("draw_contours", img)
    # plt.subplot(4, 3, 5), plt.imshow(img), plt.axis('off'), plt.title('draw_contours')

    # 按面积大小对所有的轮廓排序
    sorted_cts = sorted(cts, key=cv.contourArea, reverse=True)

    # print("寻找轮廓的个数：", len(cts))

    # 正确题的个数
    correct_count = 0

    for c in sorted_cts:
        # 周长，第1个参数是轮廓，第二个参数代表是否是闭环的图形
        peri = 0.01 * cv.arcLength(c, True)
        # 获取多边形的所有定点，如果是四个定点，就代表是矩形
        approx = cv.approxPolyDP(c, peri, True)
        # 打印定点个数
        # print("顶点个数：", len(approx))

        if len(approx) == 4:  # 矩形，识别到答题卡，进行放大
            # 透视变换提取原图内容部分
            ox_sheet = four_point_transform(img, approx.reshape(4, 2))
            # 透视变换提取灰度图内容部分
            tx_sheet = four_point_transform(gray, approx.reshape(4, 2))

            # cv.imshow("ox", ox_sheet)
            # cv.imshow("tx", tx_sheet)
            # plt.subplot(4, 3, 6), plt.imshow(ox_sheet), plt.axis('off'), plt.title('ox')
            # plt.subplot(4, 3, 7), plt.imshow(tx_sheet, cmap='gray'), plt.axis('off'), plt.title('tx')

            # 使用ostu二值化算法对灰度图做一个二值化处理
            ret, thresh2 = cv.threshold(tx_sheet, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
            # cv.imshow("ostu", thresh2)
            # plt.subplot(4, 3, 8), plt.imshow(thresh2), plt.axis('off'), plt.title('thresh2')

            # 继续寻找轮廓
            r_cnt, r_hierarchy = cv.findContours(thresh2.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # print("找到轮廓个数：", len(r_cnt))

            # 使用红色标记所有的轮廓
            # cv.drawContours(ox_sheet, r_cnt, -1, (0, 0, 255), 2)
            # cv.imshow('ox_sheet', ox_sheet)
            # 把所有找到的轮廓，给标记出来

            # 存储所有选项
            questionCnts = []
            c = 1
            for cxx in r_cnt:
                print(c)
                c += 1
                # 通过矩形，标记每一个指定的轮廓
                x, y, w, h = cv.boundingRect(cxx)
                ar = w / float(h)
                # print(f'w: {w}, h: {h}')
                if w >= 0.1 and h >= 0.1:
                    # 使用红色标记，满足指定条件的图形
                    # cv.rectangle(ox_sheet, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # 把每个选项，保存下来
                    questionCnts.append(cxx)

            # cv.imshow("ox_1", ox_sheet)
            # cv.waitKey(0)
            # plt.subplot(4, 3, 9), plt.imshow(ox_sheet), plt.axis('off'), plt.title('ox_1')

            # 按坐标从上到下排序
            questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

            # 使用np函数，按5个元素，生成一个集合
            for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):

                # 获取按从左到右的排序后的5个元素
                cnts = contours.sort_contours(questionCnts[i:i + 4])[0]

                bubble_rows = []

                # 遍历每一个选项
                for (j, c) in enumerate(cnts):
                    # 生成一个大小与透视图一样的全黑背景图布
                    # mask = np.zeros(tx_sheet.shape, dtype="uint8")
                    mask = np.zeros(tx_sheet.shape, np.uint8)
                    # cv.imshow('mask', mask)

                    # 将指定的轮廓+白色的填充写到画板上,255代表亮度值，亮度=255的时候，颜色是白色，等于0的时候是黑色
                    cv.drawContours(mask, [c], -1, 255, -1)
                    # 做两个图片做位运算，把每个选项独自显示到画布上，为了统计非0像素值使用，这部分像素最大的其实就是答案
                    mask = cv.bitwise_and(thresh2, thresh2, mask=mask)
                    # cv.imshow("c" + str(i), mask)
                    # cv.imshow('mask', mask)
                    # cv.waitKey(0)
                    # cv.destroyWindow()

                    # 获取每个答案的像素值
                    total = cv.countNonZero(mask)
                    # 存到一个数组里面，tuple里面的参数分别是，像素大小和答案的序号值
                    # print(total,j)
                    bubble_rows.append((total, j))

                bubble_rows = sorted(bubble_rows, key=lambda x: x[0], reverse=True)

                # 判断是否有没有填涂的题目
                # if bubble_rows[0][0] < 450:
                #     print(f'({q + 1})This question has not been answered.')
                #     continue
                # if bubble_rows[1][0] > 450:
                #     print(f'({q + 1})This is a single-choice question, but you have selected multiple options.')
                #     continue
                # 选择的答案序号
                choice_num = bubble_rows[0][1]

                fill_color = None

                # 如果做对就加 1
                if ANSWER_KEY_SCORE.get(q) == choice_num:
                    fill_color = (0, 255, 0)  # 正确 绿色
                    correct_count += 1
                    print(f'({q + 1})Accept', end=' ')
                else:
                    # fill_color = (0, 0, 255)  # 错误 红色
                    fill_color = (255, 0, 0)  # 错误 红色
                    print(f'({q + 1})Wrong answer!', end=' ')
                # ans = chr(ANSWER_KEY_SCORE.get(q) + ord('A'))
                # print(f"正确答案：{ans} 学生答案：{ANSWER_KEY.get(choice_num)} 数据: {bubble_rows}")
                # print(f"correct answer：{ans} your answer：{ANSWER_KEY.get(choice_num)}")

                cv.drawContours(ox_sheet, cnts[choice_num], -1, fill_color, 2)

            # cv.imshow("answer_flagged", ox_sheet)
            # plt.subplot(4, 3, 10), plt.imshow(ox_sheet), plt.axis('off'), plt.title('answer_flagged')

            text1 = "total: " + str(len(ANSWER_KEY)) + ""

            text2 = "right: " + str(correct_count)

            # text3 = "score: " + str(correct_count * 1.0 / len(ANSWER_KEY) * 100) + ""
            text3 = "score: " + str(correct_count * 1.0 / len(ANSWER_KEY_SCORE) * 100) + ""

            score = str(int(correct_count / len(ANSWER_KEY_SCORE) * 100))

            # score_dict[score] += 1

            font = cv.FONT_HERSHEY_SIMPLEX
            # cv.putText(ox_sheet, text1 + "  " + text2 + "  " + text3, (10, 30), font, 0.5, (0, 0, 255), 2)
            cv.putText(ox_sheet, text1 + "  " + text2 + "  " + text3, (10, 30), font, 0.5, (255, 0, 0), 2)

            # cv.imshow("score", ox_sheet)
            plt.subplot(1, 1, 1), plt.imshow(ox_sheet), plt.axis('off'), plt.title(f'answer_sheet_{cnt}')

            # plt.subplot(4, 3, 1), plt.imshow(img), plt.axis('off'), plt.title('image')
            # plt.subplot(4, 3, 2), plt.imshow(gray), plt.axis('off'), plt.title('gray')
            # plt.subplot(4, 3, 3), plt.imshow(gaussian_bulr), plt.axis('off'), plt.title('gaussian_bulr')
            # plt.subplot(4, 3, 4), plt.imshow(edged), plt.axis('off'), plt.title('edged')
            # plt.subplot(4, 3, 5), plt.imshow(img), plt.axis('off'), plt.title('draw_contours')
            # plt.subplot(4, 3, 6), plt.imshow(ox_sheet), plt.axis('off'), plt.title('ox')
            # plt.subplot(4, 3, 7), plt.imshow(tx_sheet), plt.axis('off'), plt.title('tx')
            # plt.subplot(4, 3, 8), plt.imshow(thresh2), plt.axis('off'), plt.title('thresh2')
            # plt.subplot(4, 3, 9), plt.imshow(ox_sheet), plt.axis('off'), plt.title('ox_1')
            # plt.subplot(4, 3, 10), plt.imshow(ox_sheet), plt.axis('off'), plt.title('answer_flagged')
            # plt.subplot(4, 3, 11), plt.imshow(ox_sheet), plt.axis('off'), plt.title('score')
            # plt.show()
            # cv.imshow(f'judge{cnt}', ox_sheet)
            # print(f'正确率为 {correct_count}/{problem_num} = {int(correct_count / problem_num * 100)}%')
            # print(f'Correct rate: {correct_count}/{problem_num} = {int(correct_count / problem_num * 100)}%')

            # print()
            # Data_analysis_file()
            break


if __name__ == '__main__':
    prework()
    plt.figure()
    for i in range(1, sheet_num + 1):
        path = './res/img/work1.png'
        cnt += 1
        main()
        # cv.waitKey(0)
        # cv.destroyWindow('score')
    # column_chart()
    # Pie_chart()
    plt.show()
    cv.waitKey(0)
    # cv.destroyWindow()
