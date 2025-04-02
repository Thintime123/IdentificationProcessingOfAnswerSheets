# -*- coding:utf-8 -*-
import copy

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib

import to_excel
from preworlk import *
from Data_analysis_file import *
from colum_chart import *
from Pie_chart import *
from to_excel import *
import _exit

(ANSWER_KEY_SCORE, ANSWER_KEY, sheet_num, path, col, row, cnt, problem_num, scores, score_dict, question_error,
 summarize) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


def main():
    # print(f'正在识别第{cnt}份答题卡：')
    print(f'The answer sheet({cnt}) is being recognized.')

    # 加载一个图片到opencv中
    img = cv.imread(path)

    # 输入答题信息
    file_name = f'./feedback/answering_info/2023xx{cnt:02d}_info.txt'
    file = open(file_name, 'w')
    file.write(f'2023xx{cnt:02d}:\n')
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
            r_cnt, r_hierarchy = cv.findContours(thresh2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # print("找到轮廓个数：", len(r_cnt))

            # 使用红色标记所有的轮廓
            # cv.drawContours(ox_sheet, r_cnt, -1, (0, 0, 255), 2)
            # cv.imshow('ox_sheet', ox_sheet)
            # 把所有找到的轮廓，给标记出来

            # 存储所有选项
            questionCnts = []
            for cxx in r_cnt:
                # 通过矩形，标记每一个指定的轮廓
                x, y, w, h = cv.boundingRect(cxx)
                ar = w / float(h)
                # print(f'w: {w}, h: {h}')

                if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                    # 使用红色标记，满足指定条件的图形
                    # cv.rectangle(ox_sheet, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # 把每个选项，保存下来
                    questionCnts.append(cxx)

            # cv.imshow("ox_1", ox_sheet)
            # cv.waitKey(0)
            # plt.subplot(4, 3, 9), plt.imshow(ox_sheet), plt.axis('off'), plt.title('ox_1')

            # 按坐标从上到下排序
            questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

            ox_sheetx = copy.deepcopy(ox_sheet)

            # 使用np函数，按5个元素，生成一个集合
            for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):

                # 获取按从左到右的排序后的5个元素
                cnts = contours.sort_contours(questionCnts[i:i + 5])[0]

                bubble_rows = []

                # 遍历每一个选项
                for (j, c) in enumerate(cnts):
                    # 生成一个大小与透视图一样的全黑背景图布
                    mask = np.zeros(tx_sheet.shape, dtype="uint8")
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
                if bubble_rows[0][0] < 450:
                    print(f'\t({q + 1})This question has not been answered.')
                    question_error[str(q)] += 1
                    file.write(f'\t({q + 1})This question has not been answered.\n')
                    summarize[cnt][q + 1] = 0
                    continue
                if bubble_rows[1][0] > 450:
                    print(f'\t({q + 1})This is a single-choice question, but you have selected multiple options.')
                    question_error[str(q)] += 1
                    file.write(
                        f'\t({q + 1})This is a single-choice question, but you have selected multiple options.\n')
                    summarize[cnt][q + 1] = 0
                    continue
                # 选择的答案序号
                choice_num = bubble_rows[0][1]

                fill_color = None
                fill_colorx = None

                # 如果做对就加 1
                if ANSWER_KEY_SCORE.get(q) == choice_num:
                    fill_color = (0, 255, 0)  # 正确 绿色
                    fill_colorx = (0, 255, 0)
                    correct_count += 1
                    print(f'\t({q + 1})Accepted!')
                    file.write(f'\t({q + 1})Accepted!\n')
                    summarize[cnt][q + 1] = 1
                else:
                    fill_color = (255, 0, 0)  # 错误 红色 plt
                    fill_colorx = (0, 0, 255) # cv
                    print(f'\t({q + 1})Wrong answer!')
                    file.write(f'\t({q + 1})Wrong answer!\n')
                    question_error[str(q)] += 1
                    summarize[cnt][q + 1] = 0

                ans = chr(ANSWER_KEY_SCORE.get(q) + ord('A'))
                # print(f"正确答案：{ans} 学生答案：{ANSWER_KEY.get(choice_num)} 数据: {bubble_rows}")
                print(f"\t\tcorrect answer：{ans} \tthe given answer：{ANSWER_KEY.get(choice_num)}")
                file.write(f"\t\tcorrect answer：{ans} \tthe given answer：{ANSWER_KEY.get(choice_num)}\n")

                cv.drawContours(ox_sheet, cnts[choice_num], -1, fill_color, 2)
                cv.drawContours(ox_sheetx, cnts[choice_num], -1, fill_colorx, 2)

            # cv.imshow("answer_flagged", ox_sheetx)
            # plt.subplot(4, 3, 10), plt.imshow(ox_sheet), plt.axis('off'), plt.title('answer_flagged')

            text1 = "total: " + str(len(ANSWER_KEY)) + ""

            text2 = "right: " + str(correct_count)

            # text3 = "score: " + str(correct_count * 1.0 / len(ANSWER_KEY) * 100) + ""
            text3 = "score: " + str(correct_count * 1.0 / len(ANSWER_KEY_SCORE) * 100) + ""

            score = str(int(correct_count / len(ANSWER_KEY_SCORE) * 100))

            score_dict[score] += 1

            font = cv.FONT_HERSHEY_SIMPLEX
            # cv.putText(ox_sheet, text1 + "  " + text2 + "  " + text3, (10, 30), font, 0.5, (0, 0, 255), 2)
            cv.putText(ox_sheet, text1 + "  " + text2 + "  " + text3, (10, 30), font, 0.5, (255, 0, 0), 2)
            cv.putText(ox_sheetx, text1 + "  " + text2 + "  " + text3, (10, 30), font, 0.5, (0, 0, 255), 2)

            # cv.imshow("score", ox_sheet)
            plt.subplot(row, col, cnt), plt.imshow(ox_sheet), plt.axis('off'), plt.title(f'answer_sheet_{cnt}')

            # 保存每张答题卡
            cv.imwrite(f'./feedback/answer_sheets_for_feedback/sheet_feedback_{cnt}.png', ox_sheetx)

            # cv.imshow(f'judge{cnt}', ox_sheet)

            rate = int(correct_count / problem_num * 100)

            print(f'Correct rate: {correct_count}/{problem_num} = {rate}%')
            file.write(f'Correct rate: {correct_count}/{problem_num} = {rate}%\n\n')
            print()

            # 每位学生的总分和正确率
            summarize[cnt][problem_num + 1] = sum(summarize[cnt][1:-1])
            summarize[cnt][problem_num + 2] = f'{summarize[cnt][problem_num + 1] / problem_num * 100:.2f}%'
            Data_analysis_file(score_dict, sheet_num)
            file.close()

            break


def work():
    global ANSWER_KEY_SCORE, ANSWER_KEY, sheet_num, path, col, row, cnt, problem_num, scores, score_dict, question_error, summarize

    (ANSWER_KEY_SCORE, ANSWER_KEY, sheet_num, path, col, row, cnt, problem_num, scores, score_dict, question_error,
     summarize) = pre()

    plt.figure()
    for i in range(1, sheet_num + 1):
        path = f'./res/img/img{i}.png'
        cnt += 1
        main()
        # cv.waitKey(0)
        # cv.destroyWindow('score')

    _exit.finish()
    column_chart(score_dict, question_error, sheet_num)
    Pie_chart(score_dict, cnt)
    to_excel.fun(summarize, sheet_num)
    plt.show()
    # cv.waitKey(0)
    # cv.destroyWindow()
    # print(summarize)
    _exit.end()
    print('\nWelcome to use this program again next time!')

if __name__ == '__main__':
    work()
