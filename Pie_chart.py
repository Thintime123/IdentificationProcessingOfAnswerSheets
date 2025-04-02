import matplotlib.pyplot as plt
import matplotlib


def Pie_chart(score_dict, cnt):
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

    patches, l_text, p_text = plt.pie(size, radius=1, explode=explode, colors=color, labels=label_list,
                                      labeldistance=1.2,
                                      autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.6)
    plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
    plt.legend()
    plt.savefig('./feedback/Pie_chart.png')


if __name__ == "__main__":
    pass
