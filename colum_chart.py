import matplotlib.pyplot as plt


def column_chart(score_dict, question_error, sheet_num):
    question_cnt = len(question_error)
    # print(f'题目数量：{question_cnt}')

    question_number = [i + 1 for i in range(question_cnt)]
    # print(f'题目序号：{question_number}')

    plt.figure()
    x = [0, 20, 40, 60, 80, 100]
    y = list(score_dict.values())

    # for key in score_dict:
    #     y.append(int(score_dict[key]))

    plt.bar(x, y, 15, label='The number of people with corresponding scores')
    plt.title(r'The situation of each score range')
    plt.xlabel('score')
    plt.ylabel('number')
    plt.legend()
    plt.savefig(r'./feedback/Situation_of_each_score_range.png')
    # plt.show()

    plt.figure()
    error_number = list(question_error.values())
    correct_number = [sheet_num - i for i in error_number]

    # print(question_number)
    # print(error_number)

    plt.bar(question_number, correct_number, width=0.35, label='correct_number')
    plt.bar([i + 0.35 for i in question_number], error_number, width=0.35, label='error_number')
    plt.title(r'Analysis charts for each question')
    plt.xlabel(r'question number')
    plt.ylabel('number')
    plt.legend()
    plt.savefig(r'./feedback/Analysis_charts_for_each_question.png')


if __name__ == "__main__":
    pass
