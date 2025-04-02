import copy


def pre():
    question_cnt = 5

    # 题号
    question_number = [str(i) for i in range(question_cnt)]

    ANSWER_KEY_SCORE = {0: 0, 1: 4, 2: 0, 3: 3, 4: 1}

    ANSWER_KEY = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    # 答题卡数量
    sheet_num = 20
    path = ''
    col = 4
    row = (sheet_num + col - 1) // col
    cnt = 0
    scores = []

    # 各个分段的人数
    score_dict = {'0': 0, '20': 0, '40': 0, '60': 0, '80': 0, '100': 0}

    # 各题error人数
    error_number = [0 for i in range(question_cnt)]
    question_error = dict(zip(question_number, error_number))

    # 各题correct人数
    question_correct = question_error

    # 每位学生每题的答题情况
    summarize0 = [0 for i in range(question_cnt + 3)]
    summarize = [copy.deepcopy(summarize0) for i in range(sheet_num + 2)]

    # 每位学生的正确率
    correct_rates = [0 for i in range(sheet_num)]

    summarize[0][0] = 'Stu_ID'
    for i in range(1, 6):
        summarize[0][i] = f'Q_{i}'
    for i in range(1, sheet_num + 1):
        summarize[i][0] = f'2023xx{i:02d}'
    summarize[0][question_cnt + 1] = 'TS'
    summarize[0][question_cnt + 2] = 'CR'
    summarize[sheet_num + 1][0] = 'aver.'

    return ANSWER_KEY_SCORE, ANSWER_KEY, sheet_num, path, col, row, 0, question_cnt, scores, score_dict, question_error, summarize


if __name__ == "__main__":
    pre()
