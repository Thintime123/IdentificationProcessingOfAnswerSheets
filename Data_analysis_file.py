def Data_analysis_file(score_dict, sheet_num):
    score_sum = sum(int(key) * value for key, value in score_dict.items())
    # with open('./feedback/Data_analysis_file.txt', 'w') as file:
    #     file.write(f'The number of students taking the exam: {sheet_num}\n')
    #     file.write(f'average score: {score_sum / sheet_num:.2f}\n')

    stu_num = sheet_num
    perfect_num = score_dict['100']
    out_num = score_dict['80'] + score_dict['100']
    passed_num = score_dict['60'] + out_num
    failed_num = stu_num - passed_num
    ave = score_sum / stu_num

    file_name = './feedback/answering_info/overall_answering_situation.txt'
    file = open(file_name, 'w')
    file.write("Overall Analysis of Students' Answering Situations:\n\n")
    file.write(f'\tThe total number of students taking the exam: {stu_num}\n\n')
    file.write(f'\tThe number of students with full marks: {perfect_num}\n\n')
    file.write(f'\tThe number of outstanding students: {out_num}\n\n')
    file.write(f'\tThe number of students who passed the exam: {passed_num}\n\n')
    file.write(f'\tThe number of students who failed the exam: {failed_num}\n\n')
    file.write(f'\tAverage score: {ave:.2f}\n')

    file.close()


if __name__ == "__main__":
    pass
