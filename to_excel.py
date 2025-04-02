import copy
import pandas as pd
# import openpyxl


def fun(summarize, sheet_num):
    problem_num = len(summarize[0]) - 3
    # print(summarize)

    for i in range(1, problem_num + 1):
        # 每题的平均分
        summarize[sheet_num + 1][i] = f'{sum(row[i] for row in summarize[1:]) / sheet_num :.2f}'
    # 总平均分
    summarize[sheet_num + 1][problem_num + 1] = f'{sum(row[-2] for row in summarize[1:]) / sheet_num:.2f}'

    # 平均正确率
    sum_rate = 0
    for i in range(1, sheet_num + 1):
        sum_rate += float(str(summarize[i][-1])[:-1])
    summarize[-1][-1] = str(f'{sum_rate / sheet_num:.2f}') + r'%'
    data = {
        summarize[0][0]: [row[0] for row in summarize[1:]],
        summarize[0][1]: [row[1] for row in summarize[1:]],
        summarize[0][2]: [row[2] for row in summarize[1:]],
        summarize[0][3]: [row[3] for row in summarize[1:]],
        summarize[0][4]: [row[4] for row in summarize[1:]],
        summarize[0][5]: [row[5] for row in summarize[1:]],
        summarize[0][6]: [row[6] for row in summarize[1:]],
        summarize[0][7]: [row[7] for row in summarize[1:]],
    }
    # df = pd.DataFrame(data)
    #
    # # 导出到 Excel 文件
    # df.to_excel('out.xlsx', index=False)

    # print(summarize[1:][0])
    # print(summarize[1])
    df = pd.DataFrame(data)
    # 将数据框导出到Excel，使用xlsxwriter引擎
    writer = pd.ExcelWriter("./feedback/out.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    # 获取xlsxwriter的workbook和worksheet对象
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    # 设置单元格格式为居中
    cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    for row_num, row in enumerate(df.values):
        for col_num, value in enumerate(row):
            worksheet.write(row_num + 1, col_num, value, cell_format)
    # 保存Excel文件
    writer._save()
