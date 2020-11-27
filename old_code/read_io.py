import csv
import os.path as op
import numpy as np


def gen_fname(base_data, month_start, month_end, f_type):
    suff_file = '.csv'
    date_str = []
    name_str = []
    file_all = []

    month = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for m in range(month_start, month_end+1):
        m_str = '%02d' % m

        for d in range(1, month[m]+1):  
            d_str = '%02d' % d
            s_str = m_str + d_str
            date_str.append(s_str)
    
    if f_type == 'regioni':
        base_path = base_data + 'dati-regioni/'
        base_file = 'dpc-covid19-ita-regioni-2020'

    elif f_type == 'province':
        base_path = base_data + 'dati-province/'
        base_file = 'dpc-covid19-ita-province-2020'

    elif f_type == 'italia':
        base_path = base_data + 'dati-andamento-nazionale/'
        base_file = 'dpc-covid19-ita-andamento-nazionale-2020'
 
    else:
        print('Error: file type does not exist.')
        exit()

    for this_date in date_str:
        this_file = base_path + base_file + this_date + suff_file
        file_all.append(this_file)

    return file_all



def read_file(f_name):
    
    with open(f_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        all_rows = []

        for row in csv_reader:
            if line_count == 0:
                #print(f'File header: {", ".join(row)} ')
                line_count += 1
            else:
                #print(f'\t{row[3]} ({row[2]}):')
                #print(f'\t{row[6:15]}')
                all_rows.append(row)

    return all_rows
    


def read_all_files(base_data, month_start, month_end, f_type):
    print('Reading all files of type: ', f_type)

    file_all = gen_fname(base_data, month_start, month_end, f_type)
    data_all = []


    for this_file in file_all:
        if op.exists(this_file):
            data = read_file(this_file)               
            data_all.append(data)

    return data_all



def get_data_italia(data, col):
    n_pts = len(data)

    selected = np.zeros((n_pts))
    giorni = np.zeros((n_pts), dtype=int)
    counts = 0

    for i_d in range(1, n_pts+1):
        giorni[i_d-1] = i_d

    for this_data in data:

        for one_data in this_data:
            reg_num = int(one_data[2])
            reg_name = one_data[3]
            column = one_data[col]
            selected[counts] = column
            counts += 1
 
    return [giorni, selected]



def get_data_regioni(data, col):

    n_reg_tot = 20
    n_pts = len(data)

    selected = np.zeros((n_reg_tot, n_pts))
    counts = np.zeros((n_reg_tot), dtype=int)
    giorni = np.zeros((n_pts), dtype=int)

    for i_d in range(1, n_pts+1):
        giorni[i_d-1] = i_d

    for this_data in data:

        for one_data in this_data:
            reg_num = int(one_data[2])
            reg_name = one_data[3]
            column = one_data[col]
        
            i = reg_num-1
    
            # Col = 3 e' il Trentino + Bolzano
            if i != 3:
                selected[i, counts[i]] = column
                counts[i] += 1
 
    return [giorni, selected]
