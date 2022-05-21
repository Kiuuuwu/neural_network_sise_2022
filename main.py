import copy

import pandas as pd
import numpy as np
import random
import math

def read_from_file(filename):
    # wczytywanie csv i podzial danych na macierze
    dataset = pd.read_csv(filename)
    dataset = pd.get_dummies(dataset, columns=['Species'])  # One Hot Encoding
    values = list(dataset.columns.values)

    which_flower = dataset[values[-3:]]
    which_flower = np.array(which_flower, dtype='float32')  # macierz postaci [1, 0, 0] gdzie 1 jest w miejscu odpowiednim dla danego gatunku irysa
    input_data = dataset[values[1:-3]]      # input_data to tablica dwuwymiarowa ze wszystkimi danymi
    input_data = np.array(input_data, dtype='float32')

    return input_data, which_flower

def change_input_to_0_1_values(data_matrix):
    # funkcja zmieniajaca dane na liczby z zakresu [0, 1]
    # wzor: (x - min) / (max - min) = y
    minimum = data_matrix.min()
    maximum = data_matrix.max()
    normalized_data_matrix = data_matrix

    if (minimum == maximum):
        normalized_data_matrix[0][0] = 0
        return normalized_data_matrix


    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            normalized_data_matrix[i][j] = (data_matrix[i][j] - minimum) / (maximum - minimum)

    return normalized_data_matrix

def generate_wages(matrix_x, matrix_y, is_bias):
    # generujemy losowo wagi z przedzialu [-1; 1] "Wagi sieci, o ile nie jest ona wczytywana z pliku, mają być inicjalizowane w sposób pseudolosowy"
    # print("generate wages:")
    w = matrix_y
    if (is_bias):
        h = matrix_x + 1 # bo jeszcze waga biasu
    else:
        h = matrix_x
    matrix_wages = [[0 for x in range(h)] for y in range(w)]
    for i in range(w):
        for j in range (h):
            matrix_wages[i][j] = random.uniform(-1, 1)   # tablica 2d z wagami dla wszystkich neuronow w warstwie (waga biasu tez)
        # print(matrix_wages[i])
    return matrix_wages

    # zakomentowany kod jest dla wag dla jednego neuronu, to bylo glupie bo potrzebuje wszystkich wag i musze je gdzies trzymac, zeby je zmienaic
    # w = 1
    # h = matrix_len + 1 # bo jeszcze waga biasu
    # matrix_wages = [[0 for x in range(w)] for y in range(h)]
    # for i in range(h):
    #     matrix_wages[i] = random.uniform(-1, 1) # tablica 1d z wagami dla danego neuronu
    # return matrix_wages
    #---------------------------------------------------------------------------------------------------------------------------------------------

# def transpose(matrix):
#     rows = len(matrix)
#     columns = len(matrix[0])
#
#     matrix_T = []
#     for j in range(columns):
#         row = []
#         for i in range(rows):
#            row.append(matrix[i][j])
#         matrix_T.append(row)
#
#     return matrix_T

def elo_bec():
    # dane i wagi z przykaldu rozpisanego na kartce tylko po to zeby sprawdzic czy program dziala
    # fixed_data_matrix = [[0 for x in range(4)] for y in range(3)]

    # fixed_data_matrix = [[1,2,1,1],[2,0,-1,-2],[0,0,2,-1]]
    # fixed_expected_result_matrix = [[3,2,2,-2]]


    fixed_wages_matrix_1 = [[[0.5, 0.25, 1, 0],[0, 0.25, 0, 1]]]
    fixed_wages_matrix_2 = [[0.25,0.5,1]]

    fixed_data_matrix = np.array([[1,2,0],[2,0,0],[1,-1,2],[1,-2,-1]], dtype='float32')
    fixed_expected_result_matrix = np.array([[3],[2],[2],[-2]], dtype='float32')

    # fixed_wages_matrix_1 = [[0.5, 0.25, 1, 0], [0, 0.25, 0, 1]]
    # fixed_wages_matrix_2 = [[0.25, 0.5, 1]]
    # fixed_data_matrix = np.array([[1, 2, 0]], dtype='float32')
    # fixed_expected_result_matrix = np.array([[3]], dtype='float32')

    # fixed_wages_matrix_1 = [[0.5,0],[0.25,0.25],[1,0],[0,1]]
    # fixed_wages_matrix_2 = [[0.25],[0.5],[1]]

    return fixed_data_matrix, fixed_expected_result_matrix, fixed_wages_matrix_1, fixed_wages_matrix_2


def sum(wages_matrix, data_row, is_bias):    # data_row to jeden rzad z data, bo chce jeden przyklad, wszystkie cechy; wages_matrix to wagi dla danego nauronu
    # suma, pierwszy etap w kazdym neuronie; zwraca sume dla danego wezla i danego przykladu
    # rownie dobrze mozna wymnozyc macierze
    # wages_matrix to jednowymiarowka na ktorej sa wagi dla kolejnych cech (i biasu tez)

    # to dziala dla liczby neuronow w warstwie ukrytej od 2 do 4: -------------------------------
    result = 0
    if (is_bias):
        result += 1 * wages_matrix[0]   # to jest wporzadku
        for i in range (0, len(wages_matrix) - 1):   # '-1' bo juz ogarnelismy bias wyzej jesli jest

            result += data_row[i] * wages_matrix[i + 1] # data row to wszystkie cechy dla danego przykladu
    else:
        for i in range(0, len(wages_matrix)):
            result += data_row[i] * wages_matrix[i]  # data row to wszystkie cechy dla danego przykladu
    return result
    # --------------------------------------------------------------------------------------------
    # result = 0
    # if (bias):
    #     result = transpose(wages_matrix) * data_row
    # print()
    # print("sumy for every neuron:")
    # for i in range(0, len(matrix_of_sums)):
    #     print(matrix_of_sums[i])
    # return result

def sigmoid_function(x):
    # funkcja sigmoidalna, drugi etap w kazdym neuronie
    return 1 / (1 + math.exp(-x))

def count_neuron(data_matrix, matrix_new_wages_hidden_layers, matrix_wages_2_layer, is_bias):
    # funkcja liczaca sume i funkcje sigmoidalna w danym neuronie
    # zaleznie od danych wejsciowych (wszystkie cechy z danego przykladu i wagi)
    # w = len(matrix_wages_2_layer)  # liczba neuronow wszystkich
    # for i in range (0, len(matrix_new_wages_hidden_layers)):
    #     w += len(matrix_new_wages_hidden_layers[i])
    #
    # h = len(data_matrix)  # liczba przykladow
    # matrix_of_sigmoid_values = [[0 for x in range(h)] for y in range(w)]
    # matrix_of_sums = [[0 for x in range(h)] for y in range(w)]
    matrix_of_sigmoid_values_for_all_layers = [None] * (len(matrix_new_wages_hidden_layers) + 1) # + 1 bo warstwa wyjsciowa jeszcze (czyli tu sa wszystkie warstwe ukryte i wyjsciowa)
    matrix_of_sums_for_all_layers = [None] * (len(matrix_new_wages_hidden_layers) + 1)

    # tmp = 0
    for k in range(0, len(matrix_new_wages_hidden_layers)): # dla kazdej warswty ukrytej
        matrix_of_current_layer = copy.deepcopy(matrix_new_wages_hidden_layers[k])  # bierzemy dana warstwe


        w = len(matrix_of_current_layer)  # liczba neuronow na danej warstwie
        h = len(data_matrix)  # liczba przykladow
        matrix_of_sigmoid_values = [[0 for x in range(h)] for y in range(w)]
        matrix_of_sums = [[0 for x in range(h)] for y in range(w)]

    # print("liczba przykladow:", h)
    # print("liczba neuronow:", w)
    # print("liczba neuronow w warstwie ukrytej:", len(matrix_wages_1_layer))
    # print("liczba neuronow w warstwie wyjsciowej:", len(matrix_wages_2_layer))

        if (k > 0):
            hlp_matrix1 = [[0 for x in range(len(matrix_of_sigmoid_values_for_all_layers[k - 1]))] for y in range(len(matrix_of_sigmoid_values_for_all_layers[k - 1][0]))]
            for j in range(0, len(hlp_matrix1)):
                for i in range(0, len(hlp_matrix1[0])):
                    hlp_matrix1[j][i] = matrix_of_sigmoid_values_for_all_layers[k - 1][i][j]



        for j in range(0, len(matrix_of_current_layer)):  # sie dzieje dla wszystkich neuronow w wastwie ukrytej
            for i in range(0, len(data_matrix)):  # sie dzieje dla wszystkich przykladow
                if (k == 0):
                    result = sum(matrix_of_current_layer[j], data_matrix[i], is_bias)  # bo wagi sa takie same dla wszystkich przykladow, data_matrix[i] to jeden przyklad (wszystkie cechy)
                else:
                    result = sum(matrix_of_current_layer[j], hlp_matrix1[i], is_bias)  # bo wagi sa takie same dla wszystkich przykladow, data_matrix[i] to jeden przyklad (wszystkie cechy)
                matrix_of_sums[j][i] = result
                matrix_of_sigmoid_values[j][i] = sigmoid_function(result)
            # tmp += 1
        matrix_of_sigmoid_values_for_all_layers[k] = matrix_of_sigmoid_values
        matrix_of_sums_for_all_layers[k] = matrix_of_sums

    # stworzyc macierz z naodwrot wierszami i kolumnami dla matrix_of_sigmoid_values
    hlp_matrix = [[0 for x in range(len(matrix_of_sigmoid_values))] for y in
                  range(len(matrix_of_sigmoid_values[0]))]
    for j in range(0, len(hlp_matrix)):
        for i in range(0, len(hlp_matrix[0])):
            hlp_matrix[j][i] = matrix_of_sigmoid_values[i][j]

    w = len(matrix_wages_2_layer)  # liczba neuronow na danej warstwie
    h = len(data_matrix)  # liczba przykladow
    matrix_of_sigmoid_values = [[0 for x in range(h)] for y in range(w)]
    matrix_of_sums = [[0 for x in range(h)] for y in range(w)]

    for j in range(0, len(matrix_wages_2_layer)):  # sie dzieje dla wszystkich neuronow w wastwie wyjsciowej
        for i in range(0, len(data_matrix)):  # sie dzieje dla wszystkich przykladow
            # result = sum(matrix_wages_2_layer[j], data_matrix[i], True)

            result = sum(matrix_wages_2_layer[j], hlp_matrix[i], is_bias)
            matrix_of_sums[j][i] = result
            matrix_of_sigmoid_values[j][i] = sigmoid_function(result)
    matrix_of_sigmoid_values_for_all_layers[-1] = matrix_of_sigmoid_values   # -1 bo wrzucamy na ostatnie miejsce
    matrix_of_sums_for_all_layers[-1] = matrix_of_sums

    # print()
    # print("matrix_of_sigmoid_values_for_all_layers:")
    # for i in range(0, len(matrix_of_sigmoid_values_for_all_layers)):
    #     for j in range (0, len(matrix_of_sigmoid_values_for_all_layers[i])):
    #         print(matrix_of_sigmoid_values_for_all_layers[i][j])

    return matrix_of_sums_for_all_layers, matrix_of_sigmoid_values_for_all_layers

def error(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, is_bias):
    # funkcja liczacza blad na danym neuronie (wynik z danej iteracji na danym neuronie - to co mialo wyjsc (z danych))

    matrix_of_all_errors = [None] * len(matrix_of_sigmoid_values_for_all_layers)    # tablica 3d, piewrszy wymiar to numer warstwy

    # print()
    # print("matrix_of_sigmoid_values_for_all_layers:")
    # for i in range(0, len(matrix_of_sigmoid_values_for_all_layers)):
    #     print(matrix_of_sigmoid_values_for_all_layers[i])
    # len(matrix_of_sigmoid_values_for_all_layers) to liczba warstw (ukrytych + wyjsciowa)

    # warstwa wyjsciowa:
    w = len(matrix_wages_2_layer)  # liczba neuronow wszystkich na warstwie wyjsciowej
    h = len(matrix_of_sigmoid_values_for_all_layers[0][0])  # liczba przykladow
    matrix_of_errors = [[0 for x in range(h)] for y in range(w)]
    # matrix_wages_last_hidden_layer = matrix_of_wages_hidden_layers[-1]  # len z tego to liczba neuronow w ostatniej warstwie ukrytej
    # print("elo:", len(matrix_of_wages_hidden_layers))    #ile jest warstw ukrytych

    for j in range (0, len(matrix_wages_2_layer)):  # sie dzieje dla wszystkich neuronow w wastwie wyjsciowej   (zaczynamy od tego bo back propagation sie robi od konca)
        matrix_of_sigmoid_values_last_hidden_layer = matrix_of_sigmoid_values_for_all_layers[-2]

        for i in range(0, h):  # sie dzieje dla wszystkich przykladow
            matrix_of_errors[j][i] =  matrix_of_sigmoid_values_for_all_layers[-1][j][i] - normalized_expected_result_matrix[i][j]

    matrix_of_all_errors[-1] = matrix_of_errors

    # for dla ostatniej warstwy ukrytej:
    w = len(matrix_of_wages_hidden_layers[-1])  # liczba neuronow wszystkich na warstwie ostatbniej ukrytej
    h = len(matrix_of_sigmoid_values_for_all_layers[0][0])  # liczba przykladow
    matrix_of_errors_last_hidden_layer = [[0 for x in range(h)] for y in range(w)]

    for j in range (0, len(matrix_of_wages_hidden_layers[-1])):  # dla wszystkich neuronow w ostatniej wastwie ukrytej #todo: tak?
        for t in range(0, len(matrix_wages_2_layer)):   # dla kazdego neuronu w warstwie wyjsciowej
            for i in range(0, h):  # sie dzieje dla wszystkich przykladow
                if (is_bias):
                    linear_combination = matrix_wages_2_layer[t][j + 1] * matrix_of_errors[t][i]
                else:
                    linear_combination = matrix_wages_2_layer[t][j] * matrix_of_errors[t][i]
                matrix_of_errors_last_hidden_layer[j][i] = linear_combination * matrix_of_sigmoid_values_for_all_layers[-2][j][i] * (1 - matrix_of_sigmoid_values_for_all_layers[-2][j][i])   # todo: tu powinno byc sigmoidal balue czy suma??

    matrix_of_all_errors[-2] = matrix_of_errors_last_hidden_layer

    # for dla kazdej warstwy ukrytej poza ostatnia:
    for k in range (len(matrix_of_wages_hidden_layers) - 2, -1, -1):     # dla kazdej warstwy ukrytej poza pierwsza

        w = len(matrix_of_wages_hidden_layers[k])  # liczba neuronow wszystkich na danej warstwie
        h = len(matrix_of_sigmoid_values_for_all_layers[0][0])  # liczba przykladow
        matrix_of_errors_hidden_layers = [[0 for x in range(h)] for y in range(w)]
        for j in range (0, len(matrix_of_wages_hidden_layers[k])):  # dla wszystkich neuronow w wastwie ukrytej
            for i in range(0, h):  # sie dzieje dla wszystkich przykladow
                linear_combination = 0
                for t in range(0, len(matrix_of_wages_hidden_layers[k + 1])):  # dla kazdego neuronu w warstwie nastepnej
                    if (is_bias):
                        linear_combination += matrix_of_wages_hidden_layers[k + 1][t][j + 1] * matrix_of_all_errors[k + 1][t][i] # daje j+1 bo bias?
                    else:
                        linear_combination += matrix_of_wages_hidden_layers[k + 1][t][j] * matrix_of_all_errors[k + 1][t][i]
                matrix_of_errors_hidden_layers[j][i] = linear_combination * matrix_of_sigmoid_values_for_all_layers[k][j][i] * (1 - matrix_of_sigmoid_values_for_all_layers[k][j][i])
            # print(matrix_of_errors_hidden_layers[j])

        matrix_of_all_errors[k] = matrix_of_errors_hidden_layers

    # print("matrix_of_all_errors:")
    # for i in range (0, len(matrix_of_all_errors)):
    #     print(matrix_of_all_errors[i])

    return matrix_of_all_errors

def liczenie_trojkacikow(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_errors, data_matrix, matrix_of_sigmoid_values_for_all_layers, is_bias): # todo: jak to nazwac??
    # tworzenie macierzy takich samych jak macierze z wagami, ktore byly wczesniej
    # i zastepowanie wartosci tych wag wynikami odpowiednich rownan:
    # (wartosc danego x (rowniez biasu)) * (odpowiadajacy blad na neuronie) + to samo tyle razy ile jest x (czyli przykladow)
    # najpierw bias, ktorego nie ma na data_matrix

    matrix_of_deltas_hidden_layer = [None] * (len(matrix_of_sigmoid_values_for_all_layers) - 1)   # tablica 3d, piewrszy wymiar to numer warstwy
    matrix_deltas_2_layer = copy.deepcopy(matrix_wages_2_layer)
    result = 0



    # UKRYTE - pierwsza warstwa ukryta
    matrix_deltas_1_layer = copy.deepcopy(matrix_of_wages_hidden_layers[0]) # bo ta macierz ma miec takie same wymiary
    if (is_bias):
        for j in range (0, len(matrix_of_wages_hidden_layers[0])):  # liczba neuronow w warwtie ukrytej
            result_bias = 0
            for i in range (0, len(data_matrix)):   # (wykona sie tyle razy ile jest przyklad
                result_bias += 1 * matrix_of_all_errors[0][j][i]
                # print("matrix_of_errors[j][i]",matrix_of_errors[j][i])
            matrix_deltas_1_layer[j][0] = result_bias    # delta dla biasu na warstwie 1

    # dla wszystkich nauronow na warstwie pierwszej i dla wszystkich przypadkow wszytskich cech liczymy delty:
    for j in range (0, len(matrix_of_wages_hidden_layers[0])):  # dla kazdego neuronu w danej warstwie
        for k in range(0, len(data_matrix[0])):  # dla kazdej cechy
            for i in range(0, len(data_matrix)):  # dla kaxdego przykladu
                result += data_matrix[i][k] * matrix_of_all_errors[0][j][i]   # matrix_of_errors[i][j] ma mi wziac kazdy przyklad pokolei dla danego neuronu
            if (is_bias):
                matrix_deltas_1_layer[j][k + 1] = result  # delta dla zapisywana w macierzy w wierszu j od kolumny k + 1
            else:
                matrix_deltas_1_layer[j][k] = result  # delta dla zapisywana w macierzy w wierszu j od kolumny k
            result = 0

    matrix_of_deltas_hidden_layer[0] = matrix_deltas_1_layer


    # UKRYTE - reszta warstw ukrytych
    for m in range(1, len(matrix_of_wages_hidden_layers)):  # dla kazdej warstwy ukrytej
        matrix_deltas_1_layer = copy.deepcopy(matrix_of_wages_hidden_layers[m])  # bo ta macierz ma miec takie same wymiary
        if (is_bias):
            for j in range(0, len(matrix_of_wages_hidden_layers[m])):  # liczba neuronow w warwtie ukrytej
                result_bias = 0
                for i in range(0, len(matrix_of_sigmoid_values_for_all_layers[m][0])):  # (wykona sie tyle razy ile jest przykladow)
                    result_bias += 1 * matrix_of_all_errors[m][j][i]
                    # print("matrix_of_errors[j][i]",matrix_of_errors[j][i])
                matrix_deltas_1_layer[j][0] = result_bias  # delta dla biasu na warstwie 1

        # dla wszystkich nauronow na warstwie pierwszej i dla wszystkich przypadkow wszytskich cech liczymy delty:
        for j in range(0, len(matrix_of_wages_hidden_layers[m])):  # dla kazdego neuronu w danej warstwie
            for k in range(0, len(matrix_of_sigmoid_values_for_all_layers[m - 1])):  # dla kazdego neuronu w poprzedniej warswtie
                for i in range(0, len(matrix_of_sigmoid_values_for_all_layers[m][0])):  # dla kaxdego przykladu
                    result += matrix_of_sigmoid_values_for_all_layers[m - 1][k][i] * matrix_of_all_errors[m][j][i]  # matrix_of_errors[i][j] ma mi wziac kazdy przyklad pokolei dla danego neuronu
                if (is_bias):
                    matrix_deltas_1_layer[j][k + 1] = result  # delta dla zapisywana w macierzy w wierszu j od kolumny k + 1
                else:
                    matrix_deltas_1_layer[j][k] = result  # delta dla zapisywana w macierzy w wierszu j od kolumny k
                result = 0

        matrix_of_deltas_hidden_layer[m] = matrix_deltas_1_layer

    # stworzyc macierz z naodwrot wierszami i kolumnami dla matrix_of_sigmoid_values dla ostatniej warstwy ukrytej
    matrix_of_sigmoid_values_last_hidden_layer = matrix_of_sigmoid_values_for_all_layers[-2]
    hlp_matrix = [[0 for x in range(len(matrix_of_sigmoid_values_last_hidden_layer))] for y in range(len(matrix_of_sigmoid_values_last_hidden_layer[0]))]
    for j in range(0, len(hlp_matrix)):
        for i in range(0, len(hlp_matrix[0])):
            hlp_matrix[j][i] = matrix_of_sigmoid_values_last_hidden_layer[i][j]


    # WYJSCIOWA
    result = 0
    if (is_bias):
        result_bias = 0
        for j in range(0, len(matrix_wages_2_layer)):
            for i in range(0, len(matrix_of_sigmoid_values_last_hidden_layer)):  # (wykona sie tyle razy ile jest neuronow na ostatniej warswie ukrytej)
                result_bias += 1 * matrix_of_all_errors[-1][j][i]  # todo: tu jest ten sam blad
            matrix_deltas_2_layer[j][0] = result_bias  # delta dla biasu na warstwie 2

    # dla wszystkich nauronow na warstwie wyjsciowej i dla wszystkich przypadkow wszytskich cech liczymy delty:
    for j in range (0, len(matrix_wages_2_layer)):  # dla kazdego neuronu warwty wyjsciowej
        for k in range(0, len(matrix_of_wages_hidden_layers[-1])):  #DLA LICZBY NEURONOW NA WARSTWIE ostatniej z ukrytych (bias dodaje pozniej)
            for i in range(0, len(hlp_matrix)):  # (wykona sie tyle razy ile jest przykladow)
                result += hlp_matrix[i][k]  \
                          * matrix_of_all_errors[-1][j][i]
            if (is_bias):
                matrix_deltas_2_layer[j][k + 1] = result  # delta dla zapisywana w macierzy w wierszu j od kolumny k + 1
            else:
                matrix_deltas_2_layer[j][k] = result  # delta dla zapisywana w macierzy w wierszu j od kolumny k
            result = 0
    # print()
    # print("delty warstwa wyjsciowa:")
    # for i in range(0, len(matrix_wages_2_layer)):
    #     print(matrix_wages_2_layer[i])

    return matrix_of_deltas_hidden_layer, matrix_deltas_2_layer



def liczenie_dija(matrix_of_deltas_hidden_layers, matrix_deltas_2_layer, data_matrix): # todo: jak to nazwac??   # len(data_matrix) to liczba przykladow
    # bierzemy kazdy trojkacik wyliczony w funkcji liczenie_trojkacikow i mnozymy * (1 / liczba przykladow w danych)
    matrix_of_all_mean_deltas_hidden_layer = [None] * len(matrix_of_deltas_hidden_layers)    # tablica 3d, piewrszy wymiar to numer warstwy
    # ^ dlaczego to jest liczba wszystkich warstw a nie jedynie ukrytych?

    # print("deltaski")
    # for i in range (0, len(matrix_of_deltas_hidden_layers)):
    #     print(matrix_of_deltas_hidden_layers[i])

    #  UKRYTE
    for k in range (0, len(matrix_of_deltas_hidden_layers)):    # dla kazdej warstwy
        matrix_deltas_1_layer = matrix_of_deltas_hidden_layers[k]   # bierzemy delty dla danej warstwy
        matrix_of_mean_deltas_1_layer = matrix_of_deltas_hidden_layers[k]
        for i in range (0, len(matrix_deltas_1_layer[0])):  # wiersze
            for j in range (0, len(matrix_deltas_1_layer)):     # kolumny
                matrix_of_mean_deltas_1_layer[j][i] = matrix_deltas_1_layer[j][i] * (1 / len(data_matrix))
        matrix_of_all_mean_deltas_hidden_layer[k] = matrix_of_mean_deltas_1_layer

    # WYJSCIOWA
    matrix_mean_deltas_2_layer = matrix_deltas_2_layer
    for i in range (0, len(matrix_deltas_2_layer[0])):  # wiersze
        for j in range (0, len(matrix_deltas_2_layer)):     # kolumny
            matrix_mean_deltas_2_layer[j][i] = matrix_deltas_2_layer[j][i] * (1 / len(data_matrix))

    # print()
    # print("dije warstwa ukryta:")
    # for i in range(0, len(matrix_dij_1_layer)):
    #     print(matrix_dij_1_layer[i])
    # print()
    # print("dije warstwa wyjsciowa:")
    # for i in range(0, len(matrix_mean_deltas_2_layer)):
    #     print(matrix_mean_deltas_2_layer[i])
    # print()

    return matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer

def gradient_descent(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer, alfa):
    # liczymy nowe wagi, ktorymi zastapimy stare wagi przy nastepnej iteracji
    # (nowa waga) = (stara waga) - (alfa) *

    # UKRYTE
    matrix_of_all_new_wages_hidden_layer = [None] * len(matrix_of_wages_hidden_layers)    # tablica 3d, piewrszy wymiar to numer warstwy

    for k in range (0, len(matrix_of_wages_hidden_layers)):
        matrix_old_wages_1_layer = matrix_of_wages_hidden_layers[k]
        matrix_new_wages_1_layer = matrix_of_wages_hidden_layers[k]
        for i in range(0, len(matrix_old_wages_1_layer[0])):  # wiersze
            for j in range(0, len(matrix_old_wages_1_layer)):  # kolumny
                matrix_new_wages_1_layer[j][i] = matrix_old_wages_1_layer[j][i] - (alfa * matrix_of_all_mean_deltas_hidden_layer[k][j][i])

        matrix_of_all_new_wages_hidden_layer[k] = matrix_new_wages_1_layer


    # WYJSCIOWE
    matrix_new_wages_2_layer = matrix_wages_2_layer
    for i in range(0, len(matrix_wages_2_layer[0])):  # wiersze
        for j in range(0, len(matrix_wages_2_layer)):  # kolumny
            matrix_new_wages_2_layer[j][i] = matrix_wages_2_layer[j][i] - (alfa * matrix_mean_deltas_2_layer[j][i])

    # print("nowe wagi warstwa ukryta:")
    # for i in range(0, len(matrix_new_wages_1_layer)):
    #     print(matrix_new_wages_1_layer[i])
    # print()
    # print("nowe wagi warstwa wyjsciowa:")
    # for i in range(0, len(matrix_new_wages_2_layer)):
    #     print(matrix_new_wages_2_layer[i])

    return matrix_of_all_new_wages_hidden_layer, matrix_new_wages_2_layer

def back_propagation(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, normalized_data_matrix, alfa, is_bias):
    # funkcja skladajaca do siebie funckje: error, liczenie_trojkacikow, liczenie_dija i gradient_descent

    # print()
    # print("matrix_of_wages_hidden_layers:")
    # for i in range(0, len(matrix_of_wages_hidden_layers)):
    #     print(matrix_of_wages_hidden_layers[i])

    matrix_of_all_errors = error(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, is_bias)
    matrix_of_deltas_hidden_layers, matrix_deltas_2_layer = liczenie_trojkacikow(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_errors, normalized_data_matrix, matrix_of_sigmoid_values_for_all_layers, is_bias)
    matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer = liczenie_dija(matrix_of_deltas_hidden_layers, matrix_deltas_2_layer, normalized_data_matrix)
    matrix_of_all_new_wages_hidden_layer, matrix_new_wages_2_layer = gradient_descent(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer, alfa)

    return matrix_of_all_new_wages_hidden_layer, matrix_new_wages_2_layer

def change_0_1_values_back_to_normal(data_matrix, normalized_matrix_for_all_layers):
    # funkcja zmieniajaca liczby z zakresu [0, 1] z powrotem na 'normalne' po wszystkich iteracjach (na pewno po wszystkich??)
    # wzor: x = y * (max - min) + min
    minimum = data_matrix.min()
    maximum = data_matrix.max()

    denormalized_matrix_for_all_layers = [None] * len(normalized_matrix_for_all_layers)    # tablica 3d, piewrszy wymiar to numer warstwy

    for k in range (0, len(normalized_matrix_for_all_layers)):
        normalized_matrix = normalized_matrix_for_all_layers[k]
        denormalized_matrix = normalized_matrix

        for i in range(len(normalized_matrix)):
            for j in range(len(normalized_matrix[i])):
                denormalized_matrix[i][j] = normalized_matrix[i][j] * (maximum - minimum) + minimum # to minimum i maximum to poprawne wartosci?

        denormalized_matrix_for_all_layers[k] = denormalized_matrix

    # print()
    # print("denormalized:")
    # for i in range(0, len(denormalized_matrix)):
    #     print(matrix_new_wages_2_layer[i])

    return denormalized_matrix_for_all_layers

def mean_square_error(matrix_of_sigmoid_values_for_all_layers, normalized_expected_result_matrix, matrix_new_wages_hidden_layers):   #matrix_new_wages_hidden_layers to jest mi potrzebne bo matrix_of_sigmoid_values_for_all_layers
    # to wyniki na kazdym neuronie, rowniez na tych z warstwy ukrytej, ktore mnie w tej funkcji nie interesuja
    # od kazdej wartosci spodziewanej odejmujemy kazda wartosc otrzymana, wszystkie roznice podnosimy do kwadratu, nastepnie te kwadraty sumujemy
    # i na końcu dzielimy przez liczbe przypadkow testowych
    nr_of_neurons_layer_1 = len(matrix_new_wages_hidden_layers[-1])
    result = 0
    for i in range (0, len(matrix_of_sigmoid_values_for_all_layers[0])): # to sie dzieje dla kazdej kolumny (kazdego przykladu)
        for j in range (len(matrix_of_sigmoid_values_for_all_layers[-1])): # to sie dzieje dla kazdego neuronu z warstwy wyjsciowej
            result += (normalized_expected_result_matrix[i][j] -
                       matrix_of_sigmoid_values_for_all_layers[-1][j][i]) ** 2
            # print()
            # print("co ma wyjsc vs co jest:")
            # print(expected_result_matrix[i][j - nr_of_neurons_layer_1], matrix_of_outcomes[j][i])
    result = result / len(matrix_of_sigmoid_values_for_all_layers[0])
    print()
    print("funkcja kosztu:",result)

# def write_to_file:
#     # zapisuje otrzymane wyniki do pliku
#
def learning(normalized_data_matrix, matrix_of_hidden_layers, matrix_wages_2_layer, normalized_expected_result_matrix, alfa, nr_of_iterations, is_bias):
    # skleja ze soba funkcje do nauki

    # CZESC DO TESTOW: -------------------------------------------------------------------------------------------------------------------------------
    # normalized_data_matrix, expected_result_matrix, matrix_of_hidden_layers, matrix_wages_2_layer = elo_bec()  # to wybombic jak sie przestane bawic
    # normalized_data_matrix = change_input_to_0_1_values(normalized_data_matrix)
    # normalized_expected_result_matrix = change_input_to_0_1_values(expected_result_matrix)
    # alfa = 0.5
    # ------------------------------------------------------------------------------------------------------------------------------------------------

    # print()
    # print("normalized_data_matrix:")
    # for i in range(0, len(normalized_data_matrix)):
    #     print(normalized_data_matrix[i])

    matrix_new_wages_hidden_layers = copy.deepcopy(matrix_of_hidden_layers) # taka sama tablica 3D, po to by trzymac w niej nowe wagi
    matrix_new_wages_2_layer = copy.deepcopy(matrix_wages_2_layer)

    for i in range (0, nr_of_iterations):
        # co sie powinno dziac w kazdej iteracji:
        # - licze sumy i wartosci funkcji sigmoid na kazdym neuronie w warstwie ukrytej
        # - robie to samo na warstwie wyjsciowej (korzystajac z wynikow z wyzej)
        # - licze errory (od konca, bo to back propagation helol i potrzebuje errorow z warstwy wyjsciowej by wiedziec jakie sa w warstwie ukrytej)
        # - licze trojkaciki ktore sa jedynie krokiem do policzenia sredniegio bledu
        # - licze Dije (srednie bledy) -> trojkacik * (1/liczba_przykladow)
        # - licze gradient descent (nowe wagi) na podstawie starej wagi, alfy i Dija
        # - mean_square_error jest mi potrzebny do sprawdzania funkcji kosztu (dzieki niej wiemy, czy siec sie uczy)
        matrix_of_sums_for_all_layers, matrix_of_sigmoid_values_for_all_layers = count_neuron(normalized_data_matrix, matrix_new_wages_hidden_layers, matrix_new_wages_2_layer, is_bias)
        matrix_new_wages_hidden_layers, matrix_new_wages_2_layer = back_propagation(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, normalized_data_matrix, alfa, is_bias)
        mean_square_error(matrix_of_sigmoid_values_for_all_layers, normalized_expected_result_matrix, matrix_new_wages_hidden_layers)

    matrix_of_sigmoid_values_for_all_layers = change_0_1_values_back_to_normal(data_matrix, matrix_of_sigmoid_values_for_all_layers)
    print()
    print("sigmoid_values for every neuron:")
    for j in range (0, len(matrix_of_sigmoid_values_for_all_layers)):
        for i in range(0, len(matrix_of_sigmoid_values_for_all_layers[j])):
            print(matrix_of_sigmoid_values_for_all_layers[j][i])


# def testing:
#     # sklada razem funkcje az do wyliczenia bledow, nastepnie przechodzi do change_0_1_values_back_to_normal
    # 1. wzorzec treningowy podawany jest na wejścia sieci,
    # 2. odbywa się jego propagacja w przód
    # 3. na podstawie wartości odpowiedzi wygenerowanej przez sieć oraz wartości pożądanego wzorca odpowiedzi następuje wyznaczenie błędów





# main / user communication
how_many_hidden_layers = (int)(input("podaj liczbe warstw ukrytych: "))
is_bias_input = input("z biasem (t), czy bez biasu (n): ")
nr_of_iterations = (int)(input("podaj liczbe iteracji: "))

if (is_bias_input == "t"):
    is_bias = True
else:
    is_bias = False

data_matrix, expected_result_matrix = read_from_file('Iris.csv')  # zbior irysow
alfa = random.uniform(0, 1) # todo: opowiedzni zakres??
normalized_data_matrix = change_input_to_0_1_values(data_matrix)
normalized_expected_result_matrix = change_input_to_0_1_values(expected_result_matrix)

# normalized_data_matrix = data_matrix

# print("liczba cech: ",len(normalized_data_matrix[0]))
# print()
# print("data matrix:")
# for i in range (0, 150):
#     print(normalized_data_matrix[i])
matrix_of_hidden_layers = [None] * how_many_hidden_layers   # lista naktora wrzucam tablice 2d wag, a wiec tablica 3d, pierwszy wymiar: ktora warstwa ukryta, drugi wymiar: ktory neuron, trzeci wymiar: ktora waga
for i in range (0, how_many_hidden_layers):
    print("podaj liczbe neuronow w warstwie ukrytej", i, ": ")
    nr_of_neurons_hidden_layer = (int)(input())



    if i == 0:    # pierwsza warstwa bierze info z data zamiast poprzedniej warstwy
        # print("liczba cech",len(normalized_data_matrix[0]))
        matrix_of_hidden_layers[i] = generate_wages(len(normalized_data_matrix[0]), nr_of_neurons_hidden_layer, is_bias)  # (liczba cech - neuronow w warstwie wejsciowej (bies dodajemy wewnatrz funkcji, spokojnie)) x (liczba neuronow w tej warstwie) NOTE: W TYM KODZIE ZAWSZE BEDA WARSTWY: 0; 1; 2
        # print(matrix_of_hidden_layers[i][0])
        # print(matrix_of_hidden_layers[i][1])

    else:
        # print("i:",i)
        # print("liczba neuronow w warstwie poprzedniej",len(matrix_of_hidden_layers[i - 1]))   # liczba neuronow w warstwie poprzedniej
        # print("liczba wag przed poprzednia warstwa",len(matrix_of_hidden_layers[i - 1][0]))   # liczba wag przed poprzednia warstwa
        # czy to nie jest przypadkiem tak ze tablica wag jest na odwrot i musialabym wziac [i - 1][][0] zeby bylo git? a tak si enie da (git znaczy ze rzad wag jeden, dla jednego neuronu)
        # if(is_bias):
        matrix_of_hidden_layers[i] = generate_wages(len(matrix_of_hidden_layers[i - 1]), nr_of_neurons_hidden_layer, is_bias)  # (liczba cech - neuronow w warstwie wejsciowej (bies dodajemy wewnatrz funkcji, spokojnie)) x (liczba neuronow w tej warstwie) NOTE: W TYM KODZIE ZAWSZE BEDA WARSTWY: 0; 1; 2
        # else:
        #     matrix_of_hidden_layers[i] = generate_wages(len(matrix_of_hidden_layers[i - 1][0]), nr_of_neurons_hidden_layer, is_bias)  # (liczba cech - neuronow w warstwie wejsciowej (bies dodajemy wewnatrz funkcji, spokojnie)) x (liczba neuronow w tej warstwie) NOTE: W TYM KODZIE ZAWSZE BEDA WARSTWY: 0; 1; 2
matrix_wages_2_layer = generate_wages(nr_of_neurons_hidden_layer, 3, is_bias)  # (liczba neuronow w warswie ukrytej) x (liczba neuronow w tej warstwie - 3, bo sa 3 rodzaje kwiatkow)
learning(normalized_data_matrix, matrix_of_hidden_layers, matrix_wages_2_layer, normalized_expected_result_matrix, alfa, nr_of_iterations, is_bias)

# print()
# for i in range (0, 150):
#     print(expected_result_matrix[i]) # tablica 2d, np w [0][0] jest 1, czyli dla pierwszego zestawu danych pierwszy kwiatek jest odpowiedzia

# skoro mamy juz nowe wagi, to w  nastepnej itearacji to ich uzywamy, therefore dostajemy mniejszy blad; wyniki to zawartosc matrix_of_sigmoid_values dla warstwy wyjsciowej chybaaaa??
# powtarzam kod zadana liczbe iteracji lub dopoki blad nie bedzie odpowiednio niewielki