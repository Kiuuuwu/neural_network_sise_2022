import copy
import pandas as pd
import numpy as np
import random
import math
import csv

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

    fixed_wages_matrix_1 = [[[0.5, 0.25, 1, 0],[0, 0.25, 0, 1]]]
    fixed_wages_matrix_2 = [[0.25,0.5,1]]

    fixed_data_matrix = np.array([[1,2,0],[2,0,0],[1,-1,2],[1,-2,-1]], dtype='float32')
    fixed_expected_result_matrix = np.array([[3],[2],[2],[-2]], dtype='float32')

    return fixed_data_matrix, fixed_expected_result_matrix, fixed_wages_matrix_1, fixed_wages_matrix_2

def for_autoencoder_data():
    # dane i wagi z przykaldu rozpisanego na kartce tylko po to zeby sprawdzic czy program dziala

    # fixed_wages_matrix_1 = [[[0.5, 0.25, 1, 0],[0, 0.25, 0, 1]]]
    # fixed_wages_matrix_2 = [[0.25,0.5,1]]

    fixed_data_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype='float32')
    fixed_expected_result_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype='float32')

    return fixed_data_matrix, fixed_expected_result_matrix


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

    return matrix_of_sums_for_all_layers, matrix_of_sigmoid_values_for_all_layers

def error(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, is_bias):
    # funkcja liczacza blad na danym neuronie (wynik z danej iteracji na danym neuronie - to co mialo wyjsc (z danych))

    matrix_of_all_errors = [None] * len(matrix_of_sigmoid_values_for_all_layers)    # tablica 3d, piewrszy wymiar to numer warstwy

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

        matrix_of_all_errors[k] = matrix_of_errors_hidden_layers

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

    return matrix_of_deltas_hidden_layer, matrix_deltas_2_layer



def liczenie_dija(matrix_of_deltas_hidden_layers, matrix_deltas_2_layer, data_matrix): # todo: jak to nazwac??   # len(data_matrix) to liczba przykladow
    # bierzemy kazdy trojkacik wyliczony w funkcji liczenie_trojkacikow i mnozymy * (1 / liczba przykladow w danych)
    matrix_of_all_mean_deltas_hidden_layer = [None] * len(matrix_of_deltas_hidden_layers)    # tablica 3d, piewrszy wymiar to numer warstwy

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

    return matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer

def gradient_descent(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer, alfa, mu, momentum_hidden_layers, momentum_output_layers):
    # liczymy nowe wagi, ktorymi zastapimy stare wagi przy nastepnej iteracji
    # (nowa waga) = (stara waga) - (alfa) *

    # UKRYTE
    matrix_of_all_new_wages_hidden_layer = [None] * len(matrix_of_wages_hidden_layers)    # tablica 3d, piewrszy wymiar to numer warstwy

    for k in range (0, len(matrix_of_wages_hidden_layers)):
        matrix_old_wages_1_layer = matrix_of_wages_hidden_layers[k]
        matrix_new_wages_1_layer = matrix_of_wages_hidden_layers[k]
        for i in range(0, len(matrix_old_wages_1_layer[0])):  # wiersze
            for j in range(0, len(matrix_old_wages_1_layer)):  # kolumny
                # matrix_new_wages_1_layer[j][i] = matrix_old_wages_1_layer[j][i] - (alfa * matrix_of_all_mean_deltas_hidden_layer[k][j][i])
                momentum_hidden_layers[k][j][i] = momentum_hidden_layers[k][j][i] * mu - (alfa * matrix_of_all_mean_deltas_hidden_layer[k][j][i]) #todo: poprawnie?
                matrix_new_wages_1_layer[j][i] = matrix_old_wages_1_layer[j][i] + momentum_hidden_layers[k][j][i]

        matrix_of_all_new_wages_hidden_layer[k] = matrix_new_wages_1_layer


    # WYJSCIOWE
    matrix_new_wages_2_layer = matrix_wages_2_layer
    for i in range(0, len(matrix_wages_2_layer[0])):  # wiersze
        for j in range(0, len(matrix_wages_2_layer)):  # kolumny
            # matrix_new_wages_2_layer[j][i] = matrix_wages_2_layer[j][i] - (alfa * matrix_mean_deltas_2_layer[j][i])
            momentum_output_layers[j][i] = momentum_output_layers[j][i] * mu - (alfa * matrix_mean_deltas_2_layer[j][i])  # todo: poprawnie?
            matrix_new_wages_2_layer[j][i] = matrix_wages_2_layer[j][i] + momentum_output_layers[j][i]

    return matrix_of_all_new_wages_hidden_layer, matrix_new_wages_2_layer

def back_propagation(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, normalized_data_matrix, alfa, is_bias, mu, momentum_hidden_layers, momentum_output_layers):
    # funkcja skladajaca do siebie funckje: error, liczenie_trojkacikow, liczenie_dija i gradient_descent

    matrix_of_all_errors = error(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, is_bias)
    matrix_of_deltas_hidden_layers, matrix_deltas_2_layer = liczenie_trojkacikow(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_errors, normalized_data_matrix, matrix_of_sigmoid_values_for_all_layers, is_bias)
    matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer = liczenie_dija(matrix_of_deltas_hidden_layers, matrix_deltas_2_layer, normalized_data_matrix)
    matrix_of_all_new_wages_hidden_layer, matrix_new_wages_2_layer = gradient_descent(matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_all_mean_deltas_hidden_layer, matrix_mean_deltas_2_layer, alfa, mu, momentum_hidden_layers, momentum_output_layers)

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

    result = result / len(matrix_of_sigmoid_values_for_all_layers[0])
    # print()
    # print("funkcja kosztu:",result)
    return result

def write_to_file_learning(mean_errors_array, matrix_new_wages_hidden_layers, matrix_new_wages_2_layer):
    # zapisuje otrzymane wyniki do pliku; osobno bledy co dana liczbe iteracji i wyuczone wagi

    f = open('mean_square_errors.csv', 'w')
    writer = csv.writer(f)
    # print("len(mean_errors_array)", len(mean_errors_array), mean_errors_array[0])
    # for i in range (0, len(mean_errors_array)):
    #     print(mean_errors_array[i])
    writer.writerow(mean_errors_array)
    f.close()

    f = open('learned_wages_hidden_layers.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(matrix_new_wages_hidden_layers)
    f.close()

    f = open('learned_wages_output_layer.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(matrix_new_wages_2_layer)
    f.close()

def write_to_file_testing(normalized_data_matrix, mean_error, normalized_expected_result_matrix, matrix_of_all_errors,
                          matrix_of_sigmoid_values_for_all_layers, matrix_wages_2_layer, matrix_of_wages_hidden_layers):

    f = open('from_testing.csv', 'w')

    writer = csv.writer(f)
    str = "input data:"
    array = []
    array.append(str)
    writer.writerow(array)
    for i in range(0, len(normalized_data_matrix)):
        writer.writerow(normalized_data_matrix[i])

    str = "mean error:"
    array = []
    array.append(str)
    writer.writerow(array)
    array = []
    array.append(mean_error)
    writer.writerow(array)

    str = "expected results:"
    array = []
    array.append(str)
    writer.writerow(array)
    for i in range (0, len(normalized_expected_result_matrix)):
        writer.writerow(normalized_expected_result_matrix[i])

    str = "errors for every neuron:"
    array = []
    array.append(str)
    writer.writerow(array)
    for i in range(0, len(matrix_of_all_errors)):
        for j in range (0, len(matrix_of_all_errors[i])):
            writer.writerow(matrix_of_all_errors[i][j])

    str = "outputs on every neuron:"
    array = []
    array.append(str)
    writer.writerow(array)
    for i in range(0, len(matrix_of_sigmoid_values_for_all_layers)):
        for j in range (0, len(matrix_of_sigmoid_values_for_all_layers[i])):
            writer.writerow(matrix_of_sigmoid_values_for_all_layers[i][j])

    str = "wages of output neurons:"
    array = []
    array.append(str)
    writer.writerow(array)
    for i in range (0, len(matrix_wages_2_layer)):
        writer.writerow(matrix_wages_2_layer[i])

    str = "wages of hidden neurons:"
    array = []
    array.append(str)
    writer.writerow(array)
    for i in range(0, len(matrix_of_wages_hidden_layers)):
        for j in range(0, len(matrix_of_wages_hidden_layers[i])):
            writer.writerow(matrix_of_wages_hidden_layers[i][j])

    f.close()

def read_wages_from_file(filename):
    matric_3d_part1 = []
    matric_3d_part2 = []

    result_list = []
    file = open(filename, newline='')
    result_list = list(csv.reader(file))
    # print("result_list")
    # print(result_list)
    result_2D=np.array(result_list, dtype='object')
    # print("result_2D")
    # print(result_2D)

    for i in range (0, len(result_2D)):
        if len(result_2D[i]) != 0:
            matric_3d_part2 = []
            for j in range (0, len(result_2D[i])):
                tmp = result_2D[i][j]   # string
                if '[' in tmp:
                    tmp = tmp[1:-1:1]
                    tmp1 = tmp.split(',')
                    tmp1 = list(map(float, tmp1))
                    matric_3d_part2.append(tmp1)
                else:
                    matric_3d_part2.append(float(tmp))
            matric_3d_part1.append(matric_3d_part2)




    # print("eksperyment:", matric_3d_part1[0][0])
    return matric_3d_part1


def learning(normalized_data_matrix, matrix_of_wages_hidden_layers, matrix_wages_2_layer, normalized_expected_result_matrix, alfa, nr_of_iterations, is_bias, is_shuffle, mu, variant, nr_of_neurons_hidden_layer):
    # skleja ze soba funkcje do nauki

    # CZESC DO TESTOW: -------------------------------------------------------------------------------------------------------------------------------
    # normalized_data_matrix, expected_result_matrix, matrix_of_hidden_layers, matrix_wages_2_layer = elo_bec()  # to wybombic jak sie przestane bawic
    # normalized_data_matrix = change_input_to_0_1_values(normalized_data_matrix)
    # normalized_expected_result_matrix = change_input_to_0_1_values(expected_result_matrix)
    # alfa = 0.5
    # ------------------------------------------------------------------------------------------------------------------------------------------------


    if (variant == 2):
        fixed_data_matrix, fixed_expected_result_matrix = for_autoencoder_data()
        normalized_data_matrix = change_input_to_0_1_values(fixed_data_matrix)
        normalized_expected_result_matrix = change_input_to_0_1_values(fixed_expected_result_matrix)
        matrix_of_wages_hidden_layers[0] = generate_wages(len(normalized_data_matrix[0]), nr_of_neurons_hidden_layer, is_bias)
        matrix_wages_2_layer = generate_wages(nr_of_neurons_hidden_layer, 4, is_bias)  # (liczba neuronow w warswie ukrytej) x (liczba neuronow w tej warstwie - 3, bo sa 3 rodzaje kwiatkow)

    # tworze macierz momentum (na starcie one maja wartosci 0, sa zmieniane pozniej):
    momentum_hidden_layers = copy.deepcopy(matrix_of_wages_hidden_layers)
    momentum_output_layers = copy.deepcopy(matrix_wages_2_layer)

    for i in range (0, len(momentum_hidden_layers)):
        for j in range (0, len(momentum_hidden_layers[i])):
            for k in range(0, len(momentum_hidden_layers[i][j])):
                momentum_hidden_layers[i][j][k] = 0

    for i in range (0, len(momentum_output_layers)):
        for j in range (0, len(momentum_output_layers[0])):
            momentum_output_layers[i][j] = 0

    matrix_new_wages_hidden_layers = copy.deepcopy(matrix_of_wages_hidden_layers) # taka sama tablica 3D, po to by trzymac w niej nowe wagi
    matrix_new_wages_2_layer = copy.deepcopy(matrix_wages_2_layer)

    mean_errors_array = []

    for i in range (0, nr_of_iterations):
        # co sie powinno dziac w kazdej iteracji:
        # - licze sumy i wartosci funkcji sigmoid na kazdym neuronie w warstwie ukrytej
        # - robie to samo na warstwie wyjsciowej (korzystajac z wynikow z wyzej)
        # - licze errory (od konca, bo to back propagation helol i potrzebuje errorow z warstwy wyjsciowej by wiedziec jakie sa w warstwie ukrytej)
        # - licze trojkaciki ktore sa jedynie krokiem do policzenia sredniegio bledu
        # - licze Dije (srednie bledy) -> trojkacik * (1/liczba_przykladow)
        # - licze gradient descent (nowe wagi) na podstawie starej wagi, alfy i Dija
        # - mean_square_error jest mi potrzebny do sprawdzania funkcji kosztu (dzieki niej wiemy, czy siec sie uczy)

        if (is_shuffle == 2):
            # zmieniamy kolejnosc wierszy z data
            hlp_matrix = [[0 for x in range(len(normalized_data_matrix))] for y in
                          range(len(normalized_data_matrix[0]))]
            for j in range(0, len(hlp_matrix)):
                for i in range(0, len(hlp_matrix[0])):
                    hlp_matrix[j][i] = normalized_data_matrix[i][j]

            hlp_matrix_1 = [[0 for x in range(len(normalized_expected_result_matrix))] for y in
                          range(len(normalized_expected_result_matrix[0]))]
            for j in range(0, len(hlp_matrix_1)):
                for i in range(0, len(hlp_matrix_1[0])):
                    hlp_matrix_1[j][i] = normalized_expected_result_matrix[i][j]

            df = pd.DataFrame({'trait_1':hlp_matrix[0],
                               'trait_2':hlp_matrix[1],
                               'trait_3':hlp_matrix[2],
                               'trait_4':hlp_matrix[3],
                               'result_1':hlp_matrix_1[0],
                               'result_2':hlp_matrix_1[1],
                               'result_3':hlp_matrix_1[2]})
            print(df.to_string())
            df_shuffled = df.sample(frac=1).reset_index(drop=True)
            shuffled_matrix = pd.DataFrame.to_numpy(df_shuffled)

            for j in range (0, len(shuffled_matrix)):
                for w in range (0, len(shuffled_matrix[j]) - 3):    # bo 3 ostatnie kolumny to dane
                    normalized_data_matrix[j][w] = shuffled_matrix[j][w]
                print()
                print(normalized_data_matrix[j])

            for j in range (0, len(shuffled_matrix)):
                for w in range (len(shuffled_matrix[j]) - 3, len(shuffled_matrix[j])):    # bo 3 ostatnie kolumny to dane
                    normalized_expected_result_matrix[j][w - (len(shuffled_matrix[j]) - 3)] = shuffled_matrix[j][w]
                print()
                print(normalized_expected_result_matrix[j])



        matrix_of_sums_for_all_layers, matrix_of_sigmoid_values_for_all_layers = count_neuron(normalized_data_matrix, matrix_new_wages_hidden_layers, matrix_new_wages_2_layer, is_bias)
        matrix_new_wages_hidden_layers, matrix_new_wages_2_layer = back_propagation(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, normalized_data_matrix, alfa, is_bias, mu, momentum_hidden_layers, momentum_output_layers)
        mean_error = mean_square_error(matrix_of_sigmoid_values_for_all_layers, normalized_expected_result_matrix, matrix_new_wages_hidden_layers)

        if (i % 10 == 0):
            mean_errors_array.append(mean_error)

    matrix_of_sigmoid_values_for_all_layers = change_0_1_values_back_to_normal(data_matrix, matrix_of_sigmoid_values_for_all_layers)

    print()
    print("sigmoid_values for every neuron:")
    for j in range (0, len(matrix_of_sigmoid_values_for_all_layers)):
        for i in range(0, len(matrix_of_sigmoid_values_for_all_layers[j])):
            print(matrix_of_sigmoid_values_for_all_layers[j][i])

    write_to_file_learning(mean_errors_array, matrix_new_wages_hidden_layers, matrix_new_wages_2_layer)



def testing(normalized_data_matrix, normalized_expected_result_matrix, matrix_of_wages_hidden_layers, matrix_wages_2_layer, is_bias, data_matrix):
    # sklada razem funkcje az do wyliczenia bledow, nastepnie przechodzi do change_0_1_values_back_to_normal
    # 1. wzorzec treningowy podawany jest na wejścia sieci,
    # 2. odbywa się jego propagacja w przód
    # 3. na podstawie wartości odpowiedzi wygenerowanej przez sieć oraz wartości pożądanego wzorca odpowiedzi następuje wyznaczenie błędów
    matrix_of_sums_for_all_layers, matrix_of_sigmoid_values_for_all_layers = count_neuron(normalized_data_matrix, matrix_of_wages_hidden_layers, matrix_wages_2_layer, is_bias)
    # licze errory dla kazdego neuronu:
    matrix_of_all_errors = error(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers, matrix_wages_2_layer, matrix_of_sums_for_all_layers, is_bias)
    # licze sredni blad kwadratowy:
    mean_error = mean_square_error(matrix_of_sigmoid_values_for_all_layers, normalized_expected_result_matrix,matrix_of_wages_hidden_layers)
    write_to_file_testing(data_matrix, mean_error, normalized_expected_result_matrix, matrix_of_all_errors, matrix_of_sigmoid_values_for_all_layers, matrix_wages_2_layer, matrix_of_wages_hidden_layers)
    confusion_matrix(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers)



def confusion_matrix(normalized_expected_result_matrix, matrix_of_sigmoid_values_for_all_layers, matrix_of_wages_hidden_layers):
    # dla wszystkich przykladow biore to co wyszlo (ostatnie 3 rzedy matrix_of_all_sigmoid_values) i sprawdzam, czy dobrze
    # rozbicie na klasy to po prosto podzial na 50/50/50


    # print("sigmoid_values for every neuron:")
    # for j in range (0, len(matrix_of_sigmoid_values_for_all_layers)):   # warstwa?
    #     for i in range(0, len(matrix_of_sigmoid_values_for_all_layers[j])): # neuron?
    #         print(matrix_of_sigmoid_values_for_all_layers[j][i])

    # print("liczba warstw:",len(matrix_of_sigmoid_values_for_all_layers))
    # print("liczba neuronow wyjsciowych:",len(matrix_of_sigmoid_values_for_all_layers[-1]))
    print("liczba przykladow:", len(matrix_of_sigmoid_values_for_all_layers[-1][0]))

    real_setosa = 0
    real_versicolor = 0
    real_virginica = 0

    for i in range (0, len(normalized_expected_result_matrix)):
        # sprawdzam ile jest kwiatkow jakiego rodzaju
        if normalized_expected_result_matrix[i][0] == 1:
            real_setosa += 1
        if normalized_expected_result_matrix[i][1] == 1:
            real_versicolor += 1
        if normalized_expected_result_matrix[i][2] == 1:
            real_virginica += 1

    # licze ile jest neuronow w warstwach ukrytych
    nr_of_neurons_hidden_layers = 0
    for i in range (0, len(matrix_of_wages_hidden_layers)):
        for j in range (0, len(matrix_of_wages_hidden_layers[i])):
            nr_of_neurons_hidden_layers += 1


    real_setosa_recognised_as_setosa = 0
    real_setosa_recognised_as_versicolor = 0
    real_setosa_recognised_as_virginica = 0

    for i in range (0, real_setosa):    # sprawdzam jako jakie kwiatki zostaly zakwalifikowane te, ktore rzeczywiscie sa setosami
        # for j in range (0, len(matrix_of_sigmoid_values_for_all_layers[-1])): # dla wszytskich neuronow wyjsciowych
        if matrix_of_sigmoid_values_for_all_layers[-1][0][i] >  matrix_of_sigmoid_values_for_all_layers[-1][1][i] and matrix_of_sigmoid_values_for_all_layers[-1][0][i] >  matrix_of_sigmoid_values_for_all_layers[-1][2][i]:   # jesli setosa jest wykryta tam, gdzie powinna
            real_setosa_recognised_as_setosa += 1
        if matrix_of_sigmoid_values_for_all_layers[-1][1][i] >  matrix_of_sigmoid_values_for_all_layers[-1][0][i] and matrix_of_sigmoid_values_for_all_layers[-1][1][i] >  matrix_of_sigmoid_values_for_all_layers[-1][2][i]:   # jesli versicolor jest wykryty tam, gdzie powinna byc setosa
            real_setosa_recognised_as_versicolor += 1
        if matrix_of_sigmoid_values_for_all_layers[-1][2][i] >  matrix_of_sigmoid_values_for_all_layers[-1][0][i] and matrix_of_sigmoid_values_for_all_layers[-1][2][i] >  matrix_of_sigmoid_values_for_all_layers[-1][1][i]:   # jesli virginica jest wykryta tam, gdzie powinna byc setosa
            real_setosa_recognised_as_virginica += 1



    real_versicolor_recognised_as_versicolor = 0
    real_versicolor_recognised_as_setosa = 0
    real_versicolor_recognised_as_virginica = 0

    for i in range (real_setosa, real_setosa + real_versicolor):    # sprawdzam jako jakie kwiatki zostaly zakwalifikowane te, ktore rzeczywiscie sa versicolor
        # for j in range (0, len(matrix_of_sigmoid_values_for_all_layers[-1])): # dla wszytskich neuronow wyjsciowych
        if matrix_of_sigmoid_values_for_all_layers[-1][0][i] >  matrix_of_sigmoid_values_for_all_layers[-1][1][i] and matrix_of_sigmoid_values_for_all_layers[-1][0][i] >  matrix_of_sigmoid_values_for_all_layers[-1][2][i]:   # jesli setosa jest wykryta tam, gdzie powinna
            real_versicolor_recognised_as_setosa += 1
        if matrix_of_sigmoid_values_for_all_layers[-1][1][i] >  matrix_of_sigmoid_values_for_all_layers[-1][0][i] and matrix_of_sigmoid_values_for_all_layers[-1][1][i] >  matrix_of_sigmoid_values_for_all_layers[-1][2][i]:   # jesli versicolor jest wykryty tam, gdzie powinna byc setosa
            real_versicolor_recognised_as_versicolor += 1
        if matrix_of_sigmoid_values_for_all_layers[-1][2][i] >  matrix_of_sigmoid_values_for_all_layers[-1][0][i] and matrix_of_sigmoid_values_for_all_layers[-1][2][i] >  matrix_of_sigmoid_values_for_all_layers[-1][1][i]:   # jesli virginica jest wykryta tam, gdzie powinna byc setosa
            real_versicolor_recognised_as_virginica += 1



    real_virginica_recognised_as_virginica = 0
    real_virginica_recognised_as_setosa = 0
    real_virginica_recognised_as_versicolor = 0

    for i in range (real_setosa + real_versicolor, real_setosa + real_versicolor + real_virginica):    # sprawdzam jako jakie kwiatki zostaly zakwalifikowane te, ktore rzeczywiscie sa virginica
        # for j in range (0, len(matrix_of_sigmoid_values_for_all_layers[-1])): # dla wszytskich neuronow wyjsciowych
        if matrix_of_sigmoid_values_for_all_layers[-1][0][i] >  matrix_of_sigmoid_values_for_all_layers[-1][1][i] and matrix_of_sigmoid_values_for_all_layers[-1][0][i] >  matrix_of_sigmoid_values_for_all_layers[-1][2][i]:   # jesli setosa jest wykryta tam, gdzie powinna
            real_virginica_recognised_as_setosa += 1
        if matrix_of_sigmoid_values_for_all_layers[-1][1][i] >  matrix_of_sigmoid_values_for_all_layers[-1][0][i] and matrix_of_sigmoid_values_for_all_layers[-1][1][i] >  matrix_of_sigmoid_values_for_all_layers[-1][2][i]:   # jesli versicolor jest wykryty tam, gdzie powinna byc setosa
            real_virginica_recognised_as_versicolor += 1
        if matrix_of_sigmoid_values_for_all_layers[-1][2][i] >  matrix_of_sigmoid_values_for_all_layers[-1][0][i] and matrix_of_sigmoid_values_for_all_layers[-1][2][i] >  matrix_of_sigmoid_values_for_all_layers[-1][1][i]:   # jesli virginica jest wykryta tam, gdzie powinna byc setosa
            real_virginica_recognised_as_virginica += 1

    print("real_setosa:",real_setosa)
    print("real_versicolor:",real_versicolor)
    print("real_virginica:", real_virginica)
    print()

    print("real_setosa_recognised_as_setosa:",real_setosa_recognised_as_setosa)
    print("real_setosa_recognised_as_versicolor",real_setosa_recognised_as_versicolor)
    print("real_setosa_recognised_as_virginica",real_setosa_recognised_as_virginica)
    print()

    print("real_versicolor_recognised_as_versicolor",real_versicolor_recognised_as_versicolor)
    print("real_versicolor_recognised_as_setosa",real_versicolor_recognised_as_setosa)
    print("real_versicolor_recognised_as_virginica",real_versicolor_recognised_as_virginica)
    print()

    print("real_virginica_recognised_as_virginica",real_virginica_recognised_as_virginica)
    print("real_virginica_recognised_as_setosa",real_virginica_recognised_as_setosa)
    print("real_virginica_recognised_as_versicolor",real_virginica_recognised_as_versicolor)
    print()

    setosa_recall = recall(real_setosa_recognised_as_setosa, real_setosa_recognised_as_versicolor + real_setosa_recognised_as_virginica)
    setosa_precision = precision(real_setosa_recognised_as_setosa, real_versicolor_recognised_as_setosa + real_virginica_recognised_as_setosa)
    setosa_f2 = f2(setosa_recall, setosa_precision)

    versicolor_recall = recall(real_versicolor_recognised_as_versicolor, real_versicolor_recognised_as_setosa + real_versicolor_recognised_as_virginica)
    versicolor_precision = precision(real_versicolor_recognised_as_versicolor, real_virginica_recognised_as_versicolor + real_setosa_recognised_as_versicolor)
    versicolor_f2 = f2(versicolor_recall, versicolor_precision)

    virginica_recall = recall(real_virginica_recognised_as_virginica, real_virginica_recognised_as_setosa + real_virginica_recognised_as_versicolor)
    virginica_precision = precision(real_virginica_recognised_as_virginica, real_setosa_recognised_as_virginica + real_versicolor_recognised_as_virginica)
    virginica_f2 = f2(virginica_recall, virginica_precision)

    macro_f1 = (setosa_f2 + versicolor_f2 + virginica_f2) / 3

    print("recall, precision, f2:")
    print("setosa:",setosa_recall,setosa_precision,setosa_f2)
    print("versicolor:",versicolor_recall,versicolor_precision,versicolor_f2)
    print("virginica:",virginica_recall,virginica_precision,virginica_f2)
    print("macro_f1:",macro_f1)


def recall(tp, fn):
    if (tp + fn == 0):
        return 0
    return tp / (tp + fn)

def precision(tp, fp):
    if (tp + fp == 0):
        return 0
    return tp / (tp + fp)

def f2(recall, precision):
    if (precision + recall == 0):
        return 0
    return 2 * (precision * recall) / (precision + recall)


# main / user communication
variant = (int)(input("irysy (1), czy autoencoder (2): "))

if(variant == 1):
    mode = (int)(input("nauka (1), czy testowanie (2): "))
    if (mode != 2):
        how_many_hidden_layers = (int)(input("podaj liczbe warstw ukrytych: "))
else:
    mode = 1    # w autoencoderze sie jedynie uczymy
    how_many_hidden_layers = 1

if mode != 2:
    is_bias_input = input("z biasem (t), czy bez biasu (n): ")
    nr_of_iterations = (int)(input("podaj liczbe iteracji: "))
    is_shuffle = (int)(input("wczytujemy wartosci kolejno (1), czy losowo (2): "))
    mu = (float)(input("podaj momentum: "))

    if (is_bias_input == "t"):
        is_bias = True
    else:
        is_bias = False

    # alfa = random.uniform(0, 1) # opowiedzni zakres??
    alfa = (float)(input("podaj wspolczynnik nauki (alfa): "))

if (mode == 1):
    data_matrix, expected_result_matrix = read_from_file('learn.csv')
else:
    data_matrix, expected_result_matrix = read_from_file('test.csv')



normalized_data_matrix = change_input_to_0_1_values(data_matrix)
normalized_expected_result_matrix = change_input_to_0_1_values(expected_result_matrix)

if (mode != 2):
    matrix_of_wages_hidden_layers = [None] * how_many_hidden_layers   # lista naktora wrzucam tablice 2d wag, a wiec tablica 3d, pierwszy wymiar: ktora warstwa ukryta, drugi wymiar: ktory neuron, trzeci wymiar: ktora waga
    for i in range (0, how_many_hidden_layers):
        print("podaj liczbe neuronow w warstwie ukrytej", i, ": ")
        nr_of_neurons_hidden_layer = (int)(input())

        if( mode == 1): #variant == 1 &&
            if i == 0:    # pierwsza warstwa bierze info z data zamiast poprzedniej warstwy
                matrix_of_wages_hidden_layers[i] = generate_wages(len(normalized_data_matrix[0]), nr_of_neurons_hidden_layer, is_bias)  # (liczba cech - neuronow w warstwie wejsciowej (bies dodajemy wewnatrz funkcji, spokojnie)) x (liczba neuronow w tej warstwie) NOTE: W TYM KODZIE ZAWSZE BEDA WARSTWY: 0; 1; 2
            else:
                matrix_of_wages_hidden_layers[i] = generate_wages(len(matrix_of_wages_hidden_layers[i - 1]), nr_of_neurons_hidden_layer, is_bias)  # (liczba cech - neuronow w warstwie wejsciowej (bies dodajemy wewnatrz funkcji, spokojnie)) x (liczba neuronow w tej warstwie) NOTE: W TYM KODZIE ZAWSZE BEDA WARSTWY: 0; 1; 2

if (mode == 1):
    # if (variant == 1):
    matrix_wages_2_layer = generate_wages(nr_of_neurons_hidden_layer, 3, is_bias)  # (liczba neuronow w warswie ukrytej) x (liczba neuronow w tej warstwie - 3, bo sa 3 rodzaje kwiatkow)
    learning(normalized_data_matrix, matrix_of_wages_hidden_layers, matrix_wages_2_layer, normalized_expected_result_matrix, alfa, nr_of_iterations, is_bias, is_shuffle, mu, variant, nr_of_neurons_hidden_layer)

else:
    # matrix_of_wages_hidden_layers, matrix_wages_2_layer = wczytaj wagi z pliku
    matrix_of_wages_hidden_layers = read_wages_from_file('learned_wages_hidden_layers.csv')
    matrix_wages_2_layer = read_wages_from_file('learned_wages_output_layer.csv')


    is_bias = False
    if len(matrix_of_wages_hidden_layers[0][0]) > 4 :   # bo sa 4 cechy kwiatkow, wiec jak jest wiecej to dlatego ze jeszcze biad
        is_bias = True
    testing(normalized_data_matrix, normalized_expected_result_matrix, matrix_of_wages_hidden_layers, matrix_wages_2_layer, is_bias, data_matrix)
