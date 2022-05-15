import pandas as pd
import numpy as np

def read_from_file:
    # wczytywanie csv i podzial danych na macierze
    dataset = pd.read_csv('Iris.csv')
    dataset = pd.get_dummies(dataset, columns=['Species'])  # One Hot Encoding
    values = list(dataset.columns.values)

def change_input_to_0_1_values:
    # funckja zmieniajaca dane na liczby z zakresu [0, 1]

def sum:
    # suma, pierwszy etap w kazdym neuronie

def sigmoidal_function:
    # funkcja sigmoidalna, drugi etap w kazdym neuronie

def count_neuron:
    # funkcja liczaca sume i funkcje sigmoidalna w danym neuronie
    # zaleznie od danych wejsciowych (wszystkie cechy z danego przykladu i wagi)

def error:
    # funkcja liczacza blad na danym neuronie (wynik z danej iteracji na danym neuronie - to co mialo wyjsc (z danych))

def liczenie_trojkacikow: # todo: jak to nazwac??
    # tworzenie macierzy takich samych jak macierze z wagami, ktore byly wczesciej
    # i zastepowanie wartosci tych wag wynikami odpowiednich rownan:
    # (waga danego x (rowniez biasu)) * (odpowiadajacy blad na neuronie) + to samo tyle razy ile jest x

def liczenie_dija: # todo: jak to nazwac??
    # bierzemy kazdy trojkacik wyliczony w funkcji liczenie_trojkacikow i mnozymy * (1 / liczba przykladow w danych)

def gradient_descent:
    # liczymy nowe wagi, ktorymi zastapimy stare wagi przy nastepnej iteracji
    # (nowa waga) = (stara waga) - (alfa) * (odpowiedni dij z fkcji liczenie_dija)

def back_propagation:
    # funkcja skladajaca do siebie funckje: error, liczenie_trojkacikow, liczenie_dija i gradient_descent

def change_0_1_values_back_to_normal:
    # funkcja zmieniajaca liczby z zakresu [0, 1] z powrotem na 'normalne' po wszystkich iteracjach (na pewno po wszystkich??)

def write_to_file:
    # zapisuje otrzymane wyniki do pliku

def learning:
    # sklada razem wszystkie powyzsze funkcje

def testing:
    # sklada razem funckje az do wyliczenia bledow, nastepnie przechodzi do change_0_1_values_back_to_normal

# main / user communication
nr_of_neurons_layer_0 = input("podaj liczbe neuronow w warstwie wejsciowej")
nr_of_neurons_layer_1 = input("podaj liczbe neuronow w warstwie wejsciowej")
nr_of_neurons_layer_2 = input("podaj liczbe neuronow w warstwie wejsciowej")
