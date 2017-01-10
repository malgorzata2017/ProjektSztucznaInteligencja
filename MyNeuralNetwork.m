clear ; close all; clc

%inicjalizowanie zmiennych
input_layer_size  = 400;  % 20x20 wejsciowy obrazek
hidden_layer_size = 25;   % 25 ukrytych jednostek
num_labels = 10;          % 10 etykiet, od 1 do 10   

fprintf('Ladowanie i wyswietlanie danych ...\n')
