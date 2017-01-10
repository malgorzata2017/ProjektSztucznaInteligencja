clear ; close all; clc

%inicjalizowanie zmiennych
input_layer_size  = 400;  % 20x20 wejsciowy obrazek
hidden_layer_size = 25;   % 25 ukrytych jednostek
num_labels = 10;          % 10 etykiet, od 1 do 10   

fprintf('Ladowanie i wyswietlanie danych ...\n')


load('data1.mat');
m = size(X, 1);

sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program sie zatrzymal. Nacisnij enter aby kontynuowac.\n');
pause;


fprintf('\nLadowanie poczatkowych wag ...\n')

load('weights.mat');

nn_params = [Theta1(:) ; Theta2(:)];
