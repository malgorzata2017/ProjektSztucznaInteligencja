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

% Przejscie sieci neuronowej w przod

lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

g = sigmoidGradient([1 -0.5 0 0.5 1]);

fprintf('\nInicjalizowanie parametrow sieci neuronowej ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

checkNNGradients;

fprintf('Program sie zatrzymal. Nacisnij enter aby kontynuowac.\n');
pause;

lambda = 3;
checkNNGradients(lambda);

