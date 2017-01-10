function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
								   
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


X = [ones(m, 1) X];
theta1_size = size(Theta1, 1);
theta2_size = size(Theta2, 1);

K = theta2_size;
classes = [1: K];

for i=1:m
    
    a1 = X(i, :);
    
    for j=1:theta1_size
        theta1 = Theta1(j, :);
        z2(j) = theta1 * a1.';    
    end
    
    a2 = sigmoid(z2);
    
    a2 = [ones(size(a2, 1), 1) a2];
    
    for j=1:theta2_size
        theta2 = Theta2(j, :);
        z3(j) = theta2 * a2.';   
    end
    
    a3 = sigmoid(z3);
    
    A1(i, :) = a1;
    A2(i, :) = a2;
    A3(i, :) = a3;
    Z2(i, :) = z2;
    
   
    part1 =  log(a3);
    part2 = log(1 - a3);
    part3 = (y(i)==classes).';    
    class_sum = ((-part1 * part3) - (part2 * (1 - part3)))/m;
        
    J = J + class_sum; 
end


sum1 = 0;
for j=1:hidden_layer_size
    for k=2:(input_layer_size+1)
        sum1 = sum1 + (Theta1(j,k)^2);
    end
end

sum2 = 0;
for j=1:num_labels
    for k=2:(hidden_layer_size+1)
        sum2 = sum2 + (Theta2(j,k)^2);
    end
end

regulized_part = (sum1 + sum2) * (lambda/(2*m));
 
J = J + regulized_part;
% -------------------------------------------------------------

%backpropagation algorithm----------------------------

for i=1:m 
  class = (y(i) == classes);
  delta3(i, :) =  A3(i, :) - class;
end  

  delta2 = (Theta2(:, 2:end)' * delta3') .* sigmoidGradient(Z2)';


Delta1 = delta2 * A1;
Delta2 = delta3' * A2;

Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1 = (Theta1 .* (lambda/m));
Theta2 = (Theta2 .* (lambda/m));

grad = grad + [Theta1(:) ; Theta2(:)];


