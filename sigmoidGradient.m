function g = sigmoidGradient(z)

g = zeros(size(z));
n = size(z, 1);
m = size(z, 2);

if(n ~=1 && m ~=1 )
    for i=1:n
        for j=1:m
            g(i, j) =  sigmoid(z(i, j)) * (1 - sigmoid(z(i, j)));
        end
    end
else        
    g =  sigmoid(z) * (1 - sigmoid(z)).';
end

end
