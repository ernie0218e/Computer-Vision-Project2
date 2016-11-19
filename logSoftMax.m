% 
% Input: parameters phi (N x D)
%        single data (D x 1)
% Output: output of softmax function lambda (N x 1)
%
function [lambda] = logSoftMax(phi, x)
    N = size(phi, 1);
    lambda = zeros(N, 1);
    
    
    y = max(phi*x);
    den = 0;
    for j = 1:N
           den = den + exp(phi(j,:)*x - y);
    end
    den = den + y;
    
    for i = 1:N
        lambda(i) = phi(i,:)*x - den;
    end    
    
end