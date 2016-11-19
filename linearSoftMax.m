% 
% Input: parameters phi (N x D)
%        single data (D x 1)
% Output: output of softmax function lambda (N x 1)
%
function [lambda] = linearSoftMax(phi, x)
    N = size(phi, 1);
    lambda = zeros(N, 1);
    
%     for i = 1:N
%        for j = 1:N
%            lambda(i) = lambda(i) + exp(phi(j,:)*x - phi(i,:)*x);
%        end
%     end
%     lambda = 1./lambda;
    
    
    y = max(phi*x);
    den = 0;
    for j = 1:N
           den = den + exp(phi(j,:)*x - y);
    end
    
    for i = 1:N
        lambda(i) = exp(phi(i,:)*x - y)/den;
    end    

%     M = bsxfun(@minus,phi*x,max(phi*x, [], 1));
%     M = exp(M);
%     lambda = bsxfun(@rdivide, M, sum(M));
    
end