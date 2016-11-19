%==========================================================================%
%
% Input: World state w (I x 1)
%        training data x (I x D)
%        parameters phi (N x D)
% Output: cost L
%         gradient g (N x D)
%
%==========================================================================%
function [L, g] = optMultiLogistic(w, x, phi)
    % initailize common const variables
    % N: number of class
    N = size(phi, 1);
    
    % I: number of data
    I = size(x, 1);
    
    % D: data dimension
    D = size(x, 2);
  
    % initialize cost to zero
    L = 0;
    
    % initialize gradient
    % g(n, :) is related to phi(n)
    g = zeros(N, D);
    
    lambda = 0.01;
    
    % for each data
    for i = 1:I
        % compute prediction y
        y = linearSoftMax(phi, x(i,:)');
%         y = logSoftMax(phi, x(i,:)');
        
        % update log likelihood
        % Take the w(i)^th element of y
        L = L - log(y(w(i)));
%         L = L - y(w(i));
        
        % update gradient
        for n = 1:N
           if w(i) == n
%                g(n, :) = g(n, :) + (1 - exp(y(n))).*x(i, :);
               g(n, :) = g(n, :) + ((y(n)) - 1).*x(i, :);
           else
%                g(n, :) = g(n, :) + (-exp(y(n))).*x(i, :);
               g(n, :) = g(n, :) + (y(n)).*x(i, :);
           end
        end
        
    end
    
    L = L ./ I + lambda/2 * sum(sum(phi.^2));
    
    g = g ./ I + lambda*phi;
    
%     L = L ./ I;
%     
%     g = g ./ I;
    
end