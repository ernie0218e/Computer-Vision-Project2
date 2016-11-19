%==========================================================================%
%
% Input: w: World state (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        precision: the algorithm stops when the difference between
%                      the previous and the new likelihood is < precision.
%        ita: learning rate of gradient decent method
%        
% Output: parameters phi (N X D)
%==========================================================================%
function [phi] = multiLogisticBFGS(w, x, N, ita, precision)
    
    D = size(x, 2);
    
    % initialize parameters phi with random var
    phi = ones(N, D);
    
    count = 1;
    
    Mat_D = zeros(D, D, N);
    
    for n = 1:N
        Mat_D(:, :, n) = eye(D, D);
    end
    
    d = zeros(N, D);
    
    [L, g] = optMultiLogistic(w, x, phi);
    
    pre_L = L;
    pre_g = g;
    
    while true

       for n = 1:N
           d(n, :) = -ita*squeeze(Mat_D(:,:,n))*g(n, :)';
       end
       
       phi = phi + d;
       
       [L, g] = optMultiLogistic(w, x, phi);
       
       if abs(pre_L - L) < precision
           break;
       else
           pre_L = L;
       end
       display(L);

       
       y = g - pre_g;
       pre_g = g;
       
       for n = 1:N
           Mat_D(:,:,n) = (eye(D, D) - (d(n, :)'*y(n, :))./(d(n, :)*y(n, :)'))...
                            *squeeze(Mat_D(:,:,n))...
                            *(eye(D, D) - (y(n, :)'*d(n, :))./(y(n, :)*d(n, :)'))...
                            + (d(n, :)'*d(n, :))./(d(n, :)*y(n, :)');
       end
       
       count = count + 1;
       display(count);
    end

end