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
function [phi] = multiLogistic(w, x, N, ita, precision)
    
    D = size(x, 2);
    
    % initialize parameters phi with random var
    phi = ones(N, D);
    pre_L = 0;
    
    count = 1;
    scale = 1;
    
    while true
       
       [L, g] = optMultiLogistic(w, x, phi);
       
       phi = phi - ita*g;
       
       if abs(pre_L - L) < precision
           break;
       else
           pre_L = L;
       end
       display(L);
       
       count = count + 1;
       display(count);
    end

end