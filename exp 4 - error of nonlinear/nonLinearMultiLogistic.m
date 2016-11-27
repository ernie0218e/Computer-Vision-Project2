%==========================================================================
%
% Input: w: World state (I x 1)
%        x: Training data (I x D)
%        N: number of class
%        precision: the algorithm stops when the difference between
%                      the previous and the new likelihood is < precision.
%        eta: learning rate of gradient decent method
%        
% Output: phi_zero - bias  (N X 1)
%         phi - parameters (N x K)
%         zeta - parameters of nonlinear function (N x D x K)
%         
%==========================================================================
function [phi_zero, phi, zeta, e, t_e] = nonLinearMultiLogistic(w, x, N, K, eta, multiplier, test_w, test_x)
    
    D = size(x, 2);
    I = size(x, 1);
    
    phi_zero = zeros(N, 1);
    
    phi = zeros(N, K);
    zeta = zeros(N, D, K);
    
    a = zeros(N, I);
    
    % set up world state matrix W
    W = zeros(N, I);
    for i = 1:I
        W(w(i), i) = 1;
    end
    
    count = 1;
    
    t_times = (K*(K+1)/2)*multiplier;
    
    e = zeros(t_times, 1);
    t_e = zeros(t_times, 1);
    
    for k = 1:K
        
        phi_t = 2*(rand(N, 1) - 0.5);
        zeta_t = 2*(rand(N, D) - 0.5);
        
        a = a - repmat(phi_zero, 1, I);
        
        eta_t = eta;
        
        times = multiplier * (K - k + 1);
        
        for t = 1:times

           [L, a_t, phi_zero_g, phi_g, zeta_g] ...
               = optNonLinearMultiLogistic(W, x, a, phi_zero, phi_t, zeta_t);
           
           if L < 1
                eta_t = L*eta;
           end
           
           phi_zero = phi_zero - eta_t*phi_zero_g;
           phi_t = phi_t - eta_t*phi_g;
           zeta_t = zeta_t - eta_t*zeta_g;
           
           e(count) = ...
               testNonlinearMulticlassLogistic(test_x, test_w, phi_zero, phi, zeta, K);
           t_e(count) = ...
               testNonlinearMulticlassLogistic(x, w - 1, phi_zero, phi, zeta, K);
           
           display(L);

           count = count + 1;
           display(count);
        end
        
        phi(:, k) = phi_t;
        zeta(:, :, k) = zeta_t;       
        a = a_t;
    end

end