

% Fucntion for 1 layer
function [A,Z_bn_unfolded,Z_hat_unfolded,gamma,Z_mu_unfolded, ivar, var_sqrt, var, eps] = ...
    batch_norm_forward_cnn_Test(Z,gamma,beta,mu,sigma)
m = size(Z,4);
Z_unfolded = reshape(Z,numel(Z)/m,m);
[Z_bn_unfolded,Z_hat_unfolded,gamma,Z_mu_unfolded, ivar, var_sqrt, var, eps] = ...
    batch_norm_forward_Test(Z_unfolded,gamma,beta,mu,sigma);
A_unfolded = Z_bn_unfolded .* linear_activation(Z_bn_unfolded);
A = reshape(A_unfolded,size(Z,1),size(Z,2),size(Z,3),m);
    



