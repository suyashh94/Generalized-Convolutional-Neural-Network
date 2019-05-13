

% Fucntion for 1 layer
function [A,Z_bn_unfolded,Z_hat_unfolded,gamma,Z_mu_unfolded, ivar, var_sqrt, var, eps] = batch_norm_forward_maxpool(Z,gamma,beta)
m = size(Z,4);
Z_unfolded = reshape(Z,numel(Z)/m,m);
[Z_bn_unfolded,Z_hat_unfolded,gamma,Z_mu_unfolded, ivar, var_sqrt, var, eps] = ...
    batch_norm_forward(Z_unfolded,gamma,beta);
A_unfolded = Z_bn_unfolded;
A = reshape(A_unfolded,size(Z,1),size(Z,2),size(Z,3),m);
    



