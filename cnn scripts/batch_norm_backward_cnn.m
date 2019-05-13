

function [dZ,d_gamma,d_beta] = batch_norm_backward_cnn(dA,Z_bn,Z_hat,gamma, Z_mu, ivar, var_sqrt, var, eps)
m = size(dA,4);
dA_unfolded = reshape(dA,numel(dA)/m,m);
dZ_bn = dA_unfolded .* linear_activation_grad(Z_bn);
[dZ_unfolded,d_gamma,d_beta] = batch_norm_backward(dZ_bn,Z_hat,gamma, Z_mu, ivar, var_sqrt, var, eps);
dZ = reshape(dZ_unfolded,size(dA,1),size(dA,2),size(dA,3),m);