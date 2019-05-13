function O = fullFilterConv(X,filt,s_x,s_y)

K = size(filt,1) - 1;
% pad_dim = K * ones(ndims(X),1);
X_padded = padarray(X,[K K],0,'both');
O = filterConv(X_padded,filt,s_x,s_y);