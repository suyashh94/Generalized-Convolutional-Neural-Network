
function O = filterConv(X,filt,s_x,s_y)
% filt is filter - n_f X n_f X n_h -  assuming a square filter
% X is input - n_x X n_y X n_h
% S is stride = 1 % default
% No zero padding

n_x = size(X,2);
n_y = size(X,1);
n_h = size(X,3);
n_f = size(filt,1);

assert(size(filt,3) == n_h);

i_In = 1;
i_out = 1;


while(i_In + n_f - 1) <= n_y
%     disp('Vertical Sliding')
    j_In = 1;
    j_out = 1;
    while(j_In + n_f - 1) <= n_x
%         disp('Horizontal sliding')
        i_In_start = i_In;
        i_In_end = i_In + n_f - 1;
        
        j_In_start = j_In;
        j_In_end = j_In + n_f - 1;
        
        temp = X(i_In_start : i_In_end, j_In_start : j_In_end,:).* filt;
      O(i_out,j_out) = (sum(temp,'all'));
      clear temp
      
      j_out = j_out + 1;
      j_In = j_In + s_x;
    end
    i_out = i_out + 1;
    i_In = i_In + s_y;
end 