
function O = max_pool(X,s)
% X - input of n_x X n_y X n_h
% s_x - stride in x; s_y = stride in y;
% Assuming same stride in x and y and strides to be non overlapping
% X = rand(4,4,3);
% s = 2;
[n_y,n_x,n_h] = size(X);
n_x_out = floor(n_x/s);
n_y_out = floor(n_y/s);

for h = 1:n_h
    for i = 1 : n_y_out
        for j = 1:n_x_out
            x_range = (j-1)*s + 1 : j*s;
            y_range = (i-1)*s + 1 : i*s;
            temp = X(y_range,x_range,h);
            O(i,j,h) = max(temp(:));
            clear x_range y_range temp
        end
    end
end
