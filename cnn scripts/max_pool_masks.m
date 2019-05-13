
function mask = max_pool_masks(X,s)
% X - maxpool input - n_x X n_y X n_h X m_match
% s - maxpool stride. Same in x and y . Non overlapping strides

% X = rand(4,4,3);
% s = 2;

m = size(X,4);
n_channels = size(X,3);
n_x = size(X,1);
n_y = size(X,2);

mask = zeros(size(X));

for training_sample = 1:m
    for depth = 1:n_channels
        for i = 1:s:n_y
            for j = 1:s:n_x
                y_range = i : i+s-1;
                x_range = j : j+s-1;
                if (max(x_range) > n_x) || (max(y_range) > n_y)
                    break;
                end
                temp = X(y_range,x_range,depth,training_sample);
                [~,I] = max(temp(:));
                [I_row, I_col] = ind2sub(size(temp),I);
                mask(y_range(I_row),x_range(I_col),depth,training_sample) = 1;
            end
            if (max(y_range) > n_y)
                break
            end
        end 
    end 
end