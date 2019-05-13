
function dBias = convBiasGrad(dZ)
%dZ - Gradient of output of one layer - o_h X o_w X n_c X m_batch  - dimension of output

n_channels = size(dZ,3);
dBias = zeros(n_channels,1);

for channels = 1:n_channels
    input = dZ(:,:,channels,:);
    dBias(channels) = sum(input,'all');
end

