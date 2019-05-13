
clear all 
clc

pic = imread('peppers.png');
pic = im2double(pic);

figure; 
imagesc(pic)

pic1D = median(pic,3);

% figure; 
% colormap('gray')
% imagesc(pic1D)
%% Valid Convolution

filt = ones(4,4,1);

convImageMatlab = convn(pic1D,flip(flip(flip(filt,1),2),3),'valid');
convImageSelf = filterConv(pic1D,filt,1,1);
temp = convImageMatlab - convImageSelf;
figure; plot(temp(:))

figure; 
subplot(1,2,1)
colormap('gray')
imagesc(convImageMatlab)
subplot(1,2,2)
colormap('gray')
imagesc(convImageSelf)


%% Full Convolution

convImageMatlabFull = convn(pic,filt,'valid');
convImageSelfFull = fullFilterConv(pic,filt,1,1);
temp = convImageMatlabFull - convImageSelfFull;
figure; plot(temp(:))

figure; 
subplot(1,2,1)
colormap('gray')
imagesc(convImageMatlabFull)
subplot(1,2,2)
colormap('gray')
imagesc(convImageSelfFull)

%% Max Pool

imageMaxpool = max_pool(pic,8);

figure; 
imagesc(imageMaxpool);

%% Conv layer backprop 

clear all
I = rand(3,3,4);
filt = rand(2,2,4);
s_x = 1;
s_y = 1;
Z = filterConv(I,filt,s_x,s_y);

dZ = rand(size(Z));
dFilt = convFilterGrad(dZ,I,filt,s_x,s_y);

dz_flipped = flip(flip(dZ,1),2);

for i = 1:size(filt,3)
    dFilt_matlab(:,:,i) = convn(I(:,:,i),dz_flipped,'valid');
end

%% Conv Layer input gradient

dI = convInputGrad(dZ,I,filt,s_x,s_y);
dI_matlab = convn(dZ,filt);

%% max_pool 

s = 2; 
Z_max_pool = max_pool(Z,s);
masks = max_pool_masks(Z,s);
%% max_pool grad

dZ_max_pool = rand(size(Z_max_pool));

dZ_if_max_pool = maxPoolGrad(dZ_max_pool,Z,masks,s);

%% Forward convolution layer

clear all; 
I = rand(4,4,3,4);
filt = rand(1,1,3,3);
s_x = 1;
s_y = 1; 

O = Forward_Convolution(I,filt,s_x,s_y);

%% Forward max pool 
s = 2; 
O_max_pool = Forward_MaxPool(O,s);
O_max_mask = max_pool_masks(O,s);

%% back prop of max pool layer 

dO_max_pool  = rand(size(O_max_pool));

dO = maxPoolGrad(dO_max_pool,O,O_max_mask,s);
%% back prop of conv layer
dFilt = convFilterGrad(dO,I,filt,s_x,s_y);
dI = convInputGrad(dO,I,filt,s_x,s_y);

