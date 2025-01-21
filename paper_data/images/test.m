% Define the 512x512 FFT matrix
fft_matrix = zeros(512);
block_size = 64;

% Set 64x64 blocks of ones in the corners
fft_matrix(1:block_size, 1:block_size) = 1; % Top-left
fft_matrix(1:block_size, end-block_size+1:end) = 1; % Top-right
fft_matrix(end-block_size+1:end, 1:block_size) = 1; % Bottom-left
fft_matrix(end-block_size+1:end, end-block_size+1:end) = 1; % Bottom-right


%%


image = ifft2(fft_matrix);

% Check if the result is real
is_real = isreal(image); % Should be true
disp(['Is the image real? ', num2str(is_real)]);

%%

avg_filter = [1, 1; 1, 1] / 4;
% Convolve the image with the averaging filter
filtered_img = conv2(image, avg_filter, 'same'); % 'same' ensures output is the same size as input

% Downsample the image by 2
downsampled_img = filtered_img(1:2:end, 1:2:end);

filtered_fft = fft2(filtered_img);
downsample_fft = fft2(downsampled_img);

%%
figure;
imshow(abs(downsample_fft));
%%
down_image = image(1:2:end, 1:2:end);
down_fft = fft2(down_image);
figure;
imshow(abs(down_fft));

figure;
imagesc(abs(fft2(avg_filter)));


%%

