function [X, varargout] = data_loader(data)
%data_loader A function for loading various datasets
%
%The various file paths below need to be replaced with the appropriate
%directories on your local computer. Some further notes on getting the
%datasets:
%   - For 'cat-12d', download the jpg image from
%     https://unsplash.com/photos/l54ZALpH2_I 
%
%   - For 'coil-downsampled' and 'coil-downsampled-7d', download the full
%     COIL-100 dataset from
%     https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php 
%     and then use the Python script in data/coil-100-downsampled/main.py
%     to process it into a .mat file.

switch data
    case 'cat-12d'
        % This is a 12-way tensorized version of the Muller cat image.
        % After reshaping, the modes are permuted in the fashion of visual
        % data tensorization.
        img = imread('data/cat/sebastian-muller-l54ZALpH2_I-unsplash.jpg');
        input_mat = mean(double(img), 3)/255;
        input_mat = input_mat(1:end-64, 1:end-64);
        X = reshape(input_mat, 4*ones(1,12));
        X = permute(X, [1 7 2 8 3 9 4 10 5 11 6 12]);
        X = reshape(X, 16*ones(1,6));
        
    case 'coil-downsampled'
        % This is a compressed variant of the COIL-100 dataset. The
        % dimensions are 32 x 32 x 3 x 7200, where the first two dimensions
        % have been compressed down from 128 x 128. The other dimensions
        % are the three color channels and the number of images and haven't
        % been compressed.
        path = "data/coil-100-downsampled/compressed_coil_100.mat";
        load(path, 'img_array', 'class_array');
        X = img_array;
        varargout{1} = class_array;
    
    case 'coil-downsampled-7d'
        % This is a 7-way tensorized version of the compressed COIL-100
        % dataset. The compressed 32 x 32 dimensions have been reshaped
        % into 4 x 4 x 4 x 4 x 4 via visual data tensorization.
        path = "data/coil-100-downsampled/compressed_coil_100.mat";
        load(path, 'img_array', 'class_array');
        X = img_array;
        X = reshape(X, [2*ones(1,10) 3 7200]);
        X = permute(X, [1 6 2 7 3 8 4 9 5 10 11 12]);
        X = reshape(X, [4 4 4 4 4 3 7200]);
        varargout{1} = class_array;

    case 'coil-100'
        path = 'E:/data_sets/images/coil-100/coil_100.mat';
        load(path, 'img_array', 'class_array');
        X = img_array;
        varargout{1} = class_array;
end

end
