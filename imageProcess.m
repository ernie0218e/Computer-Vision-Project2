%%============================================================%%
% Purpose:  Resize the image and scale each pixel to [0, 1]
% Input:    img: vectors of multiple images (N x D)
%           ratio: the ratio of new image width to original one
% Output:   data: processed vectors of multiple images (N x D)
%%============================================================%%
function [data] = imageProcess(img, ratio)

    % get number of images (N) and the size of each image (D)
    imgNum = size(img, 1);
    imgSize = size(img, 2);
    
    imgWidth = round(sqrt(imgSize));
    nimgWidth = round(imgWidth*ratio);
    data = zeros(imgNum, nimgWidth^2);
    
    for i = 1:imgNum
       temp = zeros(imgWidth, imgWidth);
       for j = 1:imgWidth
           temp(j, :) = img(i, (imgWidth*(j-1)+1:imgWidth*j));
       end
       temp = imresize(temp, ratio);
       for j = 1:nimgWidth
           data(i, (nimgWidth*(j-1)+1:nimgWidth*j)) = abs(temp(j, :)./255);
       end
    end
end