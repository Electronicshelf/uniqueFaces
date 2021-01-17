function files_list = extractFace(f_path, tline, files_list,i,batch_x,f)
%Extract files from folder to Matrix

while ischar(tline) && i < batch_x
    A = tline;
    full_path = fullfile(f_path,A);
    img = imread(full_path);
    img = imresize(img,[150 150]);
    img = rgb2gray(img);
    img = rescale(img);
%     img = imrotate(img,-90);
    imshow(img)
    [irow, icol] = size(img);
    img = reshape(img',irow*icol,1);
    files_list(i,:) = img; 
%     files_img{i} = img;
    disp(full_path)
    disp(i)
    tline = fgetl(f);
    i = i+1;  
end

