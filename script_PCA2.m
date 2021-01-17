clc
%%
% x_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/";
% % WM_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_WM.txt";
% BF_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_BF.txt";

% f = fopen(BF_path,'r');
% f_path = "/media/nyma/EXTERNAL1/fairface-img-margin025-trainval";
% batch_x = 5000;
% tline = fgetl(f);
 
% % WM_list = zeros(batch_x,22500);
% % BF_list = zeros(batch_x,22500);
% files_img={};
% i=1;
 
% % WM_list = extractFace(f_path, tline, WM_list,i,batch_x,f);
% % BF_list = extractFace(f_path, tline, BF_list,i,batch_x,f);
% fclose(f);

% load("/media/nyma/EXTERNAL1/PCADATA/BFmatlab.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/WMmatlab.mat")

WM_mean = mean(WM_list, 1);
BF_mean = mean(BF_list, 1);
 
WM_matrix = WM_list';
BF_matrix = BF_list';
 
BF_matrix_nm = BF_matrix - BF_mean';
WM_matrix_nm = WM_matrix - WM_mean';

%% PCA Extraction %%
% [BF_Vec, BF_S, BF_Val] = pca(BF_matrix_nm'); 
% pause(5)
% [WM_Vec, WM_S, WM_Val] = pca(WM_matrix_nm');

%% VECTOR ANGLE
% BF_Vec_angle = (BF_Vec)';
% WM_Vec_angle = (WM_Vec)'; 
% clc
% selecty = 3;
% angle_list = zeros(1,selecty);
% for i = 1:selecty
%     u = BF_Vec_angle(i,:);
%     v = WM_Vec_angle(i,:);
%     angle = dot(u,v)/(norm(u)* norm(v));
%     angle =  real(acosd(angle));
%     angle_list(i) = angle;
% end
% sum_angle = sum(angle_list)/selecty;
% disp(sum_angle);


BF_Vec_angle = (BF_Vec)';
WM_Vec_angle = (WM_Vec)'; 
clc
selecty = 300;
angle_list = zeros(1,selecty);
% selecty-99:selecty
for i = selecty:4999
    u = BF_Vec_angle(i,:,:);
    v = WM_Vec_angle(i,:,:);
    angle = dot(u,v)/(norm(u)* norm(v));
    angle =  real(acosd(angle));
    angle_list(i) = angle;
end
% sum_angle = sum(angle_list)/selecty;
divx = 4999 - selecty + 1;
sum_angle = sum(angle_list)/divx;
disp(sum_angle);


BF_Vec_angle = (BF_Vec)';
WM_Vec_angle = (WM_Vec)'; 
clc
selecty = 300;
angle_list = zeros(1,selecty);
% selecty-99:selecty
for i = selecty:4999
    u = BF_Vec_angle(i,:,:);
    v = WM_Vec_angle(i,:,:);
    angle = dot(u,v)/(norm(u)* norm(v));
    angle =  real(acosd(angle));
    angle_list(i) = angle;
end
% sum_angle = sum(angle_list)/selecty;
divx = 4999 - selecty + 1;
sum_angle = sum(angle_list)/divx;
disp(sum_angle);



% SEAM / BF WM/WF
% Angles between the rest eigen vectors among
% sum of the difference abslotue value  
% abs,mean......


BF_Val_ten = sum(BF_Val(1:10))/10;
WM_Val_ten = sum(WM_Val(1:10))/10;

select = 100;
selectx = select;
% %% PROJECTION %%
% BF_proj = BF_matrix_nm(:,1:4999)' * BF_Vec; 
% BF_proj  =  BF_proj  * diag(BF_Val) ;
% 
% WM_proj = WM_matrix_nm(:,1:4999)' * WM_Vec; 
% WM_proj  =  WM_proj  * diag(WM_Val) ;

%% PROJECTION %%
BF_matrix_nm  =  BF_matrix_nm(:,1:4999)';
BF_proj_BF  = BF_matrix_nm * (diag(BF_Val(1:select,:)) * BF_Vec(:,1:select)')'; 

WM_matrix_nm = WM_matrix_nm(:,1:4999)';
WM_proj_WM   = WM_matrix_nm  * (diag(WM_Val(1:select,:)) * WM_Vec(:,1:select)')'; 


%% PROJECTION FLIP %%
% select = 50;
BF_matrix_WM =  BF_matrix - WM_mean';
BF_proj_WM   =  BF_matrix_WM(:,1:4999)' * (diag(WM_Val(1:select,:)) * WM_Vec(:,1:select)')'; 


WM_matrix_BF = WM_matrix - BF_mean';
WM_proj_BF   = WM_matrix_BF(:,1:4999)' * (diag(BF_Val(1:select,:)) * BF_Vec(:,1:select)')'; 

%% Angle Vectors
% u = abs(mean(BF_Vec,1)');
% v = abs(mean(WM_Vec,1)');
% 
% for i = 1:4999
% angle = vec_Angle(u,v);
% end

%% RECONSTRUCT %%
BF_Xhat  = (BF_Vec * BF_S(1:4999,:)') + BF_mean';
img_i = mat2gray(reshape( BF_Xhat (:,10), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

WM_Xhat  = (WM_Vec * WM_S(1:4999,:)') + WM_mean';
img_i = mat2gray(reshape( WM_Xhat (:,10), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

%% FLIP RECONSTRCT %%

% WM projected on WM 
rec_WM_proj_WM = (WM_proj_WM  * pinv((diag(WM_Val(1:select,:)) * WM_Vec(:,1:select)')')); %+  BF_mean;
img = rec_WM_proj_WM ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


% BF projected on BF
rec_BF_proj_BF   =  BF_proj_BF * pinv((diag(BF_Val(1:select,:)) * BF_Vec(:,1:select)')'); %+ WM_mean; 
img = rec_BF_proj_BF;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

% WM projected on BF 
rec_WM_proj_BF = (WM_proj_BF  * pinv((diag(BF_Val(1:select,:)) * BF_Vec(:,1:select)')')); %+  BF_mean;
img = rec_WM_proj_BF ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

%% PATH 
path_WM = "/home/nyma/Pictures/WM";
path_BF = "/home/nyma/Pictures/BF";
% BF projected on WM 
rec_BF_proj_WM   =  BF_proj_WM * pinv((diag(WM_Val(1:select,:)) * WM_Vec(:,1:select)')'); %+ WM_mean; 
img = rec_BF_proj_WM ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

%% RESIDUE 
x_WM_proj_WM  =  (WM_matrix_nm') - rec_WM_proj_WM'; 
x_WM_proj_WM  = abs(x_WM_proj_WM);
score_red_WM_proj_WM =  mean(x_WM_proj_WM,1);

% pd = fitdist((score_red_WM_proj_WM'),'kernel','Kernel','normal');
% x = -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','green','LineStyle','-')

x_BF_proj_BF = (BF_matrix_nm') - rec_BF_proj_BF'; 
x_BF_proj_BF = abs(x_BF_proj_BF);
score_red_BF_proj_BF = (mean(x_BF_proj_BF,1));


for i = 1:5
    img_1 = BF_mean - rec_BF_proj_BF(i,:);
    img_1 = abs(img_1);
    img_x = mean(img_1,2);
    
    img_1 = BF_matrix(:,i); 
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    imshow(img_1) 
    disp(img_x)
    close all
    pause(2);
end



for i= 1:5
    figs = figure();
    img_1 = BF_matrix_nm(i,:);
    img_2 = rec_BF_proj_BF(i,:);
    img_3 = x_BF_proj_BF(:,i);
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_BF, ['BF_',num2str(i),'_',num2str(select),'.jpg'])); % changed how name is saved
    close all
    pause(1);
end


% red_WM_proj_BF  =  WM_matrix(:,1) - rec_WM_proj_BF(1,:)'; 
x_WM_proj_BF  =  (WM_matrix(:,1:4999) - BF_mean(:,1:4999)) - rec_WM_proj_BF'; 
x_WM_proj_BF = abs(x_WM_proj_BF);
score_red_WM_proj_BF = (mean(x_WM_proj_BF,1));

figh = figure();
hist(score_red_WM_proj_WM,100);
std_red_WM_proj_WM = std(score_red_WM_proj_WM);
text(.09, 160,['WM-WM std = ',num2str(std_red_WM_proj_WM)], 'Color', [1 0 0])
hold on;
h = findobj(gca,'Type','patch');
h.FaceColor = [1 0 0];
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
xlabel('Residue Values') ;
ylabel('Frequency'); 
hist(score_red_WM_proj_BF, 100);
std_red_WM_proj_BF = std(score_red_WM_proj_BF);
text(.09, 150,['WM-BF std = ',num2str(std_red_WM_proj_BF)], 'Color', [0 0 1])
legend({'WM projected on WM' 'WM projected on BF' });
saveas(figh, fullfile(path_WM, ['WM_rec_',num2str(select),'_','.jpg'])); % changed how name is saved
close all
pause(2);

for i= 1:5
    figs = figure();
    img_1 = WM_matrix_nm(i,:);
    img_2 = rec_WM_proj_BF(i,:);
    img_3 = x_WM_proj_BF(:,i);
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_WM, ['WM_BF_',num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(1);
end

close all
x_BF_proj_WM  =  (BF_matrix(:,1:4999) - WM_mean(:,1:4999)) - rec_BF_proj_WM'; 
x_BF_proj_WM = abs (x_BF_proj_WM);
score_red_BF_proj_WM =  (mean(x_BF_proj_WM,1));


figh= figure();
hist(score_red_BF_proj_BF,100);
std_red_BF_proj_BF = std(score_red_BF_proj_BF);
text(.1, 200,['BF-BF std = ',num2str(std_red_BF_proj_BF)], 'Color', [0 0.5 0.5])
hold on;
h = findobj(gca,'Type','patch');
h.FaceColor = [0 0.5 0.5];
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
xlabel('Residue Values') ;
ylabel('Frequency'); 
hist(score_red_BF_proj_WM, 100);
std_red_BF_proj_WM = std(score_red_BF_proj_WM);
text(.1, 190,['BF-WM std = ',num2str(std_red_BF_proj_WM)], 'Color', [0 0 0.6])
legend({'BF projected on BF' 'BF projected on WM' });
saveas(figh, fullfile(path_BF, ['WM_rec_',num2str(select),'_','.jpg'])); % changed how name is saved
close all
pause(1);

% pd = fitdist(score_red_BF_proj_WM','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','black','LineStyle','-')

for i= 1:5
    figs = figure();
    img_1 = BF_matrix_nm(i,:);
    img_2 = rec_BF_proj_WM(i,:);
    img_3 = x_BF_proj_WM(:,i) ;%-  BF_mean(:,i)';
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_BF, ['BF_WM',num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(2);
end

% img = ans;
% img_m = reshape(img, 150,150);
% img_m = imrotate(img_m,-90);
% imshow(img_m);
% score_red_WM_proj_BF =  mean(red_WM_proj_BF,1);

%% PLOT PROJ

figp = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_WM = rescale(mean(abs(WM_proj_WM),1));
plot(plot_WM(:,1:selectx),"r");
plot_WM_BF = rescale(mean(abs(WM_proj_BF),1))';
plot((plot_WM_BF(1:selectx,:)),"b");
legend({'WM projected on WM' 'WM projected on BF' });
saveas(figp, fullfile(path_WM, ['WM_WM_',num2str(select),'_','.jpg'])); % changed how name is saved
close all


%% PLOT PROJ FLIP
figx = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_BF = (mean((BF_proj_BF),1));
plot(plot_BF(:,1:selectx), "r");
plot_BF_WM = rescale(mean(abs(BF_proj_WM),1))';
plot((plot_BF_WM(1:selectx,:)),"g");
legend({'BF projected on BF' 'BF projected on WM' });
saveas(figx, fullfile(path_BF, ['BF_WM_',num2str(select),'_','.jpg'])); % changed how name is saved
close all


% img_mean= imrotate(img_mean,-90);
% img_mean = img_mean';

% image_matrix = files_list';
% image_matrix_no_mean =  image_matrix  - img_mean;

% eig_face = pca(image_matrix_no_mean);
% cov_no_mean = image_matrix_no_mean * image_matrix_no_mean';
% cov_n = cov(image_matrix_no_mean');
% [D_vec, D_val] = eig(cov_n);
% [Vectors_m, D_Values_m]= eig(cov_no_mean);

% img_i = reshape(Vectors_m(:,2), 150,150);
% normalizedImage = uint8(255*mat2gray(img_i));
% img_i = imrotate(img_i,-90);
% imshow(normalizedImage);

% imv = Vectors(:,selected)*image_matrix;
% save(fullfile(x_path,cov_image_matrix))
% save(fullfile(x_path,Vectors))
% save(fullfile(x_path,D_values))
A = [100, 200, 300, 400, 500, 600];
Ethnics = ['WF_Val','SEAM_Val','WM_Val','BF_Val'];
for i = 1:6
    x = sum(WF_Val(1:A(i),:))/A(i);
    disp(x)
%     for j = 1:4
%         x = sum(Ethnics(j(1:A(i),:)))/A(i);
%         disp(x)
%     end
end
