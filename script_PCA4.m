clc

% % clear
% x_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/";
% % WF_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_WF.txt";
% SEAM_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_SEAM.txt";
 
% f = fopen(SEAM_path,'r');
% f_path = "/media/nyma/EXTERNAL1/fairface-img-margin025-trainval";
% batch_x = 5000;
% tline = fgetl(f);
 
% % WF_list = zeros(batch_x,22500);
% % SEAM_list = zeros(batch_x,22500);
% files_img={};
% i=1;
 
% % WF_list = extractFace(f_path, tline, WF_list,i,batch_x,f);
% % SEAM_list = extractFace(f_path, tline, SEAM_list,i,batch_x,f);
% fclose(f);

% load("/media/nyma/EXTERNAL1/PCADATA/data_matlab.mat")



WF_mean = mean(WF_list, 1);
SEAM_mean = mean(SEAM_list, 1);
 
WF_matrix = WF_list';
SEAM_matrix = SEAM_list';
 
SEAM_matrix_nm = SEAM_matrix - SEAM_mean';
WF_matrix_nm = WF_matrix - WF_mean';

%% PCA Extraction %%
% [SEAM_Vec, SEAM_S, SEAM_Val] = pca(SEAM_matrix_nm'); 
% pause(5)
% [WF_Vec, WF_S, WF_Val] = pca(WF_matrix_nm');

% %% VECTOR ANGLE
% SEAM_Vec_angle = (SEAM_Vec)';
WF_Vec_angle = (WF_Vec)'; 
% clc
% angle_list = zeros(1,30);
% for i = 1:30
%     u = SEAM_Vec_angle(i,:);
%     v = WF_Vec_angle(i,:);
%     angle = dot(u,v)/(norm(u)* norm(v));
%     angle =  real(acosd(angle));
%     angle_list(i) = angle;
% end
% disp(angle_list(:));


SEAM_Vec_angle = (SEAM_Vec)';
WF_Vec_angle = (WF_Vec)'; 
clc
selecty = 600;
angle_list = zeros(1,selecty);
% selecty:4999
% selecty-99:selecty
for i = selecty:4999
    u = SEAM_Vec_angle(i,:,:);
    v = WF_Vec_angle(i,:,:);
    angle = dot(u,v)/(norm(u)* norm(v));
    angle =  real(acosd(angle));
    angle_list(i) = angle;
end
divx = 4999 - selecty + 1;
sum_angle = sum(angle_list)/divx;
disp(sum_angle);


SEAM_Val_ten = sum(SEAM_Val(1:10))/10;
WF_Val_ten = sum(WF_Val(1:10))/10;


select = 100;
selectx = select;
% %% PROJECTION %%
% SEAM_proj = SEAM_matrix_nm(:,1:4999)' * SEAM_Vec; 
% SEAM_proj  =  SEAM_proj  * diag(SEAM_Val) ;
% 
% WF_proj = WF_matrix_nm(:,1:4999)' * WF_Vec; 
% WF_proj  =  WF_proj  * diag(WF_Val) ;

%% PROJECTION %%
SEAM_matrix_nm  =  SEAM_matrix_nm(:,1:4999)';
SEAM_proj_SEAM  = SEAM_matrix_nm * (diag(SEAM_Val(1:select,:)) * SEAM_Vec(:,1:select)')'; 

WF_matrix_nm = WF_matrix_nm(:,1:4999)';
WF_proj_WF   = WF_matrix_nm  * (diag(WF_Val(1:select,:)) * WF_Vec(:,1:select)')'; 


%% PROJECTION FLIP %%
% select = 50;
SEAM_matrix_WF =  SEAM_matrix - WF_mean';
SEAM_proj_WF   =  SEAM_matrix_WF(:,1:4999)' * (diag(WF_Val(1:select,:)) * WF_Vec(:,1:select)')'; 


WF_matrix_SEAM = WF_matrix - SEAM_mean';
WF_proj_SEAM   = WF_matrix_SEAM(:,1:4999)' * (diag(SEAM_Val(1:select,:)) * SEAM_Vec(:,1:select)')'; 

%% Angle Vectors
% u = abs(mean(SEAM_Vec,1)');
% v = abs(mean(WF_Vec,1)');
% 
% for i = 1:4999
% angle = vec_Angle(u,v);
% end

%% RECONSTRUCT %%
SEAM_Xhat  = (SEAM_Vec * SEAM_S(1:4999,:)') + SEAM_mean';
img_i = mat2gray(reshape( SEAM_Xhat (:,10), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

WF_Xhat  = (WF_Vec * WF_S(1:4999,:)') + WF_mean';
img_i = mat2gray(reshape( WF_Xhat (:,10), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

%% FLIP RECONSTRCT %%

% WF projected on WF 
rec_WF_proj_WF = (WF_proj_WF  * pinv((diag(WF_Val(1:select,:)) * WF_Vec(:,1:select)')')); %+  SEAM_mean;
img = rec_WF_proj_WF ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


% SEAM projected on SEAM
rec_SEAM_proj_SEAM   =  SEAM_proj_SEAM * pinv((diag(SEAM_Val(1:select,:)) * SEAM_Vec(:,1:select)')'); %+ WF_mean; 
img = rec_SEAM_proj_SEAM;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

% WF projected on SEAM 
rec_WF_proj_SEAM = (WF_proj_SEAM  * pinv((diag(SEAM_Val(1:select,:)) * SEAM_Vec(:,1:select)')')); %+  SEAM_mean;
img = rec_WF_proj_SEAM ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

%% PATH 
path_WF = "/home/nyma/Pictures/WF";
path_SEAM = "/home/nyma/Pictures/SEAM";
% SEAM projected on WF 
rec_SEAM_proj_WF   =  SEAM_proj_WF * pinv((diag(WF_Val(1:select,:)) * WF_Vec(:,1:select)')'); %+ WF_mean; 
img = rec_SEAM_proj_WF ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

%% RESIDUE 
x_WF_proj_WF  =  (WF_matrix_nm') - rec_WF_proj_WF'; 
x_WF_proj_WF  = abs(x_WF_proj_WF);
score_red_WF_proj_WF =  mean(x_WF_proj_WF,1);

% pd = fitdist((score_red_WF_proj_WF'),'kernel','Kernel','normal');
% x = -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','green','LineStyle','-')

x_SEAM_proj_SEAM =  (SEAM_matrix_nm') - rec_SEAM_proj_SEAM'; 
x_SEAM_proj_SEAM = abs(x_SEAM_proj_SEAM);
score_red_SEAM_proj_SEAM = (mean(x_SEAM_proj_SEAM,1));

% pd = fitdist(score_red_SEAM_proj_SEAM','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','c','LineStyle','-')

for i= 1:5
    figs = figure();
    img_1 = SEAM_matrix_nm(i,:);
    img_2 = rec_SEAM_proj_SEAM(i,:);
    img_3 = x_SEAM_proj_SEAM(:,i);
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_SEAM, ['SEAM_',num2str(i),'_',num2str(select),'.jpg'])); % changed how name is saved
    close all
    pause(1);
end


% red_WF_proj_SEAM  =  WF_matrix(:,1) - rec_WF_proj_SEAM(1,:)'; 
x_WF_proj_SEAM  =  (WF_matrix(:,1:4999) - SEAM_mean(:,1:4999)) - rec_WF_proj_SEAM'; 
x_WF_proj_SEAM = abs(x_WF_proj_SEAM);
score_red_WF_proj_SEAM = (mean(x_WF_proj_SEAM,1));

figh = figure();
hist(score_red_WF_proj_WF,100);
std_red_WF_proj_WF = std(score_red_WF_proj_WF);
text(.08, 200,['WF-WF std = ',num2str(std_red_WF_proj_WF)], 'Color', [1 0 0])
hold on;
h = findobj(gca,'Type','patch');
h.FaceColor = [1 0 0];
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
xlabel('Residue Values') ;
ylabel('Magnitude'); 
std_red_WF_proj_SEAM = std(score_red_WF_proj_SEAM);
hist(score_red_WF_proj_SEAM, 100);
legend({'WF projected on WF' 'WF projected on SEAM' });
% Label the points where the curves are equal in black
text(.08, 190,['WF-SEAM std = ',num2str(std_red_WF_proj_SEAM)], 'Color', 'b')
saveas(figh, fullfile(path_WF, ['WF_rec_',num2str(std_red_WF_proj_SEAM),'_','.jpg'])); % changed how name is saved
close all
pause(2);

for i= 1:5
    figs = figure();
    img_1 = WF_matrix_nm(i,:);
    img_2 = rec_WF_proj_SEAM(i,:);
    img_3 = x_WF_proj_SEAM(:,i);
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_WF, ['WF_SEAM_',num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(1);
end

close all
x_SEAM_proj_WF  =  (SEAM_matrix(:,1:4999) - WF_mean(:,1:4999)) - rec_SEAM_proj_WF'; 
x_SEAM_proj_WF = abs (x_SEAM_proj_WF);
score_red_SEAM_proj_WF =  (mean(x_SEAM_proj_WF,1));


figh= figure();
hist(score_red_SEAM_proj_SEAM,100);
std_red_SEAM_proj_SEAM = std(score_red_SEAM_proj_SEAM);
text(.08, 130,['SEAM-SEAM std = ',num2str(std_red_SEAM_proj_SEAM)], 'Color', [0 0.5 0.5])
hold on;
h = findobj(gca,'Type','patch');
h.FaceColor = [0 0.5 0.5];
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
xlabel('Residue Values') ;
ylabel('Magnitude'); 
hist(score_red_SEAM_proj_WF, 100);
std_red_SEAM_proj_WF = std(score_red_SEAM_proj_WF);
legend({'SEAM projected on SEAM' 'SEAM projected on WF' });
text(.08, 120,['SEAM-WF std = ',num2str(std_red_SEAM_proj_WF)], 'Color', 'b')
saveas(figh, fullfile(path_SEAM, ['WF_rec_',num2str(select),'_','.jpg'])); % changed how name is saved
close all
pause(1);

% pd = fitdist(score_red_SEAM_proj_WF','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','black','LineStyle','-')

for i= 1:5
    figs = figure();
    img_1 = SEAM_matrix_nm(i,:);
    img_2 = rec_SEAM_proj_WF(i,:);
    img_3 = x_SEAM_proj_WF(:,i) ;%-  SEAM_mean(:,i)';
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_SEAM, ['SEAM_WF',num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(2);
end

% img = ans;
% img_m = reshape(img, 150,150);
% img_m = imrotate(img_m,-90);
% imshow(img_m);
% score_red_WF_proj_SEAM =  mean(red_WF_proj_SEAM,1);

%% PLOT PROJ

figp = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_WF = rescale(mean(abs(WF_proj_WF),1));
plot(plot_WF(:,1:selectx),"r");
plot_WF_SEAM = rescale(mean(abs(WF_proj_SEAM),1))';
plot((plot_WF_SEAM(1:selectx,:)),"b");
legend({'WF projected on WF' 'WF projected on SEAM' });
saveas(figp, fullfile(path_WF, ['WF_WF_',num2str(select),'_','.jpg'])); % changed how name is saved
close all


%% PLOT PROJ FLIP

figx = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_SEAM = rescale(mean(abs(SEAM_proj_SEAM),1));
plot(plot_SEAM(:,1:selectx), "r");
plot_SEAM_WF = rescale(mean(abs(SEAM_proj_WF),1))';
plot((plot_SEAM_WF(1:selectx,:)),"g");
legend({'SEAM projected on SEAM' 'SEAM projected on WF' });
saveas(figx, fullfile(path_SEAM, ['SEAM_WF_',num2str(select),'_','.jpg'])); % changed how name is saved
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




