% clc

% % clear
% x_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/";
% % raceYY_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_raceYY.txt";
% raceXX_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_raceXX.txt";

% f = fopen(raceXX_path,'r');
% f_path = "/media/nyma/EXTERNAL1/fairface-img-margin025-trainval";
% batch_x = 5000;
% tline = fgetl(f);

% % raceYY_list = zeros(batch_x,22500);
% % raceXX_list = zeros(batch_x,22500);
% files_img={};
% figh = 
% i=1;

% % raceYY_list = extractFace(f_path, tline, raceYY_list,i,batch_x,f);
% % raceXX_list = extractFace(f_path, tline, raceXX_list,i,batch_x,f);
% fclose(f);
% clc

% load("/media/nyma/EXTERNAL1/PCADATA/WF_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/WM_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/BF_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/BM_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/LM_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/LF_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/SEAF_1.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/SEAM_1.mat")

select = 4000;
selectx = select;
close all

raceYY_str = {'LF_'};
raceXX_str = {'LM_'};
% raceYY_list = WF_list;
% raceXX_list = BM_list;
race_res = {'RES_'};

c_select = {'red',[0.51 0.2 0.4 ],'blue','black','green','magenta','cyan','yellow'};
x_races  =  ["SEAM","BF","WM", "WF","SEAF","BM","LM", "LF"];

x_val  =  {raceA_Val, raceB_Val, raceC_Val, raceD_Val, raceE_Val, raceF_Val, raceG_Val, raceH_Val};
x_vec  =  {raceA_Vec, raceB_Vec, raceC_Vec, raceD_Vec, raceE_Vec, raceF_Vec, raceG_Vec, raceH_Vec};
x_s    =  {raceA_S, raceB_S, raceC_S, raceD_S, raceE_S, raceF_S, raceG_S, raceH_S};
x_list =  {SEAM_list, BF_list, WM_list, WF_list, SEAF_list, BM_list, LM_list, LF_list};


%% Race Index
c_str = strsplit(raceXX_str{1}, '_');
c_str = c_str{1};
[i_x, idx] = max(strcmp(x_races, c_str));

yc_str = strsplit(raceYY_str{1}, '_');
yc_str = yc_str{1};


%% Race Index
[i_y, idy] = max(strcmp(x_races, yc_str));

raceXX_Val = x_val{idx};
raceXX_Vec = x_vec{idx};
raceXX_S   = x_s{idx};

raceYY_Val = x_val{idy};
raceYY_Vec = x_vec{idy};
raceYY_S   = x_s{idy};

raceXX_list = x_list{idx};
raceYY_list = x_list{idy};

rec_ = raceXX_list(1:4999,:);
img = rec_;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


raceYY_mean = mean(raceYY_list, 1);
raceXX_mean = mean(raceXX_list, 1); 

raceYY_matrix = raceYY_list';
raceXX_matrix = raceXX_list';

%% Subtract Mean
raceXX_matrix_nm = raceXX_matrix - raceXX_mean';
raceYY_matrix_nm = raceYY_matrix - raceYY_mean';

%% Subtract Flip Mean
raceXX_matrix_nmyy = raceXX_matrix - raceYY_mean';
raceYY_matrix_nmxx = raceYY_matrix - raceXX_mean';

raceYY_matrix_nmyy = raceYY_matrix - raceYY_mean';
%%
%%
%% BREAK POINT CHECK
% rec_ = raceXX_matrix_nm(:,1:4999);
% img = rec_;
% img_m = reshape(img(:,100), 150,150);
% img_m = imrotate(img_m,-90);
% imshow(img_m);

%% PCA Extraction %%
% [raceXX_Vec, raceXX_S, raceXX_Val] = pca(raceXX_matrix_nm'); 
% pause(5)
% [raceYY_Vec, raceYY_S, raceYY_Val] = pca(raceYY_matrix_nm');
% 

%% VECTOR ANGLE AND DISTANCE
raceXX_Vec_angle = (raceXX_Vec);
raceYY_Vec_angle = (raceYY_Vec); 
a1 = raceXX_Vec(:,1:select);
b1 = raceYY_Vec(:,1:select);
% clc

angle_begining = subspace((a1), (b1))

% aa1 = raceXX_Vec(:,select:4999);
% bb1 = raceYY_Vec(:,select:4999);
% % clc
% 
% angle_remain = subspace((aa1), (bb1))
% distance = (a1 - b1).^2;
% distance = mean(sum(distance));
% pdist = sqrt(pdist2(a1,b1));
% pdist
% distance
% angle_x 

%% PROJECTION %%
raceXX_matrix_nm    =  raceXX_matrix_nm(:,1:4999)';
raceXX_proj_raceXX  =  raceXX_matrix_nm  * (diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'; 

raceYY_matrix_nm     = raceYY_matrix_nm(:,1:4999)';
raceYY_proj_raceYY   = raceYY_matrix_nm  * (diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')'; 



%% PROJECTION FLIP %%
% raceXX_matrix_raceYY =  raceXX_matrix - raceYY_mean';
% raceXX_matrix_raceYY =  raceXX_matrix - raceXX_mean';
% raceXX_proj_raceYY   =  (raceXX_matrix_raceYY(:,1:4999))' * (diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')'; 


% raceYY_matrix_raceXX = raceYY_matrix - raceXX_mean';
raceYY_matrix_raceXX = raceYY_matrix - raceYY_mean';
raceYY_proj_raceXX   = (raceYY_matrix_raceXX(:,1:4999))' * (diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'; 


%% RECONSTRUCT %%
% raceXX_Xhat  = (raceXX_Vec * raceXX_S(1:4999,:)') + raceXX_mean';
% img_i = mat2gray(reshape( raceXX_Xhat (:,14), 150, 150));
% img_i = imrotate(img_i,-90);
% % imshow(img_i);
% 
% raceYY_Xhat  = (raceYY_Vec * raceYY_S(1:4999,:)') + raceYY_mean';
% img_i = mat2gray(reshape( raceYY_Xhat (:,14), 150, 150));
% img_i = imrotate(img_i,-90);
% % imshow(img_i);


%% FLIP RECONSTRCT %%
% raceYY projected on raceYY 
rec_raceYY_proj_raceYY = (raceYY_proj_raceYY  * pinv((diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')')); %+  raceXX_mean;
img = rec_raceYY_proj_raceYY ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


% % raceXX projected on raceXX
% rec_raceXX_proj_raceXX   =  raceXX_proj_raceXX * pinv((diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'); %+ raceYY_mean; 
% img_m = reshape(img(100,:), 150,150);
% img_m = imrotate(img_m,-90);
% % % imshow(img_m);


% % raceYY projected on raceXX 
rec_raceYY_proj_raceXX = (raceYY_proj_raceXX  * pinv((diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')')); %+  raceXX_mean;
img = rec_raceYY_proj_raceXX ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

%% PATH
path = "/home/nyma/Pictures";
path_res="/home/nyma/Pictures/RES_";
path_raceYY = '';
path_raceXX = '';
% fullfile(path_x{2}, [raceXX_str{:}, num2str(ix),'_',num2str(select)
path_x = {path_raceYY, path_raceXX};
path_race = [raceYY_str, raceXX_str];
for i = 1:length(path_x)
    path_x{i} = fullfile(path, path_race{i});
    if ~exist(path_x{i}, 'dir')
        mkdir(path_x{i})   
    end
end

%% RESIDUE 
% x_raceXX_proj_raceXX =  raceXX_matrix_nm' - rec_raceXX_proj_raceXX'; 
% x_raceXX_proj_raceXX = abs(x_raceXX_proj_raceXX);
% score_red_raceXX_proj_raceXX = mean(x_raceXX_proj_raceXX,1);

% raceYY_matrix_mn_rmw  = raceYY_matrix - raceYY_mean';
% raceYY_matrix_XX_nm    = (raceYY_matrix_nm');

x_raceYY_proj_raceXX  = (raceYY_matrix_nmyy(:,1:4999))' - rec_raceYY_proj_raceXX; 
x_raceYY_proj_raceXX  =   abs(x_raceYY_proj_raceXX');
score_red_raceYY_proj_raceXX =  mean(x_raceYY_proj_raceXX,1);


x_raceYY_proj_raceYY  =  (raceYY_matrix_nm') - rec_raceYY_proj_raceYY'; 
x_raceYY_proj_raceYY  = abs(x_raceYY_proj_raceYY);
score_red_raceYY_proj_raceYY =  mean(x_raceYY_proj_raceYY,1);


figh = figure('Renderer', 'painters', 'Position', [700 700 900 600]);
a = histogram(score_red_raceYY_proj_raceYY, 100,'Facecolor', c_select{idy});

Spacing_lines = 3;
score_std_raceYY_proj_raceYY = std(score_red_raceYY_proj_raceYY);
title(['Residue Plot '  ' eigen faces used = ' num2str(select)])
xlabel('Residue Values') ;
ylabel('Frequency'); 
hold on;

score_std_raceYY_proj_raceXX = std(score_red_raceYY_proj_raceXX);
b = histogram(score_red_raceYY_proj_raceXX, 100,'Facecolor', c_select{idx});
 
[sumAOC, AreaB] = AOC(a.Values, a.BinEdges,a.BinWidth, b.Values, b.BinEdges, b.BinWidth);
sumAOC
AreaB

ha = plot(NaN,NaN,'or');
hb = plot(NaN,NaN,'*b');

AA  = (strcat(raceYY_str{:} ," projected on ", raceYY_str{:}, '\ std =  ', num2str(score_std_raceYY_proj_raceYY)));
AB  = (strcat(raceYY_str{:} ," projected on ", raceXX_str{:}, '\ std =  ', num2str(score_std_raceYY_proj_raceXX)));
AUC = (strcat(' Overlap =  ', num2str(sumAOC)));
AUA = (strcat(raceXX_str{:}, ' Area =  ', num2str(AreaB)));
legend({AA  AB AUC AUA});

saveas(figh, fullfile(path_x{1}, [raceXX_str{:}, 'rec_',raceYY_str{:}, '_', raceYY_str{:}, num2str(select),'_','.jpg'])); % changed how name is saved
pause(2);
% close all

% 
% for i= 1:5
%     figs = figure();
%     img_1 = raceYY_matrix_nm(i,:);
%     img_2 = rec_raceYY_proj_raceXX(i,:);
%     img_3 = x_raceYY_proj_raceXX(:,i);
%     img_1 = reshape(img_1, 150,150);
%     img_1 = imrotate(img_1,-90);
%     subplot(131);imshow(img_1)
%     img_2 = reshape(img_2, 150,150);
%     img_2 = imrotate(img_2,-90);
%     subplot(132);imshow(img_2 ) 
%     img_3 = reshape(img_3, 150,150);
%     img_3 = imrotate(img_3,-90);
%     subplot(133);imshow(img_3) 
%     saveas(figs, fullfile(path_x{1}, [raceYY_str{:}, raceXX_str{:}, num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
%     close all
%     pause(1);
% end
% 
% close all
% x_raceXX_proj_raceYY  =  (raceXX_matrix(:,1:4999) - raceYY_mean(:,1:4999)) - rec_raceXX_proj_raceYY'; 
% x_raceXX_proj_raceYY = abs (x_raceXX_proj_raceYY);
% score_red_raceXX_proj_raceYY =  (mean(x_raceXX_proj_raceYY,1));
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figh= figure();
% hist(score_red_raceXX_proj_raceXX,100);
% hold on;
% h = findobj(gca,'Type','patch');
% score_std_raceXX_proj_raceXX = std(score_red_raceXX_proj_raceXX);
% text(.09, 125,[raceXX_str{:}, 'std_','_', raceXX_str{:},' = ', num2str(score_std_raceXX_proj_raceXX)], 'Color', [0 0.5 0.5])
% h.EdgeColor = 'w'; 
% h.FaceColor = [0 0.5 0.5];
% h.EdgeColor = 'w'; 
% h.FaceAlpha = 0.9;
% title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
% xlabel('Residue Values') ;
% ylabel('Frequency'); 
% hist(score_red_raceXX_proj_raceYY, 100);
% score_std_raceXX_proj_raceYY = std(score_red_raceXX_proj_raceYY);
% text(.09, 115,[raceXX_str{:},'std_', '_', raceYY_str{:},' = ', num2str(score_std_raceXX_proj_raceYY)], 'Color','b')
% BB = (strcat(raceXX_str{:} ,'projected_on_', raceXX_str{:}));
% BA = (strcat(raceXX_str{:} ,'projected_on_', raceYY_str{:}));
% legend({BB BA});
% % legend({'raceXX projected on raceXX' 'raceXX projected on raceYY' });
% saveas(figh, fullfile(path_x{2}, [raceYY_str{:}, 'rec_BFyXX', num2str(select),'_','.jpg'])); % changed how name is saved
% close all
% pause(1);
% 
% % pd = fitdist(score_red_raceXX_proj_raceYY','kernel','Kernel','normal');
% % x =  -.5:.1:.5;
% % y = pdf(pd,x);
% % plot(x,y,'Color','black','LineStyle','-')
% 
% for i= 1:5
%     figs = figure();
%     img_1 = raceXX_matrix_nm(i,:);
%     img_2 = rec_raceXX_proj_raceYY(i,:);
%     img_3 = x_raceXX_proj_raceYY(:,i) ;%-  raceXX_mean(:,i)';
%     img_1 = reshape(img_1, 150,150);
%     img_1 = imrotate(img_1,-90);
%     subplot(131);imshow(img_1)
%     img_2 = reshape(img_2, 150,150);
%     img_2 = imrotate(img_2,-90);
%     subplot(132);imshow(img_2 ) 
%     img_3 = reshape(img_3, 150,150);
%     img_3 = imrotate(img_3,-90);
%     subplot(133);imshow(img_3) 
%     saveas(figs, fullfile(path_x{2}, [raceXX_str{:},raceYY_str{:}, num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
%     close all
%     pause(2);
% end
% 
% % img = ans;
% % img_m = reshape(img, 150,150);
% % img_m = imrotate(img_m,-90);
% % imshow(img_m);
% % score_red_raceYY_proj_raceXX =  mean(red_raceYY_proj_raceXX,1);
% 
% %% PLOT PROJ
% 
% figp = figure();
% hold on
% title(" Projections")
% xlabel('Number of Vectors') 
% ylabel('Projection Coefficients') 
% plot_raceYY = rescale(mean(abs(raceYY_proj_raceYY),1));
% plot(plot_raceYY(:,1:selectx),"r");
% plot_raceYY_raceXX = rescale(mean(abs(raceYY_proj_raceXX),1))';
% % plot((plot_raceYY_raceXX(1:selectx,:)),"b");
% AA = (strcat(raceYY_str{:} ,'projected_on_', raceYY_str{:}));
% AB = (strcat(raceYY_str{:} ,'projected_on_', raceXX_str{:}));
% legend({AA AB});
% % legend({'raceYY projected on raceYY' 'raceYY projected on raceXX' });
% saveas(figp, fullfile(path_x{1}, [raceYY_str{:},'_', raceYY_str{:},num2str(select),'_','.jpg'])); % changed how name is saved
% close all
% 
% 
% %% PLOT PROJ FLIP
% 
% figx = figure();
% hold on
% title(" Projections")
% xlabel('Number of Vectors') 
% ylabel('Projection Coefficients') 
% plot_raceXX = rescale(mean(abs(raceXX_proj_raceXX),1));
% plot(plot_raceXX(:,1:selectx), "r");
% plot_raceXX_raceYY = rescale(mean(abs(raceXX_proj_raceYY),1))';
% plot((plot_raceXX_raceYY(1:selectx,:)),"g");
% BB = (strcat(raceXX_str{:} ,'projected_on_', raceXX_str{:}));
% BA = (strcat(raceXX_str{:} ,'projected_on_', raceYY_str{:}));
% legend({BB BA});
% % legend({'raceXX projected on raceXXspectralcluster' 'raceXX projected on raceYY' });
% saveas(figx, fullfile(path_x{2}, [raceXXB_str{:},'_', raceYY_str{:},num2str(select),'_','.jpg'])); % changed how name is saved
% close all
% 
% 
% % img_mean= imrotate(img_mean,-90);
% % img_mean = img_mean';
% 
% % image_matrix = files_list';
% % image_matrix_no_mean =  image_matrix  - img_mean;
% 
% % eig_face = pca(image_matrix_no_mean);
% % cov_no_mean = image_matrix_no_mean * image_matrix_no_mean';
% % cov_n = cov(image_matrix_no_mean');
% % [D_vec, D_val] = eig(cov_n);
% % [Vectors_m, D_Values_m]= eig(cov_no_mean);
% 
% % img_i = reshape(Vectors_m(:,2), 150,150);
% % normalizedImage = uint8(255*mat2gray(img_i));
% % img_i = imrotate(img_i,-90);
% % imshow(normalizedImage);
% 
% % imv = Vectors(:,selected)*image_matrix;
% % save(fullfile(x_path,cov_image_matrix))
% % save(fullfile(x_path,Vectors))
% % save(fullfile(x_path,D_values))
% 
% 
% 
% % idx = spectralcluster(raceYY_Vec,7);
