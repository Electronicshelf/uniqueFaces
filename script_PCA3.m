clc
% % clear
% x_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/";
% % raceC_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_raceC.txt";
% raceG_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_raceG.txt";

% f = fopen(raceG_path,'r');
% f_path = "/media/nyma/EXTERNAL1/fairface-img-margin025-trainval";
% batch_x = 5000;
% tline = fgetl(f);

% % raceC_list = zeros(batch_x,22500);
% % raceG_list = zeros(batch_x,22500);
% files_img={};
% figh = 
% i=1;

% % raceC_list = extractFace(f_path, tline, raceC_list,i,batch_x,f);
% % raceG_list = extractFace(f_path, tline, raceG_list,i,batch_x,f);
% fclose(f);
% clc


% load("/media/nyma/EXTERNAL1/PCADATA/data_matlab.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/BFmatlab.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/WMmatlab.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/BM.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/BMmatlab.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/LM.mat")
% load("/media/nyma/EXTERNAL1/PCADATA/LF.mat")

raceC_str = {'WM_'};
raceG_str = {'LM_'};
race_res = {'RES_'};
raceC_list = WM_list;
raceG_list = LM_list;

select = 600;
selectx = select;
c_select = {'red','yellow','blue','black','green','magenta','cyan','yellow'};
x_races  =  ["SEAM","BF","WM", "WF","SEAF","BM","LM", "LF"];
% x_races  =  ["SEAM","Black Female","White Male", "White Female","BM","Black Male","Latino Male", "Latino Female"];
c_str = strsplit(raceG_str{1},'_');
c_str = c_str{1};

%% Hyper Parameters'b'
[i_x, idx] = max(strcmp(x_races, c_str));
% x_index = find(x_races , c_str{1});
% index = find((x_races == c_str));
% cc = find(ismember(x_races , c_str{1}));
% c_select{idx}

% raceC_list = WM_list;
% raceG_list = LF_list;

raceC_mean = mean(raceC_list, 1);
raceG_mean = mean(raceG_list, 1); 

raceC_matrix = raceC_list';
raceG_matrix = raceG_list';

raceG_matrix_nm = raceG_matrix - raceG_mean';
raceC_matrix_nm = raceC_matrix - raceC_mean';

%% PCA Extraction %%
% [raceG_Vec, raceG_S, raceG_Val] = pca(raceG_matrix_nm'); 
% pause(5)
% [raceC_Vec, raceC_S, raceC_Val] = pca(raceC_matrix_nm');


%% VECTOR ANGLE
% raceG_Vec_angle = (raceG_Vec)';
% raceC_Vec_angle = (raceC_Vec)'; 
% a = raceG_Vec_angle;
% b = raceC_Vec_angle;
clc
% angle_x = subspace(raceC_Vec_angle, raceG_Vec_angle);
% 
% angle_y = mPrinAngles(raceC_Vec_angle,raceG_Vec_angle);
% angle_z = real(acosd(dot(a, b) / (norm(a) * norm(b))));
% u=a';
% v=b';

% CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
% ThetaInDegrees = real(acosd(CosTheta));
% angle_list = zeros(1,30);
% raceG_Vec_P1 = raceG_Vec(:,1:select) * raceG_Vec(:,1:select)';
% raceC_Vec_P1 = raceC_Vec(:,1:select) * raceC_Vec(:,1:select)';
% norm_dist = norm(raceG_Vec_P1-raceC_Vec_P1);
% for i = 1:  4999
%     u = raceG_Vec_angle(i,:);
%     v = raceC_Vec_angle(i,:);
%     angle = dot(u,v)/(norm(u)* norm(v));
%     angle =  real(acosd(angle));
%     angle_list(i) = angle;
% end%% Hyper Parameters'b'
% c_select =  {'blue','yellow','red','magenta', 'green', 'cyan', 'black', 'brown'};
% x_races  =  ["SEAM","Black Female","White Male", "White Female","BM","Black Male","Latino Male", "Latino Female"];
% c_str = 
% [i_x, idx] = max(strcmp(x_races, c_str));
% disp(angle_list(:));

clc
% xangle_list = zeros(1,4999);
% dist_x = zeros(1,4999);
% for i = 1:  4999
%     u = raceG_Vec_angle(i,:);
%     v = raceC_Vec_angle(i,:);
%     angle = (u*v');
%     angle =  real(acosd(angle));
%     xangle_list(i) = angle;
%     dist_x(i) = sind(angle);
% end
% disp(xangle_list(:));
% disp(dist_x());

% %% PROJECTION %%
% raceG_proj = raceG_matrix_nm(:,1:4999)' * raceG_Vec; 
% raceG_proj  =  raceG_proj  * diag(raceG_Val) ;
% 
% raceC_proj = raceC_matrix_nm(:,1:4999)' * raceC_Vec; 
% raceC_proj  =  raceC_proj  * diag(raceC_Val) ;



%% PROJECTION %%
raceG_matrix_nm  =  raceG_matrix_nm(:,1:4999)';
raceG_proj_raceG  = raceG_matrix_nm * (diag(raceG_Val(1:select,:)) * raceG_Vec(:,1:select)')'; 

raceC_matrix_nm = raceC_matrix_nm(:,1:4999)';
raceC_proj_raceC   = raceC_matrix_nm  * (diag(raceC_Val(1:select,:)) * raceC_Vec(:,1:select)')'; 


%% PROJECTION FLIP %%
raceG_matrix_raceC =  raceG_matrix - raceC_mean';
raceG_proj_raceC   =  raceG_matrix_raceC(:,1:4999)' * (diag(raceC_Val(1:select,:)) * raceC_Vec(:,1:select)')'; 


raceC_matrix_raceG = raceC_matrix - raceG_mean';
raceC_proj_raceG   = raceC_matrix_raceG(:,1:4999)' * (diag(raceG_Val(1:select,:)) * raceG_Vec(:,1:select)')'; 

%% Angle Vectors
% u = abs(mean(raceG_Vec,1)');
% v = abs(mean(raceC_Vec,1)');
% 
% for i = 1:4999
% angle = vec_Angle(u,v);
% end

%% RECONSTRUCT %%
raceG_Xhat  = (raceG_Vec * raceG_S(1:4999,:)') + raceG_mean';
img_i = mat2gray(reshape( raceG_Xhat (:,14), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

raceC_Xhat  = (raceC_Vec * raceC_S(1:4999,:)') + raceC_mean';
img_i = mat2gray(reshape( raceC_Xhat (:,78), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

%% FLIP RECONSTRCT %%
% raceC projected on raceC 
rec_raceC_proj_raceC = (raceC_proj_raceC  * pinv((diag(raceC_Val(1:select,:)) * raceC_Vec(:,1:select)')')); %+  raceG_mean;
img = rec_raceC_proj_raceC ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


% raceG projected on raceG
rec_raceG_proj_raceG   =  raceG_proj_raceG * pinv((diag(raceG_Val(1:select,:)) * raceG_Vec(:,1:select)')'); %+ raceC_mean; 
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% % imshow(img_m);


% % raceC projected on raceG 
rec_raceC_proj_raceG = (raceC_proj_raceG  * pinv((diag(raceG_Val(1:select,:)) * raceG_Vec(:,1:select)')')); %+  raceG_mean;
img = rec_raceC_proj_raceG ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);

%% PATH
path = "/home/nyma/Pictures";
path_res="/home/nyma/Pictures/RES_";
path_raceC = '';
path_raceG = '';
% fullfile(path_x{2}, [raceG_str{:}, num2str(ix),'_',num2str(select)
path_x = {path_raceC, path_raceG};
path_race = [raceC_str, raceG_str];
for i = 1:length(path_x)
    path_x{i} = fullfile(path, path_race{i});
    if ~exist(path_x{i}, 'dir')
        mkdir(path_x{i})   
    end
end

% select = 50;
raceG_matrix_raceC =  raceG_matrix - raceC_mean';
raceG_proj_raceC   =  raceG_matrix_raceC(:,1:4999)' * (diag(raceC_Val(1:select,:)) * raceC_Vec(:,1:select)')'; 


raceC_matrix_raceG = raceC_matrix - raceG_mean';
raceC_proj_raceG   = raceC_matrix_raceG(:,1:4999)' * (diag(raceG_Val(1:select,:)) * raceG_Vec(:,1:select)')'; 

%% Angle Vectors
% u = abs(mean(raceG_Vec,1)');
% v = abs(mean(raceC_Vec,1)');r(path_x{i})   

% path_raceC = "/home/nyma/Pictures/raceC";
% path_raceG = "/home/nyma/Pictures/raceG";
% raceG projected on raceC 

rec_raceG_proj_raceC   =  raceG_proj_raceC * pinv((diag(raceC_Val(1:select,:)) * raceC_Vec(:,1:select)')'); %+ raceC_mean; 
img = rec_raceG_proj_raceC ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


%% RESIDUE 
x_raceC_proj_raceC  =  (raceC_matrix_nm') - rec_raceC_proj_raceC'; 
x_raceC_proj_raceC  = abs(x_raceC_proj_raceC);
score_red_raceC_proj_raceC =  mean(x_raceC_proj_raceC,1);

% pd = fitdist((score_red_raceC_proj_raceC'),'kernel','Kernel','normal');
% x = -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','green','LineStyle','-')

x_raceG_proj_raceG =  raceG_matrix_nm' - rec_raceG_proj_raceG'; 
x_raceG_proj_raceG = abs(x_raceG_proj_raceG);
score_red_raceG_proj_raceG = mean(x_raceG_proj_raceG,1);

% pd = fitdist(score_red_raceG_proj_raceG','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','c','LineStyle','-')


%% Generate Residue Plot
% clc
% res_list = zeros(3999);
% for i = 1:5
%     img_1 = raceG_matrix_nm(i,:);
%     img_2 = rec_raceG_proj_raceG(i,:);
%     img_3 = x_raceG_proj_raceG(:,i);
%     img_1 = reshape(img_1, 150,150);
%     img_1 = imrotate(img_1,-90);
%     img_2 = reshape(img_2, 150,150);
%     img_2 = imrotate(img_2,-90);
%     img_3 = reshape(img_3, 150,150);
%     img_3 = imrotate(img_3,-90);
%     res_img_3 = immse(img_1,img_2);
%     disp(res_img_3 )
%     res_list(i) = res_img_3 ;
%     i
%     pause(1);
% end
% %%

% bx=   [0.0090    0.0107    0.0119    0.0128    0.0137    0.0162    0.0181    0.0202    0.0239    0.0254    0.0303    0.0477    0.0512];
% res_list =  res_list(:,1);
% [vx , ix] = max(res_list);
% xy = find(res_list>=0.0512);
% clc
% xstr=num2str(0.0613);
% length(xy)
% for i = 1:length(xy)
%     figs = figure();
%     ix=xy(i);
%     img_xx = raceG_matrix_nm(ix,:);
%     img_xx = reshape(img_xx, 150,150);
%     img_xx = imrotate(img_xx,-90);
%     imshow(img_xx);
%     saveas(figs, fullfile(path_res, ['Res_', num2str(ix),'_', xstr,'.jpg']));
% %     pause(1);
%     close all
%     if i == 20
%         break;
%     end
% end

%%
ffa = 1;

% for i= 1:5
%     figs = figure();
%     img_1 = raceG_matrix_nm(i,:);
%     img_2 = rec_raceG_proj_raceG(i,:);
%     img_3 = x_raceG_proj_raceG(:,i);
%     img_1 = reshape(img_1, 150,150);
%     img_1 = imrotate(img_1,-90);
%     subplot(131);imshow(img_1)
%     img_2 = reshape(img_2, 150,150);
%     img_2 = imrotate(img_2,-90);
%     subplot(132);imshow(img_2 ) 
%     img_3 = reshape(img_3, 150,150);
%     img_3 = imrotate(img_3,-90);
%     subplot(133);imshow(img_3) 
%     saveas(figs, fullfile(path_x{2}, [raceG_str{:}, num2str(i),'_',num2str(select),'.jpg'])); % changed how name is saved
%     close all
%     pause(2);
% end

x_raceC_proj_raceG  =  (raceC_matrix(:,1:4999) - raceG_mean(:,1:4999)) - rec_raceC_proj_raceG'; 
x_raceC_proj_raceG  = abs(x_raceC_proj_raceG);
score_red_raceC_proj_raceG =  mean(x_raceC_proj_raceG,1);


figh = figure();
hist(score_red_raceC_proj_raceC, 100,'Facecolor', 'blue');
% h = findobj(gca,'Type','patch');
% h.FaceColor = [1 0 0];
% ex = xlim;
% ey = ylim;
% centerX = ex/2;
% centerY = ey/2;
score_std_raceC_proj_raceC = std(score_red_raceC_proj_raceC);
% text(centerX(2)+0.009, centerY(2), [raceC_str{:},' std_','_', raceC_str{:},' = ', num2str(score_std_raceC_proj_raceC)], 'Color', 'black');
% h.EdgeColor = 'w'; 
% h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)])
xlabel('Residue Values') ;
ylabel('Frequency'); 
hold on;

score_std_raceC_proj_raceG = std(score_red_raceC_proj_raceG);
% text(centerX(2), centerY(2)+10,[raceC_str{:},' std_', '_', raceG_str{:},' = ', num2str(score_std_raceC_proj_raceG)], 'Color',  c_select{idx});
histogram(score_red_raceC_proj_raceG, 100,'Facecolor', c_select{idx});
AA = (strcat(raceC_str{:} ," projected on ", raceC_str{:}, '\ std =  ', num2str(score_std_raceC_proj_raceC)));
AB = (strcat(raceC_str{:} ," projected on ", raceG_str{:}, '\ std =  ', num2str(score_std_raceC_proj_raceG)));
legend({AA  AB});
saveas(figh, fullfile(path_x{1}, [raceG_str{:}, 'rec_xy', num2str(select),'_','.jpg'])); % changed how name is saved

pause(2);
% close all

for i= 1:5
    figs = figure();
    img_1 = raceC_matrix_nm(i,:);
    img_2 = rec_raceC_proj_raceG(i,:);
    img_3 = x_raceC_proj_raceG(:,i);
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_x{1}, [raceC_str{:}, raceG_str{:}, num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(1);
end

close all
x_raceG_proj_raceC  =  (raceG_matrix(:,1:4999) - raceC_mean(:,1:4999)) - rec_raceG_proj_raceC'; 
x_raceG_proj_raceC = abs (x_raceG_proj_raceC);
score_red_raceG_proj_raceC =  (mean(x_raceG_proj_raceC,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figh= figure();
hist(score_red_raceG_proj_raceG,100);
hold on;
h = findobj(gca,'Type','patch');
score_std_raceG_proj_raceG = std(score_red_raceG_proj_raceG);
text(.09, 125,[raceG_str{:}, 'std_','_', raceG_str{:},' = ', num2str(score_std_raceG_proj_raceG)], 'Color', [0 0.5 0.5])
h.EdgeColor = 'w'; 
h.FaceColor = [0 0.5 0.5];
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
xlabel('Residue Values') ;
ylabel('Frequency'); 
hist(score_red_raceG_proj_raceC, 100);
score_std_raceG_proj_raceC = std(score_red_raceG_proj_raceC);
text(.09, 115,[raceG_str{:},'std_', '_', raceC_str{:},' = ', num2str(score_std_raceG_proj_raceC)], 'Color','b')
BB = (strcat(raceG_str{:} ,'projected_on_', raceG_str{:}));
BA = (strcat(raceG_str{:} ,'projected_on_', raceC_str{:}));
legend({BB BA});
% legend({'raceG projected on raceG' 'raceG projected on raceC' });
saveas(figh, fullfile(path_x{2}, [raceC_str{:}, 'rec_', num2str(select),'_','.jpg'])); % changed how name is saved
close all
pause(1);

% pd = fitdist(score_red_raceG_proj_raceC','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','black','LineStyle','-')

for i= 1:5
    figs = figure();
    img_1 = raceG_matrix_nm(i,:);
    img_2 = rec_raceG_proj_raceC(i,:);
    img_3 = x_raceG_proj_raceC(:,i) ;%-  raceG_mean(:,i)';
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_x{2}, [raceG_str{:},raceC_str{:}, num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(2);
end

% img = ans;
% img_m = reshape(img, 150,150);
% img_m = imrotate(img_m,-90);
% imshow(img_m);
% score_red_raceC_proj_raceG =  mean(red_raceC_proj_raceG,1);

%% PLOT PROJ

figp = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_raceC = rescale(mean(abs(raceC_proj_raceC),1));
plot(plot_raceC(:,1:selectx),"r");
plot_raceC_raceG = rescale(mean(abs(raceC_proj_raceG),1))';
% plot((plot_raceC_raceG(1:selectx,:)),"b");
AA = (strcat(raceC_str{:} ,'projected_on_', raceC_str{:}));
AB = (strcat(raceC_str{:} ,'projected_on_', raceG_str{:}));
legend({AA AB});
% legend({'raceC projected on raceC' 'raceC projected on raceG' });
saveas(figp, fullfile(path_x{1}, [raceC_str{:},'_', raceC_str{:},num2str(select),'_','.jpg'])); % changed how name is saved
close all


%% PLOT PROJ FLIP

figx = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_raceG = rescale(mean(abs(raceG_proj_raceG),1));
plot(plot_raceG(:,1:selectx), "r");
plot_raceG_raceC = rescale(mean(abs(raceG_proj_raceC),1))';
plot((plot_raceG_raceC(1:selectx,:)),"g");
BB = (strcat(raceG_str{:} ,'projected_on_', raceG_str{:}));
BA = (strcat(raceG_str{:} ,'projected_on_', raceC_str{:}));
legend({BB BA});
% legend({'raceG projected on raceGspectralcluster' 'raceG projected on raceC' });
saveas(figx, fullfile(path_x{2}, [raceGB_str{:},'_', raceC_str{:},num2str(select),'_','.jpg'])); % changed how name is saved
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



% idx = spectralcluster(raceC_Vec,7);
