clc
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

select = 40;
selectx = select;
close all

raceYY_str = {'WF_'};%
raceXX_str = {'SEAF_'};
% raceXX_list = WF_list;
% raceYY_list = BM_list;
race_res = {'RES_'};

c_select = {'red',[0.51 0.2 0.4 ],'blue','black','green','magenta','cyan','yellow'};
x_races  =  ["SEAM","BF","WM", "WF","SEAF","BM","LM", "LF"];

x_val  =  {raceA_Val, raceB_Val, raceC_Val, raceD_Val, raceE_Val, raceF_Val, raceG_Val, raceH_Val};
x_vec  =  {raceA_Vec, raceB_Vec, raceC_Vec, raceD_Vec, raceE_Vec, raceF_Vec, raceG_Vec, raceH_Vec};
x_s    =  {raceA_S, raceB_S, raceC_S, raceD_S, raceE_S, raceF_S, raceG_S, raceH_S};
x_list =  {SEAM_list, BF_list, WM_list, WF_list, SEAF_list, BM_list, LM_list, LF_list};

%% Race Index
c_str = strsplit(raceXX_str{1},'_');
c_str = c_str{1};
[i_x, idx] = max(strcmp(x_races, c_str));

yc_str = strsplit(raceYY_str{1},'_');
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

raceYY_mean = mean(raceYY_list, 1);
raceXX_mean = mean(raceXX_list, 1); 

raceYY_matrix = raceYY_list';
raceXX_matrix = raceXX_list';

raceXX_matrix_nm = raceXX_matrix - raceXX_mean';
raceYY_matrix_nm = raceYY_matrix - raceYY_mean';

%% PCA Extraction %%
% [raceXX_Vec, raceXX_S, raceXX_Val] = pca(raceXX_matrix_nm'); 
% pause(5)
% [raceYY_Vec, raceYY_S, raceYY_Val] = pca(raceYY_matrix_nm');


%% VECTOR ANGLE
raceXX_Vec_angle = (raceXX_Vec);
raceYY_Vec_angle = (raceYY_Vec); 
a1 = raceXX_Vec_angle(:,1:select);
b1 = raceYY_Vec_angle(:,1:select);
% clc

angle_x = subspace(abs(a1), abs(b1));
angle_x 

% angle_y = mPrinAngles(a,b);
% angle_z = real(acosd(dot(a, b) / (norm(a) * norm(b))));
% angle_z

% u = a';
% v = b';
% 
% CosThet = acos(abs(a'*b));
% 
% CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
% CosTheta
% aaaa = 1;
% ThetaInDegrees = real(acosd(CosTheta));

% angle_list = zeros(1,select);
% raceXX_Vec_P1 = raceXX_Vec(:,1:select) * raceXX_Vec(:,1:select)';
% raceYY_Vec_P1 = raceYY_Vec(:,1:select) * raceYY_Vec(:,1:select)';
% norm_dist = norm(raceXX_Vec_P1-raceYY_Vec_P1);

% for i = 1:  4999
%     u = raceXX_Vec_angle(i,:);
%     v = raceYY_Vec_angle(i,:);
%     angle = dot(u,v)/(norm(u)* norm(v));
% %     angle =  real(acosd(angle));
%     angle_list(i) = angle;
% end

%% Hyper Parameters
% disp(angle_list(:));

% clc
% xangle_list = zeros(1,select);
% dist_x = zeros(1,select);
% for i = 1:  select
%     u = raceXX_Vec_angle(i,:);
%     v = raceYY_Vec_angle(i,:);
%     angle = (u*v');
% %     angle =  real(acosd(angle));
%     xangle_list(i) = angle;
%     dist_x(i) = sind(angle);
% end


% %% PROJECTION %%
% raceXX_proj = raceXX_matrix_nm(:,1:4999)' * raceXX_Vec; 
% raceXX_proj  =  raceXX_proj  * diag(raceXX_Val) ;
% 
% raceYY_proj = raceYY_matrix_nm(:,1:4999)' * raceYY_Vec; 
% raceYY_proj  =  raceYY_proj  * diag(raceYY_Val) ;


%% PROJECTION %%
raceXX_matrix_nm  =  raceXX_matrix_nm(:,1:4999)';
raceXX_proj_raceXX  = raceXX_matrix_nm * (diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'; 

raceYY_matrix_nm = raceYY_matrix_nm(:,1:4999)';
raceYY_proj_raceYY   = raceYY_matrix_nm  * (diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')'; 


%% PROJECTION FLIP %%
raceXX_matrix_raceYY =  raceXX_matrix - raceYY_mean';
raceXX_proj_raceYY   =  raceXX_matrix_raceYY(:,1:4999)' * (diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')'; 


raceYY_matrix_raceXX = raceYY_matrix - raceXX_mean';
raceYY_proj_raceXX   = raceYY_matrix_raceXX(:,1:4999)' * (diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'; 

%% Angle Vectors
% u = abs(mean(raceXX_Vec,1)');
% v = abs(mean(raceYY_Vec,1)');
% 
% for i = 1:4999
% angle = vec_Angle(u,v);
% end

%% RECONSTRUCT %%
raceXX_Xhat  = (raceXX_Vec * raceXX_S(1:4999,:)') + raceXX_mean';
img_i = mat2gray(reshape( raceXX_Xhat (:,14), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

raceYY_Xhat  = (raceYY_Vec * raceYY_S(1:4999,:)') + raceYY_mean';
img_i = mat2gray(reshape( raceYY_Xhat (:,78), 150,150));
img_i = imrotate(img_i,-90);
% imshow(img_i);

%% FLIP RECONSTRCT %%
% raceYY projected on raceYY 
rec_raceYY_proj_raceYY = (raceYY_proj_raceYY  * pinv((diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')')); %+  raceXX_mean;
img = rec_raceYY_proj_raceYY ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


% raceXX projected on raceXX
rec_raceXX_proj_raceXX   =  raceXX_proj_raceXX * pinv((diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'); %+ raceYY_mean; 
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% % imshow(img_m);


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

% select = 50;
raceXX_matrix_raceYY =  raceXX_matrix - raceYY_mean';
raceXX_proj_raceYY   =  raceXX_matrix_raceYY(:,1:4999)' * (diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')'; 


raceYY_matrix_raceXX = raceYY_matrix - raceXX_mean';
raceYY_proj_raceXX   = raceYY_matrix_raceXX(:,1:4999)' * (diag(raceXX_Val(1:select,:)) * raceXX_Vec(:,1:select)')'; 

%% Angle Vectors
% u = abs(mean(raceXX_Vec,1)');
% v = abs(mean(raceYY_Vec,1)');  

% path_raceYY = "/home/nyma/Pictures/raceYY";
% path_raceXX = "/home/nyma/Pictures/raceXX";
% raceXX projected on raceYY 

rec_raceXX_proj_raceYY   =  raceXX_proj_raceYY * pinv((diag(raceYY_Val(1:select,:)) * raceYY_Vec(:,1:select)')'); %+ raceYY_mean; 
img = rec_raceXX_proj_raceYY ;
img_m = reshape(img(100,:), 150,150);
img_m = imrotate(img_m,-90);
% imshow(img_m);


%% RESIDUE 
x_raceYY_proj_raceYY  =  (raceYY_matrix_nm') - rec_raceYY_proj_raceYY'; 
x_raceYY_proj_raceYY  = abs(x_raceYY_proj_raceYY);
score_red_raceYY_proj_raceYY =  mean(x_raceYY_proj_raceYY,1);

% pd = fitdist((score_red_raceYY_proj_raceYY'),'kernel','Kernel','normal');
% x = -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','green','LineStyle','-')

x_raceXX_proj_raceXX =  raceXX_matrix_nm' - rec_raceXX_proj_raceXX'; 
x_raceXX_proj_raceXX = abs(x_raceXX_proj_raceXX);
score_red_raceXX_proj_raceXX = mean(x_raceXX_proj_raceXX,1);

% pd = fitdist(score_red_raceXX_proj_raceXX','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','c','LineStyle','-')


%% Generate Residue Plot
% clc
% res_list = zeros(3999);
% for i = 1:5
%     img_1 = raceXX_matrix_nm(i,:);
%     img_2 = rec_raceXX_proj_raceXX(i,:);
%     img_3 = x_raceXX_proj_raceXX(:,i);
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
%     img_xx = raceXX_matrix_nm(ix,:);
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
%     img_1 = raceXX_matrix_nm(i,:);
%     img_2 = rec_raceXX_proj_raceXX(i,:);
%     img_3 = x_raceXX_proj_raceXX(:,i);
%     img_1 = reshape(img_1, 150,150);
%     img_1 = imrotate(img_1,-90);
%     subplot(131);imshow(img_1)
%     img_2 = reshape(img_2, 150,150);
%     img_2 = imrotate(img_2,-90);
%     subplot(132);imshow(img_2 ) 
%     img_3 = reshape(img_3, 150,150);
%     img_3 = imrotate(img_3,-90);
%     subplot(133);imshow(img_3) 
%     saveas(figs, fullfile(path_x{2}, [raceXX_str{:}, num2str(i),'_',num2str(select),'.jpg'])); % changed how name is saved
%     close all
%     pause(2);
% end

x_raceYY_proj_raceXX  =  (raceYY_matrix(:,1:4999) - raceXX_mean(:,1:4999)) - rec_raceYY_proj_raceXX'; 
x_raceYY_proj_raceXX  = abs(x_raceYY_proj_raceXX);
score_red_raceYY_proj_raceXX =  mean(x_raceYY_proj_raceXX,1);


figh = figure();
a = histogram(score_red_raceYY_proj_raceYY, 100,'Facecolor', c_select{idy});
% h = findobj(gca,'Type','patch');
% h.FaceColor = [1 0 0];
% ex = xlim;
% ey = ylim;
% centerX = ex/2;
% centerY = ey/2;
Spacing_lines =3;
score_std_raceYY_proj_raceYY = std(score_red_raceYY_proj_raceYY);
% text(centerX(2)+0.009, centerY(2), [raceYY_str{:},' std_','_', raceYY_str{:},' = ', num2str(score_std_raceYY_proj_raceYY)], 'Color', 'black');
% h.EdgeColor = 'w'; 
% h.FaceAlpha = 0.9;

title(['Residue Plot '  ' eigen faces used = ' num2str(select)])
xlabel('Residue Values') ;
ylabel('Frequency'); 
hold on;

score_std_raceYY_proj_raceXX = std(score_red_raceYY_proj_raceXX);
b = histogram(score_red_raceYY_proj_raceXX, 100,'Facecolor', c_select{idx});

% area_a = sum(a.Values)*a.BinWidth;
% area_b = sum(b.Values)*b.BinWidth;
% 
% xsuby = abs(a.Values - b.Values);
% xandy = (a.Values + b.Values);
% 
% intersection = (xandy - xsuby)/100;


sumAOC = AOC(a.Values, a.BinEdges , b.Values, b.BinEdges);
sumAOC

% figure()
% x_inter = histogram(intersection, 100,'Facecolor', c_select{idx});
% area_x = sum(x_inter.Values)*x_inter.BinWidth;
% sm = 0;
% for i = 1:100
%     x_a =  a.Values(i) * a.BinWidth;
% %     a.BinWidth
%     x_b = b.Values(i)  * b.BinWidth;
% %     b.BinWidth
%     x_c = min(x_a,x_b);
%     sm = sm + x_c; 
% end
% sm = sm/min(sum(a.Values),sum(b.Values));
% sm

    
area_c = abs(area_a - area_b); 

% disp(area_c);
% c = abs(score_red_raceYY_proj_raceYY - score_red_raceYY_proj_raceXX);
% d = histogram(c, 100,'Facecolor', c_select{idx});
ha = plot(NaN,NaN,'or');
% area = sum(d.Values)*d.BinWidth;
% d.BinWidth
% sum(d.Values)
% d.Values
% cc_ = area;
% disp(cc_);

AA = (strcat(raceYY_str{:} ," projected on ", raceYY_str{:}, '\ std =  ', num2str(score_std_raceYY_proj_raceYY)));
AB = (strcat(raceYY_str{:} ," projected on ", raceXX_str{:}, '\ std =  ', num2str(score_std_raceYY_proj_raceXX)));
% OV = (strcat('Intersection = ' ," " , num2str(sm)));

% delete(d);
legend({AA  AB });
% add your text:

saveas(figh, fullfile(path_x{1}, [raceXX_str{:}, 'rec_', num2str(select),'_','.jpg'])); % changed how name is saved
pause(2);


% close all

for i= 1:5
    figs = figure();
    img_1 = raceYY_matrix_nm(i,:);
    img_2 = rec_raceYY_proj_raceXX(i,:);
    img_3 = x_raceYY_proj_raceXX(:,i);
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_x{1}, [raceYY_str{:}, raceXX_str{:}, num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(1);
end

close all
x_raceXX_proj_raceYY  =  (raceXX_matrix(:,1:4999) - raceYY_mean(:,1:4999)) - rec_raceXX_proj_raceYY'; 
x_raceXX_proj_raceYY = abs (x_raceXX_proj_raceYY);
score_red_raceXX_proj_raceYY =  (mean(x_raceXX_proj_raceYY,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figh= figure();
hist(score_red_raceXX_proj_raceXX,100);
hold on;
h = findobj(gca,'Type','patch');
score_std_raceXX_proj_raceXX = std(score_red_raceXX_proj_raceXX);
text(.09, 125,[raceXX_str{:}, 'std_','_', raceXX_str{:},' = ', num2str(score_std_raceXX_proj_raceXX)], 'Color', [0 0.5 0.5])
h.EdgeColor = 'w'; 
h.FaceColor = [0 0.5 0.5];
h.EdgeColor = 'w'; 
h.FaceAlpha = 0.9;
title(['Residue Plot '  ' eigen faces used = ' num2str(select)] )
xlabel('Residue Values') ;
ylabel('Frequency'); 
hist(score_red_raceXX_proj_raceYY, 100);
score_std_raceXX_proj_raceYY = std(score_red_raceXX_proj_raceYY);
text(.09, 115,[raceXX_str{:},'std_', '_', raceYY_str{:},' = ', num2str(score_std_raceXX_proj_raceYY)], 'Color','b')
BB = (strcat(raceXX_str{:} ,'projected_on_', raceXX_str{:}));
BA = (strcat(raceXX_str{:} ,'projected_on_', raceYY_str{:}));
legend({BB BA});
% legend({'raceXX projected on raceXX' 'raceXX projected on raceYY' });
saveas(figh, fullfile(path_x{2}, [raceYY_str{:}, 'rec_BFyXX', num2str(select),'_','.jpg'])); % changed how name is saved
close all
pause(1);

% pd = fitdist(score_red_raceXX_proj_raceYY','kernel','Kernel','normal');
% x =  -.5:.1:.5;
% y = pdf(pd,x);
% plot(x,y,'Color','black','LineStyle','-')

for i= 1:5
    figs = figure();
    img_1 = raceXX_matrix_nm(i,:);
    img_2 = rec_raceXX_proj_raceYY(i,:);
    img_3 = x_raceXX_proj_raceYY(:,i) ;%-  raceXX_mean(:,i)';
    img_1 = reshape(img_1, 150,150);
    img_1 = imrotate(img_1,-90);
    subplot(131);imshow(img_1)
    img_2 = reshape(img_2, 150,150);
    img_2 = imrotate(img_2,-90);
    subplot(132);imshow(img_2 ) 
    img_3 = reshape(img_3, 150,150);
    img_3 = imrotate(img_3,-90);
    subplot(133);imshow(img_3) 
    saveas(figs, fullfile(path_x{2}, [raceXX_str{:},raceYY_str{:}, num2str(i),'_', num2str(select),'_','.jpg'])); % changed how name is saved
    close all
    pause(2);
end

% img = ans;
% img_m = reshape(img, 150,150);
% img_m = imrotate(img_m,-90);
% imshow(img_m);
% score_red_raceYY_proj_raceXX =  mean(red_raceYY_proj_raceXX,1);

%% PLOT PROJ

figp = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_raceYY = rescale(mean(abs(raceYY_proj_raceYY),1));
plot(plot_raceYY(:,1:selectx),"r");
plot_raceYY_raceXX = rescale(mean(abs(raceYY_proj_raceXX),1))';
% plot((plot_raceYY_raceXX(1:selectx,:)),"b");
AA = (strcat(raceYY_str{:} ,'projected_on_', raceYY_str{:}));
AB = (strcat(raceYY_str{:} ,'projected_on_', raceXX_str{:}));
legend({AA AB});
% legend({'raceYY projected on raceYY' 'raceYY projected on raceXX' });
saveas(figp, fullfile(path_x{1}, [raceYY_str{:},'_', raceYY_str{:},num2str(select),'_','.jpg'])); % changed how name is saved
close all


%% PLOT PROJ FLIP

figx = figure();
hold on
title(" Projections")
xlabel('Number of Vectors') 
ylabel('Projection Coefficients') 
plot_raceXX = rescale(mean(abs(raceXX_proj_raceXX),1));
plot(plot_raceXX(:,1:selectx), "r");
plot_raceXX_raceYY = rescale(mean(abs(raceXX_proj_raceYY),1))';
plot((plot_raceXX_raceYY(1:selectx,:)),"g");
BB = (strcat(raceXX_str{:} ,'projected_on_', raceXX_str{:}));
BA = (strcat(raceXX_str{:} ,'projected_on_', raceYY_str{:}));
legend({BB BA});
% legend({'raceXX projected on raceXXspectralcluster' 'raceXX projected on raceYY' });
saveas(figx, fullfile(path_x{2}, [raceXXB_str{:},'_', raceYY_str{:},num2str(select),'_','.jpg'])); % changed how name is saved
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



% idx = spectralcluster(raceYY_Vec,7);
