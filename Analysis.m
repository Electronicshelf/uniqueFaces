
clc
close all
val = {raceA_Val, raceB_Val, raceC_Val, raceD_Val, raceE_Val, raceF_Val, raceG_Val, raceH_Val};
hold on
x_take = 15;

for i = 1:8
    val_A = val{i};
%     disp(val_A)
    val_A = (val_A./sum(val_A));
%     val_A = (val_A);
    val_A = (val_A(1:x_take,:));
    plot(cumsum(val_A));
%     brighten(summer,-.9)
    grid on
%     a.FaceAlpha = 0.3;
    disp(val_A);
end
title(['Eigen Value Plot Normalised with ' , num2str(x_take) ,' Eigen Values'] );
legend({'SEAM','Black Female','White Male', 'White Female','SEAF','Black Male','Latino Male', 'Latino Female'});


clc
close all
val = {raceA_Val, raceB_Val, raceC_Val, raceD_Val, raceE_Val, raceF_Val, raceG_Val, raceH_Val};

hold on
x_take = 10;
clc
val_list = zeros(1,8);
A = zeros(x_take ,8);
for i = 1:x_take 
%     val_x = val{i};
%     disp(i);
    n = 1;
    for j = 1:8
%         disp(i);
%         disp(n);
        val_A = val{n};
        val_x = val_A(i);
        val_c = val_x;
%         disp(val_c);
        val_c = (val_c./sum(val_A));
%         val_c = (val_c);
        val_list(n) = val_c;
        n = n + 1;
         A(i,:) = val_list;
    end
    disp(val_list)
    val_list = zeros(1,8);
   
%     grid on
end
bar(A)
title(['Eigen Value Plot Normalised with ' , num2str(x_take) ,' Eigen Values'] );
legend({'SEAM','Black Female','White Male', 'White Female','SEAF','Black Male','Latino Male', 'Latino Female'});
xlx =['SEAM','Black Female','White Male', 'White Female','SEAF','Black Male','Latino Male', 'Latino Female'];





% WF_val_meanx   = [12.4033, 6.4454, 4.3679, 3.3070, 2.6620,2.2282];
% SEAM_Val_meanx = [11.0782, 5.7779, 3.9223, 2.9725, 2.3943,2.005];
% WM_Val_meanx   = [12.5543, 6.5298, 4.4263, 3.3517, 2.6980,2.2583];
% BF_Val_meanx   = [10.4283, 5.4425, 3.6975, 2.8040, 2.2599,1.8933];


% A = ([WF_val_meanx(1), SEAM_Val_meanx(1), WM_Val_meanx(1), BF_Val_meanx(1)]);
% B = ([WF_val_meanx(2), SEAM_Val_meanx(2), WM_Val_meanx(2), BF_Val_meanx(2)]);
% C = ([WF_val_meanx(3), SEAM_Val_meanx(3), WM_Val_meanx(3), BF_Val_meanx(3)]);
% D = ([WF_val_meanx(4), SEAM_Val_meanx(4), WM_Val_meanx(4), BF_Val_meanx(4)]);
% E = ([WF_val_meanx(5), SEAM_Val_meanx(5), WM_Val_meanx(5), BF_Val_meanx(5)]);
% F = ([WF_val_meanx(6), SEAM_Val_meanx(6), WM_Val_meanx(6), BF_Val_meanx(6)]);


% Y = [A
%      B
%      C
%      D
%      E
%      F];
% figure
% bar(Y)
% title(['Eigen Value Comparison Plot'  '(100 - 600)'] )
% xlabel('Eigenvectors Chosen') ;
% xline(1,'--b','100');
% xline(2,'--b','200');
% xline(3,'--b','300');
% xline(4,'--b','400');
% xline(5,'--b','500');
% xline(6,'--b','600');
% ylabel('Magnitude'); 
% legend({'White Female','SEAM','White Male','Black Female' });
% 
% 
% hold on
% grid on
% plot(WF_val_meanx)
% plot(SEAM_Val_meanx)
% plot(WM_Val_meanx)
% plot(BF_Val_meanx)
% title(['Eigen value comparison plot'  '(100 - 600)'] )
% xlabel('Eigenvectors chosen') ;
% xline(1,'--b','100');
% xline(2,'--b','200');
% xline(3,'--b','300');
% xline(4,'--b','400');
% xline(5,'--b','500');
% xline(6,'--b','600');
% ylabel('Magnitude'); 
% legend({'White Female','SEAM','White Male','Black Female' });
% set(gca,'XLim',[0 7]);
% 
% hold on
% grid on
% bar(WF_val_meanx)
% bar(SEAM_Val_meanx)
% bar(WM_Val_meanx)
% bar(BF_Val_meanx)
% title(['Eigen value comparison plot'  '(100 - 600)'] )
% xlabel('Eigenvectors chosen') ;
% xline(1,'--b','100');
% xline(2,'--b','200');
% xline(3,'--b','300');
% xline(4,'--b','400');
% xline(5,'--b','500');
% xline(6,'--b','600');
% ylabel('Magnitude'); 
% legend({'White Female','SEAM','White Male','Black Female' });
% set(gca,'XLim',[0 7]);


  A =  [0.2111,    0.1999 ,   0.2161  ,  0.2148,    0.1956 ,   0.1960 ,   0.2083,    0.1994];

  B =   [0.1676 ,   0.1638 ,   0.1886 ,   0.1775 ,   0.1502  ,  0.1831  ,  0.1657  ,  0.1603];

  C =   [0.1044 ,   0.1180,    0.1014 ,   0.1093 ,   0.1347 ,   0.1045,    0.1021  ,  0.1235];

  D =  [0.0560,    0.0542 ,   0.0558 ,   0.0487  ,  0.0535  ,  0.0519  ,  0.0574  ,  0.0504];

  E =  [0.0333   , 0.0395  ,  0.0294 ,   0.0303 ,   0.0379,    0.0365   , 0.0334   , 0.0376];




Y = [A
     B
     C
     D
     E
     ];
figure
bar(Y)
title(['Eigen Value Comparison Plot'  '(100 - 600)'] )
xlabel('Eigenvectors Chosen') ;
xline(1,'--b','100');
xline(2,'--b','200');
xline(3,'--b','300');
xline(4,'--b','400');
xline(5,'--b','500');

ylabel('Magnitude'); 
legend({'White Female','SEAM','White Male','Black Female' });



% raceC_str = {'WM_'};
% raceD_str = {'WF_'};
% 
% raceC_list = WM_list;
% raceD_list = WF_list;
% 
% raceC_mean = mean(raceC_list, 1);
% raceD_mean = mean(raceD_list, 1); 
% 
% raceC_matrix = raceC_list';
% raceD_matrix = raceD_list';
% 
% raceD_matrix_nm = raceD_matrix - raceD_mean';
% raceC_matrix_nm = raceC_matrix - raceC_mean';

%% PCA Extraction %%
% [raceD_Vec, raceD_S, raceD_Val] = pca(raceD_matrix_nm'); 
% pause(5)
% [raceC_Vec, raceC_S, raceC_Val] = pca(raceC_matrix_nm');


% select = 200;
% %% VECTOR ANGLE
% raceD_Vec_angle = (raceD_Vec)';
% raceC_Vec_angle = (raceC_Vec)'; 
% a=raceD_Vec_angle;
% b=raceC_Vec_angle;
% clc
% angle_x = subspace(raceC_Vec_angle, raceD_Vec_angle);
% angle_y = mPrinAngles(raceC_Vec_angle,raceD_Vec_angle);
% angle_z = real(acosd(dot(a, b) / (norm(a) * norm(b))));
% u=a';
% v=b';
% CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
% ThetaInDegrees = real(acosd(CosTheta));
% angle_list = zeros(1,30);

select = 200;

raceB_Vec_P1 = raceB_Vec(:,1:select) * raceB_Vec(:,1:select)';
raceA_Vec_P1 = raceA_Vec(:,1:select) * raceA_Vec(:,1:select)';
raceD_Vec_P1 = raceD_Vec(:,1:select) * raceD_Vec(:,1:select)';
raceC_Vec_P1 = raceC_Vec(:,1:select) * raceC_Vec(:,1:select)';


% norm_distAB = norm(raceA_Vec_P1-raceB_Vec_P1);
% norm_distAC = norm(raceA_Vec_P1-raceC_Vec_P1);
% norm_distAD = norm(raceA_Vec_P1-raceD_Vec_P1);
% 
% 
% norm_distBA = norm(raceB_Vec_P1-raceA_Vec_P1);
% norm_distBC = norm(raceB_Vec_P1-raceC_Vec_P1);
% norm_distBD = norm(raceB_Vec_P1-raceD_Vec_P1);
% 
% 
% norm_distCA = norm(raceC_Vec_P1-raceA_Vec_P1);
% pause(5)
% norm_distCB = norm(raceC_Vec_P1-raceB_Vec_P1);
% pause(5)
% norm_distCD = norm(raceC_Vec_P1-raceD_Vec_P1);
% 
% 
% norm_distDA = norm(raceD_Vec_P1-raceA_Vec_P1);
% pause(5)
% norm_distDB = norm(raceD_Vec_P1-raceB_Vec_P1);
% pause(5)
% norm_distDC = norm(raceD_Vec_P1-raceC_Vec_P1);


raceC_Vec_angle = (raceC_Vec)';
raceD_Vec_angle = (raceD_Vec)'; 

xangle_list = zeros(1,4999);
dist_x = zeros(1,4999);
for i = 1:  4999
    u = raceC_Vec_angle(i,:);
    v = raceD_Vec_angle(i,:);
    angle = abs(u*v');
    angle =  (acosd(angle));
    xangle_list(i) = angle;
    dist_x(i) = sind(angle);
end
disp(xangle_list(:));
disp(dist_x(:));

normAB = asind(dist_AB);




%% Angle Radians GRAM
AB_angle = subspace(raceA_Vec,raceB_Vec);
AC_angle = subspace(raceA_Vec,raceC_Vec);
AD_angle = subspace(raceA_Vec,raceD_Vec);
BC_angle = subspace(raceB_Vec,raceC_Vec);
BD_angle = subspace(raceB_Vec,raceD_Vec);
CD_angle = subspace(raceC_Vec,raceD_Vec);

%% Angle Radians Covariance
xAB_angle = subspace(raceA_Vec',raceB_Vec');
xAC_angle = subspace(raceA_Vec',raceC_Vec');
xAD_angle = subspace(raceA_Vec',raceD_Vec');
xBC_angle = subspace(raceB_Vec',raceC_Vec');
xBD_angle = subspace(raceB_Vec',raceD_Vec');
xCD_angle = subspace(raceC_Vec',raceD_Vec');

%% Angle Degrees
d_AB_angle = rad2deg(AB_angle);
d_AC_angle = rad2deg(AC_angle);
d_AD_angle = rad2deg(AD_angle);
d_BC_angle = rad2deg(BC_angle);
d_BD_angle = rad2deg(BD_angle);
d_CD_angle = rad2deg(CD_angle);


%% Distance
a_AB_angle = sind(d_AB_angle);
a_AC_angle = sind(d_AC_angle);
a_AD_angle = sind(d_AD_angle);
a_BC_angle = sind(d_BC_angle);
a_BD_angle = sind(d_BD_angle);
a_CD_angle = sind(d_CD_angle);



%% Angle Radians 200
selecty = 20;
vec_Races = {raceA_Vec, raceB_Vec, raceC_Vec, raceD_Vec, raceE_Vec, raceF_Vec, raceG_Vec, raceH_Vec};
races_D = ['A','B','C','D','E','F','G','H'];

clc
selecty = 200;
x_take = 8;
c = 8;
v = 1;
for i = 1:c
%     val_x = val{i};
%     disp(i);
    n = 1;
    for j = v:8
%         disp(i);
%         disp(j);
        a = vec_Races{i}; 
        b = vec_Races{j};
        s = subspace(a(:,1:selecty), b(:,1:selecty));
        cc = races_D(i);
        dd = races_D(j);
        s = num2str(s);
%         xx = [cc, dd, ',', s];
        xx=s;
        disp(xx);
%         val_c = (val_c./sum(val_A));
%         val_c = (val_c);
%         val_list(n) = val_c;
        n = n + 1;
%          A(i,:) = val_list;
    end
    c = c-1;
    v = v+1;
%     disp(val_list)
%     val_list = zeros(1,8);  
%     grid on
end

zAB_angle = subspace(raceA_Vec(:,1:selecty),raceB_Vec(:,1:selecty));
zAC_angle = subspace(raceA_Vec(:,1:selecty),raceC_Vec(:,1:selecty));
zAD_angle = subspace(raceA_Vec(:,1:selecty),raceD_Vec(:,1:selecty));
zBC_angle = subspace(raceB_Vec(:,1:selecty),raceC_Vec(:,1:selecty));
zBD_angle = subspace(raceB_Vec(:,1:selecty),raceD_Vec(:,1:selecty));
zCD_angle = subspace(raceC_Vec(:,1:selecty),raceD_Vec(:,1:selecty));



