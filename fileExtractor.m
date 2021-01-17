
% x_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/";
A_path = "/media/nyma/EXTERNAL1/PCA_UNIQUE/name_WM.txt";
f = fopen(A_path,'r');
f_path = "/media/nyma/EXTERNAL1/fairface-img-margin025-trainval";
batch_x = 5000;
tline = fgetl(f);
 
C_list = zeros(batch_x,22500);
files_img={};
i=1;
 
C_list = extractFace(f_path, tline, C_list,i,batch_x,f);
WM_list = C_list;
fclose(f);

% {'SEAM','Black Female','White Male', 'White Female','SEAF','Black Male','Latino Male', 'Latino Female'});
raceC_str = {'WM_'};
raceE_str = {'SEAF_'};

raceC_list = WM_list;
raceE_list = SEAF_list;

raceC_mean = mean(raceC_list, 1);
raceE_mean = mean(raceE_list, 1); 

raceC_matrix = raceC_list';
raceE_matrix = raceE_list';

raceE_matrix_nm = raceE_matrix - raceE_mean';
raceC_matrix_nm = raceC_matrix - raceC_mean';

%% PCA Extraction %%
[raceE_Vec, raceE_S, raceE_Val] = pca(raceE_matrix_nm'); 
pause(5)
[raceC_Vec, raceC_S, raceC_Val] = pca(raceC_matrix_nm');
