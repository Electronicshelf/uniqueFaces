function sumAOC = AOC(aValues, aBinEdges , bValues, bBinEdges)
%% Area of under Curve
avalues = aValues;
aEdges = aBinEdges;

bvalues = bValues;
bEdges = bBinEdges;

n = bEdges(1);
[val, i_x]= min(abs(aEdges-n));
minVal = aEdges(i_x);


end_idx = i_x;
a_idx = avalues(end_idx:end);
a_end_idx = length(a_idx);
b_idx = bvalues(1:a_end_idx);

% for i = 1:length(a_idx)
%     if(b_idx(i) <a_idx(i))
%         i   
% %     else    
% %             i = i + 48
%     end
%     
% end
    
% hold on
A_O_C = min(a_idx, b_idx);
sumAOC = sum(A_O_C)/(sum(avalues));
sumAOC;
end