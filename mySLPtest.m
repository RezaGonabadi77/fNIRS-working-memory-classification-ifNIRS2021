function [output] = mySLPtest(mdl,datatest)
w=mdl.w;
userlabel=mdl.userlabel;
%% test
datatest= [-ones(1,size(datatest,2));datatest];
%
temp= round(w*datatest);
%% convertin labels to form user labels
vect= vec2ind(temp);
for i=1:numel(userlabel)
    ind= find(vect==i);
    output(ind)=userlabel(i);
end

end

