function [output] = mySLPtestR(mdl,datatest)
w=mdl.w;
%% test
datatest= [-ones(1,size(datatest,2));datatest];
%
output= w*datatest;

end

