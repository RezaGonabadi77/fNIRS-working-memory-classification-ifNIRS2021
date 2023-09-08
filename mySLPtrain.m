function [mdl] = mySLPtrain(datatrain,dtrain)
%% converting labels to form of NN
userlabel= unique(dtrain);
N=numel(userlabel);
for i=1:N
    ind= find(dtrain==userlabel(i));
    label(ind)= i;
end   
label = full(ind2vec(label,N));
%% train
datatrain= [-ones(1,size(datatrain,2));datatrain];
X= datatrain;
d= label;
Rx= X*X'; % autocorrelation
Rdx= (d)*X'; % cross corrlation

w=  Rdx* inv(Rx);
mdl.w=w;
mdl.userlabel=userlabel;
end