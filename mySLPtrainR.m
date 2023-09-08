function [mdl] = mySLPtrainR(Xtrain,Ytrain)
%% train
Xtrain= [-ones(1,size(Xtrain,2));Xtrain];
X= Xtrain;
d= Ytrain;
Rx= X*X'; % autocorrelation
Rdx= (d)*X'; % cross corrlation

w=  Rdx* inv(Rx+ (eye(size(Rx))+eps));
mdl.w=w;
end