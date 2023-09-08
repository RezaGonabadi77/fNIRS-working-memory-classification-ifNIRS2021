function [output] = myMejorityVoting(P,userlabel)
NumClass= numel(userlabel);
for i=1:NumClass
    tp= (P==userlabel(i));
    num(i,:)= sum(tp,1);
end
[~,indx]= max(num);
y= indx;
for i=1:NumClass
    ind= find(y==i);
    output(ind)= userlabel(i);
end
end

