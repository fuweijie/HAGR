clear
load('Data_MNIST8M_HAGR.mat')

% Z01 Z12 Z23 are k-sparse inter-layer adjacency matrices in three-layer hirarchical anchor graph.
% For ANNS-based graph construction, please download the source code
% at FLANN 'http://www.cs.ubc.ca/research/flann/' and compile the it based on your own PC.

m1=300000;
Zpro=(Z12*Z23);
ZH=Z01*Zpro;
temp=Z01'*ZH;
temp_s=sum(Z1).^-1;
temp_s=sparse(1:m1,1:m1,temp_s);
rL=Zpro'*temp-temp'*temp_s*temp;
acc=zeros(5,1);
for iter = 1:5
    [F, A, err] = AnchorGraphReg(Z, rL, label', label_index(iter,:), 0.01);
    acc(iter) = 1-err;
end
fprintf('\n The average classification accuracy of HAGR is %.2f%%.\n', mean(acc)*100);