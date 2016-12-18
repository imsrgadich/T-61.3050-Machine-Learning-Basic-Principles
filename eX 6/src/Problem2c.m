clc;clearvars;

X=load('C:\Users\Srikanth\Documents\MATLAB\MLBP\eX 6\kmeans_data\kmeans_data_2.txt');
[m,n] =size(X);
%figure,scatter(X(:,1),X(:,2))
rng(1000)

for i = 1:10 
    A = [];
    k = i;
    for j = 1:k
        a = [15*rand-8 12*rand-8 12*rand-3 11*rand-2 12*rand-4];
        A=  [A;a];
    end

   
    
    opts = statset('Display','final');
    %[idx,C] = kmeans(X,2,'Distance','cityblock',...
     %   'Replicates',1,'Options',opts, 'start',a);
    [idx,C] = kmeans(X,k,'start',A);
    
    for ii = 1:k
        
        temp = X(idx==ii,:);
        
        centroid_repmat = repmat(C(ii,:),size(temp,1),1);
        
        error(ii) = sum(sum((temp - centroid_repmat).^2,2));
        
    end
    
    min_error(i) = min(error);
    
end

figure;
plot(1:10,min_error)
