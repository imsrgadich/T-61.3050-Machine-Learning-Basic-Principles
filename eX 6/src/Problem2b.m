clc;clearvars;
 
X=load('C:\Users\Srikanth\Documents\MATLAB\MLBP\eX 6\kmeans_data\kmeans_data_1.txt');
 
%figure,scatter(X(:,1),X(:,2))
rng(1000)
 
A = [];
 
for i = 1:3    
    
    a = [14*rand-4 3*(rand-1) ;14*rand-4 3*(rand-1)];
    
    opts = statset('Display','final');
    %[idx,C] = kmeans(X,2,'Distance','cityblock',...
     %   'Replicates',1,'Options',opts, 'start',a);
    [idx,C] = kmeans(X,2,'start',a);
    A=[A;a];
    figure;
    plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
    hold on
    plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
    plot(C(:,1),C(:,2),'kx',...
        'MarkerSize',15,'LineWidth',3)
    legend('Cluster 1','Cluster 2','Centroids',...
        'Location','NW')
    title 'Cluster Assignments and Centroids'
    scatter(a(:,1),a(:,2),40,'MarkerEdgeColor',[.5 .8 .8],...
              'MarkerFaceColor',[.5 .9 .9],...
              'LineWidth',1.5)
    hold off
    
end
 
% Plot used from the Matlab example %
