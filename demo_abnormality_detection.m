% demo Abnormality Detection using PCA approach
% email: ntienvu@gmail.com

%% user's input
% load the Feature matrix
load('demoFeature.mat');
% Feature is the matrix N x D  (N instance x D dimension)

% provide the threshold input
mythreshold=50;

% remove empty feature (this step is optional)
temp=sum(Feature);
idx=find(temp==0);
Feature(:,idx)=[];


%% estimating principle and residual subspaces

% compute Covariance Matrix
C=cov(Feature);

% plot covariance matrix
figure;
imagesc(C);
title('Covariance Matrix of Events','fontsize',16);
set(gca,'fontsize',14);
xlabel('Feature Dimension');
ylabel('Feature Dimension');
%saveTightFigure(h,'figs/CovariancematrixOfNormalEvents');

[U, S, V]=svd(C);
dim=size(U,1);

% plot the estimated subspaces
h=figure;
imagesc(U);
title('Estimated Subspaces','fontsize',16);
set(gca,'fontsize',14);
xlabel('Number of Subspace');
ylabel('Feature Dimension');
%saveTightFigure(h,'figs/EstimatedSubspaces.eps');

%principal subspace
nPrinSubspace=25; % need to adjust this number
U1 = U(:,1:nPrinSubspace);
%residual subspace
U2 = U(:,nPrinSubspace+1:end);

% plot the principle subspaces
h=figure('position',[100 100 300 330]);
imagesc(U1);
title('Principle Subspaces','fontsize',14);
set(gca,'fontsize',14);
xlabel('Number of Subspace');
ylabel('Feature Dimension');
%saveTightFigure(h,'figs/PrincipleSubspaces.eps');

% plot the residual subspaces
h=figure('position',[100 100 200 330]);
imagesc(U2);
title('Residual Subspaces','fontsize',12);
set(gca,'fontsize',12);
str=num2str([nPrinSubspace+1:5:dim]');
set(gca,'XTick',[1:5:dim-nPrinSubspace]);
set(gca,'XTickLabel',str);
xlabel('Number of Subspace');
ylabel('Feature Dimension');
%saveTightFigure(h,'figs/ResidualSubspaces.eps');

%% evaluation step

% project test feature onto residual subspace
residual=Feature*U2*U2';

% sum of residual
myresidual=sum(residual');

% we consider the magnitude of the signal
myresidual=abs(myresidual);

% find abnormal items which get higher residual than threshold
isAbnormal=find(myresidual>mythreshold);
predLabel=zeros(1,length(myresidual));
predLabel(isAbnormal)=1;
histc(predLabel,0:1)

h=figure;
plot(myresidual);
hold on;
plot(mythreshold*ones(1,length(myresidual))','r','linewidth',2);
hold on;
ylabel('Residual','fontsize',14);
xlabel('Data Point','fontsize',14);
title('Residual Energy For Abnomal Detection','fontsize',14);
set(gca,'fontsize',14);
legend('Residual Signal','Threshold');
%saveTightFigure(h,'figs/Residual.eps');

