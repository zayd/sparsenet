function S = cgf_fitS(A,X,noise_var, beta, sigma, tol, ...
                      disp_ocbsol, disp_patnum, disp_stats)
% cgf_fitS -- fit internal vars S to the data X using fast congugate gradient
%   Usage
%     S = cgf_fitS(A,X,noise_var,beta,sigma,
%                  [tol, disp_ocbsol, disp_patnum, disp_stats])
%   Inputs
%      A             basis functions
%      X             data vectors
%      noise_var     variance of the noise (|x-As|^2)
%      beta          steepness term for prior
%      sigma         scaling term for prior
%      tol           solution tolerance (default 0.001)
%      disp_ocbsol   display info from the fitting process
%      disp_patnum   display the pattern number
%      disp_stats    display summary statistics for the fit
%   Outputs
%      S             the estimated coefficients

maxiter=100;

[L,M] = size(A);
N = size(X,2);

if ~exist('tol','var');		tol = 0.001;			end
if ~exist('disp_ocbsol','var');	disp_ocbsol = 0;		end
if ~exist('disp_patnum','var');	disp_patnum = 1;		end
if ~exist('disp_stats','var');	disp_stats = 1;			end

Sinit=A'*X;
normA2=sum(A.*A)';
for i=1:N
  Sinit(:,i)=Sinit(:,i)./normA2;
end

lambda=1/noise_var;

S = zeros(M,N);
tic
[S niters nf ng] = cgf(A,X,Sinit,lambda,beta,sigma,tol,maxiter,...
    disp_ocbsol,disp_patnum);
t = toc;

if (disp_stats)
  fprintf(' aits=%6.2f af=%6.2f ag=%6.2f  at=%7.4f\n', ...
      niters/N, nf/N, ng/N, t/N);
end
