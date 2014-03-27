
B = 3;

L = 9;  
Lsz = sqrt(L);

M = 4;

Nsz = 5;
N = Nsz^2;

Osz = Nsz + Lsz-1; 
O = Osz^2;

Z = 1;
Wsz = length(1:Z:Nsz);
W = Wsz^2;

%reinit
reinit_jumpstart

a = randn(M*W,B);
X = randn(N,B);

lambda = 0.1;
gamma = 0.2;

checkgrad('objfun_phi', phi(:), 0.01, a, psi, E, Dl, X, lambda, gamma)

