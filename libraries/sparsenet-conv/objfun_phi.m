function [f,g] = objfun_phi(x0,a,psi,E,Dl,X,lambda,gamma);

[N B] = size(X);
W = size(E,2);
M = size(a,1)/W;
L = size(E,1)/N;

phi = reshape(x0,L,M);

% compute the shifted, masked basis set
for w = 1:W
    psi_w = sparse(N,M);
    nl = Dl{w};
    psi(nl(:,1),(w-1)*M+1:w*M) = phi(nl(:,2),:);
    fprintf('\r%d / %d',w,W);
end
fprintf('\r');

EI = psi*a;
R = X-EI;

f = 0.5*( sum(R(:).^2) + lambda*sum(abs(a(:))) )/B + ...
     0.5*gamma*sum(phi(:).^2);

dphi = zeros(L,M);
for b = 1:B
    a_b = reshape(a(:,b),M,W)';
    Ea_b = E*a_b;
    Ea_b = reshape(Ea_b,N,L*M);

    dphi = dphi - reshape(R(:,b)'*Ea_b,L,M);
    fprintf('\r%d / %d',b,B);
end
fprintf('\r');

dphi = dphi / B;

g = dphi + gamma*phi;
g = g(:);


