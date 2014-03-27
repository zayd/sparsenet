
phi = randn(L,M);
phi = phi*diag(1./sqrt(sum(phi.^2)));

eval(sprintf('load cache/E_%dx%dx%dz%d.mat E',O,N,L,Z));
eval(sprintf('load cache/Dl_%dx%dx%dz%d.mat Dl',O,N,L,Z));
eval(sprintf('load cache/psi_%dx%dx%dz%d.mat psi',N,M,O,Z));

update = 1;

eta_log = [];
objtest_log = [];

