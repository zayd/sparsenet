function psi = update_psi(phi, psi, Dl, M, N, V)

% compute the shifted, masked basis set
for v = 1:V
    psi_v = sparse(N,M);
    nl = Dl{v};

    psi(nl(:,1),(v-1)*M+1:v*M) = phi(nl(:,2),:);

    fprintf('\r%d / %d',v,V);
end
fprintf('\n');

