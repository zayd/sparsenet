phi = randn(L,M);
phi = phi*diag(1./sqrt(sum(phi.^2)));

D = cell(W,1);
Dl = cell(W,1);
E = sparse(N*L,W);

%% Build a re-mapping matrix that takes every colum of phi (with a small spatial
%% extent), and copies it into a location in psi (with a larger spatial extent).
%% The idea here is to make a generic remapping procedure that can be defined
%% by a matrix, so translation invariance can be encoded in exactly the same
%% way as rotation and scale invariance. To build those in, simply add code
%% that performs these remappings of the phi basis.

%% I realize that this is a somewhat horrible function and not easy to
%% understand. By flipping the "if 0" to "if 1" statements around the figures
%% you can get an idea of what's going on, because I plot the location that
%% each column of phi is remapped to, as it is remapping. Thus, you can easily
%% verify the correctness of the program.
%% --bjc 2011-03-09

mask = zeros(Osz,Osz);
mask((Lsz+1)/2:Nsz+(Lsz+1)/2-1,(Lsz+1)/2:Nsz+(Lsz+1)/2-1) = ones(Nsz,Nsz);

time_loop = 0;
tic
i = 1;
for p = 1:Z:Nsz
    tic
    for q = 1:Z:Nsz
        DiO = zeros(O,L);
        DiN = zeros(N,L);

        k = 1;

        for r = p:p+Lsz-1
            for c = q:q+Lsz-1
                j = (r-1)*Osz+c;

                DiO(j,k) = 1;
                k = k+1;
            end
        end


        DiO = reshape(DiO,Osz,Osz,L);
        for l = 1:L
            DiO(:,:,l) = DiO(:,:,l) .* mask;
        end
        DiN = DiO((Lsz+1)/2:(Lsz+1)/2+Nsz-1,(Lsz+1)/2:(Lsz+1)/2+Nsz-1,:);
        DiN = reshape(DiN,N,L);



        [r,c] = find(DiN > 0);
        Dl{i} = [r,c];

        D{i} = sparse(double(DiN));
        E(:,i) = DiN(:);


        if 1
            psi_i = D{i} * phi;
            figure(3); colormap(gray);
            imagesc(reshape(psi_i(:,1),Nsz,Nsz), [-1 1]); axis image;

            drawnow
        end



        fprintf('\r%d %d / %d %d lt %.4f',p,q,Nsz,Nsz,time_loop);
        fprintf(' et %.4f',(Nsz-p) * time_loop/60);

        i = i+1;
    end
    time_loop = toc;
    fprintf('\n');
    whos D E
end


fprintf('\n');
toc




tic
psi = sparse([],[],[],N,M*W,L*M*W);
for i = 1:W
    if 1
        psi_i = sparse(D{i} * phi);
        if 0
            figure(3); imagesc(reshape(psi_i(:,1),Nsz,Nsz)); axis image off;
            drawnow;
            pause
        end
        psi(:,(i-1)*M+1:i*M) = psi_i;

    else
        psi_i = sparse(N,M);
        nl = Dl{i};

        if (1)
            psi_i(nl(:,1),:) = phi(nl(:,2),:);
            psi(:,(i-1)*M+1:i*M) = psi_i;
        else
            psi(nl(:,1),(i-1)*M+1:i*M) = phi(nl(:,2),:);
        end
    end

    fprintf('\r%d / %d',i,W);
end
fprintf('\n',i,W);
toc

[sucess,msg,msgid] = mkdir(sprintf('cache'));

eval(sprintf('save cache/E_%dx%dx%dz%d.mat E',O,N,L,Z));
eval(sprintf('save cache/Dl_%dx%dx%dz%d.mat Dl',O,N,L,Z));
eval(sprintf('save -v7.3 cache/psi_%dx%dx%dz%d.mat psi',N,M,O,Z));

update = 1;

