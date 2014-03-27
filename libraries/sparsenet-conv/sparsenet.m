

for t = 1:num_trials

    tic

    switch datasource
        case 'images'

            X = zeros(N,B);

            % extract subimages at random from this image to make data vector X
            for b = 1:B
                i = ceil(K*rand);
                r = buff + ceil((imsz-Nsz-2*buff)*rand);
                c = buff + ceil((imsz-Nsz-2*buff)*rand);

                X(:,b) = reshape(IMAGES(r:r+Nsz-1,c:c+Nsz-1, i), N, 1);
            end


        case 'movies'
            X = zeros(N,B);
            for i = 1:B
                if (~exist('F','var') || mod(i+t*B,load_interval) == 0)
                    % choose a movie for this batch
                    j = ceil(num_chunks*rand);
                    fprintf('%d ', j);
                    F = read_chunk(data_root,j,imsz,imszt);
                end

                while (var(X(:,i)) < var_thresh)
                    frame_num = ceil((imszt-1)*rand);

                    G = F(:,:,frame_num);
                    G = G - mean(G(:));
                    G = sqrt(0.1) * G / std(G(:));

                    if Nsz == 128
                        X(:,i) = G(:);
                    else
                        r = topmargin+Nsz/2+buff+ceil((imsz-Nsz-(topmargin+2*buff))*rand);
                        c = Nsz/2+buff+ceil((imsz-Nsz-2*buff)*rand);
                        X(:,i) = reshape(G(r-Nsz/2:r+Nsz/2-1,c-Nsz/2:c+Nsz/2-1),N,1);
                    end

                    fprintf('%.4f ', var(X(:,i)));
                end
                fprintf('\n');
        end

    end

    time_choose = toc;

    tic
    switch mintype_inf
        case 'mintotol'
            a1 = zeros(M*W,B);
            a0 = zeros(M*W,1);
            for b = 1:B
                [a1(:,b),fX,ri] = mintotol(a0(:),'objfun_a',100,...
                                           tol_coef,psi,X(:,b),lambda);
                fprintf('\r%d',b);
            end
            a = a1;

        case 'lbfgsb'
       
            a1 = zeros(M*W,B);
            a0 = zeros(M*W,1);

            for b = 1:B
                [a1(:,b),fx,exitflag,userdata] = lbfgs(@objfun_a,a0,lb,ub,nb,opts, ...
                                                       psi,X(:,b),lambda);

                fprintf(' %d / %d\n', b, B);
            end
            a = a1;

        case 'lasso'
            a = lasso(X, full(psi), param);
            a = full(a);
        case 'l1_ls'
            for b = 1:B
                a(:,b) = l1_ls(psi, X(:,b), lambda, tol_coef);
            end


    end
    time_inf = toc;

    snr = 10 * log10 ( sum(X(:).^2) / sum(sum((X-psi*a).^2)) );


    tic

    % update bases
    switch mintype_lrn
        case 'gd'

            [obj0,g] = objfun_phi(phi,a,psi,E,Dl,X,lambda,gamma);
            dphi = reshape(g, L, M);
            phi1 = phi - eta*dphi;
            [obj1,g] = objfun_phi(phi1,a,psi,E,Dl,X,lambda,gamma);


            %% pursue a constant change in angle
            angle_phi = acos(phi1(:)' * phi(:) / sqrt(sum(phi1(:).^2)) / sqrt(sum(phi(:).^2)));
            if angle_phi < target_angle
                eta = eta*1.01;
            else
                eta = eta*0.99;
            end

            eta_log = eta_log(1:update-1);
            eta_log = [ eta_log ; eta ];


        case 'minimize'
            [obj0,g] = objfun_phi(phi,a,psi,E,Dl,X,lambda,gamma);
            [phi1,fX,ri] = minimize(phi(:),'objfun_phi',max_ls,a,psi,E,Dl,X,lambda,gamma);
            phi1 = reshape(phi1, L, M);
            [obj1,g] = objfun_phi(phi1,a,psi,E,Dl,X,lambda,gamma);
    end

    if (obj1 > obj0)
        fprintf('warning: objfun increased\n');
    end

    phi = phi1;

    time_updt = toc;

    tic

    % compute the shifted, masked basis set
    psi = update_psi(phi, psi, Dl, M, N, W);

    time_bf = toc;

    % display
    
    tic
    if mod(t,display_every) == 0
        % Display the bfs
        array = render_network(phi, Mrows);
 
        sfigure(1); colormap(gray);
        imagesc(array, [-1 1]);
        axis image off;

        sfigure(2);
        phi_norm = sum(phi.^2);
        bar(phi_norm);
        axis tight;
 

        sfigure(4);
        EI = psi*a;

        clim = max(abs([ EI(:,1) ; X(:,1) ]));
        subplot(1,3,1),imagesc(reshape(EI(:,1),Nsz,Nsz),[-clim clim]);
            title('EI'); colormap(gray),axis image off;
        subplot(1,3,2),imagesc(reshape(X(:,1),Nsz,Nsz),[-clim clim]);
            title('X'); colormap(gray),axis image off;
        subplot(1,3,3),imagesc(reshape(X(:,1)-EI(:,1),Nsz,Nsz),[-clim clim]);
            title('E'); colormap(gray),axis image off;


        sfigure(5);
        bar(a(:,1));
        axis tight;


        sfigure(6); colormap(gray); clf;
        a_xy = reshape(a(:,1), M, Wsz, Wsz);

        %% show a max of 25 xy maps -- drawing many takes a long time
        Mdpy = min(Mrows, 5);
        for m = 1:Mdpy^2
            subplot(Mdpy, Mdpy, m);
                imagesc(squeeze(a_xy(m,:,:)));
                axis image; colorbar;
        end


        sfigure(7);
        plot(1:update, eta_log, 'r-');
        axis tight; grid on;
        title('\eta');


        if mod(t,save_every) == 0
            array_frame = uint8(255*((array+1)/2)+1);

            [sucess,msg,msgid]=mkdir(sprintf('state/%s', paramstr));
 
            imwrite(array_frame, ...
                sprintf('state/%s/l1bf_up=%06d.gif',paramstr,update), ...
                'gif');

            eval(sprintf('save state/%s/phi_%dx%d.mat phi',paramstr,L,M));

            saveparamscmd = sprintf('save state/%s/params.mat', paramstr);
            saveparamscmd = sprintf('%s lambda', saveparamscmd);
            saveparamscmd = sprintf('%s gamma', saveparamscmd);
            saveparamscmd = sprintf('%s eta', saveparamscmd);
            saveparamscmd = sprintf('%s eta_log', saveparamscmd);
            %saveparamscmd = sprintf('%s objtest_log', saveparamscmd);
            saveparamscmd = sprintf('%s L', saveparamscmd);
            saveparamscmd = sprintf('%s M', saveparamscmd);
            saveparamscmd = sprintf('%s N', saveparamscmd);
            saveparamscmd = sprintf('%s Z', saveparamscmd);
            saveparamscmd = sprintf('%s W', saveparamscmd);
            saveparamscmd = sprintf('%s mintype_inf', saveparamscmd);
            saveparamscmd = sprintf('%s update', saveparamscmd);
            eval(saveparamscmd);


            if 0
                X_frame = X(:,1) - min(X(:,1));
                X_frame = X_frame / max(X_frame(:));
                X_frame = uint8(255*X_frame);
                imwrite(reshape(X_frame,Nsz,Nsz), ...
                    sprintf('state/%s/X_up=%06d.gif',paramstr,update), ...
                    'gif');

                EI_frame = EI(:,1) - min(EI(:,1));
                EI_frame = EI_frame / max(EI_frame(:));
                EI_frame = uint8(255*EI_frame);
                imwrite(reshape(EI_frame,Nsz,Nsz), ...
                    sprintf('state/%s/EI_up=%06d.gif',paramstr,update), ...
                    'gif');
            end
        end
        drawnow;

    end
    time_disp = toc;


    % normalize bases to have length 1
    phi = phi*diag(1./sqrt(sum(phi.^2)));

    psi = update_psi(phi, psi, Dl, M, N, W);


    fprintf('%s up %d ch %.2f if %.2f ud %.2f dy %.2f bf %.2f', ...
        paramstr,update,time_choose,time_inf,time_updt,time_disp,time_bf);
    fprintf(' l0 %.3f', mean(abs(sign(a(:)))) );
    fprintf(' o0 %.4f o1 %.4f snr %.4f ang %.4f\n',obj0,obj1,snr,angle_phi);


    update = update + 1;
end

eval(sprintf('save state/%s/matlab_up=%06d.mat', paramstr, update)); 

