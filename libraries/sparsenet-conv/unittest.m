clear


L = 144;
L = 121;
L = 225;
Lsz = sqrt(L);

M = 25;
M = 64;
M = 36;
M = 16;
M = 100;
M = 900;
M = 400;
Mrows = sqrt(M);

Nsz = 128;
Nsz = 64;
Nsz = 100;
Nsz = 200;
Nsz = 300;
Nsz = 40;
N = Nsz^2;

Osz = Nsz + Lsz-1;
O = Osz^2;

%% "subconvolution" indexing interval
Z = 10;
Z = 3;
Z = 1;

Wsz = length(1:Z:Nsz);
W = Wsz^2;

tol_coef = 0.01;
tol_coef = 1.0;
tol_bf = 0.0001;
tol_bf = 0.2;

datasource = 'movies';
datasource = 'images';

switch datasource
    case 'images'
        load ../data/IMAGES.mat
        [imsz,imsz,K] = size(IMAGES);
        buff = 4;

    case 'movies'
        data_root = '../data/vid075-chunks';
        data_root = '../data/vid075-whiteframes';

        load_interval = 10;
        num_chunks = 4;
        num_chunks = 56;
        imsz = 128;
        imszt = 64;
        buff = 4;
        topmargin = 15;
end


lambda = 0.3;
lambda = 0.1;
gamma = 0.01;
gamma = 0;


target_angle = 0.03;


mintype_inf = 'mintotol';
mintype_inf = 'lasso';
mintype_inf = 'l1_ls';
mintype_inf = 'lbfgsb';

mintype_lrn = 'minimize';
mintype_lrn = 'gd';



switch mintype_inf
    case 'lbfgsb'

        lb  = zeros(1,M*W); % lower bound
        ub  = zeros(1,M*W); % upper bound
        nb  = ones(1,M*W);  % bound type (lower only)

        opts = lbfgs_options('iprint', -1, 'maxits', 20, ...
                             'factr', tol_coef, ...
                             'cb', @cb_a);
    case 'lasso'

        %% SPAMS lasso params
        param.L = min(M*W,N);
        param.lambda = lambda;
        param.mode = 2;         % constrain norms <= 1
        param.numThreads = 8;   % -1 = use all cores, uses OpenMP for threading
        lasso = @mexLasso;      % use LARS, CD requires sparse guess

end

max_ls = 1;

var_thresh = 0.02;
eta = 0.1;
eta = 0.05;
eta = 0.01;

paramstr = sprintf('L=%03d_M=%03d_%s',L,M,datestr(now,30));

% run this to create the .mat files for _jumpstart
%reinit
reinit_jumpstart



display_every = 1;
save_every = 10;


num_trials = 500;

for B = 1:5; sparsenet; end

num_trials = 1000;
sparsenet

for target_angle = 0.03:-0.01:0.01 ; sparsenet ; end

