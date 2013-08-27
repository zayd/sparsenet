% sparsenet.m - simulates the sparse coding algorithm
% 
% Before running you must first define A and load IMAGES.
% See the README file for further instructions.

num_trials=10000;
batch_size=100;

num_images=size(IMAGES,3);
image_size=size(IMAGES,1);
BUFF=4;

[L M]=size(A);
sz=sqrt(L);

eta = 1.0;
noise_var= 0.01;
beta= 2.2;
sigma=0.316;
tol=.0001;

VAR_GOAL=0.1;
S_var=VAR_GOAL*ones(M,1);
var_eta=.001;
alpha=.02;
gain=sqrt(sum(A.*A))';

X=zeros(L,batch_size);

display_every=10;

h=display_network(A,S_var);

% main loop

for t=1:num_trials

    % choose an image for this batch
    
    i=ceil(num_images*rand);
    this_image=IMAGES(:,:,i);
    
    % extract subimages at random from this image to make data vector X
    
    for i=1:batch_size
        r=BUFF+ceil((image_size-sz-2*BUFF)*rand);
        c=BUFF+ceil((image_size-sz-2*BUFF)*rand);
        X(:,i)=reshape(this_image(r:r+sz-1,c:c+sz-1),L,1);
    end
    
    % calculate coefficients for these data via conjugate gradient routine
    
    S=cgf_fitS(A,X,noise_var,beta,sigma,tol);
    
    % calculate residual error
    
    E=X-A*S;
    
    % update bases
    
    dA=zeros(L,M);
    for i=1:batch_size
        dA = dA + E(:,i)*S(:,i)';
    end
    dA = dA/batch_size;
    
    A = A + eta*dA;
    
    % normalize bases to match desired output variance
    
    for i=1:batch_size
        S_var = (1-var_eta)*S_var + var_eta*S(:,i).*S(:,i);
    end
    gain = gain .* ((S_var/VAR_GOAL).^alpha);
    normA=sqrt(sum(A.*A));
    for i=1:M
        A(:,i)=gain(i)*A(:,i)/normA(i);
    end
    
    % display
    
    if (mod(t,display_every)==0)
        display_network(A,S_var,h);
    end
    
end
