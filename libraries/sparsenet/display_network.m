function h=display_network(A,S_var,h)
%
%  display_network -- displays the state of the network (weights and 
%                     output variances)
%
%  Usage:
%
%    h=display_network(A,S_var,h);
%
%    A = basis function matrix
%    S_var = vector of coefficient variances
%    h = display handle (optional)

figure(1)

[L M]=size(A);

sz=sqrt(L);

buf=1;

if floor(sqrt(M))^2 ~= M
  m=sqrt(M/2);
  n=M/m;
else
  m=sqrt(M);
  n=m;
end

array=-ones(buf+m*(sz+buf),buf+n*(sz+buf));

k=1;

for i=1:m
  for j=1:n
    clim=max(abs(A(:,k)));
    array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz])=...
	reshape(A(:,k),sz,sz)/clim;
    k=k+1;
  end
end

if exist('h','var')
  set(h,'CData',array);
else
  h=imagesc(array,'EraseMode','none',[-1 1]);
  axis image off
end


if exist('S_var','var')
  figure(2)
  subplot(211)
  bar(S_var), axis([0 M+1 0 max(S_var)])
  title('s variance')
  subplot(212)
  normA=sqrt(sum(A.*A));
  bar(normA), axis([0 M+1 0 max(normA)])
  title('basis norm (L2)')
end

drawnow
