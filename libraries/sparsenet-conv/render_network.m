function array = render_network(phi, rows)

[L M] = size(phi);

sz = sqrt(L);

buf = 1;

m = rows;
n = M/rows;

array = -ones(buf+m*(sz+buf),buf+n*(sz+buf));

k = 1;

for i = 1:m
    for j = 1:n
        clim = max(abs(phi(:,k)));

        array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz]) = ...
            reshape(phi(:,k),sz,sz)/clim;

        k = k+1;
    end
end

