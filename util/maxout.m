function [h, idh, z] = maxout(x,W,b)

[d,m,k] = size(W);
[n, d] = size(x);


z = zeros(n,m,k)*nan;
for i = 1:m
    for j = 1:k
        z(:,i,j) = x*W(:,i,j); %+b(j,i);
    end
end

%add biases
z = z + repmat(shiftdim(b,-1),[100, 1,1]);

h = squeeze(max(z,[],2));
idh = ismember(z,h); %use reshape(z(idh),[1 k n])
end

