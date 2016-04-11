clear
I = imread('barbara256.png');
I_patch = zeros(64,32*32);
% modify the value of f here, compression value
f = 0.9;
n = 64;
m = ceil(f*n);
count = 0;
% creating non-overlapping patches of size 8x8 from the image
for i = 1:32
    for j = 1:32
        count = count+1;
        I_patch(:,count) = reshape(I((i-1)*8+1:i*8,(j-1)*8+1:j*8),[1,64]);
    end
end

phi  = randn(64);
phi = phi(1:m,:);
y = phi*double(I_patch);
U = kron(dctmtx(8)',dctmtx(8)');
A = phi*U;
%normalize all the column vectors, could this be the cause of the error?
for i = 1:64
    A(:,i) = A(:,i)/norm(A(:,i));
end

%% Algorithm for compressed sensing

I_rec = [];
for i = 1:1024    
    y_i = y(:,i);
    r = y_i;
    T = [];
    inds = [];
    count = 0;
    while ( norm(r) > 0.01)
        count = count +1;       
        % to ensure that the no. of s does not exceed m, otherwise
        % pseudoinverse doesn't exist
        if (size(inds,2) > m || count > 100)break;
        end
        % select the column of A that is maximally correlated with r
        [~,ind] = max(abs(r'*A));      
        % avoid repeated selection of the same indices.. which is happening
        % for some reason
        if sum(find(ind==inds))==0
            inds = [inds,ind];
        else
            continue
        end
        T_i = zeros(n,1);
        T_i(ind) = 1;
        T = [T,T_i];
        s = pinv(A*T)*y_i;
        r = y_i-(A*T)*s;
    end
    t = 64-size(s,1);
    s = [s;zeros(t,1)];
    I_rec = [I_rec; (U*s)'];
    norm(r)
end
%% Checking without CS
% this works properly, indicating that the DCT transform is working fine,
% its the estimation of theta, where we are going wrong
%{
t = U'*I_patch;
[~,ind] = sort(t);
for i = 1:1024
    for j = 1:n-m
        t(ind(j),i) = 0;
    end    
end
I_rec = U*t;
%}

%% To recreate the initial image
%
X = zeros(size(I));
count = 0;
for i = 1:32
    for j = 1:32
        count = count +1;
        X((i-1)*8+1:i*8,(j-1)*8+1:j*8) = reshape(I_rec(count,:),[8,8]);
    end
end
min = min(min(X));
max = max(max(X));
X = (X-min)*255/(max-min);
imshow(uint8(X))
%}