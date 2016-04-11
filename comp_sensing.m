clear
I = imread('barbara256.png');
[mc,nc] = size(I);
p = 8; %patch size
stride = 4; %should divide p, m and n to leave 0 remainder
I_patch = zeros(p*p,(mc/stride-p/stride +1)*(nc/stride-p/stride +1));
% modify the value of f here, compression value
f = 0.9;
n = p*p;
m = ceil(f*n);
count = 0;
% creating overlapping patches of size 8x8 from the image
for i = 1:(mc/stride-p/stride +1)
    for j = 1:(nc/stride-p/stride +1)
        count = count+1;
        I_patch(:,count) = reshape(I((i-1)*stride+1:(i-1)*stride +8,(j-1)*stride+1:(j-1)*stride+8),[1,64]);
    end
end

phi  = randn(p*p);
phi = phi(1:m,:);
y = phi*double(I_patch);
U = kron(dctmtx(p)',dctmtx(p)');
A = phi*U;

for i = 1:p*p
    A(:,i) = A(:,i)/norm(A(:,i));
end

%% Algorithm for compressed sensing

I_rec = [];
for i = 1:(mc/stride-p/stride +1)*(nc/stride-p/stride +1)
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
        % avoid repeated selection of the same indices.. just in case
        if sum(find(ind==inds))==0
            inds = [inds,ind];
        else
            count
            continue
        end
        T_i = zeros(n,1);
        T_i(ind) = 1;
        T = [T,T_i];
        s = pinv(A*T)*y_i;
        r = y_i-(A*T)*s;       
        i
    end
    t = 64-size(s,1);
    s = [s;zeros(t,1)];
    I_rec = [I_rec; (U*s)'];
end

%% To recreate the initial image
%
X = zeros(size(I));
count = 0;
for i = 1:(mc/stride-p/stride +1)
    for j = 1:(nc/stride-p/stride +1)
        count = count +1;
        X((i-1)*stride+1:(i-1)*stride +8,(j-1)*stride+1:(j-1)*stride+8) = ...
            X((i-1)*stride+1:(i-1)*stride +8,(j-1)*stride+1:(j-1)*stride+8) ...
            + reshape(I_rec(count,:),[8,8]);
    end
end
%boundary correction
X_as = X;
for i = 1:stride:p-stride
    X(:,i:i+stride-1)= 2*X(:,i:i+stride-1);
    X(i:i+stride-1,:)= 2*X(i:i+stride-1,:);
    X(mc-i-stride+2:mc-i+1,:)= 2*X(mc-i-stride+2:mc-i+1,:);
    X(:,mc-i-stride+2:mc-i+1)= 2*X(:,mc-i-stride+2:mc-i+1);
end
    
minx = min(min(X));
maxx = max(max(X));
X = (X-minx)*255/(maxx-minx);
imshow(uint8(X))

%}