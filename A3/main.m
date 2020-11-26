n=8;
Corr=zeros(n,n);
for i=1:n
for j=1:n
Corr(i,j)=(-1)^abs(i-j)/(abs(i-j)+1);
end
end
sigma=zeros(n,1);
mu=zeros(n,1);
sigma(1)=2;
mu(1)=3;
for i=1:n-1
sigma(i+1)=sigma(i)+2*rand;
mu(i+1)=mu(i)+1;
end
D=diag(sigma);
C2=D*Corr*D;
C=0.5*(C2+C2');
mu
sigma
D
C

r = 3.00;

f_vector = zeros(n, 1);
e_vector = ones(n, 1);
Aeq = [mu,e_vector]';
Aeq
beq = [r; 1];
beq
%H = [1 -1; -1 2]; 
%f = [-2; -6];
%Aeq = [1 1];
%beq = 0;
Aeq
[x,fval,exitflag,output,lambda] = quadprog(C,f_vector,[],[],Aeq,beq);
x
