


r = 3.00;
n_of_rs = 25;
rs = linspace(3.00, 9.00, n_of_rs);
optimal_sigmas = zeros(n_of_rs, 1);
optimal_mus = zeros(n_of_rs, 1);
n=8;
lb = zeros(n, 1);
seed = 1337;
rng(1337);

for r_index=1:n_of_rs
    % Generate new data
    
    
    f_vector = zeros(n, 1);
    e_vector = ones(n, 1);
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
    % Find optimal values with quadprog
	Aeq = [mu,e_vector]';
    beq = [rs(r_index); 1];
    [x,fval,exitflag,output,lambda] = quadprog(C,f_vector,[],[],Aeq,beq,lb);
    optimal_sigmas(r_index) = sqrt(fval);
    optimal_mus(r_index) = mu' * x;

end

optimal_sigmas
optimal_mus
scatter(optimal_sigmas, optimal_mus, [], rs, "filled")

%plot(optimal_sigmas, optimal_mus)
xlabel('optimal mu')
ylabel('optimal sigma')
title('Exercise 1')

