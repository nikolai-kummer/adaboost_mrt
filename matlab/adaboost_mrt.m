function [netArray,beta_t]=adaboost_mrt(mode,inputs,outputs_or_model,itt,beta_or_num_per_t,phi)
%  Training mode:
%    [estimateclass,model,index_used]=adaboost('train',datafeatures,dataclass,itt)

%    datafeatures : An Array with size number_samples x number_features
%    dataclass : An array with the class off all examples, the class
%                 can be -1 or 1
%    itt : The number of training itterations
%    model : A struct with the cascade of weak-classifiers
%    estimateclass : The by the adaboost model classified data

switch(mode)
    case 'train'    % Train the adaboost model
        % Set the data class
        y=outputs_or_model;
        N=beta_or_num_per_t; %number of items to sample
        %model=struct;
        m=length(inputs);
        %Initialize Dt(1) for even weights
        Dt=ones(m,1)/m;
        D=ones(size(y))'/m;
        
        n=1; %n is the power (1,2,3 for linear square or cubic)
        
        ep=zeros(size(y,1),itt); %Error rate epsilon
        beta_t=zeros(size(y,1),itt); %Weight Updating parameter
        netArray=cell(itt,1);
        are=zeros(itt,m);
        % Do all model training itterations
        for t=1:itt
            %Sample without replacements:
            %ks = sort(randsample([1:m],N,true,D)); %FIX THIS true
            if sum(isnan(Dt))>0
                disp('Something broke')
                break
            else
                ks=(sort(sampleWeighted(N,Dt,[1:m])));
                %index_used=[index_used,ks];
                %index_used=[];
            end
            %Weak Learner:
            %% Radial Basis Neural Network:
            %netArray{t} = newgrnn(inputs(ks)',y(ks)',1);
            %% Multi-layer Perceptron
            netArray{t} = newff(inputs(:,ks),y(:,ks),30);
            %netArray{t}.trainParam.showWindow=1;
            netArray{t}.trainParam.epochs=100;
            netArray{t} = train(netArray{t},inputs(:,ks),y(:,ks));
            
            %% Calculate Errors:
            %absolute error rate:
            %Multiclass error rate (Mahalanobis Distance)
            ypredict=sim(netArray{t},inputs);
            er=ypredict-y;
            are = bsxfun(@times, abs(er'), 1./std(er'));
            %are(t,:)=sqrt(mahal(er',er'));
            ind_m=are>phi;
            
            %% PLOT THIS
            %             maxi=10;
            %             mini=-10;
            %             ti=mini:(maxi-mini)/60:maxi;
            %             [xi,yi]=meshgrid(ti,ti);
            %             zi=griddata(ypredict(1,:),ypredict(2,:),inputs(1,:),xi,yi);
            %             surf(xi,yi,zi)
            %%
            
            %differ=(ypredict-y);
            %are=differ*cov(differ')*differ'
            %%%%are(t,:)=(abs((sim(netArray{t},inputs(ks,:)')-y(ks,:)')./y(ks,:)'));
            
            %Error rate as a function of standard deviation:
            %are(t,:)=(abs((sim(netArray{t},inputs(:,:)')-y(:,:)')))./std(abs((sim(netArray{t},inputs(:,:)')-y(:,:)')'));
            
            %Absolute Error
            %are(t,:)=(abs((sim(netArray{t},inputs(ks)')-y(ks)')));
            
            %not good: are(t,:)=mahal((sim(netArray{t},inputs(ks)'))',inputs(ks));
            
            
            
            %subplot(1,2,1);plot(inputs(ks),sim(netArray{t},inputs(ks)'),inputs(ks),y(ks));
            
            %one: ind_m=are(t,:)>phi; %missclassified index Er>phi;
            
            %ep(t)=sum( D(ind_m) ); %epsilon, error rate, sum of missclassifieds
            ep(:,t)=sum(D.*ind_m);
            beta_t(:,t)=ep(:,t).^n;
            
            %Calculate: D(~ind_m)=D(~ind_m)*beta_t(:,t);
            temp_b=bsxfun(@times, ~ind_m, beta_t(:,t)');
            temp_b(temp_b==0)=1;
            D=D.*temp_b;
            D=bsxfun(@times, D, 1./sum(D));% normalize
            Dt=mean(D,2);
            %Dt=max(D')';
            Dt=Dt./sum(Dt); %normalize
            %             clf
            %             plot(D)
            %             hold on
            %             plot(Dt,'r-','LineWidth',1.5);
            drawnow
            
        end
        
    case 'apply' %Apply the data
        netArray=outputs_or_model;
        beta_t=beta_or_num_per_t;
        num=0;
        den=0;
        for pl =1:itt
            if beta_t(pl)>10e-8 %check for zero value
                %num=num+log10(1/beta_t(pl))*sim(netArray{pl},inputs);
                num=num+bsxfun(@times,sim(netArray{pl},inputs),log10(1./beta_t(:,pl)));
                %bsxfun(@times,sim(netArray{pl},inputs),log10(1./beta_t(:,pl)))
                den=den+log10(1./beta_t(:,pl));
            end
        end
        %return final
        netArray=bsxfun(@times,num,1./den);
    otherwise
        error('adaboost_mrt:inputs','unknown mode');
end

end

%% Sample with replacement
function R=sampleWeighted(N,w,a)
%sample weighted numbers. N is the numbers of things to generate and w
% N     Number of items
% w     weight vector
% a
try
    %Method 1, Faster, sometimes fails
    [~,R] = histc(rand(1,N),cumsum([0;w(:)./sum(w)]));
    R = a(R);
catch
    %Method 2, slower, sometimes fails?
    disp('Method 1 broke');
    R = sort(a( sum( bsxfun(@ge, rand(N,1), cumsum(w'./sum(w))), 2) + 1 ));
end
end

