
[XNETa,Ssuma,minAba,sXNETa,orig_index,acc,FF,FP,supp]=NB_FDR(Asign_frac,sAsign_frac,method,data,init,datadir,FDR,zetavec,boot);
XNET(:,:)=XNETa;
Ssum(:,:)=Ssuma;
minAb(:,:)=minAba;
sXNET(:,:)=sXNETa;
ACC(:,:)=acc;
FREQ(:,:)=FF;

function booAsign = signage(data,zetavec,Apos,init,boot)
    for i=1:init
        for j=1:size(data.Y,1)
            for k=1:size(data.Y,1)
                for l=1:length(zetavec)
                    if Apos{i}(j,k,l)>Apos{i}(j,k,l+length(zetavec))
                        booAsign{i}(j,k,l) =max(Apos{i}(j,k,l),(Apos{i}(j,k,l+length(zetavec))))/boot;
                    else
                        booAsign{i}(j,k,l) =-1*max(Apos{i}(j,k,l),(Apos{i}(j,k,l+length(zetavec))))/boot;
                    end
                end
            end
        end
    end

function [XNET,Ssum,minAb,sXNET,orig_index,accumulated,binned_freq,FP_rate_cross,support_cross]= NB_FDR(Afrac,shuf_Afrac,method,data,z,init,datadir,FDR,zetavec,boot)
    [accumulated,supover,shuover,overlaps_support,overlaps_shuffle,freq] = accumulate(Afrac,shuf_Afrac,init,z,data);
    [y,overlap_100,final_net_X,overlap_cross,support_cross,FP_rate_cross,orig_index]=plottable(z,freq,init,accumulated,overlaps_support,FDR);

	[XNET,Ssum,minAb,sXNET]=networks(Afrac,shuf_Afrac,boot,init,z,data,orig_index,FDR,datadir,method,zetavec);
	%no output networks or overall statistics, just scores compared to Gold Standard -- uncomment networks and plotttt for actual network tsv output
	[binned_freq] = gsUtilities.plot_frequency_of_bins(freq,[0:init]/init,'-');%hold on;

	% binned_freq,support_cross, FP_rate_cross, accumulated
	% [cc,dd]=plotttt(z,y,init,method,boot,data,zetavec,freq,support_cross, FP_rate_cross,accumulated,datadir,FDR,XNET);
end

function [accumulated,supover,shuover,overlaps_support,overlaps_shuffle,freq]= accumulate(booAlink,booShuffleAlink,init,z,data)
	estimated_support_net = [];
    estimated_shuffle_net = [];
    overlaps_support = [];
    overlaps_shuffle = [];
    % overlaps_support_all_zeta = [];
    % overlaps_shuffle_all_zeta = [];

    for i=1:length(booAlink) %or size(booAlink,1)
    	[estimated_support_net,overlaps_support]=structureboot(booAlink,i,z,estimated_support_net,init,data);
    	[estimated_shuffle_net,overlaps_shuffle]=structureboot(booShuffleAlink,i,z,estimated_shuffle_net,init,data);
    end

    % MData = analyse.CompareModels(ones(data.N));
    % MData = [MData, mt1];
    % Mshuf = [Mshuf, mt2];

    freq = [estimated_support_net,estimated_shuffle_net];
	[binned_freq] = gsUtilities.plot_frequency_of_bins(freq,[0:init]/init,'-');%hold on;

    % overlaps_support_all_zeta = cat(3,overlaps_support_all_zeta,overlaps_support);
    % overlaps_shuffle_all_zeta = cat(3,overlaps_shuffle_all_zeta,overlaps_shuffle);

    supover = [];
    shuover = [];
    for k=1:init
		[supover]=structuresupport(overlaps_support,k,init);
		[shuover]=structuresupport(overlaps_shuffle,k,init);
    end
    shuover(isnan(shuover)) = 0;
    supover(isnan(supover)) = 0;
    accumulated = [supover',shuover'];
end

function [estimated_support_net,overlaps_support]=structureboot(booAlink,iii,z,estimated_support_net,init,data)
		% mt1 = analyse.CompareModels(ones(data.N));
		tmp = booAlink{iii}(:,:,z);
        estimated_support_net = [estimated_support_net; tmp(:)];
        overlaps_support(iii,:) = tmp(:);
        % tmp(round(tmp*init)*(1/init)~=1) = 0;
        % mt1 = [mt1; analyse.CompareModels(ones(data.N),tmp)];
end

function [supover]=structuresupport(overlaps_support,k,init)
        tmp_intersect = sum(mand((overlaps_support >= k/init)));
        tmp_union = sum(mor((overlaps_support >= k/init)));
        supover(k+1) = tmp_intersect/tmp_union;
end


function [y,overlap_100,final_net_X,overlap_cross,support_cross,FP_rate_cross,orig_index] = plottable(j,freq,init,accumulated,overlaps_support,FDR)

	[binned_freq] = gsUtilities.plot_frequency_of_bins(freq,[0:init]/init,'-');hold on;
    y = ylim();

    diff_freq = sign(binned_freq(:,1)-binned_freq(:,2));
    non_0_freq = mor(binned_freq);
    non_0_freq_index = find(non_0_freq);
    diff_freq = diff_freq(non_0_freq_index);
    when_shuffle_cross_freq = find(diff_freq == -1);
	% orig_index = non_0_freq_index(when_shuffle_cross_freq(end));

    %Find crossing of shuffled and plain data, based on 5% FNDR
    [orig_index] = FDRcutoff(binned_freq,accumulated,FDR);
    % for n=1:length(binned_freq)-1
    % 	sum(binned_freq(end-n:end,2))*20 > sum(binned_freq(end-n:end,1))
    % end
    % orig_index=length(binned_freq)-n;

    overlap_at_cross_freq(j,:) = binned_freq(orig_index,:);
    overlap_cross(j,:) = accumulated(orig_index,:);
    support_cross(j) = (orig_index-1)/init;
    final_net_X(j,:) = mand((overlaps_support >= (orig_index-1)/init));

    overlap_100(j,:) = accumulated(end,:);
    % overlap_99(j,:) = accumulated(init*0.99,:);
    % overlap_90(j,:) = accumulated(init*0.9,:);

    tmp = sum(binned_freq(orig_index:end,:),1);
    FP_rate_cross(j) = tmp(2)/tmp(1);
end

function [orig_index] = FDRcutoff(binned_freq,accumulated,FDR)
	pbs=[];
	bbc=[];
	ccc=[];
	for n=1:length(binned_freq)-1
	    pbs(n)= trapz((length(accumulated)-n):length(accumulated),binned_freq(end-n:end,2));
	    bbc(n)= trapz((length(accumulated)-n):length(accumulated),binned_freq(end-n:end,1));
	    % ccc(n)=bbc(n)/pbs(n);
	end
	ccc=pbs./bbc;
	n=nnz(ccc<(FDR/100));
	orig_index=length(binned_freq)-n;
end

function out = mor(in,dim);
if ~exist('dim','var')
    dim = 1;
end
ndim = length((size(in)));
if ndim < dim
    error(' Index exceeds matrix dimensions.\n\n input have no dimension %d',dim)
end


indim = ['in(:',repmat(',:',1,ndim-1),')'];
indim(2+dim*2) = 'i';

i=1;
tmp = eval(indim);
i=2;
out = or(tmp,eval(indim));

for i=3:size(in,dim)
    out = or(out,eval(indim));
end
end

function out = mand(in,dim);
in(isnan(in)) = 0;  
if ~exist('dim','var')
    dim = 1;
end
ndim = length((size(in)));
if ndim < dim
    error(' Index exceeds matrix dimensions.\n\n input have no dimension %d',dim)
end


indim = ['in(:',repmat(',:',1,ndim-1),')'];
indim(2+dim*2) = 'i';

i=1;
tmp = eval(indim);
i=2;

out = and(tmp,eval(indim));

for i=3:size(in,dim)
    out = and(out,eval(indim));
end
end
function [XNET,Ssum,minAb,sXNET] =networks(Afrac,shuf_Afrac,boot,init,z,data,orig_index,FDR,datadir,method,zetavec);

	% booAlink2 = restructurematrix(Afrac);
	% booShuffleAlink2 = restructurematrix(shuf_Afrac);
	% % record of every signed direction support
	booAsign = restructurematrix(Afrac);
	booShuffleAsign = restructurematrix(shuf_Afrac);


	% calculate frequencies
	% [binned_freq,bins]=calc_bin_freq(booAlink2{z},init);
	% [binned_freq(2,:),bins]=calc_bin_freq(booShuffleAlink2{z},init);

	[binned_freq2,bins2]=calc_bin_freq(booAsign{z},init);
	[binned_freq2(2,:),bins2]=calc_bin_freq(booShuffleAsign{z},init);

	% Find crossover index
	% X_network= network_at_support_level(bins(orig_index),booAlink2{z});

	X_network= network_at_support_level(bins2(orig_index),booAsign{z});
	
	% the (1000) bootstrapped intersection at given support cutoff
	XNET = double(mand(X_network,3));
	% take lower bound of bootstrapped networks (were 100 but swapped for 35) 
	% as consensus for final network ie link overlap
	% LOver = min(booAlink2{z},[],3);
	% % take sign of the sum of sign support to be sign
	Ssum = sign(sum(booAsign{z},3));
	% take the minimum absolute value of the sign support 
	minAb = min(abs(booAsign{z}),[],3);
	% sXNET = double(mand(X_network2,3));
	
	ddd=(sum(X_network,3));
	cc=ddd>((boot*init)-FDR)/boot; 
	dd=ddd<(-((boot*init)-1)/boot);dd=dd*-1;
	sXNET=cc+dd;

	FDRR=sprintf('%f',1-FDR/100);
	% if exist [datadir,method,'_',date,'_FDR',FDRR(1:4),'/','cytoscape_net/']==0
	% 	mkdir [datadir,method,'_',date,'_FDR',FDRR(1:4),'/','cytoscape_net/'];
	% else
	% filename = fullfile([datadir,method,'_',date,'_FDR',FDRR(1:4),'/','cytoscape_net/'],[method,'_network_L',num2str(length(XNET)),'_M',num2str(data.M),'_support',num2str(orig_index/10),'_',sprintf('%0.2e',zetavec(z))]);
	% % datastruct.export2Cytoscape(XNET,data.names,fullfile([method,'_FDR',FDRR(1:4),'_L',num2str(length(XNET)),'_M',num2str(data.M),'_support',num2str(orig_index/10),'_',sprintf('%0.2e',zetavec(z))]),Ssum,minAb,sXNET);
	% datastruct.export2Cytoscape(XNET,data.names,filename,Ssum,minAb,sXNET);
	% end
end


function [nevomatrix]= restructurematrix(matrix);
nit= length(matrix);
for i=1:size(matrix{1},3)

	for j =1:nit
		nevomatrix{i}(:,:,j) = matrix{j}(:,:,i);
	end
end
end

function [freq,bins]= calc_bin_freq (matrix,init)

[counts,bins] = hist(matrix(:),[0:init]/init);
freq = counts./repmat(sum(counts),size(counts,1),1);

end

function [cutoff_net]= network_at_support_level(support_level,support_network)

	cutoff_net = support_network;
	cutoff_net2 = support_network;
	cutoff_net(find(cutoff_net < support_level)) = 0;
	cutoff_net2(find(cutoff_net2 > -(support_level))) = 0;

	cutoff_net=cutoff_net+cutoff_net2;
end


function [filedir] = plotttt(j,y,init,method,boot,data,zetavec,binned_freq,support_cross, FP_rate_cross, accumulated,datadir,FDR,Net);
	FDRR=sprintf('%f',1-FDR/100);
    TIT = [method,'_',num2str(boot),'_FDR',FDRR(1:4),'_frequency_M',num2str(data.M),'_L',num2str(nnz(Net)),'_Z',sprintf('%0.2e',zetavec(j))];
    freq= binned_freq(1:(init*boot)/10:end,:);freq=cat(1,freq,binned_freq(init*boot,:));
    titles = {'Frequencies and overlap'};
    titles_block1 = {'bins','Data','shuffle','overlap','overlap_shuffle'};
    values_block1 = [[0:init]'/init,binned_freq,accumulated];
    titles_block2 = {'yrange','sparsity','# links','zeta','support at cross'};
    values_block2 = [y',[1;1]*zetavec(j),[1;1]*num2str(nnz(Net)),[1;1]*support_cross(j),[1;1]*FP_rate_cross(j)];

    % if ploting
    % cc=[titles_block1,values_block1]
    % dd=[titles_block2,values_block2]
    % filedir=fullfile([datadir,method,'_',date,'_FDR',FDRR(1:4)]);
    % gsUtilities.export2gnuplot([(TIT),'.dat'],titles,[],titles_block1,values_block1,titles_block2,values_block2);
end

