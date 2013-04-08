function batches = nnevaldata2batches(opts,data)
    %Divides train/validation/test set into batches to fit into memory
    %(primarily a concern on gpus).
    
    %for backward compability
    if isfield(opts, 'ntrainforeval')
        opts.batchsizeforeval = opts.ntrainforeval;
        warning('Use opts.batchsizeforeval instead of opts.ntrainforeval')
    end
    
    m = size(data,1);
    
    if isfield(opts, 'batchsizeforeval')
        batch_size = opts.batchsizeforeval;
    else
        batch_size = m;
    end
    
    nbatch = ceil(m/batch_size)-1;
    
    if isfield(opts, 'maxevalbatches') && ~isempty(opts.maxevalbatches)
        if nbatch > opts.maxevalbatches
            nbatch = opts.maxevalbatches;
        end
    end
    
    batches = cell(1,nbatch);
    idx = randperm(m);
    for i = 1:nbatch
        start_nr = batch_size*(i-1)+1;
        end_nr = batch_size*(i);
        batches{i} = data(idx(start_nr:end_nr),:);
        batches{i} = data(idx(start_nr:end_nr),:);        
    end
    
    %add the remainder batch if any
    if end_nr < m && m-end_nr <= batch_size %last condition is to ensure that the remainder is not used if opts.maxevalbatches is set
        batches{i+1} = data(idx(end_nr+1:end),:);
    end
    
end

