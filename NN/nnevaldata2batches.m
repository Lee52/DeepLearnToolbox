function [batches_x, batches_y] = nnevaldata2batches(opts,data_x, data_y)
    %Divides train/validation/test set into batches to fit into memory
    %(primarily a concern on gpus).
    
    %for backward compability
    if isfield(opts, 'ntrainforeval')
        opts.batchsizeforeval = opts.ntrainforeval;
        warning('Use opts.batchsizeforeval instead of opts.ntrainforeval')
    end
    
    m = size(data_x,1);
    
    if isfield(opts, 'batchsizeforeval')
        batch_size = opts.batchsizeforeval;
    else
        batch_size = m;
    end
    
    nbatch = ceil(m/batch_size);
    
    if isfield(opts, 'maxevalbatches') && ~isempty(opts.maxevalbatches)
        if nbatch > opts.maxevalbatches
            nbatch = opts.maxevalbatches;
        end
    end
    
    batches_x = cell(1,nbatch);
    batches_y = cell(1,nbatch);
    idx = randperm(m);
    for i = 1:nbatch
        start_nr = batch_size*(i-1)+1;
        end_nr = batch_size*(i);
        if end_nr > m
            end_nr = m;
        end

        batches_x{i} = data_x(idx(start_nr:end_nr),:);
        batches_y{i} = data_y(idx(start_nr:end_nr),:); 
    end

    
end

