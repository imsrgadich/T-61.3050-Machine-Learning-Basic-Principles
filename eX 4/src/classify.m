function predictions = classify(data, p1, p2, mu1s, mu2s, sigma1s, sigma2s)
    predictions = zeros(size(data, 1), 8);

    for i=1:8
        for j=1:size(data, 1)
            p1l = p1 * normpdf(data(j, i), mu1s(i), sigma1s(i));
            p2l = p2 * normpdf(data(j, i), mu2s(i), sigma2s(i));
            if p1l > p2l
                predictions(j,i) = 0;
            else
                predictions(j,i) = 1;
            end
        end    
    end
