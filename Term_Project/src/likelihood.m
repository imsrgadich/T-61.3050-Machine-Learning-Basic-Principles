function s = likelihood(data,dataTarget,w)
  
    h = sigmoid(data,w);
    
    d =dataTarget;
    
    s = sum(d.*log(h)+ (1-d).*(1-log(1-h)));
    

end