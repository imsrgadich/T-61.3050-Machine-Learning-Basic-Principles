function a = sigmoid(x,w)

a = 1./(1+exp(-x*w'));

end