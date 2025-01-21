close all 
clear all 
a = [0:511];
m = [0:511]/512;
k = sinc(2*m);
plot(a,k)

close all 
clear all 
m = zeros(500,1);
m(1:100) = 1;
plot(m)
m_h = fft(m);
plot(abs(m_h))
