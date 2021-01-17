function angle = vec_Angle(u,v)
CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1,-1));
disp(CosTheta)
ThetaInDegrees = real(acosd(CosTheta));
angle = ThetaInDegrees;
end