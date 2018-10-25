import numpy as np
import math
from shapely.geometry.polygon import LinearRing

#C1 = np.array([[1, -4], [-4, 3]])
C1 = np.array([[0.21908399, -0.1329367], [-0.1329367, 0.09937114]])
C2 = np.array([[3, 2], [2, 5]])

#[[ 0.21908399 -0.01329367]
# [-0.01329367  0.09937114]]
#eVa_C2, eVe_C1 = np.linalg.eig(C2)

# 2x2 rotation matrix (eigenvectors as columns) should have the form
# [[cos(theta), -sin(theta)],
#   [sin(theta), cos(theta)]]

def get_theta_from_cov(C):
    eVa, eVe = np.linalg.eig(C)

    if C[0,1] > 0.0:
        if abs(round(math.degrees(math.acos(eVe[0,0])), 3)) > 90:
            theta = (180 - abs(round(math.degrees(math.acos(eVe[0,0])), 3)))
        else:
            theta = abs(round(math.degrees(math.acos(eVe[0,0])), 3))
        #eve_theta_00 = round( math.degrees(math.acos(eVe[0,0])), 3)
        #eve_theta_10 = round(abs(math.degrees(math.asin(eVe[1,0]))), 3)
        #eve_theta_01 = round(abs(math.degrees(math.asin(-eVe[0,1]))), 3)
        #eve_theta_11 = round( math.degrees(math.acos(eVe[1,1])), 3)

        #if eve_theta_00 == eve_theta_10 == eve_theta_01 == eve_theta_11:
        #    theta = eve_theta_00
        #else:
        #    print(1)
        #    print( eve_theta_00, eve_theta_10, eve_theta_01, eve_theta_11)

    elif C[0,1] < 0.0:
        #print(  round(math.degrees(math.acos(eVe[0,0])), 3) )
        if abs(round(math.degrees(math.acos(eVe[0,0])), 3)) > 90:
            theta = -(180 - abs(round(math.degrees(math.acos(eVe[0,0])), 3)))
        else:
            theta = -abs(round(math.degrees(math.acos(eVe[0,0])), 3))


        #eve_theta_00 = -round(math.degrees(math.acos(eVe[0,0])), 3)
        #eve_theta_10 = -round(abs(math.degrees(math.asin(eVe[1,0]))), 3)
        #eve_theta_01 = -round(abs(math.degrees(math.asin(-eVe[0,1]))), 3)
        #eve_theta_11 = -round(math.degrees(math.acos(eVe[1,1])), 3)

        #if eve_theta_00 == eve_theta_10 == eve_theta_01 == eve_theta_11:
        #    theta = eve_theta_00
        #else:
        #    print(2)
        #    print( eve_theta_00, eve_theta_10, eve_theta_01, eve_theta_11)

    else:
        theta = 0

    return theta


# code from stackoverflow
# https://stackoverflow.com/questions/15445546/finding-intersection-points-of-two-ellipses-python
def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result

def intersections(a, b):
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)

    x = [p.x for p in mp]
    y = [p.y for p in mp]
    return x, y


theta1 = get_theta_from_cov(C1)
theta2 = get_theta_from_cov(C2)

print(theta1)

# format (x0, y0, a, b, angle)
ellipses = [(1, 1, 2, 1, 45), (10, 10, 5, 1.5, -30)]

a, b = ellipse_polyline(ellipses)
x, y = intersections(a, b)

if len(x) != 0:
    separate_ellipses = False
else:
    separate_ellipses = True

print(separate_ellipses)





def test_angle():
    # Scaling matrix
    sx, sy = 0.7, 3.4
    Scale = np.array([[sx, 0], [0, sy]])

    # Rotation matrix
    theta = -0.77*np.pi
    print(theta)
    c, s = np.cos(theta), np.sin(theta)
    Rot = np.array([[c, -s], [s, c]])

    # Transformation matrix
    T = Scale.dot(Rot)

    # Apply transformation matrix to X
    x = np.random.normal(0, 1, 500)
    y = np.random.normal(0, 1, 500)
    X = np.vstack((x, y)).T
    X = X - np.mean(X, 0)

    Y = X.dot(T)

    cov_Y = (np.cov(Y.T))

    eVa_cov_Y, eVe_cov_Y = np.linalg.eig(cov_Y)
    eve_00 = (math.degrees(math.acos(eVe_cov_Y[0,0])))
    eve_10 = (180 - abs(math.degrees(math.asin(eVe_cov_Y[1,0]))))
    eve_01 = (180 - abs(math.degrees(math.asin(-eVe_cov_Y[0,1]))))
    eve_11 = (math.degrees(math.acos(eVe_cov_Y[1,1])))

    print(eve_11)
    #print(cov_Y)
    #print(Rot)

    # eve_00, eve_10, eve_01, eve_11 are all within a degree or two of math.degrees(theta)
    # which is what we should get. It's a little off bc we're working with rndm samples
