import numpy as np
import matplotlib.pyplot as plt

class Bspline():
   def B(self, t, k, i, time_knots):            # B-spline function t is the variable, k is the degree, i is the index
      if k == 0:
         return 1.0 if time_knots[i] <= t < time_knots[i+1] else 0.0
      c1, c2 = 0, 0
      d1 = time_knots[i+k] - time_knots[i]
      d2 = time_knots[i+k+1] - time_knots[i+1]
      if d1 > 0:
         c1 = ((t - time_knots[i]) / d1) * self.B(t, k-1, i, time_knots)
      if d2 > 0:
         c2 = ((time_knots[i+k+1] - t) / d2) * self.B(t, k-1, i+1, time_knots)
      return c1 + c2
   
   # def in_last_interval(self, knot, t):
   #    if len(knot) < 2:
   #       return False
   #    return knot[-2] <= t <= knot[-1]

   def bspline(self, time_knots, c, k):         # time_knots: knot vector, c: control points, k: degree
      n = len(time_knots) - k - 1         
      assert (n >= k+1) and (len(c) >= n)
      x = np.linspace(0, max(time_knots)-0.5, 100)
      # x = np.linspace(k, len(c), 100)
      trajec = np.zeros((len(x), len(c[0])))
      for m in range(len(x)):
         for n in range(len(c)):
            trajec[m] += self.B(x[m], k, n, time_knots) * c[n]
      trajec = np.delete(trajec, -1, axis=0)
      print(len(trajec))
      print(self.B(max(time_knots), k, 3, time_knots), c[1])
      return trajec
   
class Bspline_basis():
   def bspline_basis(self, control_points, knots, degree=3):

      n = len(control_points) - 1
      m = len(knots) - 1
      assert (n + degree + 1) == m
      t = np.linspace(0, max(knots), 101)
      # t = np.linspace(degree, len(control_points), 100)
      # t = np.linspace(degree, len(control_points), 100)
      
      trajec = np.zeros((len(t), len(control_points[0])))
      for k in range(len(t)):
         for i in range(n+1-degree):
            ii = i + degree
            # print(control_points[ii-degree:ii+1], knots[ii], knots[ii+1], t[m])
            trajec[k] += self.C(control_points[ii-degree:ii+1], ii, knots, t[k])[0]
            # trajec[k] += self.C_new(control_points[ii-degree:ii+1], ii, knots, t[k])[0]

      trajec = np.delete(trajec, -1, axis=0)
      return trajec
   
   def C_new(self, cp, ii, knots, t):
      floor_index = self.find_floor(knots, t)
      ti = knots[floor_index]

      if floor_index == len(knots) - 1:
         ti_plus_1 = knots[floor_index]
      else:
         ti_plus_1 = knots[floor_index+1]

      if t - ti == 0 and ti_plus_1 - ti == 0:
         u_t = 0
      else:
         u_t = (t - ti) / (ti_plus_1 - ti)

      UU = np.array([[1, u_t, u_t ** 2, u_t **3]])
      M_BS_4 = np.array([[1, 4, 1, 0],
                         [-3, 0, 3, 0],
                         [3, -6, 3, 0],
                         [-1, 3, -3, 1]])/6
      C_t = UU @ M_BS_4 @ cp
      # print(C_t)
      return C_t

   def find_floor(self, array, value):
      array = np.asarray(array)
      idx = (np.abs(array - value)).argmin()
      if array[idx] > value:
         idx = idx - 1
      return idx

   def C(self, cp, ii, knots, t):
      ti = knots[ii]
      ti_plus_1 = knots[ii+1]
      ti_plus_2 = knots[ii+2]
      ti_plus_3 = knots[ii+3]
      ti_minus_1 = knots[ii-1]
      ti_minus_2 = knots[ii-2]

      indicator = 1 if ti <= t < ti_plus_1 else 0

      common_denominator = (ti_plus_2 - ti_minus_1) * (ti_plus_1 - ti_minus_1)
      m00 = (ti_plus_1 - ti) ** 2 / ((ti_plus_1 - ti_minus_1) * (ti_plus_1 - ti_minus_2))
      m02 = (ti - ti_minus_1) ** 2 / common_denominator
      m01 = 1 - m00 - m02
      m03, m13, m23 = 0, 0, 0
      m10, m20, m30 = -3 * m00, 3 * m00, -m00
      m12 = 3 * (ti_plus_1 - ti) * (ti - ti_minus_1) / common_denominator
      m11 = 3 * m00 - m12
      m22 = 3 * (ti_plus_1 - ti) ** 2 / common_denominator
      m21 = -3 * m00 - m22
      m33 = (ti_plus_1 - ti) ** 2 / ((ti_plus_3 - ti) * (ti_plus_2 - ti))
      m32 = -m22/3 - m33 - (ti_plus_1 - ti) ** 2 / ((ti_plus_2 - ti)*(ti_plus_2 - ti_minus_1))
      m31 = m00  - m32 - m33

      M_BS_4 = np.array([[m00, m01, m02, m03],
                           [m10, m11, m12, m13],
                           [m20, m21, m22, m23],
                           [m30, m31, m32, m33]])
      # print(M_BS_4)

      # if indicator == 1:
      #    print(M_BS_4, indicator, t)

      A_3 = np.array([[-0.4302, 0.4568, -0.02698, 0.0004103],
                [0.8349, -0.4568, -0.7921, 0.4996],
                [-0.8349, -0.4568, 0.7921, 0.4996],
                [0.4302, 0.4568, 0.02698, 0.0004103]])
      inverse_A_3 = np.linalg.inv(A_3)
            
      
      u_t = (t - ti) / (ti_plus_1 - ti)
      UU = np.array([[1, u_t, u_t ** 2, u_t **3]])
      

      rotated_M_BS_4 = list(zip(*M_BS_4[::-1]))

      C_t = UU @ M_BS_4 @ cp
      
      # C_t = UU @ A_3 @ cp

      
      return C_t * indicator
   


def func_2d_test():           # 2D test
   # cv = np.array([[ 50.,  25.],
   # [ 59.,  12.],
   # [ 50.,  10.],
   # [ 57.,   2.],
   # [ 40.,   4.],
   # [ 40.,   14.]])
   # cv = np.array([[1, 1], [0, 8], [5, 10], [9, 7], [4, 3]])
   cv = np.array([[1, 1], [0, 8], [5, 10], [9, 7]])
   # cv = np.array([[ 1.50035581e-11,  2.00000001e+00],
   #                [ 1.02864764e+00, -1.79427048e+00],
   #                [ 1.00000000e+01,  3.36148721e-15],
   #                [ 4.65233065e+00, -1.06953388e+00]])

   k = 3
   t = np.array([0]*k + list(range(len(cv)-k+1)) + [len(cv)-k]*k,dtype='int')
   # t2 = np.array([0., 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.])
   t2 = np.array(list(range(len(cv)+k+1)))+2
   # t2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
   print(t)

   ### B-spline
   plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
   traj = Bspline()
   bspline_curve = traj.bspline(t, cv, k)
   bspline_curve_2 = traj.bspline(t2, cv, k)
   plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
   plt.gca().set_aspect('equal', adjustable='box')
   plt.plot(bspline_curve[:,0], bspline_curve[:,1], label='Uniform Clamped Time Knots')
   plt.plot(bspline_curve_2[:,0], bspline_curve_2[:,1], label='Unifrom Time Knots')
   plt.legend(loc='lower right')
   plt.grid(axis='both')
   plt.show()
   # print(bspline_curve)

   ### B-spline basis
   plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
   traj_prime = Bspline_basis()
   bspline_curve_prime = traj_prime.bspline_basis(cv, t2, k)
   plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
   plt.gca().set_aspect('equal', adjustable='box')
   plt.plot(bspline_curve_prime[:,0], bspline_curve_prime[:,1], label='B-spline Curve')
   len_bspline_curve_prime = len(bspline_curve_prime)
   half_len = int(len_bspline_curve_prime/2)
   # plt.arrow(bspline_curve_prime[0,0], bspline_curve_prime[0,1], bspline_curve_prime[1,0]-bspline_curve_prime[0,0], bspline_curve_prime[1,1]-bspline_curve_prime[0,1], head_width=0.5, head_length=0.5, fc='k', ec='k')
   # plt.arrow(bspline_curve_prime[half_len,0], bspline_curve_prime[half_len,1], bspline_curve_prime[half_len+1,0]-bspline_curve_prime[half_len,0], bspline_curve_prime[half_len+1,1]-bspline_curve_prime[half_len,1], head_width=0.5, head_length=0.5, fc='k', ec='k')
   plt.legend(loc='upper left')
   plt.grid(axis='both')
   plt.show()


def func_3d_test():
   cv = np.array([[0, 0, 0], [0, 4, 0], [2, 5, 0], [4, 5, 0], [5, 4, 0], [5, 1, 0], [4, 0, 0], [1, 0, 3], [0, 0, 4], [0, 2, 5], [0, 4, 5], [4, 5, 5], [5, 5, 4], [5, 5, 0]])
   k = 3
   t = np.array([0]*k + list(range(len(cv)-k+1)) + [len(cv)-k]*k,dtype='int')
   traj = Bspline()
   bspline_curve3d = traj.bspline(t, cv, k)
   fig = plt.figure(dpi=128)
   ax = fig.add_subplot(projection='3d')
   ax.plot(bspline_curve3d[:, 0], bspline_curve3d[:, 1], bspline_curve3d[:, 2], 'r-', label='Bezier Curve')
   ax.plot(cv[:, 0], cv[:, 1], cv[:, 2], 'o-', label='Control Points')
   plt.show()

def func_2d_closrue_test():           # 2D test
   cv = np.array([[0, 0], [0, 8], [5, 10], [9, 7], [4, 3], [8,0], [0,0]])

   k = 3
   t = np.array([0]*k + list(range(len(cv)-k+1)) + [len(cv)-k]*k,dtype='int')
   print(t)
   plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
   traj = Bspline()
   bspline_curve = traj.bspline(t, cv, k)
   plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
   plt.gca().set_aspect('equal', adjustable='box')
   plt.plot(bspline_curve[:,0], bspline_curve[:,1], label='B-spline Curve')
   plt.legend(loc='upper left')
   plt.grid(axis='both')
   print(bspline_curve)
   plt.show()

   # plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
   # traj_prime = Bspline_basis()
   # bspline_curve_prime = traj_prime.bspline_basis(cv, t, k)
   # plt.xticks([ ii for ii in range(-20, 20)]), plt.yticks([ ii for ii in range(-20, 20)])
   # plt.gca().set_aspect('equal', adjustable='box')
   # plt.plot(bspline_curve_prime[:,0], bspline_curve_prime[:,1], label='B-spline Curve')
   # plt.legend(loc='upper left')
   # plt.grid(axis='both')
   # print(bspline_curve_prime)
   # plt.show()

if __name__ == '__main__':
   func_2d_test()
   # func_3d_test()
   # func_2d_closrue_test()

