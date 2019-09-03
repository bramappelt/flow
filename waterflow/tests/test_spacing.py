import unittest
import numpy as np
from waterflow.utility.spacing import spacing

class TestSpacing(unittest.TestCase):
    ''' Testing the nodal distance spacing function '''
    def test_spacing_linear(self):
        nx = 5; ny = 5; Lx = 10; Ly = 10
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=True)
        self.assertEqual([ 0. ,  2.5,  5. ,  7.5, 10. ], x_spa.tolist())
        self.assertEqual([ 0. ,  2.5,  5. ,  7.5, 10. ], y_spa.tolist())        

    def test_spacing_nonlinear(self):
        nx = 5; ny = 5; Lx = 10; Ly = 10
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False, loc=[(2,2)],
                               power=1, weight=2)
        self.assertEqual([ 0.  ,  3.75,  5.  ,  6.25, 10. ], x_spa.tolist())
        self.assertEqual([ 0.  ,  3.75,  5.  ,  6.25, 10. ], y_spa.tolist())

    def test_spacing_loc(self):
        nx = 11; ny = 11; Lx = 10; Ly = 10
        x_spa_ans = [0.,1.4375,2.875,3.75,4.5,5.,5.5,6.25,7.125,8.5625,10.]
        # well in middle (5,5)
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False, loc=[(5,5)],
                       power=3, weight=2)
        self.assertEqual(x_spa_ans, x_spa.tolist())
        self.assertEqual(x_spa_ans, y_spa.tolist())
        # well not in middle (5,4)
        y_spa_ans = [0.,1.875,2.75,3.5,4.,4.5,5.25,6.125,7.4167,8.7083,10.]
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False, loc=[(5,4)],
               power=3, weight=2)
        self.assertEqual(x_spa_ans, x_spa.tolist())
        self.assertEqual(y_spa_ans, np.around(y_spa, 4).tolist())

    def test_spacing_power_weight(self):
        nx = 11; ny = 11; Lx = 10; Ly = 10
        x_spa_ans = [0.,1.9375,2.875,3.75,4.5,5.,5.5,6.25,7.125,8.0625,10.]
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False,
                       loc=[(5,5)], power = 4, weight = 2)
        self.assertEqual(x_spa_ans, np.around(x_spa, 4).tolist())
        self.assertEqual(x_spa_ans, np.around(y_spa, 4).tolist())

        x_spa_ans2 = [0.,4.0056,4.3813,4.679,4.8889,5.,5.1111,5.321,5.6187,5.9944,10.]
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False,
               loc=[(5,5)], power = 4, weight = 9)
        self.assertEqual(x_spa_ans2, np.around(x_spa, 4).tolist())
        self.assertEqual(x_spa_ans2, np.around(y_spa, 4).tolist())

    def test_spacing_1D_linear(self):
        nx = 5; Lx = 10
        x_spa, y_spa = spacing(nx, Lx)
        self.assertEqual([ 0. ,  2.5,  5. ,  7.5, 10. ], x_spa.tolist())
        self.assertEqual([], y_spa.tolist())

    def test_spacing_1D_nonlinear(self):
        nx = 5; Lx = 10
        x_spa, y_spa = spacing(nx, Lx, linear=False, loc = [2], power = 1, weight = 2)
        self.assertEqual([ 0.  ,  3.75,  5.  ,  6.25, 10. ], x_spa.tolist())
        self.assertEqual([], y_spa.tolist())

    def test_spacing_1D_multiple_wells(self):
        nx = 22; Lx = 10
        x_spa_ans = [0.,0.6526,1.3051,1.9577,2.2222,2.381,2.5397,2.8042,4.3386,4.6032,
                     4.7619,4.9206,5.1852,6.7196,6.9841,7.1429,7.3016,7.5661,8.1746,
                     8.7831,9.3915,10.]
        x_spa, y_spa = spacing(nx, Lx, linear=False, loc=[5, 10, 15],
                                 power = 2, weight = 3)
        self.assertEqual(x_spa_ans, np.around(x_spa, 4).tolist())
        self.assertEqual([], y_spa.tolist())

    def test_spacing_2D_multiple_wells(self):
        nx = 22; Lx = 10; ny = 22; Ly = 10
        x_spa_ans = [0.,0.6526,1.3051,1.9577,2.2222,2.381,2.5397,2.8042,4.3386,4.6032,
                     4.7619,4.9206,6.2434,6.5079,6.6667,6.8254,7.0899,7.672,8.254,
                     8.836,9.418,10.]
        y_spa_ans = [0.,0.6526,1.3051,1.9577,2.2222,2.381,2.5397,2.8042,4.3386,4.6032,
                     4.7619,4.9206,5.1852,6.7196,6.9841,7.1429,7.3016,7.5661,8.1746,
                     8.7831,9.3915,10.]
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False, 
                         loc=[(5,10), (10,5), (14,15)], power = 2, weight = 3)
        self.assertEqual(x_spa_ans, np.around(x_spa, 4).tolist())
        self.assertEqual(y_spa_ans, np.around(y_spa, 4).tolist())

    def test_spacing_equal_exes(self):
        nx = 22; Lx = 10; ny = 22; Ly = 10
        x_spa, y_spa = spacing(nx, Lx, ny, Ly, linear=False, 
                         loc=[(5,5), (10,10), (15,15)], power = 2, weight = 3)
        self.assertEqual(np.all(x_spa==y_spa), True)

        
if __name__ == "__main__":
    unittest.main()
