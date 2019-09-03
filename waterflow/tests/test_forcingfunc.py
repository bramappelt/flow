import unittest
import waterflow.utility.forcingfunc as ff


class TestForcing(unittest.TestCase):
    ''' Test polynomials and check if points are intersected '''
    def test_polynomial(self):
        # degree 1 (x = constant)
        pts = [[1, 1]]
        _, _, _, func = ff.polynomial(pts)
        self.assertAlmostEqual(pts[0][1], func(pts[0][0]))
        self.assertAlmostEqual(pts[0][1], func(-5))
        self.assertAlmostEqual(pts[0][1], func(85))
        # degree 2 (linear)
        pts2 = [[-9, 5], [0, 0]]
        _, _, _, func = ff.polynomial(pts2)
        self.assertAlmostEqual(pts2[0][1], func(pts2[0][0]))
        self.assertAlmostEqual(pts2[1][1], func(pts2[1][0]))
        # degree 3 (parabola)
        pts3 = [[-8, -7], [5, 5], [2, 9]]
        _, _, _, func = ff.polynomial(pts3)
        self.assertAlmostEqual(pts3[0][1], func(pts3[0][0]))
        self.assertAlmostEqual(pts3[1][1], func(pts3[1][0]))
        self.assertAlmostEqual(pts3[2][1], func(pts3[2][0]))
        
        
        
if __name__ == "__main__":
    unittest.main()       
