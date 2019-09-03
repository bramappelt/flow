import unittest
import numpy as np
from scipy.integrate import quad

from waterflow.flow1d.flowFE1d import Flow1DFE as FE
from waterflow.utility.spacing import spacing
from waterflow.utility.forcingfunc import polynomial as poly

'''
_aggregate_forcing, _aggregate_boundaries, _CMAT, solve, waterbalance
'''

ksat = 1
kfun = lambda x: ksat + x * 0.1

def fluxfunction(x, s, gradient):
    return -1 * gradient * ksat

_, _, _, rainfun = poly([(0, 0), (20, 0.01)])

MS = FE("structured")
struc = [20, 21]
MS_states = np.repeat(0, 21)

MU = FE("unstructured")
unstruc, _ = spacing(21, 20, linear=False, loc=[6, 15], power=4, weight=5)

class TestFlow1dFE(unittest.TestCase):
    ''' Test the Flow1dFE package '''
    
    def test_set_field1d(self):
        MS.set_field1d(linear=struc)
        self.assertEqual(list(np.linspace(0, struc[0], struc[1])),
                         list(MS.nodes))
        self.assertEqual([i*0 for i in range(struc[1])], list(MS.states))

        MU.set_field1d(array=unstruc)
        self.assertEqual(list(unstruc), list(MU.nodes))
        self.assertEqual([0*i for i in range(len(unstruc))], list(MU.states))
        
    def test_set_systemfluxfunction(self):
        MS.set_systemfluxfunction(fluxfunction)
        self.assertEqual(fluxfunction, MS.systemfluxfunc)
        
        MU.set_systemfluxfunction(fluxfunction)
        self.assertEqual(fluxfunction, MU.systemfluxfunc)
        
    def test_set_initial_states(self):
        MS.set_initial_states(MS_states)
        self.assertEqual(list(MS_states), list(MS.states))
        
        MS.set_field1d(linear=struc)
        MS.set_initial_states(5.0)
        self.assertEqual([5.0 for x in range(len(MS.nodes))], list(MS.states))

    def test_wrap_bf_linear(self):
        MS.set_field1d(linear=struc)
        MU.set_field1d(array=unstruc)
        
        l_struc0 = MS.wrap_bf_linear(0, "left")
        r_struc0 = MS.wrap_bf_linear(1, "right")
        
        self.assertEqual(1.0, l_struc0(0))
        self.assertEqual(0.0, r_struc0(1))
        
        self.assertEqual(0.5, l_struc0(0.5))
        self.assertEqual(0.5, r_struc0(1.5))

        l_unstruc0 = MU.wrap_bf_linear(0, "left")
        r_unstruc0 = MU.wrap_bf_linear(1, "right")
        
        self.assertEqual(1, l_unstruc0(0))
        self.assertEqual(0, r_unstruc0(MU.nodes[1]-MU.nodes[0]))
        
        self.assertAlmostEqual(0.5, l_unstruc0((MU.nodes[1]-MU.nodes[0])/2))
        self.assertAlmostEqual(0.5, r_unstruc0((MU.nodes[1]-MU.nodes[0])/2
                                               +MU.nodes[1]))

    def test_BC_methods(self):
        MS.set_field1d(linear=struc)
               
        MS.add_dirichlet_BC(5, "west")
        self.assertEqual((5, "Dirichlet", 0), MS.BCs["west"])
        
        MS.add_dirichlet_BC(4, "west")
        self.assertEqual((4, "Dirichlet", 0), MS.BCs["west"])
        
        MS.add_dirichlet_BC(3, "east")
        self.assertEqual((3, "Dirichlet", -1), MS.BCs["east"])
        
        MS.add_neumann_BC([5.0, 4.5], ["east", "west"])
        self.assertEqual((5.0, "Neumann", -1), MS.BCs["east"])
        self.assertEqual((4.5, "Neumann", 0), MS.BCs["west"])
        
        MS.remove_BC("west")
        self.assertEqual((5.0, "Neumann", -1), MS.BCs["east"])
        
        MS.remove_BC()
        self.assertEqual({}, MS.BCs)
        
        with self.assertRaises(Exception) as context:
            MS.remove_BC("wrong name")
        self.assertTrue("No boundary named wrong name.", context.exception)
    
    def test_add_spatialflux(self):
        MS.set_field1d(linear=struc)
        MS.add_spatialflux(0.003, "rch1")
        self.assertAlmostEqual(0.003*20, sum(MS.spatflux["rch1"]))
        self.assertTrue(MS.spatflux["rch1"][0], MS.spatflux["rch1"][-1])
        self.assertNotAlmostEqual(MS.spatflux["rch1"][0],
                                  MS.spatflux["rch1"][-2])
        
        MS.add_spatialflux(rainfun, "rchfun")
        self.assertAlmostEqual(0.5*20*0.01, sum(MS.spatflux["rchfun"]))
        
        with self.assertRaises(Exception) as context:
            MS.remove_spatialflux("wrong name")
        self.assertTrue("wrong name is not a spatialflux.", context.exception)
        
        MS.remove_spatialflux("rch1", "rchfun")
        self.assertEqual({}, MS.spatflux)
        
    def test_add_pointflux(self):
        MS.set_field1d(linear=struc)
         
        MS.add_pointflux(5, 6.2, "well1")
        self.assertEqual(5, sum(MS.pointflux["well1"]))
        self.assertAlmostEqual(4, MS.pointflux["well1"][6])
        self.assertAlmostEqual(1, MS.pointflux["well1"][7])
        
        MS.add_pointflux(4.4, 5.1, "well2")
        self.assertEqual(4.4, sum(MS.pointflux["well2"]))
        self.assertAlmostEqual(3.96, MS.pointflux["well2"][5])
        self.assertAlmostEqual(0.44, MS.pointflux["well2"][6])
        
        MS.remove_pointflux("well1")
        self.assertTrue("well1" not in MS.pointflux.keys())
        
        MS.remove_pointflux()
        self.assertEqual({}, MS.pointflux)
        
        MU.set_field1d(array=unstruc)
        MU.set_systemfluxfunction(fluxfunction)
        
        MU.add_pointflux([5.5,12.3],[5.5, 5.9], "wells")
        self.assertAlmostEqual(17.8, sum(MU.pointflux["wells"]))
        self.assertAlmostEqual(4.58333333, MU.pointflux["wells"][4])
        self.assertAlmostEqual(7.06666667, MU.pointflux["wells"][5])
        self.assertAlmostEqual(6.15, MU.pointflux["wells"][6])
        
        MU.remove_pointflux()
        self.assertEqual({}, MU.pointflux)
    
        
if __name__ == "__main__":
    unittest.main()

