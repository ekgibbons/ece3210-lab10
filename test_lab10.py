import random
import time 
import unittest

import numpy as np

import ltid
import ltid_sol

class TestDT(unittest.TestCase):
    def test_recu(self):
        print("\nTesting recursive solution")
        M = random.randint(2,5)
        N = random.randint(M,7)
        
        b = np.random.uniform(-5,5,size=M)
        a = np.random.uniform(-5,5,size=N)

        k = np.arange(128)
        
        f = np.random.uniform(-10,10,size=len(k))
        y_my = ltid.recu_solution(f, b, a)
        y_sol = ltid_sol.recu_solution(f, b, a)
          
        np.testing.assert_array_almost_equal(y_my, y_sol)

        
    def test_recu_warn(self):
        print("\nTesting recursive solution warning")
        N = random.randint(2,5)
        M = random.randint(N+1,7)
        
        b = np.random.uniform(-5,5,size=M)
        a = np.random.uniform(-5,5,size=N)

        k = np.arange(128)

        f = np.random.uniform(-10,10,size=len(k))
        with self.assertRaises(ValueError):
            ltid.recu_solution(f, b, a)

            
    def test_impulse_response(self):
        print("\nTesting impulse function")
        M = random.randint(2,5)
        N = random.randint(M,7)
        
        b = np.random.uniform(-5,5,size=M)
        a = np.random.uniform(-5,5,size=N)
        
        N = 64
        h_my = ltid.find_impulse_response(b, a, N)
        h_sol = ltid_sol.find_impulse_response(b, a, N)
        
        np.testing.assert_array_almost_equal(h_my, h_sol)
        

    def test_impulse_response_warn(self):
        print("\nTesting impulse function warning")
        N = random.randint(2,5)
        M = random.randint(N+1,7)

        b = np.random.uniform(-5,5,size=M)
        a = np.random.uniform(-5,5,size=N)
        
        N = 64
        
        with self.assertRaises(ValueError):
            ltid.find_impulse_response(b, a, N)

            
    def test_z_transform(self):
        print("\nTesting z-transform")

        k = np.arange(128)

        y_my = ltid.find_z_transform(k)
        y_sol = ltid_sol.find_z_transform(k)

        np.testing.assert_array_almost_equal(y_my, y_sol)


    def test_frequency_response(self):
        print("\nTesting frequency response")
        
        M = random.randint(2,5)
        N = random.randint(M,7)
        
        b = np.random.uniform(-5,5,size=M)
        a = np.random.uniform(-5,5,size=N)

        H_my, Omega_my = ltid.frequency_response(b,a)
        H_sol, Omega_sol= ltid_sol.frequency_response(b,a)

        np.testing.assert_array_almost_equal(Omega_my, Omega_sol)
        np.testing.assert_array_almost_equal(H_my, H_sol)


    def test_frequency_response_warn(self):
        print("\nTesting frequency response warning")
        N = random.randint(2,5)
        M = random.randint(N+1,7)

        b = np.random.uniform(-5,5,size=M)
        a = np.random.uniform(-5,5,size=N)
        
        with self.assertRaises(ValueError):
            ltid.frequency_response(b, a, N)


    def test_scipy_loops(self):
        print("\nVarious inclusion checks")
        
        with open("ltid.py", "r") as file:
            content = file.read()

        scipy_check = False
        # check if string present or not
        if "scipy" in content:
            scipy_check = True

        self.assertFalse(scipy_check,
                         "You are illegally using "
                         "the scipy library")

        for_count = content.count("for ")
        while_count = content.count("while ")

        loop_count = for_count + while_count

        self.assertLess(loop_count,2,
                        "You are using %i loops" %
                        loop_count)

        
if __name__ == "__main__":
    unittest.main()
