#!/usr/bin/env python3
"""
Test script to demonstrate the new addition behavior for QUBO and linear_problem classes.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from qibo_comb_optimisation.optimisation_class.optimisation_class import QUBO, linear_problem

def test_qubo_addition():
    print("=== Testing QUBO Addition ===")
    
    # Create two QUBO objects
    Qdict1 = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
    Qdict2 = {(0, 0): 2.0, (1, 1): 1.0}
    
    qp1 = QUBO(0, Qdict1)
    qp2 = QUBO(1, Qdict2)
    
    print(f"QUBO 1: Qdict = {qp1.Qdict}, offset = {qp1.offset}")
    print(f"QUBO 2: Qdict = {qp2.Qdict}, offset = {qp2.offset}")
    
    # Add them together
    qp3 = qp1 + qp2
    
    print(f"Result: Qdict = {qp3.Qdict}, offset = {qp3.offset}")
    print(f"Original QUBO 1 unchanged: Qdict = {qp1.Qdict}, offset = {qp1.offset}")
    print(f"Original QUBO 2 unchanged: Qdict = {qp2.Qdict}, offset = {qp2.offset}")
    print()

def test_linear_problem_addition():
    print("=== Testing Linear Problem Addition ===")
    
    # Create two linear problem objects
    A1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([5, 6])
    A2 = np.array([[1, 1], [1, 1]])
    b2 = np.array([1, 1])
    
    lp1 = linear_problem(A1, b1)
    lp2 = linear_problem(A2, b2)
    
    print(f"Linear Problem 1: A = \n{lp1.A}, b = {lp1.b}")
    print(f"Linear Problem 2: A = \n{lp2.A}, b = {lp2.b}")
    
    # Add them together
    lp3 = lp1 + lp2
    
    print(f"Result: A = \n{lp3.A}, b = {lp3.b}")
    print(f"Original Linear Problem 1 unchanged: A = \n{lp1.A}, b = {lp1.b}")
    print(f"Original Linear Problem 2 unchanged: A = \n{lp2.A}, b = {lp2.b}")
    print()

if __name__ == "__main__":
    test_qubo_addition()
    test_linear_problem_addition()
    print("All tests completed successfully!") 