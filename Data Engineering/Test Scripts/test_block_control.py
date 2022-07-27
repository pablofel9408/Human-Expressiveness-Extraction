from math import pi
import numpy as np
from roboticstoolbox.models.DH import Puma560
import bdsim

bd = bdsim.BlockDiagram()

puma = Puma560()
q0 = [0, pi/4, pi, 0, pi/4, 0]

# define the blocks
jacobian = bd.JACOBIAN(robot=puma, frame='0', inverse=True, name='Jacobian')
velocity = bd.CONSTANT([0, 0.5, 0, 0, 0, 0]) # task-space velocity
qdot = bd.PROD('**', matrix=True)
integrator = bd.INTEGRATOR(x0=q0)
robot = bd.ARMPLOT(robot=puma, q0=q0, name='plot')

# connect the blocks
bd.connect(jacobian, qdot[0])
bd.connect(velocity, qdot[1])
bd.connect(qdot, integrator)
bd.connect(integrator, jacobian, robot)

bd.compile()   # check the diagram
bd.report()    # list all blocks and wires
bd.run(1.5)    # simulate for 1.5s

bd.done()