#!/usr/bin/env python

from matplotlib import rc
from daft import PGM, Node, Plate
rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = PGM([7.0, 4.2], origin=[0.5, 0.2], observed_style='inner')

# x_1 and c distributions on top line
pgm.add_node(Node("x1dist", r"$x_1^{\mathrm{dist}}$", 3, 4))
pgm.add_node(Node("cdist", r"$c^{\mathrm{dist}}$", 4, 4))

# Per-SN parameters:  top line in the plate
pgm.add_node(Node("x1itrue", r"$x_{1,i}^\mathrm{true}$", 3, 3))
pgm.add_node(Node("citrue", r"$c_i^\mathrm{true}$", 4, 3))
pgm.add_node(Node("x1iobs", r"$x_{1,i}^\mathrm{obs}$", 2, 3, observed=True))
pgm.add_node(Node("ciobs", r"$c_i^\mathrm{obs}$", 5, 3, observed=True))

# Per-SN parameters: second line in the plate
pgm.add_node(Node("x0itrue", r"$x_{0,i}^\mathrm{true}$", 3.5, 2))
pgm.add_node(Node("mui", r"$\mu_i$", 2, 2))

# Per-SN parameters: third line in the plate
pgm.add_node(Node("zi", r"$z_i$", 2, 1, observed=True))
pgm.add_node(Node("x0iobs", r"$x_{0,i}^\mathrm{obs}$", 3.5, 1, observed=True))

# Plate
pgm.add_plate(Plate([1.5, 0.5, 4, 3],
                    label=r"supernovae $i = 1, \cdots, N_{SN}$",
                    shift=-0.1))

# Cosmological parameters
pgm.add_node(Node("cosmology", r"$\Omega$", 1, 2))

# nuisance parameters
pgm.add_node(Node("nuisance",
                  r"$\alpha, \beta, \mathcal{M}, \sigma_{\mathrm{int}}$",
                  6.5, 2, aspect=3.0))

# Add in the edges.
pgm.add_edge("x1dist", "x1itrue")
pgm.add_edge("cdist", "citrue")
pgm.add_edge("x1itrue", "x1iobs")
pgm.add_edge("citrue", "ciobs")
pgm.add_edge("x1itrue", "x0itrue")
pgm.add_edge("citrue", "x0itrue")

pgm.add_edge("cosmology", "mui")
pgm.add_edge("mui", "x0itrue")
pgm.add_edge("nuisance", "x0itrue")

pgm.add_edge("zi", "mui")
pgm.add_edge("x0itrue", "x0iobs")

# Render and save.
pgm.render()
pgm.figure.savefig("sn-cosmology-pgm.png", dpi=150)
