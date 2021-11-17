# Micro_Mapping
Library dedicated to functions involved in mapping objects between microscopes and different magnifications

The demo file shows the mapping pipeline, that contains 4 main steps.
1. Get the dimensions of a single tile image.
2. Find the tile position fro the nd2 metadata, and arrange into a mapping matrix.
   This step can take time as reading the metadata can take about 1.5s per tile.
3. Make sure the correct bounderies are matched. The mapping matrix may require some flipping or transposition.
4. Run a regression to determine the optimal affine trasformation to map the coordinates of the two sets of tiles.
   the functions requires at least 3 target points that are correnspond to the same position in both sets of tiles,
   to act as targets.
   The transformation here accounts for x,y translation, global rotation, and x,y scaling.
   The default minimization algorithm is Sequential Least Squares Programming (SLSQP), with all zero initial conditions.
   Objective function is a standard method of least squares

Demo file works with demo data, soon to be uploaded.

Dependencies

numpy
pandas
tqdm
matplotlib
scipy
pillow
pims_nd2
