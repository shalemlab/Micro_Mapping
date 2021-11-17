import Mapping_Functions as mf

path_10X = 'D:/Mapping/Try_2/10X'
path_40X = 'D:/Mapping/Try_2/40X'

# Define target points. Minimum of three points is advised
p_10X = [
[3, 881, 59],
[1, 875, 1067],
[0, 1059, 1545]]
p_40X = [
[36,1085, 23],
[7, 1097, 1790],
[2, 1784, 1411]]

# 1. Get image height and width
h_10X, w_10X = mf.Get_Image_Size(path_10X, verbose = False)
h_40X, w_40X = mf.Get_Image_Size(path_40X, verbose = False)

# 2. Check image locations from metadata, and arrage in a 2D map
M_10X, Tile_List_10X = mf.Arrange_Tiles(path_10X, plot_location = False)
M_40X, Tile_List_40X = mf.Arrange_Tiles(path_40X, plot_location = False)

# 3. Make sure the tile are continous, as they may need transposition or flipping
M_10X = mf.Check_Tile_Configuration('D:/Mapping/Try_2/10X', M_10X, file_type='nd2', plot_align = False, print_config = False, plot_tiles = False)
M_40X = mf.Check_Tile_Configuration('D:/Mapping/Try_2/40X', M_40X, file_type='nd2', plot_align = False, print_config = False, plot_tiles = False)

# 4. Calculate mapping degrees of freedom (DOF)
P_10X = mf.Local_to_Global(p_10X, M_10X, [h_10X, w_10X])
P_40X = mf.Local_to_Global(p_40X, M_40X, [h_40X, w_40X])
DOF = mf.Fit_By_Points(P_10X, P_40X, verbose=False)

print(DOF)

# Use mapping to find any other point
p_10X = [[2, 505, 1490],
         [1, 212, 970]]

P_10X = mf.Local_to_Global(p_10X, M_10X, [h_10X, w_10X])
P_40X = mf.model_TRS(P_10X, DOF, angle='degree')
p_40X = mf.Global_to_Local(P_40X, M_40X, [h_40X, w_40X])

# Plot corresponding 40X cells from 10X cells
mf.Plot_Point_Mapping(p_10X, p_40X, Tile_List_10X, Tile_List_40X, path_10X, path_40X)





