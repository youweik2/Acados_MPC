import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# grid_1d = [0, 100, 100, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1,\
#             -1, 0, 100, -1, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
#             0, 100, 0, 0, 0, 100, 100, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 100, 0, -1, -1, 0, 0, 0, 0, \
#             0, 0, 100, -1, 100, 100, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 100, \
#             100, 100, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 100, 100, 100, 0, 0,\
#                 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
#             -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, \
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
#             0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, \
#             0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
#             0, 0, -1, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, \
#             100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, \
#             100, 0, -1, 0, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 100, 100, 100, \
#             -1, -1, 0, 100, 100, 0, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
#             -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
#             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

# grid_1d = np.array(grid_1d)
# grid = grid_1d.reshape(22, 20)

# print(len(grid))

# square_size = 0.5

# # Create a color map where 0 is white, 50 is gray, and 100 is black
# color_map = {0: "white", -1: "gray", 100: "black"}
# cx, cy  = 0, 0

# for i in range(len(grid)):
#     for j in range(len(grid[i])):
#         color = color_map[grid[i][j]]
#         # Draw each square with the appropriate color
#         plt.gca().add_patch(plt.Rectangle((cx + j * square_size, cy + i * square_size), square_size, square_size, color=color))

# plt.axis('equal')
# plt.legend()
# plt.show()

# print(grid)

def get_gvalue(cur_x, cur_y, cx, cy, grid):

    hpg, wpg  = 22, 20
    h = 0.5

    # Reshape the flattened grid into a 50x50 symbolic matrix
    gridmap = ca.reshape(grid, hpg, wpg)

    # Compute symbolic grid indices
    grid_x = ca.fmax(ca.floor((cur_x + cx) / h), 0)
    grid_y = ca.fmax(ca.floor((cur_y + cy) / h), 0)
    

    # Handle boundary cases symbolically
    grid_x = ca.if_else(grid_x < 0, 0, ca.if_else(grid_x >= wpg, hpg - 1, grid_x))
    grid_x1 = ca.if_else(grid_x + 1 >= wpg, wpg - 1, grid_x + 1)
    grid_y = ca.if_else(grid_y < 0, 0, ca.if_else(grid_y >= hpg, wpg - 1, grid_y))
    grid_y1 = ca.if_else(grid_y + 1 >= wpg, hpg - 1, grid_y + 1)

    print(grid_x, grid_y, grid_x1, grid_y1)

    def symbolic_lookup(matrix, row_idx, col_idx):
        result = 0
        for i in range(hpg):
            for j in range(wpg):
                result += ca.if_else(
                    ca.logic_and(row_idx == i, col_idx == j),
                    ca.if_else(matrix[i, j] == -1, 50, matrix[i, j]),
                    0
                )
        return result
    
    # Access grid values using the updated symbolic_lookup
    gxy = symbolic_lookup(gridmap, grid_x, grid_y)
    gxpy = symbolic_lookup(gridmap, grid_x1, grid_y)
    gxyp = symbolic_lookup(gridmap, grid_x, grid_y1)
    gxpyp = symbolic_lookup(gridmap, grid_x1, grid_y1)

    # Compute weights
    I_x = ca.floor((cur_x + cx) / h)
    I_y = ca.floor((cur_y + cy) / h)
    R_x = (cur_x + cx) / h - I_x
    R_y = (cur_y + cy) / h - I_y

    # Symbolic matrix and vector operations
    m_x = ca.vertcat(1 - R_x, R_x)
    m_g = ca.horzcat(ca.vertcat(gxy, gxpy), ca.vertcat(gxyp, gxpyp))
    m_y = ca.vertcat(1 - R_y, R_y)

    # Compute the value
    g_value = ca.mtimes([m_x.T, m_g, m_y])
    return g_value


grid_1d = [0, 100, 100, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1,\
            -1, 0, 100, -1, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
            0, 100, 0, 0, 0, 100, 100, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 100, 0, -1, -1, 0, 0, 0, 0, \
            0, 0, 100, -1, 100, 100, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 100, \
            100, 100, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 100, 100, 100, 0, 0,\
                0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
            0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
            0, 0, -1, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, \
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, \
            100, 0, -1, 0, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 100, 100, 100, \
            -1, -1, 0, 100, 100, 0, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
            -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    
grid_1d = np.array(grid_1d)
grid = (grid_1d.reshape(20, 22)).T


# for i in range(100):
#     value = get_gvalue(-5 + 0.1 * i, 2.5, 5.5, 1.2825, grid)
#     print(value)


print(get_gvalue(-4.46, 3.7, 5.5, -1.2825, grid))