import torch

def create_element_vector(atomic_number):
    vector = torch.zeros(118)
    vector[atomic_number - 1] = 1
    return vector

H = create_element_vector(1)
C = create_element_vector(6)
N = create_element_vector(7)
O = create_element_vector(8)
S = create_element_vector(16)

water = torch.stack([H, H, O])
ethane = torch.stack([C, C, H, H, H, H, H, H])
methane = torch.stack([C, H, H, H, H])
ammonia = torch.stack([N, H, H, H])
hydrogen_sulfide = torch.stack([H, H, S])
carbon_dioxide = torch.stack([C, O, O])

water_edge_index = torch.tensor([
    [0, 1, 2, 2],
    [2, 2, 0, 1],
], dtype=torch.long)

ethane_edge_index = torch.tensor([
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7],
    [2, 1, 6, 7, 3, 4, 5, 0, 0, 1, 1, 1, 0, 0],
], dtype=torch.long)

methane_edge_index = torch.tensor([
    [0, 0, 0, 0, 1, 2, 3, 4],
    [1, 2, 3, 4, 0, 0, 0, 0],
], dtype=torch.long)

ammonia_edge_index = torch.tensor([
    [0, 0, 0, 1, 2, 3],
    [1, 2, 3, 0, 0, 0],
], dtype=torch.long)

hydrogen_sulfide_edge_index = torch.tensor([
    [0, 1, 2, 2],
    [2, 2, 0, 1],
], dtype=torch.long)

carbon_dioxide_edge_index = torch.tensor([
    [0, 0, 1, 2],
    [1, 2, 0, 0],
], dtype=torch.long)

boiling_point = torch.tensor([100, 
                             78.37,
                             -161.5,
                             -33.34,
                             -60], dtype=torch.float)   # Celsius

data_list = [
    (water, water_edge_index, boiling_point[0]),
    (ethane, ethane_edge_index, boiling_point[1]),
    (methane, methane_edge_index, boiling_point[2]),
    (ammonia, ammonia_edge_index, boiling_point[3]),
    (hydrogen_sulfide, hydrogen_sulfide_edge_index, boiling_point[4]),
]
