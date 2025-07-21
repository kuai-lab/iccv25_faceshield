import torch
import torch.nn.functional as F

def make_dct_basis(N, device):
    x, y= torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')
    u, v = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')

    basis = torch.cos((2 * x + 1) * u.unsqueeze(-1).unsqueeze(-1) * torch.pi / (2 * N)) * \
            torch.cos((2 * y + 1) * v.unsqueeze(-1).unsqueeze(-1) * torch.pi / (2 * N))
            
    N_tensor = torch.arange(N, device=device)
    alpha_u = torch.where(N_tensor == 0, torch.sqrt(torch.tensor(1.0 / N, device=device)), torch.sqrt(torch.tensor(2.0 / N, device=device)))
    alpha_v = torch.where(N_tensor == 0, torch.sqrt(torch.tensor(1.0 / N, device=device)), torch.sqrt(torch.tensor(2.0 / N, device=device)))
    basis = torch.einsum('u,v,uvhw->uvhw', alpha_u, alpha_v, basis)# u,v -> abuv
    
    return basis

def encode(dct_blocks, DCT_basis):
    R_dct_block = torch.einsum('abcd,cdef->abef', dct_blocks[0], DCT_basis)
    G_dct_block = torch.einsum('abcd,cdef->abef', dct_blocks[1], DCT_basis)
    B_dct_block = torch.einsum('abcd,cdef->abef', dct_blocks[2], DCT_basis)
    return torch.stack([R_dct_block, G_dct_block, B_dct_block])


def decode(dct_blocks, IDCT_basis):
    R_idct_block = torch.einsum('abef,cdef->abcd', dct_blocks[0], IDCT_basis)
    G_idct_block = torch.einsum('abef,cdef->abcd', dct_blocks[1], IDCT_basis)
    B_idct_block = torch.einsum('abef,cdef->abcd', dct_blocks[2], IDCT_basis)
    return torch.stack([R_idct_block, G_idct_block, B_idct_block])

def padding(tensor, N):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    b, c, height, width = tensor.shape
    padding_length = 0
    padding_width = 0
    
    if height % N != 0:
        padding_length = N - height % N
    if width % N != 0:
        padding_width  = N - width % N
    padded_data = F.pad(tensor, (0, padding_width, 0, padding_length))
    
    return padded_data, (padding_length, padding_width)

def blockfy(tensor, N):
    padded_data, pad_size = padding(tensor, N)
    b, channel, height, width = padded_data.shape

    num_blocks_height = height // N
    num_blocks_width = width // N

    unfolded = padded_data.unfold(2, N, N).unfold(3, N, N)
    blocks = unfolded.contiguous().view(channel, num_blocks_height, num_blocks_width, N, N)

    return blocks, pad_size

def deblockfy(blocks, pad_size):
    channel, num_blocks_height, num_blocks_width, N, N = blocks.shape
    
    height = num_blocks_height * N
    width = num_blocks_width * N

    blocks_reshaped = blocks.permute(0, 1, 3, 2, 4).reshape(1, channel, height, width)

    tensor = blocks_reshaped[:,:,:height-pad_size[0], :width-pad_size[1]]
    return tensor

def dct_pass_filter(device):
    low_pass_filter = torch.tensor([
        [ 1,  1,  1,  1,  1,  1,  0,  0],
        [ 1,  1,  1,  1,  1,  0,  0,  0],
        [ 1,  1,  1,  1,  1,  0,  0,  0],
        [ 1,  1,  1,  1,  0,  0,  0,  0],
        [ 1,  1,  1,  0,  0,  0,  0,  0],
        [ 1,  1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ], device=device)
    
    high_pass_filter = torch.tensor([
        [ 0,  0,  0,  0,  0,  0,  1,  1],
        [ 0,  0,  0,  0,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  1,  1,  1,  1],
        [ 0,  0,  0,  1,  1,  1,  1,  1],
        [ 0,  0,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1]
    ], device=device)

    Low_pass_filter = low_pass_filter[None, None, None, :, :]
    High_pass_filter = high_pass_filter[None, None, None, :, :]
    return Low_pass_filter, High_pass_filter