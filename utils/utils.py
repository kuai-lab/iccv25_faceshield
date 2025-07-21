from PIL import Image, ImageOps
import torch
import os, glob, math, dlib, cv2, random
from torchvision import transforms
import torch.nn.functional as F
import os.path as osp
from insightface.model_zoo import model_zoo
from imutils import face_utils
from scipy.spatial import ConvexHull
import numpy as np

def print_vram_usage(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def compute_vae_encodings(images, vae, device, gt=False):
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(device, dtype=vae.dtype)
    if gt:
        with torch.no_grad():
            model_input = vae.encode(pixel_values).latent_dist.sample()
    else:
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input

def resize_and_pad_image(image_path, size=512):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    if h < w:
        new_height = size
        new_width = int((size / h) * w)
    else:
        new_width = size
        new_height = int((size / w) * h)
            
    image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    padding_width = size - new_width
    padding_height = size - new_height
    image_padded = ImageOps.expand(image_resized, (padding_width // 2, padding_height // 2, padding_width - padding_width // 2, padding_height - padding_height // 2), fill=(0, 0, 0))
    return image_padded

def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def save_png(save_name, image):
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    image = image.permute(1, 2, 0)
    image = (image* 255).byte().cpu().numpy()
    to_pil_image = transforms.ToPILImage()
    save_image = to_pil_image(image)
    save_image.save(save_name+".png")
    
def dynamic_grad(grad, max=2.0, min=0.1):
    abs_grad = torch.abs(grad)
    min_val = torch.min(abs_grad[abs_grad > 0])
    max_val = torch.max(abs_grad)
    scaled_grad = (max-min) * (abs_grad - min_val) / (max_val - min_val) + min
    return scaled_grad

def input_diversity(input_tensor, prob=0.5, image_size=512):
    if random.uniform(0, 1) > prob:
        return input_tensor  # Return original input with probability (1 - prob)

    # Compute padding dimensions
    pad_top = random.randint(0, image_size // 10)  # Padding up to 10% of the size
    pad_bottom = random.randint(0, image_size // 10)
    pad_left = random.randint(0, image_size // 10)
    pad_right = random.randint(0, image_size // 10)

    # Apply random padding
    padded = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Crop back to the original size to ensure consistency
    cropped = padded[:, :, pad_top:pad_top+image_size, pad_left:pad_left+image_size]
    
    return cropped

# ============ Face Detection & Mask ============ #
def face_detection_mask(src_img, save_dir, max_size, device):
    transform = transforms.Compose([transforms.ToTensor()])

    # detector = dlib.get_frontal_face_detector()
    img_a_whole = cv2.imread(src_img)
    img_a_whole = cv2.cvtColor(img_a_whole, cv2.COLOR_BGR2RGB)
    img_whole_tensor = transform(img_a_whole).to(device)
    
    _, h, w = img_whole_tensor.shape
    coord = (0, h, 0, w)
    img_crop_tensor = img_whole_tensor.clone()
    
    # resize face & whole
    whole_image_r, face_image_r, (y1, y2, x1, x2) = resize_face3(img_whole_tensor, img_crop_tensor, coord, max_size)
    face_image = face_image_r.unsqueeze(0)
    back_image_r = whole_image_r.clone()
    back_image_r[:,y1:y2,x1:x2] = 1

    # resize = face_image_r.shape[1:]
    # face_image_np = face_image_r.permute(1,2,0).cpu().numpy()
    # face_image_g = cv2.cvtColor(face_image_np, cv2.COLOR_RGB2GRAY)
    # face_image_g = (face_image_g / np.max(face_image_g) * 255).astype('uint8')
    # face_bbox = detector(face_image_g, 2)[0]
    
    # # mask
    # mask_np, land_mask_np = lmk_mask(face_image_g, face_bbox, max_size, resize, landmark_path)
    # mask_image_r = torch.from_numpy(mask_np.astype(np.uint8))
    # land_mask = torch.from_numpy(land_mask_np.astype(np.uint8))
    
    save_png(f"{save_dir}/source", face_image_r)
    # save_png(f"{save_dir}/mask(cropped)", mask_image_r.unsqueeze(0))
    # save_png(f"{save_dir}/back(cropped)", back_image_r)

    return face_image, _, _, back_image_r, (y1, y2, x1, x2)

def lmk_mask(face_image, face_bbox, base_res, resize, landmark_path):
    
    # landmark predictor
    landmark_predictor = dlib.shape_predictor(landmark_path)
    
    landmark = landmark_predictor(face_image, face_bbox)
    landmark_np = face_utils.shape_to_np(landmark)
    landmark_tensor = torch.tensor(landmark_np) / base_res
    
    mask = np.zeros(resize)
    
    def extract_convex_hull(landmark, mask, base_res):
        landmark = landmark * base_res
        hull = ConvexHull(landmark)
        points = [landmark[hull.vertices, :1], landmark[hull.vertices, 1:]]
        points = np.concatenate(points, axis=-1).astype('int32')
        mask = cv2.fillPoly(mask, pts=[points], color=(255,255,255))
        mask = mask > 0
        return mask
    
    mask = extract_convex_hull(landmark_tensor, mask, base_res)

    # extract l_eye r_eye nose mouth
    left_eye_indices = list(range(36, 42))  # Indices for left eye
    right_eye_indices = list(range(42, 48))  # Indices for right eye
    nose_indices = list(range(27, 36))  # Indices for nose
    mouth_indices = list(range(48, 68))  # Indices for mouth
    
    land_mask = np.zeros(resize)
    land_mask1 = extract_convex_hull(landmark_tensor[left_eye_indices], land_mask, base_res)
    land_mask2 = extract_convex_hull(landmark_tensor[right_eye_indices], land_mask, base_res)
    land_mask3 = extract_convex_hull(landmark_tensor[nose_indices], land_mask, base_res)
    land_mask4 = extract_convex_hull(landmark_tensor[mouth_indices], land_mask, base_res)
    land_mask = land_mask1 + land_mask2 + land_mask3 + land_mask4
    
    return mask, land_mask

def resize_face3(whole_image, face_image, coord, max_size):
    _, oh, ow = whole_image.shape
    _, h, w = face_image.shape
    if h > max_size and w > max_size:
        if h > w:
            new_height = max_size
            ratio = (max_size / h)
            new_width = int(ratio * w)
        else:
            new_width = max_size
            ratio = (max_size / w)
            new_height = int(ratio * h)
        def resize_tensor(tensor, new_height, new_width):
            if tensor.dim() == 2:
                tensor = tensor[None,None,:]
                resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
                return resized_tensor.squeeze()
                
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
                resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
                return resized_tensor.squeeze()
        
        whole_image = resize_tensor(whole_image, round(ratio*oh), round(ratio*ow))
        face_image = resize_tensor(face_image, round(ratio*h), round(ratio*w))

        # coord y1 y2 x1 x2
        new_y1 = int(coord[0] * ratio)
        new_y2 = new_y1 + round(ratio*h)
        new_x1 = int(coord[2] * ratio)
        new_x2 = new_x1 + round(ratio*w)
        coord = (new_y1, new_y2, new_x1, new_x2)
    
    return  whole_image, face_image, coord

# ===================================================== #

def crop_miniumum_rectangle(src_img, mask, save_dir, max_size, device):    
    transform = transforms.Compose([transforms.ToTensor()])

    image = Image.open(src_img).convert("RGB")
    mask = Image.open(mask)
        
    image = transform(image).to(device)
    mask = transform(mask)[0].to(device)
    
    coords = torch.nonzero(mask, as_tuple=False)

    assert coords.numel() > 0, "Error: The tensor 'coords' must have at least one element."
    
    ## calculate Top-Left (y1,x1) / Bottom-Right (y2,x2) corners
    y1x1 = torch.min(coords, dim=0).values
    y2x2 = torch.max(coords, dim=0).values
    y1, x1 = y1x1[0].item(), y1x1[1].item()
    y2, x2 = y2x2[0].item(), y2x2[1].item()

    face_h, face_w = (y2-y1), (x2-x1)
    if max_size is not None:
        if face_h > max_size or face_w > max_size:
            image, mask, (y1, x1), (y2, x2) = resize_face(image, mask, face_h, face_w, max_size)

    cropped_image_tensor = image[:, y1:y2, x1:x2]
    cropped_mask_tensor = mask[y1:y2, x1:x2]
    back_tensor = image.clone()
    back_tensor[:,y1:y2,x1:x2] = 1
        
    save_png(f"{save_dir}/source(cropped)", cropped_image_tensor)
    save_png(f"{save_dir}/mask(cropped)", cropped_mask_tensor.unsqueeze(0).repeat(3,1,1))
    save_png(f"{save_dir}/back(cropped)", back_tensor)

    c, h, w = cropped_image_tensor.shape
    unet_mask = F.interpolate(cropped_mask_tensor[None,None,:], size=(math.ceil(h/64), math.ceil(w/64)), mode='bilinear', align_corners=False)
    unet_mask[unet_mask!=0] = 1
    unet_mask = unet_mask.repeat(1,1280,1,1)
    
    # cropped_mask=cropped_mask_tensor[None,None,:].repeat(1,3,1,1)
    
    return cropped_image_tensor.unsqueeze(0), cropped_mask_tensor, unet_mask, back_tensor, (y1, y2, x1, x2)

def resize_face2(whole_image, face_image, max_size):
    
    
    _, oh, ow = whole_image.shape
    _, h, w = face_image.shape
    
    if h > w:
        new_height = max_size
        ratio = (max_size / h)
        new_width = int(ratio * w)
    else:
        new_width = max_size
        ratio = (max_size / w)
        new_height = int(ratio * h)
    
    def resize_tensor(tensor, new_height, new_width):
        if tensor.dim() == 2:
            tensor = tensor[None,None,:]
            resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
            return resized_tensor.squeeze()
            
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
            return resized_tensor.squeeze()
        
    
    whole_image_r, face_image_r = map(lambda tensor: resize_tensor(tensor, int(ratio*oh), int(ratio*ow)), [whole_image, face_image])
    return  whole_image_r, face_image_r
    

def resize_face(image, mask, h, w, max_size):
    print(f"Resize the face image with a maximum size of {max_size}")
    _, oh, ow = image.shape
    if h > w:
        new_height = max_size
        ratio = (max_size / h)
        new_width = int(ratio * w)
    else:
        new_width = max_size
        ratio = (max_size / w)
        new_height = int(ratio * h)
    print(f"Your original face size(h,w): {(h, w)}, Resized face size(h,w): {(new_height, new_width)}, Resize ratio: {ratio:.2f}")

    def resize_tensor(tensor, new_height, new_width):
        if tensor.dim() == 2:
            tensor = tensor[None,None,:]
            resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
            return resized_tensor.squeeze()
            
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
            return resized_tensor.squeeze()
        
    
    image, mask = map(lambda tensor: resize_tensor(tensor, int(ratio*oh), int(ratio*ow)), [image, mask])
    mask[mask!=1] = 0
    
    coords = torch.nonzero(mask, as_tuple=False)
    assert coords.numel() > 0, "Error: The tensor 'coords' must have at least one element."
    y1x1 = torch.min(coords, dim=0).values
    y2x2 = torch.max(coords, dim=0).values
    y1, x1 = y1x1[0].item(), y1x1[1].item()
    # y2, x2 = y2x2[0].item(), y2x2[1].item()
    
    return image, mask, (y1, x1), (y1+new_height, x1+new_width)
    
    
def merge_image(face, back, save_dir, coor):    
    (y1, y2, x1, x2) = coor
    
    back[:,y1:y2,x1:x2] = face.squeeze(0)
    save_png(f"{save_dir}/adv(merged)", back)
    return back

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    
def save_arguments_to_log_file(args, log_file_path="arguments_log.txt"):
    with open(log_file_path, "w") as log_file:
        log_file.write("************** Received arguments **************\n")
        for key, value in vars(args).items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")

# ======== Gaussian Blur Utils ======== #
def scale_tensor(tensor):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    elif len(tensor.shape) == 2:
        tensor = tensor[None,None,:,:]

    max_val = tensor.max()
    min_val = tensor.min()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def create_line_mask(save_path, image):
    grad_x, grad_y = calculate_gradients(image)
    grad_x_abs = torch.abs(grad_x)
    grad_y_abs = torch.abs(grad_y)
    mask = torch.where((grad_x_abs == grad_x_abs.max()) | (grad_y_abs == grad_y_abs.max()), 1.0, 0.0)
    
    kernel = torch.ones((3, 1, 9, 9), device=mask.device)
    dilated_mask = F.conv2d(mask, kernel, padding=kernel.shape[-1]//2, groups=3)
    dilated_mask = torch.clamp(dilated_mask, 0, 1)

    return dilated_mask

def calculate_gradients(image, device='cuda'):
    sobel_x = torch.tensor([
        [1, 0, -1], 
        [2, 0, -2], 
        [1, 0, -1]], dtype=torch.float32)[None,None,:,:].repeat(3,1,1,1).to(device)
    sobel_y = torch.tensor([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]], dtype=torch.float32)[None,None,:,:].repeat(3,1,1,1).to(device)
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=3)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=3)
    return grad_x, grad_y

def apply_gaussian(save_path, image, mask, kernel_size=5, sigma=1, device='cuda'):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    elif len(image.shape) == 2:
        image = image[None,None,:,:]
        
    kernel_2d = gaussian_kernel(kernel_size, sigma).to(device)
    blurred_image = F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=3)
    adjusted_image = blurred_image * mask + image * (1 - mask)
    
    return adjusted_image

def gaussian_kernel(size, sigma):
    x = torch.arange(size).float() - (size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)[None,None,:,:].repeat(3,1,1,1)

# ======== Loss Function ======== #

def get_loss_function(loss_fn_name):
    if loss_fn_name is False:
        return False
    if loss_fn_name == "kl":
        return CustomKLDivergenceLoss()
    elif loss_fn_name == "was":
        return WassersteinLoss()
    elif loss_fn_name == "AdaIN":
        return AdaINLoss()
    elif loss_fn_name == "l1":
        return F.l1_loss
    elif loss_fn_name == "l2":
        return F.mse_loss
    elif loss_fn_name == "AdaIN_mean":
        return AdaINLoss_mean()
    elif loss_fn_name == "AdaIN_std":
        return AdaINLoss_std()
    elif loss_fn_name == "cosine":
        return CosineLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")

class CustomKLDivergenceLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean'):
        super(CustomKLDivergenceLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = 1e-10
        
    def forward(self, input, target):
        """
        [batch heads res*res seq_len]
        """
        input = F.log_softmax(input, dim=-1)
        target = F.softmax(target, dim=-1)
        loss = F.kl_div(input, target.detach(), reduction=self.reduction)
        return loss
    
class WassersteinLoss(torch.nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        
        loss = torch.abs(input - target).sum()
        
        return loss
    
class AdaINLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaINLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        
        mean_loss = F.mse_loss(input_mean, target_mean)
        std_loss = F.mse_loss(input_std, target_std)
        
        loss = mean_loss + std_loss
        return loss

    def calc_mean_std(self, features):
        mean = features.mean(dim=[-1], keepdim=True)
        std = features.std(dim=[-1], keepdim=True) + self.eps
        
        return mean, std
    
class AdaINLoss_mean(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaINLoss_mean, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        input_mean = self.calc_mean(input)
        target_mean = self.calc_mean(target)
        
        mean_loss = F.mse_loss(input_mean, target_mean)        
        return mean_loss

    def calc_mean(self, features):
        mean = features.mean(dim=[-1], keepdim=True)
        return mean
    
class AdaINLoss_std(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaINLoss_std, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        input_std = self.calc_std(input)
        target_std = self.calc_std(target)
        std_loss = F.mse_loss(input_std, target_std)
        
        return std_loss

    def calc_std(self, features):
        std = features.std(dim=[-1], keepdim=True) + self.eps
        
        return std
    
class CosineLoss(torch.nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, input, target):
        cos_loss = F.cosine_similarity(input, target).mean()
        
        return 1- cos_loss

class AttentionStore():
    def __init__(self):
        self.attn_map = list()
        
    def __call__(self, value):
        self.attn_map.append(value)
        
class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                            threshold=self.det_thresh,
                                            max_num=max_num,
                                            metric='default')

        x1, y1, x2, y2 = int(bboxes[0][0]), int(bboxes[0][1]), int(bboxes[0][2]), int(bboxes[0][3])
        
        width, height = bboxes[0][2] - bboxes[0][0], bboxes[0][3] - bboxes[0][1]

        pad = 15
        cropped_image_tensor = img[y1-pad:y2+pad,x1-pad:x2+pad,  :]
        
        return cropped_image_tensor, (y1-pad, y2+pad, x1-pad, x2+pad)