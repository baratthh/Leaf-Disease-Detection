import os
import io
import time
import base64
import tempfile
from functools import partial

from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import math
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_dotenv_file(*paths):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            continue


load_dotenv_file(
    os.path.join(PROJECT_ROOT, '.env'),
    os.path.join(os.path.dirname(__file__), '.env')
)

SITE_CONTENT = {
    'home_title': 'Leaf Disease Detection with Higher-Order Fuzzy Sets',
    'home_subtitle': 'A compact overview of the project, the workflow, and the detection entry point.',
    'abstract': (
        'In real field conditions, leaf disease classification is complicated by background clutter, changing illumination, '
        'and disease lesions that may cover only a fraction of the leaf surface. This project compares intuitionistic, '
        'Pythagorean, and Fermatean fuzzy preprocessing across ResNeXt101-32x8d and EfficientNetV2-S on a twelve-class '
        'plant disease task, using HSV value enhancement and a controlled training protocol.'
    ),
    'key_points': [
        'Higher-order fuzzy preprocessing enhances the Value channel before CNN inference.',
        'ResNeXt101-32x8d gives the strongest overall accuracy.',
        'EfficientNetV2-S is the lighter deployment option with strong efficiency.'
    ],
    'portfolio_url': 'https://barathk.vercel.app/',
    'personal_url': os.environ.get('PERSONAL_PROFILE_URL', 'https://github.com/baratthh'),
    'github_url': 'https://github.com/baratthh/Leaf-Disease-Detection',
    'huggingface_url': 'https://huggingface.co/baratthh/Leaf-disease-detection',
    'pdf_name': 'Integrating-Higher-Order-Fuzzy-Sets-with-ResNeXt101-for-Leaf-Disease-Detection.pdf',
    'pdf_asset': 'assets/home/Integrating-Higher-Order-Fuzzy-Sets-with-ResNeXt101-for-Leaf-Disease-Detection.pdf',
    'workflow_svg': 'assets/home/sop_workflow.svg',
    'comparison_image': 'static/assets/home/fuzzy_comparison.png',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        D = int(planes * (base_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, layers, num_classes=1000, cardinality=32, base_width=8):
        super(ResNeXt, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, 32, 8, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes, 32, 8))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnext101_32x8d(num_classes=1000):
    return ResNeXt(Bottleneck, [3,4,23,3], num_classes=num_classes, cardinality=32, base_width=8)

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels, activation=nn.SiLU):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, squeeze_channels, 1),
            activation(inplace=True),
            nn.Conv2d(squeeze_channels, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.squeeze(x)

class StochasticDepth(nn.Module):
    def __init__(self, p, mode="row"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        survival = torch.bernoulli(torch.ones(x.shape[0], 1, 1, 1, device=x.device) * (1 - self.p))
        return x * survival / (1 - self.p)

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, stochastic_depth_prob, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded_channels = int(in_channels * expand_ratio)
        layers = []
        activation_layer = nn.SiLU

        if expanded_channels != in_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
                norm_layer(expanded_channels),
                activation_layer(inplace=True)
            ))
            layers.append(nn.Sequential(
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                norm_layer(out_channels)
            ))
        else:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
                norm_layer(out_channels),
                activation_layer(inplace=True)
            ))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        self.out_channels = out_channels

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
        return result

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, stochastic_depth_prob, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded_channels = int(in_channels * expand_ratio)
        activation_layer = nn.SiLU
        layers = []

        if expanded_channels != in_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                norm_layer(expanded_channels),
                activation_layer(inplace=True)
            ))

        layers.append(nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                      stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            norm_layer(expanded_channels),
            activation_layer(inplace=True)
        ))

        squeeze_channels = max(1, in_channels // 4)
        layers.append(SqueezeExcitation(expanded_channels, squeeze_channels))

        layers.append(nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        ))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        self.out_channels = out_channels

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += x
        return result

class EfficientNetV2(nn.Module):
    def __init__(self, inverted_residual_setting, dropout=0.2, stochastic_depth_prob=0.2, num_classes=1000, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []
        firstconv_output_channels = inverted_residual_setting[0][0]
        layers.append(nn.Sequential(
            nn.Conv2d(3, firstconv_output_channels, 3, stride=2, padding=1, bias=False),
            norm_layer(firstconv_output_channels),
            nn.SiLU(inplace=True)
        ))

        total_stage_blocks = sum(setting[3] for setting in inverted_residual_setting)
        stage_block_id = 0

        for in_channels, out_channels, block_type, num_layers, expand_ratio, kernel_size, stride in inverted_residual_setting:
            stage = []
            for i in range(num_layers):
                curr_stride = stride if i == 0 else 1
                curr_in_channels = in_channels if i == 0 else out_channels
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                if block_type == "fused":
                    block = FusedMBConv(curr_in_channels, out_channels, expand_ratio, kernel_size, curr_stride, sd_prob, norm_layer)
                else:
                    block = MBConv(curr_in_channels, out_channels, expand_ratio, kernel_size, curr_stride, sd_prob, norm_layer)

                stage.append(block)
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1][1]
        lastconv_output_channels = 1280
        layers.append(nn.Sequential(
            nn.Conv2d(lastconv_input_channels, lastconv_output_channels, 1, bias=False),
            norm_layer(lastconv_output_channels),
            nn.SiLU(inplace=True)
        ))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def efficientnet_v2_s(num_classes=1000):
    inverted_residual_setting = [
        [24, 24, "fused", 2, 1, 3, 1],
        [24, 48, "fused", 4, 4, 3, 2],
        [48, 64, "fused", 4, 4, 3, 2],
        [64, 128, "mbconv", 6, 4, 3, 2],
        [128, 160, "mbconv", 9, 6, 3, 1],
        [160, 256, "mbconv", 15, 6, 3, 2],
    ]
    return EfficientNetV2(inverted_residual_setting, dropout=0.2, stochastic_depth_prob=0.2, num_classes=num_classes, norm_layer=partial(nn.BatchNorm2d, eps=1e-03))

def fermatean_fuzzy_batch(images):
    images_np = images.permute(0,2,3,1).cpu().numpy()
    processed = []
    for img in images_np:
        img = img.astype(np.float32)
        img_norm = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
        h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        a = 0.6
        mem = np.clip(v,0.001,0.999)
        tan_term = np.tan(a)*mem
        denom = np.where(tan_term!=-1,1+tan_term,1e-6)
        mem3 = np.power(mem,3)
        non3 = np.power((1-mem)/denom,3)
        hes = np.cbrt(np.clip(1-mem3-non3,0,1))
        v_enh = np.clip(v+hes,0,1)
        v_enh = cv2.normalize(v_enh, None, 0,1.0,cv2.NORM_MINMAX).astype(np.float32)
        enhanced = cv2.cvtColor(cv2.merge([h,s,v_enh]), cv2.COLOR_HSV2RGB)
        enhanced = np.clip(enhanced,0,1).astype(np.float32)
        processed.append(torch.from_numpy(enhanced).permute(2,0,1))
    return torch.stack(processed).to(images.device)

def pythagorean_fuzzy_batch(images):
    images_np = images.permute(0,2,3,1).cpu().numpy()
    processed=[]
    for img in images_np:
        img_f=img.astype(np.float32)
        img_norm=cv2.normalize(img_f,None,0,1.0,cv2.NORM_MINMAX)
        hsv=cv2.cvtColor(img_norm,cv2.COLOR_RGB2HSV)
        h,s,v=cv2.split(hsv)
        a=0.6
        mem=np.clip(v,0.001,0.999)
        tan_term=np.tan(a)*mem
        denom=np.where(tan_term!=-1,1+tan_term,1e-6)
        mem2=np.square(mem)
        non2=np.square((1-mem)/denom)
        hes=np.sqrt(np.clip(1-mem2-non2,0,1))
        v_enh=np.clip(v+hes,0,1)
        v_enh=cv2.normalize(v_enh,None,0,1.0,cv2.NORM_MINMAX).astype(np.float32)
        enhanced=cv2.cvtColor(cv2.merge([h,s,v_enh]),cv2.COLOR_HSV2RGB)
        enhanced=np.clip(enhanced,0,1).astype(np.float32)
        processed.append(torch.from_numpy(enhanced).permute(2,0,1))
    return torch.stack(processed).to(images.device)

def intuitionistic_fuzzy_batch(images):
    images_np = images.permute(0,2,3,1).cpu().numpy()
    processed = []
    for img in images_np:
        img_f = img.astype(np.float32)
        img_norm = cv2.normalize(img_f, None, 0, 1.0, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        a = 0.5
        mem = v.copy()
        tan_term = np.tan(a) * mem
        denom = np.where(tan_term != -1, 1 + tan_term, 1e-6)
        nonmem = (1 - mem) / denom
        hes = np.clip(1 - mem - nonmem, 0, 1)
        v_enhanced = np.clip(mem + hes, 0, 1).astype(np.float32)
        if v_enhanced.shape != h.shape:
            v_enhanced = cv2.resize(v_enhanced, (h.shape[1], h.shape[0]))
        enhanced_hsv = cv2.merge([h, s, v_enhanced])
        enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        enhanced_rgb = np.clip(enhanced_rgb, 0, 1).astype(np.float32)
        processed.append(torch.from_numpy(enhanced_rgb).permute(2, 0, 1))
    return torch.stack(processed).to(images.device)

def tensor_to_preview_b64(tensor):
    image = tensor.detach().clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
    if image.shape[2] == 3:
        image = (image * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def preprocess_for_method(image_tensor, method_name):
    if method_name == 'fermatean':
        enhanced_tensor = fermatean_fuzzy_batch(image_tensor)
        label = 'Fermatean Fuzzy'
    elif method_name == 'pythagorean':
        enhanced_tensor = pythagorean_fuzzy_batch(image_tensor)
        label = 'Pythagorean Fuzzy'
    else:
        enhanced_tensor = intuitionistic_fuzzy_batch(image_tensor)
        label = 'Intuitionistic Fuzzy'

    return enhanced_tensor, label

METHOD_KEYS = ("intuitionistic", "pythagorean", "fermatean")
METHOD_TITLE = {
    "intuitionistic": "Intuitionistic Fuzzy",
    "pythagorean": "Pythagorean Fuzzy",
    "fermatean": "Fermatean Fuzzy",
}
ARCH_TITLE = {
    "resnext": "ResNeXt101",
    "efficientnet": "EfficientNetV2-S",
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ORIGINAL CLASS NAMES (UNCHANGED)
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Background_without_leaves",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"
]

# READABLE MAPPING
def format_class_name(raw_name):
    if "___" in raw_name:
        plant, condition = raw_name.split("___", 1)
        plant = plant.replace("_", " ").title()
        condition = condition.replace("_", " ").replace("(", " (").title()
        if condition.lower() == "healthy":
            return f"Healthy {plant}"
        return f"{plant} - {condition}"
    return raw_name.replace("_", " ").title()

friendly_names = [format_class_name(name) for name in class_names]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hugging Face model source
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "ldd_final_model_cache")
)

# Model configurations
model_configs = {
    "resnext": {
        "models": {
            "fermatean": {"model": None, "filename": "Res_FFS_1.pth"},
            "pythagorean": {"model": None, "filename": "Res_PFS_1.pth"},
            "intuitionistic": {"model": None, "filename": "Res_IFS_1.pth"}
        },
        "architecture": resnext101_32x8d
    },
    "efficientnet": {
        "models": {
            "fermatean": {"model": None, "filename": "Eff_FFS_1.pth"},
            "pythagorean": {"model": None, "filename": "Eff_PFS_1.pth"},
            "intuitionistic": {"model": None, "filename": "Eff_IFS_1.pth"}
        },
        "architecture": efficientnet_v2_s
    }
}

# INFERENCE TRANSFORM
infer_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def load_model_from_hf(arch_key, method_key):
    model_info = model_configs[arch_key]["models"][method_key]
    if model_info["model"] is not None:
        return model_info["model"]

    if not HF_REPO_ID:
        raise RuntimeError(
            "HF_REPO_ID is not set. Configure the Hugging Face model repo before running predictions."
        )

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=model_info["filename"],
        token=HF_TOKEN,
        cache_dir=HF_CACHE_DIR,
    )
    model = model_configs[arch_key]["architecture"](len(class_names)).to(device)
    state = torch.load(local_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model_info["model"] = model
    return model

def get_model_status(architecture=None, method='all'):
    if architecture in model_configs:
        architectures = [architecture]
    else:
        architectures = list(model_configs.keys())

    methods = METHOD_KEYS if method == 'all' else (method,)

    status = {}
    loaded = 0
    total = 0

    for arch_key in architectures:
        status[arch_key] = {}
        for method_key in methods:
            is_loaded = model_configs[arch_key]["models"][method_key]["model"] is not None
            status[arch_key][method_key] = {"loaded": is_loaded}
            total += 1
            if is_loaded:
                loaded += 1

    return {
        "status": status,
        "loaded": loaded,
        "total": total,
    }

torch.set_grad_enabled(False)

# AUTO-GENERATE SAMPLE IMAGES FROM DIRECTORY
def generate_sample_images():
    """Auto-generate sample images from static/sample_images directory"""
    sample_images = []
    sample_dir = 'static/sample_images'
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        return sample_images
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    files = []
    
    for filename in os.listdir(sample_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            files.append(filename)
    
    files.sort()  # Sort alphabetically
    
    # Generate friendly names from filenames
    for filename in files:
        # Remove extension and replace underscores
        name_base = os.path.splitext(filename)[0]
        name_parts = name_base.replace('_', ' ').title()
        
        # To create more readable names
        if 'apple' in name_base.lower():
            if 'scab' in name_base.lower():
                friendly_name = "Apple Scab"
            elif 'black' in name_base.lower() and 'rot' in name_base.lower():
                friendly_name = "Apple Black Rot"
            elif 'cedar' in name_base.lower():
                friendly_name = "Apple Cedar Rust"
            elif 'healthy' in name_base.lower() or 'h' in name_base.lower():
                friendly_name = "Healthy Apple"
            else:
                friendly_name = f"Apple Sample"
        elif 'grape' in name_base.lower():
            if 'black' in name_base.lower() and 'rot' in name_base.lower():
                friendly_name = "Grape Black Rot"
            elif 'esca' in name_base.lower():
                friendly_name = "Grape Esca"
            elif 'blight' in name_base.lower():
                friendly_name = "Grape Leaf Blight"
            elif 'healthy' in name_base.lower():
                friendly_name = "Healthy Grape"
            else:
                friendly_name = f"Grape Sample"
        elif 'potato' in name_base.lower():
            if 'eb' in name_base.lower() or 'early' in name_base.lower():
                friendly_name = "Potato Early Blight"
            elif 'lb' in name_base.lower() or 'late' in name_base.lower():
                friendly_name = "Potato Late Blight"
            elif 'healthy' in name_base.lower() or 'h' in name_base.lower():
                friendly_name = "Healthy Potato"
            else:
                friendly_name = f"Potato Sample"
        else:
            friendly_name = name_parts
        
        sample_images.append({
            "filename": filename,
            "name": friendly_name,
            "description": f"Sample image: {friendly_name}"
        })
    
    return sample_images

def get_research_assets():
    pdf_name = SITE_CONTENT['pdf_name']

    pdf_path = os.path.join(PROJECT_ROOT, 'static', SITE_CONTENT['pdf_asset'])

    return {
        'has_pdf': os.path.exists(pdf_path),
        'pdf_name': pdf_name,
    }


@app.route('/')
def home():
    return render_template('home.html', site_content=SITE_CONTENT, **get_research_assets())

@app.route('/static/sample_images/<filename>')
def sample_image(filename):
    return send_from_directory('static/sample_images', filename)

@app.route('/predict')
def predict_page():
    sample_images = generate_sample_images()
    return render_template('predict.html', sample_images=sample_images, site_content=SITE_CONTENT)


@app.route('/api/model-status', methods=['GET'])
def api_model_status():
    architecture = request.args.get('architecture', '').lower() or None
    method = request.args.get('method', 'all').lower()

    if architecture is not None and architecture not in model_configs:
        return jsonify({"success": False, "error": "Invalid architecture selected"}), 400

    if method != 'all' and method not in METHOD_KEYS:
        return jsonify({"success": False, "error": "Invalid preprocessing method selected"}), 400

    info = get_model_status(architecture=architecture, method=method)
    return jsonify({
        "success": True,
        "mode": "lazy-load",
        **info,
    })

@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    try:
        method = request.form.get('method', 'intuitionistic').lower()
        if method != 'all' and method not in METHOD_KEYS:
            return jsonify({"error": "Invalid preprocessing method selected"}), 400

        sample_image = request.form.get('sample_image')

        if sample_image:
            sample_path = os.path.join('static', 'sample_images', sample_image)
            if not os.path.exists(sample_path):
                return jsonify({"error": "Sample image not found"}), 404
            img = Image.open(sample_path).convert('RGB')
        else:
            file = request.files.get('image')
            if not file:
                return jsonify({"error": "No image provided"}), 400
            img = Image.open(file.stream).convert('RGB')

        original_tensor = infer_tf(img).unsqueeze(0).to(device)
        methods_to_run = METHOD_KEYS if method == 'all' else (method,)
        enhanced_images = {}
        for method_key in methods_to_run:
            enhanced_tensor, method_label = preprocess_for_method(original_tensor, method_key)
            enhanced_images[method_label] = tensor_to_preview_b64(enhanced_tensor)

        return jsonify({
            "success": True,
            "method": "All Fuzzy Methods" if method == 'all' else METHOD_TITLE[method],
            "original_image": tensor_to_preview_b64(original_tensor),
            "enhanced_images": enhanced_images,
        })
    except Exception as e:
        print(f"Preprocess error: {e}")
        return jsonify({"error": "Preprocessing failed. Please try again."}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        start_time = time.time()

        architecture = request.form.get('architecture', 'resnext').lower()
        if architecture not in model_configs:
            return jsonify({"error": "Invalid architecture selected"}), 400

        method = request.form.get('method', 'intuitionistic').lower()
        if method != 'all' and method not in METHOD_KEYS:
            return jsonify({"error": "Invalid preprocessing method selected"}), 400

        sample_image = request.form.get('sample_image')
        if sample_image:
            sample_path = os.path.join('static', 'sample_images', sample_image)
            if not os.path.exists(sample_path):
                return jsonify({"error": "Sample image not found"}), 404
            img = Image.open(sample_path).convert('RGB')
        else:
            file = request.files.get('image')
            if not file:
                return jsonify({"error": "No image provided"}), 400
            img = Image.open(file.stream).convert('RGB')

        img_tensor = infer_tf(img).unsqueeze(0).to(device)

        results = []
        methods_to_run = METHOD_KEYS if method == 'all' else (method,)

        with torch.no_grad():
            for method_key in methods_to_run:
                enhanced_tensor, method_label = preprocess_for_method(img_tensor, method_key)
                model = load_model_from_hf(architecture, method_key)
                output = model(enhanced_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

                results.append({
                    "method": method_label,
                    "prediction": friendly_names[predicted_idx.item()],
                    "confidence": confidence.item(),
                    "confidence_percent": f"{confidence.item() * 100:.1f}%"
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)

        processing_time = time.time() - start_time

        return jsonify({
            "success": True,
            "results": results,
            "architecture": ARCH_TITLE[architecture],
            "method": "All Fuzzy Methods" if method == 'all' else METHOD_TITLE[method],
            "processing_time": f"{processing_time:.2f}s"
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500

if __name__ == '__main__':
    print(" Plant Disease Detection Server Starting...")
    print(f" Access the app at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
