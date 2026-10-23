"""Best-effort Qwen sampling display, using Imagine native RGB projection.

Source: stable-diffusion.cpp/src/runtime/latent-preview.h. No ComfyUI module is
imported here. This RGB display does not modify model tensors, sigma or RNG.
"""
import math
import time
import struct

FACTORS = [[0.00860495522, 0.01219501462, -0.00321337196], [0.0188923309, 0.01246581216, 0.01074959482], [0.1255941446, 0.1176879344, -0.0332212352], [0.0418238528, 0.1043427754, 0.0121666316], [0.02025338, 0.01453670296, -0.000224336044], [-0.01896720702, -0.020609903, -0.0322728584], [0.00438984796, -0.01374969766, 0.02849196], [-0.0374495856, -0.0286777126, -0.069319281], [0.01511914734, 0.0242979386, 0.055387887], [-0.1138629518, -0.020391466, 0.001550520522], [-0.0233650696, -0.0417292018, -0.0362361182], [-0.0351603342, -0.0243595924, -0.00216261038], [0.01093355288, -0.0373466924, 0.0024131535], [0.01778704744, -0.00401984678, -0.0343259192], [0.0486059334, 0.0253144, 0.0672564966], [0.0309463558, 0.0277963166, 0.0520869622], [0.0374485008, 0.0551753676, 0.0225853902], [-0.0090809962, -0.004756176, 0.00636443612], [-0.0270455652, -0.0384966954, -0.00905908082], [-0.00553493756, 0.01484553684, -0.0211502468], [0.01319502562, 0.00948005666, 0.0483789212], [-0.00931847104, -0.00276452734, -0.01011985302], [0.0180478258, 0.01614954356, -0.020942469], [-0.0214530434, -0.00272961176, 0.0217887476], [-0.0636772304, -0.0208893548, 0.0479167742], [-0.0250321236, -0.0286715676, 0.0530110146], [-0.01853078078, 0.01647272818, -0.00207747588], [0.0023101082, 0.01228800748, 0.01303505006], [-0.01243671408, -0.0258638728, -0.0379116264], [0.0059893471, 0.0064256355, -0.01234514304], [-0.0296733996, -0.023469805, 0.00060018212], [-0.0322019498, -0.0529200462, -0.00344987414], [-0.00205026458, -0.00846599446, 0.00455971038], [-0.01082227064, 0.0315661948, -0.0677753362], [0.0645553474, 0.1109666998, 0.0674744864], [0.01036801108, -0.0048484121, -0.001529168474], [0.01264353566, 0.01548126338, -0.00966374324], [-0.0223892408, -0.00871751526, -0.00030642167], [0.0271322742, 0.03496524, -0.0089692858], [0.0512178672, 0.0173080034, 0.00804227746], [0.01210987192, 0.00758025926, -0.00281712586], [0.189727839, 0.1210261828, 0.062603892], [0.0208058822, 0.00547548182, 0.01262955638], [0.00813332858, 0.01015930914, 0.0130177129], [-0.000927236014, -0.00152540594, -0.00599213302], [0.01663314616, -0.00582789626, 0.0163958132], [-0.0252546342, -0.0604193732, -0.1606919922], [-0.091722686, -0.0409201224, -0.0959576198], [0.0282963112, -0.01387223872, -0.01648814464], [0.0552316818, 0.0967547788, 0.0413586632], [0.00922849292, 0.00451467542, -0.0529172378], [0.0558600768, 0.0122988308, -0.01445942422], [0.000210660902, -0.01295958782, -0.01804761764], [0.035813625, -0.047250597, -0.1156405142], [-0.0506390696, -0.0471914842, 0.0349791468], [-0.0480143168, 0.00628389868, -0.0545163826], [0.0315499582, 0.0564846606, -0.0430850488], [-0.0362330316, -0.01267788554, 0.0061024772], [0.0038627542, 0.00911055916, -0.00758526008], [-0.0447103298, -0.00835411408, 0.01545872328], [-0.015006738, 0.00270612302, -0.00784361356], [-0.0221755048, -0.0513344748, -0.0475317424], [-0.01036656294, -0.00422146068, -0.0213499052], [0.01788952706, 0.0119194419, 0.0397205238]]
BIAS = [-0.043293118, -0.02695978, -0.11986706]


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _triangle_weights(source, target):
    ratio = _f32(source / target)
    support = max(ratio, 1.)
    rows = []
    for position in range(target):
        center = _f32(_f32(_f32(position) + .5) * ratio)
        left = min(source - 1, max(0, math.floor(_f32(center - support))))
        right = min(source, max(left + 1, math.ceil(_f32(center + support))))
        center = _f32(center - .5)
        row = []
        total = 0.
        for index in range(left, right):
            distance = abs(_f32(_f32(_f32(index) - center) / support))
            value = _f32(1. - distance) if distance < 1. else 0.
            row.append((index, value))
            total = _f32(total + value)
        rows.append([(index, _f32(value / total)) for index, value in row])
    return rows


def _resize_axis(torch, image, axis, target):
    weights = _triangle_weights(image.shape[axis], target)
    shape = list(image.shape)
    shape[axis] = target
    output = torch.zeros(shape, dtype=torch.float32, device=image.device)
    weight_shape = [1, 1, 1]
    weight_shape[axis] = target
    for tap in range(max(map(len, weights))):
        entries = [row[tap] if tap < len(row) else (0, 0.) for row in weights]
        indices = torch.tensor([index for index, _ in entries], dtype=torch.int64, device=image.device)
        values = torch.tensor([value for _, value in entries], dtype=torch.float32, device=image.device).reshape(weight_shape)
        output = output + image.index_select(axis, indices) * values
    return output


def render(torch, denoised, maximum):
    x = denoised.detach().to(dtype=torch.float32)
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 64:
        raise ValueError("Unsupported preview tensor layout")
    x = x[0]
    rgb = torch.zeros((3, x.shape[1], x.shape[2]), dtype=torch.float32, device=x.device)
    for channel, weights in enumerate(FACTORS):
        for component, weight in enumerate(weights):
            rgb[component].add_(x[channel] * weight)
    for component, bias in enumerate(BIAS):
        rgb[component].add_(bias)
    rgb = ((rgb * .5 + .5).clamp(0., 1.) * 255.).to(dtype=torch.uint8)
    if not 1 <= maximum <= 768:
        raise ValueError("Invalid preview size")
    rgb = rgb.permute(1, 2, 0)
    width, height = rgb.shape[1], rgb.shape[0]
    if max(width, height) > maximum:
        ratio = maximum / max(width, height)
        # Positive f64 round, matching Rust round rather than Python ties-even.
        target_w, target_h = max(1, math.floor(width * ratio + .5)), max(1, math.floor(height * ratio + .5))
        rgb = _resize_axis(torch, rgb.to(dtype=torch.float32), 0, target_h)
        rgb = _resize_axis(torch, rgb, 1, target_w)
        rgb = (rgb.clamp(0., 255.) + .5).floor().to(dtype=torch.uint8)
    pixels = rgb.contiguous().cpu()
    return pixels.shape[1], pixels.shape[0], pixels.numpy().tobytes()


class State:
    def __init__(self, settings, clock=time.monotonic):
        self.clock, self.start = clock, clock()
        self.last, self.sequence = 0., 0
        self.settings = None
        if isinstance(settings, dict) and settings.get("enabled") is True:
            interval = settings.get("interval_seconds")
            edge = settings.get("max_resolution", 512)
            if (type(interval) in (float, int) and math.isfinite(interval) and interval > 0
                    and type(edge) is int):
                self.settings = (interval, min(768, max(1, edge)))

    def update(self, step, denoised, emit, renderer):
        if self.settings is None:
            return
        elapsed = self.clock() - self.start
        interval, edge = self.settings
        if elapsed - self.last < interval:
            return
        self.last = elapsed
        try:
            width, height, pixels = renderer(denoised, edge)
            if not (0 < width <= edge and 0 < height <= edge and len(pixels) == width * height * 3):
                raise ValueError("Invalid preview frame")
        except Exception:
            emit("preview_warning")
            return
        self.sequence += 1
        emit("preview", payload=pixels, metadata={"sequence": self.sequence, "step": step,
             "width": width, "height": height, "frames": 1,
             "frame_duration_ms": 1000, "source_duration_seconds": 1.})
