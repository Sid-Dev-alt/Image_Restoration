import os
import sys
import uuid
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using inference device:", device)

# Lazy loaded models
nafnet_gopro = None
restormer_motion = None
restormer_defocus = None
mprnet_deblur = None

def free_unused_models(keep_model_name):
    global nafnet_gopro, restormer_motion, restormer_defocus, mprnet_deblur
    import gc
    
    if keep_model_name != "nafnet_gopro" and nafnet_gopro is not None:
        print("Freeing NAFNet-GoPro from memory...")
        nafnet_gopro = None
    if keep_model_name != "restormer_motion" and restormer_motion is not None:
        print("Freeing Restormer-Motion from memory...")
        restormer_motion = None
    if keep_model_name != "restormer_defocus" and restormer_defocus is not None:
        print("Freeing Restormer-Defocus from memory...")
        restormer_defocus = None
    if keep_model_name != "mprnet_deblur" and mprnet_deblur is not None:
        print("Freeing MPRNet-Deblur from memory...")
        mprnet_deblur = None
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_nafnet_gopro():
    global nafnet_gopro
    if nafnet_gopro is None:
        free_unused_models("nafnet_gopro")
        print("Loading NAFNet-GoPro...")
        nafnet_path = os.path.abspath("models/nafnet")
        sys.path.insert(0, nafnet_path)
        from basicsr.models.archs.NAFNet_arch import NAFNet

        nafnet_gopro = NAFNet(
            img_channel=3,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 28],
            dec_blk_nums=[1, 1, 1, 1]
        )
        ckpt = torch.load("checkpoints/nafnet_gopro.pth", map_location="cpu")
        state_dict = ckpt.get("params", ckpt)
        nafnet_gopro.load_state_dict(state_dict, strict=True)
        del ckpt
        del state_dict
        import gc; gc.collect()
        nafnet_gopro.eval()
        nafnet_gopro.to(device)
        print("NAFNet-GoPro loaded successfully.")

        # Clean up
        sys.path.remove(nafnet_path)
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("basicsr"):
                del sys.modules[mod_name]
    return nafnet_gopro

def get_restormer_motion():
    global restormer_motion
    if restormer_motion is None:
        free_unused_models("restormer_motion")
        print("Loading Restormer-Motion...")
        restormer_path = os.path.abspath("models/restormer")
        sys.path.insert(0, restormer_path)
        from basicsr.models.archs.restormer_arch import Restormer
        restormer_motion = load_restormer("checkpoints/restormer_motion.pth")
        sys.path.remove(restormer_path)
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("basicsr"):
                del sys.modules[mod_name]
    return restormer_motion

def get_restormer_defocus():
    global restormer_defocus
    if restormer_defocus is None:
        free_unused_models("restormer_defocus")
        print("Loading Restormer-Defocus...")
        restormer_path = os.path.abspath("models/restormer")
        sys.path.insert(0, restormer_path)
        from basicsr.models.archs.restormer_arch import Restormer
        restormer_defocus = load_restormer("checkpoints/restormer_defocus.pth")
        sys.path.remove(restormer_path)
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("basicsr"):
                del sys.modules[mod_name]
    return restormer_defocus

def get_mprnet_deblur():
    global mprnet_deblur
    if mprnet_deblur is None:
        free_unused_models("mprnet_deblur")
        print("Loading MPRNet-Deblur...")
        mprnet_path = os.path.abspath("models/mprnet/Deblurring")
        sys.path.insert(0, mprnet_path)
        from MPRNet import MPRNet as MPRNetModel
        mprnet_deblur = load_mprnet("checkpoints/mprnet_deblurring.pth")
        sys.path.remove(mprnet_path)
    return mprnet_deblur

def load_restormer(checkpoint_path):
    restormer_path = os.path.abspath("models/restormer")
    if restormer_path not in sys.path:
        sys.path.insert(0, restormer_path)
    from basicsr.models.archs.restormer_arch import Restormer
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("params", ckpt)
    model.load_state_dict(state_dict, strict=True)
    del ckpt
    del state_dict
    import gc; gc.collect()
    model.eval()
    return model.to(device)

def load_mprnet(checkpoint_path):
    mprnet_path = os.path.abspath("models/mprnet/Deblurring")
    if mprnet_path not in sys.path:
        sys.path.insert(0, mprnet_path)
    from MPRNet import MPRNet as MPRNetModel
    model = MPRNetModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    del ckpt
    del state_dict
    import gc; gc.collect()
    model.eval()
    return model.to(device)

# Directories setup
UPLOAD_DIR = os.path.abspath("input/uploads")
RESTORED_DIR = os.path.abspath("output/restored")
ZIP_DIR = os.path.abspath("output/zips")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

# In-memory progress tracking
sessions = {}

app = FastAPI(title="RestorAI Pipeline Server")

# Helper utility to check image files
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

def classify_degradation(img_bgr, filename=""):
    # 1. Filename keyword check as primary
    fn = filename.lower()
    if "defocus" in fn or "focus" in fn:
        return "restormer_defocus"
    if "motion" in fn or "handheld" in fn or "shake" in fn:
        return "nafnet_gopro"
    if "mixed" in fn or "severe" in fn:
        return "mprnet_deblur"

    # 2. Image analysis fallback
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        threshold = np.max(mag) * 0.1
        edge_angles = angle[mag > threshold]
        
        if len(edge_angles) < 100:
            return "restormer_motion"
            
        edge_angles = edge_angles % 180
        hist, _ = np.histogram(edge_angles, bins=18, range=(0, 180))
        hist = hist / np.sum(hist)
        max_peak = np.max(hist)
        
        # Heuristic routing based on blur severity & gradient concentration
        if lap_var > 300.0:
            return "restormer_motion"
        elif max_peak > 0.11:
            return "nafnet_gopro"
        else:
            return "restormer_defocus"
    except Exception as e:
        print(f"Error during degradation classification: {str(e)}")
        return "nafnet_gopro"  # safe default fallback

def preprocess_for_cpu_limit(img_rgb, max_dim=512):
    h_orig, w_orig = img_rgb.shape[:2]
    is_cpu = (device.type == "cpu")
    if is_cpu and max(h_orig, w_orig) > max_dim:
        scale = float(max_dim) / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * scale), int(w_orig * scale)
        h_new = (h_new // 16) * 16
        w_new = (w_new // 16) * 16
        if h_new == 0: h_new = 16
        if w_new == 0: w_new = 16
        img_in = cv2.resize(img_rgb, (w_new, h_new), interpolation=cv2.INTER_AREA)
        return img_in, (h_orig, w_orig), True
    return img_rgb, (h_orig, w_orig), False

def postprocess_for_cpu_limit(img_rgb, orig_size, was_scaled):
    if was_scaled:
        h_orig, w_orig = orig_size
        return cv2.resize(img_rgb, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    return img_rgb

# Inference wrappers
def run_nafnet_inference(img_rgb):
    img_in, orig_size, was_scaled = preprocess_for_cpu_limit(img_rgb)
    img_norm = img_in.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    model = get_nafnet_gopro()
    with torch.no_grad():
        output_tensor = model(img_tensor)
    output_np = output_tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = (output_np * 255.0).round().astype(np.uint8)
    return postprocess_for_cpu_limit(out_img, orig_size, was_scaled)

def run_restormer_inference(model_or_getter, img_rgb, window_size=8):
    img_in, orig_size, was_scaled = preprocess_for_cpu_limit(img_rgb)
    img_norm = img_in.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, h, w = img_tensor.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    img_padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    model = model_or_getter() if callable(model_or_getter) else model_or_getter

    with torch.no_grad():
        output = model(img_padded)

    output = output[:, :, :h, :w]  # crop back
    output_np = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = (output_np * 255.0).round().astype(np.uint8)
    return postprocess_for_cpu_limit(out_img, orig_size, was_scaled)

def run_mprnet_inference(img_rgb, factor=8):
    img_in, orig_size, was_scaled = preprocess_for_cpu_limit(img_rgb)
    img_norm = img_in.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    h, w = img_tensor.shape[2], img_tensor.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    img_padded = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')

    model = get_mprnet_deblur()

    with torch.no_grad():
        restored = model(img_padded)

    output = torch.clamp(restored[0], 0, 1)
    output = output[:, :, :h, :w]  # crop back

    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = (output_np * 255.0).round().astype(np.uint8)
    return postprocess_for_cpu_limit(out_img, orig_size, was_scaled)

# Async Background Job
def process_images_task(session_id: str, model_name: str):
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    session_restore_dir = os.path.join(RESTORED_DIR, session_id)
    os.makedirs(session_restore_dir, exist_ok=True)

    files = [f for f in os.listdir(session_upload_dir) if is_image_file(f)]
    sessions[session_id]["total"] = len(files)

    try:
        for filename in files:
            input_path = os.path.join(session_upload_dir, filename)
            output_path = os.path.join(session_restore_dir, filename)

            img_bgr = cv2.imread(input_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Determine model to use
            img_model = model_name
            if model_name == "auto":
                img_model = classify_degradation(img_bgr, filename)
                print(f"Auto-selected model '{img_model}' for image '{filename}'")

            if img_model == "nafnet_gopro":
                restored_rgb = run_nafnet_inference(img_rgb)
            elif img_model == "restormer_motion":
                restored_rgb = run_restormer_inference(get_restormer_motion, img_rgb)
            elif img_model == "restormer_defocus":
                restored_rgb = run_restormer_inference(get_restormer_defocus, img_rgb)
            elif img_model == "mprnet_deblur":
                restored_rgb = run_mprnet_inference(img_rgb)
            else:
                raise ValueError(f"Unknown model architecture: {img_model}")

            restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, restored_bgr)

            # Update progress status
            sessions[session_id]["processed"] += 1
            sessions[session_id]["images"].append({
                "filename": filename,
                "model_used": img_model
            })

        sessions[session_id]["status"] = "done"
    except Exception as e:
        print(f"Error processing session {session_id}: {str(e)}")
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)

# Model for process trigger request
class ProcessRequest(BaseModel):
    session_id: str
    model: str

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise HTTPException(status_code=404, detail="static/index.html not found.")

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    session_id = uuid.uuid4().hex
    session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_upload_dir, exist_ok=True)

    saved_files = []
    for file in files:
        if is_image_file(file.filename):
            base_filename = os.path.basename(file.filename)
            file_path = os.path.join(session_upload_dir, base_filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(base_filename)

    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid images uploaded.")

    sessions[session_id] = {
        "status": "idle",
        "processed": 0,
        "total": len(saved_files),
        "images": [],
        "error": None
    }

    return {"session_id": session_id, "files": saved_files}

@app.post("/api/process")
async def process_images(req: ProcessRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id
    model_name = req.model

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    sessions[session_id]["status"] = "processing"
    sessions[session_id]["processed"] = 0
    sessions[session_id]["images"] = []

    background_tasks.add_task(process_images_task, session_id, model_name)
    return {"status": "started"}

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "processed_images": session["processed"],
        "total_images": session["total"],
        "images": session["images"],
        "error": session["error"]
    }

@app.get("/api/image/original/{session_id}/{filename}")
async def get_original_image(session_id: str, filename: str):
    file_path = os.path.join(UPLOAD_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Original image not found.")

@app.get("/api/image/restored/{session_id}/{filename}")
async def get_restored_image(session_id: str, filename: str):
    file_path = os.path.join(RESTORED_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Restored image not found.")

@app.get("/api/download/{session_id}")
async def download_restored(session_id: str):
    session_restore_dir = os.path.join(RESTORED_DIR, session_id)
    if not os.path.exists(session_restore_dir):
        raise HTTPException(status_code=404, detail="Restored folder not found.")

    zip_output_base = os.path.join(ZIP_DIR, f"restored_{session_id}")
    zip_path = shutil.make_archive(zip_output_base, "zip", session_restore_dir)

    return FileResponse(
        zip_path,
        media_type="application/x-zip-compressed",
        filename="restored_images.zip"
    )

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
