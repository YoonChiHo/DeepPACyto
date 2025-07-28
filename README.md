# DeepPACyto

**DeepPACyto** is a deep learning-based inference framework for label-free cytopathology using photoacoustic (PA) and Pap-stained (PAP) images.  
The pipeline performs **automatic preprocessing**, **virtual staining**, **super-resolution**, and **cluster-level analysis**, and stores all results in a dedicated results folder.

---

## Environment Requirements

- Execute the project inside a Docker container based on:
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`

---

## Directory Structure

Place your input data as follows:
```
dataset/
├── original_data/
│ └── {opt.name}/ # Raw input images (required)
├── cropped_data/
│ └── {opt.name}/ # Auto-generated cropped patches after preprocessing
results/
└── {opt.name}/ # Output results (virtual staining, SR, clustering)
```

**Pretrained models:**  
The pretrained models must be placed inside:
`checkpoints/`

---

## Workflow

1. **Automatic Preprocessing**  
   The input image is automatically divided into smaller patches for efficient processing.  
   Cropped patches are saved to:
`dataset/cropped_data/{opt.name}/`

2. **Virtual Staining & Super-resolution**  
If `--target_img pa` is set, the model performs virtual staining and super-resolution on PA images to synthesize high-quality PAP-like images.

3. **Cluster Detection and Analysis**  
After image enhancement, the system detects and analyzes cellular clusters to aid in cytopathological interpretation.

4. **Final Results**  
All outputs (virtual stained images, SR images, and analysis results) are saved to:
`results/{opt.name}/`

---

## Running the Code

```bash
python main.py --target_img {target_img} --target_mode {target_mode} --name {target_name}
```
### Parameters
--target_img:
pa: Run virtual staining + super-resolution + analysis starting from PA images.
pap: Run analysis directly on PAP images (without virtual staining).

--target_mode: Controls resolution and analysis mode:
lr: Low-resolution input and analysis.
lrwcell: Low-resolution with cell-level analysis.
hr: High-resolution reconstruction and analysis.
hrwcell: High-resolution with enhanced cell-level cluster analysis.

--name: The name (ID) of the input sample directory (must match the folder under dataset/original_data/).

