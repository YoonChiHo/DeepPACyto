# DeepPACyto

**DeepPACyto** is a deep learning-based inference framework for label-free cytopathology using photoacoustic (PA) and Pap-stained (PAP) images.  
The pipeline performs **automatic preprocessing**, **virtual staining**, **super-resolution**, and **cluster-level analysis**, and stores all results in a dedicated results folder.

---

## 🔧 Environment Requirements

- Execute the project inside a Docker container based on:
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`

---

## 📁 Directory Structure

Place your input data as follows:
``
dataset/
├── original_data/
│ └── {opt.name}/ # Raw input images (required)
├── cropped_data/
│ └── {opt.name}/ # Auto-generated cropped patches after preprocessing
results/
└── {opt.name}/ # Output results (virtual staining, SR, clustering)
``
