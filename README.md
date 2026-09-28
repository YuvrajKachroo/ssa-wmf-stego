# Adaptive Image Steganography using 2D-SSA and WMF

An adaptive spatial image steganography system based on **2D Singular Spectrum Analysis (2D-SSA)**, **Weighted Median Filtering (WMF)**, and **Syndrome-Trellis Codes (STC)**.

The project is based on the research paper:

> Xie et al., *A New Cost Function for Spatial Image Steganography Based on 2D-SSA and WMF*, IEEE Access, 2021.

---

## Overview

Traditional LSB steganography uses a relatively simple pixel-modification strategy. This project instead analyzes the spatial structure of a cover image and assigns different modification costs to different pixels.

The main objective is to hide a secret message while controlling image distortion and using an adaptive cost function for the embedding process.

### Pipeline

```text
                 Cover Image
                      │
                      ▼
                 ┌─────────┐
                 │ 2D-SSA  │
                 └────┬────┘
                      │
                      ▼
             Component Selection
                      │
                      ▼
              Suitability Map ζ
                      │
                      ▼
                 ┌─────────┐
                 │   WMF   │
                 └────┬────┘
                      │
                      ▼
              Adaptive Cost Map ρ
                      │
                      ▼
                 ┌─────────┐
 Secret Message →│   STC   │
                 └────┬────┘
                      │
                      ▼
                 Stego Image
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Image Quality             Extraction
 MSE / PSNR / SSIM                 │
                                  ▼
                           Secret Message
