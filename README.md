# Virtual Dressing Room (AI-Powered Virtual Try-On)

## Overview

This project was selected and developed as my **Final Year Project (FYP)**, with a strong emphasis on research, experimentation, and technical evaluation of different virtual try-on methodologies under real-world constraints such as limited computational resources, deployment feasibility, and system scalability.

The Virtual Dressing Room is an AI-powered virtual try-on system that allows users to visualize how different clothing items would look on a person without physically wearing them. This project is designed as a research-oriented and practical implementation that combines dataset-based virtual try-on pipelines with real-time pose estimation and an external AI service for high-quality custom uploads.

The main goal of this project is not to claim a single perfect solution, but to explore, compare, and integrate multiple virtual try-on approaches ranging from dataset-driven pipelines to real-time pose-based inference and API-powered high-quality rendering while keeping the overall system runnable in CPU-only environments.

The implementation is based on learning from and adapting ideas from well-known works and datasets such as VITON-HD, Zalando-HD (resized), and pose-estimation-based virtual try-on techniques, which are then unified into a single, well-structured and modular system.

---

## Key Features

* AI-based virtual clothing try-on without physical fitting
* Multiple independent pipelines for different use cases
* Dataset-driven automatic cloth-person pairing
* Real-time try-on support using pose estimation (OpenPose)
* High-quality custom try-on using Kling AI API
* Fully CPU-based execution (no GPU dependency)
* Modular and extensible architecture for research and experimentation

---

## Project Motivation

Online fashion shopping suffers from high return rates because users cannot accurately visualize how clothes will look on them. This project aims to reduce that uncertainty by:

* Allowing users to preview outfits virtually
* Supporting both **predefined datasets** and **custom user uploads**
* Exploring the trade-offs between academic pipelines and production-grade AI APIs

This system is especially useful for:

* Academic research and final-year projects
* Proof-of-concept virtual try-on systems
* AI-powered e-commerce experimentation

---

## System Architecture (High-Level)

The project is divided into **three independent pipelines**, each serving a different purpose:

1. **Dataset-Based Virtual Try-On Pipeline (Zalando-HD / VITON-HD inspired)**
2. **Real-Time Pose-Based Try-On Pipeline (OpenPose)**
3. **API-Based High-Quality Virtual Try-On Pipeline (Kling AI)**

Each pipeline can be used separately depending on the required quality, performance, and input type.

---

## Pipeline 1: Dataset-Based Virtual Try-On (Primary Pipeline)

### Description

This is the **most structured and stable pipeline** in the project. It uses **preprocessed images from the Zalando-HD resized dataset**, inspired by the VITON-HD workflow. The pipeline is designed to work with **paired datasets**, where each clothing image corresponds to a person image.

### How It Works

* The dataset consists of **person images** and **cloth images** that are already aligned or resized
* Images are preprocessed (resizing, normalization, background handling)
* When a dataset pair is selected:

  * The system automatically picks the corresponding person and clothing image
  * The clothing is virtually wrapped onto the person image
  * The output is a visually aligned try-on result

### Key Characteristics

* Works best with **paired and structured datasets**
* Highly consistent results due to controlled inputs
* Inspired by VITON-HD and Zalando-HD preprocessing pipelines
* Suitable for academic experiments and benchmarking

### Limitations

* Requires pre-aligned datasets
* Not suitable for arbitrary user-uploaded images

---

## Pipeline 2: Real-Time Virtual Try-On Using OpenPose

### Description

This pipeline focuses on **real-time interaction** rather than perfect visual quality. It uses **OpenPose** to detect human body keypoints and estimate the pose of the user in real time.

### How It Works

* Camera input is captured in real time
* OpenPose extracts body keypoints (shoulders, arms, torso, etc.)
* Clothing overlays are adjusted according to detected pose
* The output updates dynamically as the user moves

### Key Characteristics

* Enables real-time try-on experience
* Pose-based clothing alignment
* Designed mainly for **demonstration and experimentation**

### Limitations

* Not a fully professional-grade pipeline
* Visual quality depends heavily on pose detection accuracy
* Clothing fitting is approximate, not pixel-perfect

---

## Pipeline 3: API-Based Virtual Try-On Using Kling AI

### Description

This pipeline provides the **highest-quality results** in the project by leveraging the **Kling AI API**. It is designed for **custom user uploads**, where both the person image and clothing image are provided by the user.

### How It Works

* User uploads a person image and a clothing image
* Images are sent securely to the Kling AI API
* The API performs advanced virtual try-on processing
* The resulting image shows the clothing realistically wrapped on the person

### Key Characteristics

* Supports arbitrary user-uploaded images
* Produces clean, professional-quality results
* No manual pairing or dataset constraints

### Important Notes

* Requires a valid **Kling AI API key** (stored securely using environment variables)
* The API key is **never pushed to GitHub**
* Executed entirely on **CPU**, making it accessible on low-resource systems

---

## Dataset and Resources Used

This project does not claim ownership of the following resources. Instead, they are used **for learning, experimentation, and academic purposes**:

* **Zalando-HD (Resized)** – Used for structured dataset-based try-on
* **VITON-HD** – Inspired the preprocessing and pairing pipeline
* **OpenPose** – Used for real-time pose estimation
* **Kling AI API** – Used for high-quality custom virtual try-on

Each component has been adapted carefully to fit the constraints and goals of this project.

---

## Preprocessing Pipeline

The preprocessing stage plays a critical role in ensuring consistent results:

* Image resizing to standard dimensions
* Normalization for model compatibility
* Dataset organization into person–cloth pairs
* Automatic selection of corresponding images

Preprocessed data enables faster experimentation and stable outputs.

---

## Hardware & Performance

* **Execution Mode:** CPU-only
* **GPU:** Not required
* **System Compatibility:** Works on standard laptops

The project is intentionally designed to remain lightweight and accessible for students and researchers without high-end hardware.

---

## Environment Variables

For security reasons, API keys are not stored in the codebase.

Create a `.env` file in the project root:

* `KLING_API_KEY=your_api_key_here`

Make sure the `.env` file is added to `.gitignore`.

---

## Use Cases

* Virtual try-on demos for e-commerce
* Academic research and experimentation
* Final Year Project (FYP)
* AI-based fashion technology prototypes

---

## Limitations & Future Improvements

* Improve real-time pipeline accuracy
* Integrate a fully end-to-end deep learning try-on model
* Enhance clothing segmentation and warping
* Add support for multiple body types
* Optimize inference speed

---

## Disclaimer

This project is intended for **educational and research purposes only**. All datasets, models, and APIs used belong to their respective owners.

---

## Author

**Ali Naqi**
AI & Data Science Practitioner

---

## Acknowledgements

Special thanks to the open-source community and research works behind VITON-HD, Zalando-HD, OpenPose, and Kling AI for providing valuable resources that made this project possible.
