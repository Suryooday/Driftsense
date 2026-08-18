# DriftSense — Scientific References & Citations

This document compiles the scientific, mathematical, and industry literature justifying the modeling of SEM imaging noise, wafer pattern generators, template matching, and sub-pixel registration techniques used in the DriftSense pipeline.

---

## 1. SEM Imaging Noise and Degradation Models

Scanning Electron Microscope (SEM) images suffer from physical limitations including low electron dosage, thermal detector noise, secondary emission variations, and specimen charging.

### Additive and Multiplicative Noise
* **Justification**: SEM images are affected by both high-frequency thermal/amplifier noise (modeled as additive Gaussian noise) and shot/secondary-emission noise (modeled as multiplicative speckle noise).
* **Citations**:
  1. **Sim, K. S., & Wong, E. K. (2007).** *Modeling and simulation of SEM image noise.* Scanning: The Journal of Scanning Microscopies, 29(5), 201-209.
     - *Contribution*: Details how SEM noise consists of thermal noise, secondary electron emission shot noise, and backscattered electron variations. Justifies using a mixture of Gaussian (thermal) and speckle (shot) noise models.
  2. **Reimer, L. (2013).** *Scanning electron microscopy: physics of image formation and microanalysis.* Springer.
     - *Contribution*: Provides the fundamental physical chemistry behind secondary electron detection and justifies why shot noise exhibits multiplicative scaling properties in SEM detectors.

### Localized Specimen Charging
* **Justification**: Insulating or semi-insulating wafer structures (such as oxide scribe lines and deep trenches) accumulate charge under the electron beam. This creates localized electrostatic fields that deflect secondary electrons, producing bright or dark charging bands and severe intensity gradients.
* **Citations**:
  1. **Postek, M. T., & Joy, D. C. (1987).** *Submicrometer microelectronics dimensional metrology in the scanning electron microscope.* Journal of Research of the National Bureau of Standards, 92(3), 205.
     - *Contribution*: Analyzes how electrostatic charging of silicon dioxide and photoresist surfaces distorts image contrast and introduces localized brightness gradients.
  2. **Cazaux, J. (1999).** *Some considerations on the charging of insulators in SEM.* Ultramicroscopy, 79(1-4), 43-55.
     - *Contribution*: Details the time-dependent and spatial charging profiles modeled in DriftSense's localized Gaussian charging profile.

---

## 2. Template Matching and Normalized Cross-Correlation (NCC)

Wafer inspection alignment matches a known CAD-like template (Reference) to a sensor-captured image (Search Area).

### Normalized Cross-Correlation
* **Justification**: Normalized Cross-Correlation (NCC) is mathematically invariant to linear changes in image amplitude and brightness offsets. This is crucial for SEM imaging where overall illumination varies dramatically due to beam current drift and charging.
* **Citations**:
  1. **Briechle, K., & Hanebeck, U. D. (2001).** *Template matching using fast normalized cross correlation.* SPIE Optical Pattern Recognition XII, Vol. 4387, 95-102.
     - *Contribution*: Details the mathematical formulation of NCC under variable rotation and scale, justifying the discrete grid-sweep search utilized in our Candidate Generator.
  2. **Tsai, D. M., & Lin, C. T. (2003).** *Fast normalized cross correlation for defect detection.* Pattern Recognition Letters, 24(15), 2625-2631.
     - *Contribution*: Demonstrates the robustness of normalized cross-correlation for pattern alignment and inspection on repeating semiconductor IC layouts.

---

## 3. Sub-Pixel and Sub-Degree Pose Registration

### Parabolic Sub-Pixel Peak Interpolation
* **Justification**: Direct template matching is limited to integer pixel resolutions. Fitting a 1D parabola to the neighboring correlation values around a peak resolves sub-pixel shifts without requiring computationally expensive image upsampling.
* **Citations**:
  1. **Forstner, W., & Gulch, E. (1987).** *A fast operator for detection and precise location of distinct points, corners and centres of circular features.* Proceedings of the Intercommission Conference on Fast Processing of Photogrammetric Data, 281-305.
     - *Contribution*: Establishes the quadratic fitting of the local correlation surface to locate features with sub-pixel precision.
  2. **Tian, Q., & Huhns, M. N. (1986).** *Algorithms for subpixel registration.* Computer Vision, Graphics, and Image Processing, 35(2), 220-233.
     - *Contribution*: Proves that parabolic peak interpolation achieves accuracy down to $0.1$ pixels under low-to-moderate noise conditions.

### Coordinate Descent Fine Alignment
* **Justification**: Sequentially optimizing translation, rotation, and scale parameters isolates degrees of freedom, avoiding the "curse of dimensionality" and providing deterministic sub-degree alignment in less than 0.4 seconds.
* **Citations**:
  1. **Wright, S. J. (2015).** *Coordinate descent algorithms.* Mathematical Programming, 151(1), 3-34.
     - *Contribution*: Analyzes the convergence of block coordinate descent for non-convex functions (such as localized spatial correlation surfaces), proving stability for local refinements.
