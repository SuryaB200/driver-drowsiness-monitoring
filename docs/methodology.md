# Technical Methodology & Mathematical Formulations

## 1. Eye Aspect Ratio (EAR) Formulation

The Eye Aspect Ratio (EAR) algorithm enables computationally lightweight eye blink and micro-sleep tracking by computing the ratio between vertical eyelid distances and horizontal intercanthal distance.

For a 6-point eye contour $\{P_1, P_2, P_3, P_4, P_5, P_6\}$:
- $P_1$: Outer corner
- $P_4$: Inner corner
- $P_2, P_3$: Superior eyelid coordinates
- $P_5, P_6$: Inferior eyelid coordinates

$$\text{EAR} = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \cdot \|P_1 - P_4\|}$$

Where $\|\cdot\|$ denotes 2D Euclidean distance:
$$\|A - B\| = \sqrt{(A_x - B_x)^2 + (A_y - B_y)^2}$$

### Dual-Eye Derivation and Temporal Filtering:
1. Calculate $\text{EAR}_{\text{left}}$ and $\text{EAR}_{\text{right}}$ independently.
2. Formulate bilateral mean:
   $$\text{EAR}_{\text{avg}} = \frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2}$$
3. Apply Exponential Moving Average (EMA) smoothing to eliminate single-frame camera noise:
   $$\text{EAR}_{\text{smoothed}}^{(t)} = \alpha \cdot \text{EAR}_{\text{avg}}^{(t)} + (1 - \alpha) \cdot \text{EAR}_{\text{smoothed}}^{(t-1)}$$
   *(Default: $\alpha = 0.35$)*

---

## 2. Mouth Aspect Ratio (MAR) Formulation

To detect sustained yawning without triggering false alarms during vocal speech, the Mouth Aspect Ratio is derived from 8 inner lip contour points:

$$\text{MAR} = \frac{\|P_{13} - P_{14}\| + \|P_{81} - P_{178}\| + \|P_{311} - P_{402}\|}{2 \cdot \|P_{61} - P_{291}\|}$$

- **Speech Syllable Discrimination:** While speaking causes transient spikes ($\text{MAR} > 0.65$ lasting $<0.4\text{s}$), physiological yawning produces sustained aperture openings lasting $\ge 1.8\text{s}$. The timer resets whenever MAR falls below threshold.

---

## 3. Head Pose Estimation (Perspective-n-Point)

Head orientation indicates driver distraction (e.g. looking away at mirrors, passenger, or mobile phone).

### 3D Reference Face Model:
Six canonical anthropometric points in millimeter space:
1. Nose tip: $(0.0, 0.0, 0.0)$
2. Chin: $(0.0, -330.0, -65.0)$
3. Left eye outer corner: $(-225.0, 170.0, -135.0)$
4. Right eye outer corner: $(225.0, 170.0, -135.0)$
5. Left mouth corner: $(-150.0, -150.0, -125.0)$
6. Right mouth corner: $(150.0, -150.0, -125.0)$

### Pinhole Camera Matrix $\mathbf{K}$:
$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
Where focal length $f_x = f_y \approx \text{frame width}$, and principal point $(c_x, c_y) = (\text{width}/2, \text{height}/2)$.

`cv2.solvePnP` computes rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$. Using `cv2.Rodrigues` and `cv2.RQDecomp3x3`, the rotation matrix is converted into Euler angles:
- **Yaw ($\psi$):** Horizontal rotation ($>28^\circ \implies$ Looking Left/Right)
- **Pitch ($\theta$):** Vertical tilt ($<-22^\circ \implies$ Looking Downward)
- **Roll ($\phi$):** In-plane head tilt

---

## 4. MobileNetV3-Small Deep Learning Model

### Backbone Architecture:
- Depthwise separable convolutions reduce computation by factorizing standard convolution into spatial filtering followed by $1 \times 1$ pointwise channel projection.
- Squeeze-and-Excitation (SE) modules dynamically weight channel feature maps based on global contextual dependencies.
- Hardswish non-linear activation:
  $$h\text{-swish}(x) = x \frac{\text{ReLU6}(x + 3)}{6}$$

### Training Objective:
Cross-Entropy Loss with AdamW optimizer:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c}) + \lambda \|\mathbf{w}\|_2^2$$
Learning rate is scheduled via `ReduceLROnPlateau` ($lr=10^{-4}$, factor=0.5, patience=2).

---

## 5. Subject-Independent Partitioning Protocol

Subject overlap between training and testing sets is a primary cause of inflated academic accuracy in computer vision. When facial identities overlap, neural networks learn facial features rather than genuine drowsiness patterns.

To prevent this:
- **Train Split:** Subjects 1 through 8 (66.7% of subjects)
- **Validation Split:** Subjects 9 and 10 (16.7% of subjects)
- **Test Split:** Subjects 11 and 12 (16.7% of subjects)
- **Subject Leakage:** Strictly 0.0%. Test subjects are never seen during training or validation.
