# Mindset-mode-identification-with-deep-sets-
A convolutional neural network architecture based on the two frameworks PointNet and Deep Sets to identify and segment pulsation modes in artificial pre-main sequence delta Scuti stars. Created as a part of my master's thesis project at the University of Innsbruck in 2025.

# Features
- CNN architecture combining principles of Deep Sets and PointNet
- Frequency segmentation and ridge identification in artificial pre-main sequence delta Scuti stars
- Works with input of frequency lists and the large separation/folding frequency delta nu

# Augmented and idealized data
- In the first step, the data from the stellar grid is not altered and models are trained only with this data
- In the second training step, we introduce three augmentation functions: drop (D) to randomly drop frequencies, shake (S) to introduce gaussian noise to the frequencies and fake (F) to inject spurious frequencies that don't correspond to pulsations. 

# Main sources
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017) (https://arxiv.org/abs/1612.00593)
Qi, C. R., Su, H., Mo, K., & Guibas, L. J.

Deep Sets (2017) (https://arxiv.org/abs/1703.06114)
Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A.

PMS δ Scuti pulsation models without rotational splitting (2022) (https://www.aanda.org/articles/aa/abs/2022/08/aa43242-22/aa43242-22.html)
Steindl, T., Zwintz, K., & Müllner, M.
