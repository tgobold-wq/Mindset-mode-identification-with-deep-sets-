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
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (https://arxiv.org/abs/1612.00593)
Deep Sets (https://arxiv.org/abs/1703.06114)
Pulsational instability of pre-main-sequence models from accreting protostars (https://www.aanda.org/articles/aa/abs/2022/08/aa43242-22/aa43242-22.html)

