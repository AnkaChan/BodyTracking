# For texturemap learning
#### Use: 200618_26_LearnTexture_AllAtOnce_ContourMasked.ipynb

## Modify paths where it says '# Paths: modify here'.
They are modified to run without a problem. If not, please modify some to suit your paths.

## Merging background image
Pixel values of the rendered image are simply replaced with clean plate images (inside 'def forward()' of 'class Model()').
This should get an improvement, such that the gradients will flow.

## Some options to note before running:
#### - loss types inside 'def forward()' in 'class Model()'
#### - usage of contour mask in 'def forward()' in 'class Model()'

Other than these, the code should be straightforward. If not, please ask.

