# 1. For texturemap learning
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

# 2. For per-vertex mesh deformation
#### Use: 16h_DeformMesh_using14gOutput_LoR_Sil.ipynb

## Modify paths where it says '# Paths: modify here'.
They are modified to run without a problem. If not, please modify some to suit your paths.

## Merging background image
Pixel values of the rendered image are simply replaced with clean plate images (inside 'def forward()' of 'class Model()').
This should get an improvement, such that the gradients will flow.

## Laplacian pyramid
reference: http://www.cs.toronto.edu/~jepson/csc320/notes/pyramids.pdf

## Some options to note before running 'def forward()' in 'class Model()':
#### - loss types
#### - regularization types
#### - parameters related to laplacian pyramid
Please ask if some codes break.
