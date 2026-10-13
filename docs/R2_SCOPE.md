# R2 scope decision

The permitted alternative was selected: retain VGG-16 as the only evaluated
architecture and frame the study explicitly as an architecture-specific case
study. VGG-16 provides five clearly delimited sequential convolutional stages
without residual connections, parallel branches, or attention modules, which
supports transparent layer-wise inspection. Claims must therefore refer to the
evaluated pretrained VGG-16 configuration and must not be generalized to CNNs
or vision models as a whole.
