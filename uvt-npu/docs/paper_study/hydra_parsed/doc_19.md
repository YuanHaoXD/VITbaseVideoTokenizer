The final CKNNA score is obtained via normalized local alignment:

 $$ \mathrm{CKNNA}(\mathbf{K},\mathbf{L})=\frac{S(\mathbf{K},\mathbf{L})}{\sqrt{S(\mathbf{K},\mathbf{K})S(\mathbf{L},\mathbf{L})}}. $$ 

In practice, we uniformly sample 10,000 images from the ImageNet validation set (Russakovsky et al., 2014) and compute CKNNA with k = 10. As observed in (Huh et al., 2024), restricting the comparison to small neighborhoods yields a more sensitive measure of representational agreement.