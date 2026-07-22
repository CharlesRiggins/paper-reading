## Appendix F. Parameter split

For a Transformer block with $h$ heads, model width $d$, MLP width $d_{\text{MLP}}$, and head width $d_h=d/h$, the relevant trainable tensors are

$$
W_{QKV}^{(i)}\in\mathbb{R}^{3\times d\times d_h},\quad
W_O\in\mathbb{R}^{d\times d},\quad b_O\in\mathbb{R}^{d},
$$
$$
W_1\in\mathbb{R}^{d\times d_{\text{MLP}}},\quad b_1\in\mathbb{R}^{d_{\text{MLP}}},\quad
W_2\in\mathbb{R}^{d_{\text{MLP}}\times d},\quad b_2\in\mathbb{R}^{d}.
$$

Normalization-layer parameters are omitted for notational simplicity and treated as joint. Selecting $h_{\text{forget}}$ heads and $d_{\text{forget}}$ MLP hidden units creates slices:

$$
W_1=[W_1^{\text{forget}}\;W_1^{\text{retain}}],\qquad
W_2=\begin{bmatrix}W_2^{\text{forget}}\\W_2^{\text{retain}}\end{bmatrix},\qquad
W_O=\begin{bmatrix}W_O^{\text{forget}}\\W_O^{\text{retain}}\end{bmatrix},
$$

with the same partition for $b_1$. The paper defines

$$
\theta_{\text{forget}}=\{W_1^{\text{forget}},b_1^{\text{forget}},W_2^{\text{forget}},W_O^{\text{forget}},W_{QKV}^{(i)}:i\le h_{\text{forget}}\},
$$

and assigns the complementary MLP slices, output biases, and remaining attention heads to $\theta_{\text{retain}}$. Thus the target allocation corresponds to *entire head-level* QKV weights plus the matching attention output slice, and to matching input/output MLP-unit slices—not a loose collection of individual scalar weights.

## Appendix G. Leakage definition

An SGTM run sees

$$
\mathbf{D}_{\text{SGTM}}=\{\mathbf{D}_{\text{forget}},\mathbf{D}_{\text{retain}},\mathbf{D}_{\text{unlabeled}}\},
$$

where the unlabelled set secretly contains both true domains:

$$
\mathbf{D}_{\text{unlabeled}}
=\mathbf{D}^{\text{retain}}_{\text{unlabeled}}
\cup\mathbf{D}^{\text{forget}}_{\text{unlabeled}}.
$$

To measure how much unlabelled target content became usable outside the ablated component, the authors compare it with standard training on the *same retain training data* and a variable number of target examples. They choose the standard-training target set size that gives the same forget-domain loss as the SGTM run. Leakage is

$$
\operatorname{Leakage}(\theta_{\text{SGTM}})=
\frac{\left|\mathbf{D}^{\text{forget}}_{\text{standard}}\right|}
{\left|\mathbf{D}^{\text{forget}}_{\text{unlabeled}}\right|},
$$

subject to

$$
\ell(\theta_{\text{SGTM}},\mathcal{D}_{\text{forget}})
=\ell(\theta_{\text{standard}},\mathcal{D}_{\text{forget}}).
$$

The numerator is inferred by training several standard baselines with increasingly many target tokens and linearly interpolating between the closest losses. A leakage of 0 means no equivalent target exposure can be detected in the surviving model; a leakage of 1 means missed target samples were as effective as ordinary training. A filtering run with a fixed proportion of missed targets has leakage 1 by this definition.

This definition makes a careful *performance-equivalence* claim rather than observing an internal information quantity directly. It is therefore useful for comparing exposure under a fixed experimental distribution, but it depends on the loss-to-exposure calibration curve and should not be interpreted as a universal measure of information in parameters.
