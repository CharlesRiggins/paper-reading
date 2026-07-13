## Appendix E. Experimental Ablations

Appendix E checks whether the main results are artifacts of parameter count, learning-rate warmup, or a simpler symmetry-breaking attempt based on fixing biases.

### E.1. Matching Learnable Parameters

Because $W$-Asymmetric networks fix some weights to constants, they often have fewer learnable parameters than standard networks with the same nominal architecture. The paper reports the learnable parameter counts and then trains smaller standard networks matched to the $W$-Asymmetric parameter count.

| Experiment / architecture | Standard / $\sigma$-Asym | $W$-Asym |
|---|---:|---:|
| Section 5.1 `MLP` | `935,434` | `834,570` |
| Section 5.1 `ResNet 1x` | `272,474` | `230,024` |
| Section 5.1 `ResNet 8x` | `17,289,866` | `16,273,946` |
| Section 5.1 `GNN` | `176,424` | `171,576` |
| Section 5.2 `MLP-8` | `3,242,146` | `3,324,466` |
| Section 5.2 `MLP-16` | `5,960,242` | `5,796,002` |
| Section 5.2 `ResNet-20 1x` | `1,356,098` | `1,143,858` |
| Section 5.2 `ResNet-20 2x` | `5,410,386` | `5,044,756` |
| Section 5.2 `ResNet-110 1x` | `8,620,418` | `7,371,378` |
| Section 5.2 `ResNet-110 2x` | `34,512,276` | `32,014,996` |
| Section 5.3 `ResNet` | `78,042` | `60,634` |
| Section 5.4 `MLI ResNet` | `78,042` | `60,634` |

The matched-parameter metanetwork results show little change for smaller standard ResNets; $W$-Asym remains much easier to predict.

| Metanetwork | ResNet $R^2$ | ResNet $\tau$ | Smaller ResNet $R^2$ | Smaller ResNet $\tau$ | $W$-Asym ResNet $R^2$ | $W$-Asym ResNet $\tau$ |
|---|---:|---:|---:|---:|---:|---:|
| `MLP` | $.330\pm .04$ | $.389\pm .03$ | $.348\pm .07$ | $.400\pm .02$ | **$.594\pm .12$** | **$.864\pm .01$** |
| `DMC` | $.950\pm .01$ | $.787\pm .02$ | $.943\pm .01$ | $.779\pm .01$ | **$.967\pm .01$** | **$.911\pm .01$** |
| `DeepSets` | $.855\pm .01$ | $.617\pm .03$ | $.849\pm .01$ | $.627\pm .01$ | **$.936\pm .00$** | **$.858\pm .00$** |
| `StatNN` | $.976\pm .00$ | $.866\pm .00$ | $.976\pm .00$ | $.869\pm .00$ | **$.978\pm .00$** | **$.935\pm .01$** |

For Bayesian networks, reducing the standard `ResNet20` parameter count also does not explain the $W$-Asym improvement after `25` epochs:

| Base network | Test accuracy |
|---|---:|
| $W$-Asym ResNet20 | **$49.3\pm 0.4$** |
| Standard `ResNet20` | $46.8\pm 0.9$ |
| Smaller standard `ResNet20` | $46.5\pm 1.1$ |

### E.2. Changing Number of Warmup Steps

Prior work shows learning-rate warmup can affect linear mode connectivity [3]. The paper repeats interpolation-barrier tests with `1` warmup epoch versus `20` warmup epochs using `Adam` with learning rate `1e-2` for `ResNet20` and `1e-3` for GNNs. The qualitative result is unchanged: $W$-Asym interpolates substantially better than Git-ReBasin-aligned standard networks.

| Warmup | `ResNet20` Git-ReBasin | $W$-Asym ResNet20 | `GNN` Git-ReBasin | $W$-Asym GNN |
|---|---:|---:|---:|---:|
| `1` epoch | $4.2\pm .80$ | **$.673\pm .29$** | $.249\pm .04$ | $.075\pm .04$ |
| `20` epochs | $2.0\pm .21$ | $.934\pm .72$ | $.292\pm .04$ | **$.074\pm .02$** |

### E.3. Failure to Break Symmetries by Fixing Biases

A simpler way to distinguish neurons might be to fix hidden biases. The authors test this idea on standard `ResNet20` on `CIFAR-10`, drawing fixed biases from intervals $[-k,k]$. This fails: barriers are much larger than $W$-Asym (`$.934\pm .72`) and $\sigma$-Asym (`$2.521\pm .46`).

| $k$ | Loss barrier |
|---:|---:|
| `1` | $5.81\pm 3.67$ |
| `3` | $3.76\pm 0.38$ |
| `9` | $4.62\pm 1.03$ |
| `27` | $10.6\pm 4.29$ |

This supports the idea that effective symmetry breaking must intervene in the weight / computation-graph structure, not merely attach fixed identifiers to neurons via bias terms.
