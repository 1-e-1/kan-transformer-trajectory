# KAN Transformer Trajectory

Independent 8-DoF supervised KAN-Transformer policies, DAgger and trajectory evaluation.

Primary method: supervised behavior cloning and DAgger. Separate KAN actors and optimizers are retained. CBF, spline planning and CAN execution have separate focused repositories.

## Setup

Use a dedicated Python environment and install a compatible PyTorch/torchvision pair for your platform, then:

```sh
python -m pip install -r requirements.txt
```

Run from the repository root. See [method scope](METHOD_SCOPE.md), [publication notes](PUBLICATION_NOTES.md) and the retained original README for entry points and historical status.

## Distribution limits

This release contains source, selected configuration, original tests and research documentation. It does not bundle datasets, pretrained weights, robot scenes or third-party simulation libraries. Historical success rates and paper figures are not new measurements from this release. Simulator and hardware claims retain their original limitations. Do not launch full training queues just to test an installation.

## License

Original material is licensed under [MIT](LICENSE). Third-party libraries, assets and datasets retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
