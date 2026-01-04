# Holmes Mark-1 CI (GitHub Actions)

This folder contains a starter GitHub Actions workflow that runs your NUC-side orchestrator script.

## How to use

1) Copy `holmes-mark1-ci.yml` into your repository at:

```
.github/workflows/holmes-mark1-ci.yml
```

2) Add the following GitHub Actions secrets:

- `HOLMES_NUC_HOST`: Hostname or IP for the NUC (example: `mark1nuc.yourdomain` or `203.0.113.10`)
- `HOLMES_NUC_USER`: SSH user on the NUC (example: `morris`)
- `HOLMES_NUC_SSH_KEY`: Private SSH key (Ed25519 or RSA) that can connect to the NUC

Optional:

- `WARP_FORGE_NUC_REPO_DIR`: Overrides the repo location on the NUC (default in the scripts: `~/surfworks/warpforge`)

## NUC prerequisites

- Passwordless SSH from the GitHub runner to the NUC using `HOLMES_NUC_SSH_KEY`.
- The NUC can SSH to the NVIDIA and AMD boxes using host aliases `nvidia` and `amd` (or set `TARGET_HOST_OVERRIDE` when invoking the scripts).
- `wakeonlan` installed on the NUC if you want the scripts to wake the GPU boxes.
