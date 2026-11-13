# Registration-free 3D organ motion from a single 2D plane

Code for **"Registration-Free 3D Organ Motion Modelling from a Single 2D Plane with Test-Time Refinement"** (Marsien Boade, Samuel Kadoury et al.).

From one live 2D image plane, the method estimates the dense 3D displacement field of the whole organ in real time. It has three parts:

1. **Teacher.** It sees the full reference volume $V_0$ and the target volume $V_t$, and learns a compact motion token grid (6³ positions × 128 channels) together with a decoder from tokens to a 3D displacement field. It is trained by image similarity alone (local NCC, smoothness, folding penalty), so no registration-derived field is used as a training target.
2. **Student.** It predicts the teacher's tokens for frame $t$ from the live plane at frames $t-2$, $t-1$ and $t$, the reference plane, and the reference volume's encoder features.
3. **Test-time refinement.** Per frame, one Adam step on a single 128-d shift shared by all tokens, fitting the warped reference to the observed plane. It is warm-started from the previous frame.

The teacher's decoder, run on the refined tokens, gives the output field.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.5.1 and CUDA 12.1 on one NVIDIA RTX 6000 Ada (48 GB). Training one sequence at a time needs about 15 GB of GPU memory for lung (10 frames) and more for long cardiac sequences.

## Data format

Each 4D sequence is stored as one file per frame in a single directory:

```
<data_dir>/<subject>_frame00_ct.npz    # reference frame (frame 0)
<data_dir>/<subject>_frame01_ct.npz
...
```

Each file holds a `96 × 96 × 96` float volume under the key `ct`; leading singleton dimensions are allowed. Frame 0 is the reference. On loading, every volume is clipped to its 1st–99th intensity percentiles and rescaled to [0, 1]. Sequences with fewer than 3 frames are skipped.

The surrogate plane is the central slice (index 48) along `--plane_axis`:

| Data | Plane | `--plane_axis` |
|---|---|---|
| Lung 4D-CT, liver 4D-MRI (navigator) | coronal | 1 |
| Cardiac cine MRI (ACDC) | mid-ventricular short axis | 0 |

A split file is JSON: `{"train": [...], "val": [...], "test": [...]}`, listing subject identifiers. `splits/acdc_split.json` is the ACDC split used in the paper: 110 / 20 / 20 subjects, balanced over the five pathology groups. The lung 4D-CT data are not public.

## Usage

The commands below use the paper's settings, shown for lung.

**1. Train the teacher**, at most 60 epochs with early stopping after 10. The checkpoint is the best validation NCC.
```bash
python train_teacher.py --data lung:/data/lung_ct:/data/lung_split.json --out runs/lung
```

**2. Dump the teacher's tokens and reference features** for the student.
```bash
python dump_tokens.py --teacher runs/lung --data_dir /data/lung_ct --split /data/lung_split.json
```

**3. Train the student.**
```bash
python train_student.py --teacher runs/lung --data_dir /data/lung_ct --plane_axis 1 --out runs/lung/student
```

**4. Refine and evaluate on the test split.** This reports lNCC, SSIM, PSNR and the folding ratio for no motion, the student, the refined student and the teacher.
```bash
python refine.py --teacher runs/lung --student runs/lung/student --data_dir /data/lung_ct --split /data/lung_split.json \
                 --plane_axis 1 --lambda_r 10 --out runs/lung/test_results.json
```

For cardiac data, use `--plane_axis 0 --lambda_r 0` and `--data cardiac:/data/acdc:splits/acdc_split.json`. 
Refinement settings:

| Option | Default | Meaning |
|---|---|---|
| `--steps` | 1 | Adam steps per frame |
| `--eta` | 0.05 | step size |
| `--lam` | 1 | shift penalty |
| `--lambda_fold` | 1000 | folding penalty |
| `--lambda_r` | 0 | increment-consistency weight |
| `--cold` | off | start each frame's shift from zero instead of warm-starting |
| `--save_fields` | – | directory for the refined fields (float16, per sequence) |

### Joint and fine-tuned teachers

Repeat `--data` to train one teacher on several organs. The checkpoint criterion then becomes the mean of the per-organ validation NCCs.
```bash
python train_teacher.py --data lung:/data/lung_ct:/data/lung_split.json --data cardiac:/data/acdc:splits/acdc_split.json \
                        --data liver:/data/liver:/data/liver_split.json --out runs/joint
```

To fine-tune a teacher on one organ:
```bash
python train_teacher.py --data liver:/data/liver:/data/liver_split.json --init_from runs/joint/checkpoint_best.pth --lr 5e-5 --out runs/liver_ft
```

Other teacher options:

| Option | Default | Meaning |
|---|---|---|
| `--token_grid` | 6 | token grid side. The student assumes 6. |
| `--token_dim` | 128 | token channels |
| `--full_decoder` | off | predict the field at 96³ instead of at 48³ with upsampling |
| `--resume` | off | continue from `checkpoint_latest.pth` |

## Repository layout

```
mmrf/data.py      loading, normalisation, plane extraction
mmrf/nets.py      CNN encoder, token encoder, FiLM decoder, spatial transformer, student
mmrf/losses.py    local NCC (3D, 2D), smoothness, folding penalty and ratio, evaluation metrics
train_teacher.py  teacher training (single organ, joint, fine-tuning)
dump_tokens.py    teacher tokens and pooled reference features
train_student.py  student training
refine.py         student inference, test-time refinement, evaluation
```

Displacement fields are in voxels, `(B, 3, D, H, W)`, with $V_0(x + u_t(x)) \approx V_t(x)$.

## Citation

The citation will be added on publication.
