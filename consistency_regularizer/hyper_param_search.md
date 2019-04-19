## settings 1 (hs-1)

* default-params in the code.

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_1 --checkpoint_dir checkpoints/hs_1/
```

## settings 2 (hs-2)

* flatten images so as to benefit from 1d transformer
* made batch_size smaller
* other params default

```bash
git checkout cr_1d_data_hs2
python main_CR.py --tensorboard_dir tensorboard/hs_2/ --checkpoint_dir checkpoints/hs_2/ --batch-size 128
```

## setting 3 (hs-3)