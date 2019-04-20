## settings 1 (hs-1)

* default-params in the code.

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_1 --checkpoint_dir checkpoints/hs_1/
```
* best f1: 0.6359

## settings 2 (hs-2)

* flatten images (1d data and 1d conv model)
* made batch_size smaller
* other params default

```bash
git checkout cr_1d_data_hs2
python main_CR.py --tensorboard_dir tensorboard/hs_2/ --checkpoint_dir checkpoints/hs_2/ --batch-size 128
```

* best f1: 0.6058

## setting 3 (hs-3)

* prediction module without transformer. 
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_3 --checkpoint_dir checkpoints/hs_3/ --no-transformer
```

* best f1: 0.58387

## setting 4 (hs-4)

* regularization coefficient: alpha = 0.01 (dafault was 0.005)
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_4 --checkpoint_dir checkpoints/hs_4/ --alpha 0.01
```

* best f1:  0.647


## setting 5 (hs-5)

* regularization coefficient: alpha = 0 (dafault was 0.005).
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_5 --checkpoint_dir checkpoints/hs_5/ --alpha 0.0
```

* best f1:  0.660


## setting 6 (hs-6)

* regularization coefficient: alpha = 0.1 (dafault was 0.05)
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_6 --checkpoint_dir checkpoints/hs_6/ --alpha 0.1
```

* best f1:  0.6495

## setting 7 (hs-7)

* dropout = 0.6 (dafault was 0.5)
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_7 --checkpoint_dir checkpoints/hs_7/ --dropout 0.6
```

* best f1: 0.64335


## setting 8 (hs-8)

* dropout = 0.2 (dafault was 0.5)
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_8 --checkpoint_dir checkpoints/hs_8/ --dropout 0.2
```

* best f1: 0.623

## setting 9 (hs-9)

* remove entropy loss
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_9 --checkpoint_dir checkpoints/hs_9/ --no-entropy
```

* best f1: 0.6226

## setting 10 (hs-10)

* remove entropy loss, and regularization coefficient: alpha = 0.0
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_10 --checkpoint_dir checkpoints/hs_10/ --no-entropy --alpha 0
```

* best f1: 0.61908

## setting 11 (hs-11)

* regularization coefficient: alpha = 0.1 and dropout = 0.6
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_11 --checkpoint_dir checkpoints/hs_11/ --alpha 0.1 --dropout 0.6
```

* best f1: 0.6687


## setting 12 (hs-12)

* regularization coefficient: alpha = 0.5 and dropout = 0.6
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_12 --checkpoint_dir checkpoints/hs_12/ --alpha 0.5 --dropout 0.6
```

* best f1: 0.67209


## setting 13 (hs-13)

* regularization coefficient: alpha = 1 and dropout = 0.6
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_13 --checkpoint_dir checkpoints/hs_13/ --alpha 1 --dropout 0.6
```

## setting 14 (hs-14)

* regularization coefficient: alpha = 0.5 and dropout = 0.7
* other params default

```bash
python main_CR.py --tensorboard_dir tensorboard/hs_14 --checkpoint_dir checkpoints/hs_14/ --alpha 0.5 --dropout 0.7
```
