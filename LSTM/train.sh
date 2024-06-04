python train.py --train_data ../dataset/proc/adele/train.txt \
                --test_data ../dataset/proc/adele/test.txt \
                --seq_len 128 \
                --d_model 256 \
                --n_layer 2 \
                --gpu 0 \
                --batch 8 \
                --lr 3e-4 \
                --epoch 2 \
                --ckpt ./model/model-adele.pth
                # --bidirectional \