python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py \
    --workers 8 \
    --device 0,1,2,3 \
    --sync-bn \
    --batch-size 16 \
    --data data/dota_v1_poly.yaml \
    --img 640 640 \
    --cfg cfg/training/yolov7.yaml \
    --weights '' \
    --name yolov7 \
    --hyp data/hyp.finetune_dota.yaml \
    --noautoanchor


# test
python test.py \
    --data data/dota_v1_poly.yaml \
    --img 640 \
    --batch 32 \
    --conf 0.001 --iou-thres 0.65 \
    --device 6  \
    --weights ./runs/train/yolov7/weights/last.pt \
    --name yolov7_640_val