
# autoanchor & 30epoch - 0.695
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py \
    --workers 8 \
    --device 0,1,2,3 \
    --sync-bn \
    --epochs 30 \
    --batch-size 16 \
    --data data/dota_v1_poly.yaml \
    --img 1024 1024 \
    --cfg cfg/training/yolov7.yaml \
    --weights ./pretrained/yolov7.pt \
    --name yolov7-30epoch \
    --hyp data/hyp.finetune_dota.yaml \
    --noautoanchor 

# autoanchor & 12e & yolov7-w6 - 
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9529 train.py \
    --workers 8 \
    --device 0,1,2,3 \
    --sync-bn \
    --epochs 12 \
    --batch-size 16 \
    --data data/dota_v1_poly.yaml \
    --img 1024 1024 \
    --cfg cfg/training/yolov7.yaml \
    --weights '' \
    --name temp \
    --hyp data/hyp.finetune_dota.yaml \
    --noautoanchor 

# autoanchor & 12e - 
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9528 train_aux.py \
    --workers 8 \
    --device 0,1,2,3 \
    --sync-bn \
    --epochs 12 \
    --batch-size 16 \
    --data data/dota_v1_poly.yaml \
    --img 1024 1024 \
    --cfg cfg/training/yolov7-w6_fuck.yaml \
    --weights ./pretrained/yolov7-w6.pt \
    --name yolov7-w6-12epoch \
    --hyp data/hyp.finetune_dota.yaml \
    --noautoanchor 
    

python -m torch.distributed.launch --nproc_per_node 8 
--master_port 9527 train_aux.py 
--workers 8 
--device 0,1,2,3,4,5,6,7 
--sync-bn --batch-size 128 --data data/coco.yaml 
--img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml

# inference
python detect.py --weights ./runs/train/yolov73/weights/last.pt --conf 0.25 --img-size 640 --device 6 --source ./dataset/demo/P0032.png

python detect.py --weights ./runs/train/yolov7-30epoch/weights/last.pt --conf 0.25 --img-size 1024 --device 6 --source ./dataset/demo/


# test
python test.py --data data/dota_v1_poly.yaml --img 1024 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights ./runs/train/yolov7-30epoch/weights/best.pt --name test