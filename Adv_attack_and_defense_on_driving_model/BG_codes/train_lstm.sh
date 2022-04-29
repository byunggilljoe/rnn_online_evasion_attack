#python BG_codes/train_lstm.py  --root_dir ../udacity-data --batch_size 32 --epochs 15 --train 1 --lr 0.0001  --trial 0 &
#python BG_codes/train_lstm.py  --root_dir ../udacity-data --batch_size 32 --epochs 15 --train 1 --lr 0.0001  --trial 1 &
#python BG_codes/train_lstm.py  --root_dir ../udacity-data --batch_size 32 --epochs 15 --train 1 --lr 0.0001  --trial 2 &

python BG_codes/train_lstm.py  --epochs 120 --train 1 --trial 3 &
python BG_codes/train_lstm.py  --epochs 120 --train 1 --trial 4 &
python BG_codes/train_lstm.py  --epochs 120 --train 1 --trial 5 &
