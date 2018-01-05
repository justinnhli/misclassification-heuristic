test:
	/home/justinnhli/bin/python3 clusterun.py \
		--num-old-labels=20 \
		--num-new-labels=50 \
		--dataset-size=1000 \
		--random-seed-index=1 \
		'/home/justinnhli/bin/python3 colors.py --num-old-labels="$$num_old_labels" --num-new-labels="$$num_new_labels" --dataset-size="$$dataset_size" --random-seed-index="$$random_seed_index"'

color-dataset:
	/home/justinnhli/bin/python3 clusterun.py \
		--num-old-labels=[10, 20, 50, 100, 200] \
		--num-new-labels=[20, 50, 100, 200] \
		--dataset-size=[1000, 10000, 100000, 1000000] \
		--random-seed-index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] \
		'/home/justinnhli/bin/python3 colors.py --num-old-labels="$$num_old_labels" --num-new-labels="$$num_new_labels" --dataset-size="$$dataset_size" --random-seed-index="$$random_seed_index"'
