for i in `seq 1 10`
do
	inkscape max_cluster_"$i".svg --export-area-page --batch-process --export-type=pdf --export-filename=max_cluster_"$i".pdf
done