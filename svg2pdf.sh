for f in `ls *.svg`
do
	inkscape "$f" --export-area-page --batch-process --export-type=pdf --export-filename=`$(basename -- "$f")`.pdf
done