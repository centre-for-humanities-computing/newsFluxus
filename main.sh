#!/usr/bin/env bash

start=`date +%s`
echo NEWSFULUX PIPELINE INIT
while true;do echo -n '>';sleep 1;done &

python src/bow_mdl.py --dataset dat/sample.ndjson --language da --bytestore 100 --sourcename sample --verbose 100
python src/signal_extraction.py --model mdl/da_sample_model.pcl

kill $!; trap 'kill $!' SIGTERM
echo
echo ':)'

end=`date +%s`
runtime=$((end-start))
echo $runtime