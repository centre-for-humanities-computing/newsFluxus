#!/usr/bin/env bash

start=`date +%s`
echo NEWSFULUX PIPELINE INIT
while true;do echo -n '>';sleep 1;done &

# representation learning
python src/bow_mdl.py\
    --dataset dat/sample.ndjson\
    --language da\
    --bytestore 100\
    --estimate "10 250 10"\
    --sourcename sample\
    --verbose 100

python src/signal_extraction.py\
    --model mdl/da_sample_model.pcl\
    --window=7

# application example
#python src/news_uncertainty.py\
#    --dataset mdl/da_sample_signal.json\
#    --window 7

kill $!; trap 'kill $!' SIGTERM
echo
echo ':)'

end=`date +%s`
runtime=$((end-start))
echo "Total time:" $runtime "sec"